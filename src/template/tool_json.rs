//! A JSON value that remembers the key order it was parsed from (#597).
//!
//! A request's `tools` array reaches the model through the template's
//! `| tojson`, so its **bytes** are part of the prompt. transformers renders the
//! client's own key order (`json.dumps` of the dict it was given), while the
//! engine's JSON type is a `BTreeMap` and sorts keys — `{"type": …, "function":
//! …}` would come out as `{"function": …, "type": …}` and every prompt carrying
//! tools would drift from the reference.
//!
//! Two alternatives were rejected:
//!
//! * `serde_json`'s `preserve_order` feature: it is process-wide, and the
//!   JSON-Schema→GBNF compiler iterates a schema's map, so the generated grammar
//!   (and the language it accepts) would move with a render-only decision.
//! * echoing the raw request text: `| tojson` re-serializes a *parsed* value, so
//!   a raw slice cannot serve a template that indexes into the tools.
//!
//! So the order lives in the value: `serde`'s `MapAccess` yields entries in
//! document order, and both `Serialize` (which the template's `tojson` consumes)
//! and the accessors below walk that order.
//!
//! Number formatting is `serde_json`'s, deliberately: a value that reaches the
//! prompt has been parsed by CPython too, and for the forms a tool definition
//! uses (integers, short decimals) the two agree. An exponent-form float can
//! still differ in spelling (`1e30` here, `1e+30` in CPython) — recorded as a
//! known limit in the design record rather than hidden.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::fmt;

/// One JSON value, object entries in the order they were written.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolJson {
    Null,
    Bool(bool),
    /// A number as `serde_json` parsed it, re-serialized by `serde_json`'s own
    /// formatter so digits are never reformatted by hand.
    Number(serde_json::Number),
    String(String),
    Array(Vec<ToolJson>),
    Object(Vec<(String, ToolJson)>),
}

impl ToolJson {
    /// The object's entries in document order, or `None` for a non-object.
    pub fn as_object(&self) -> Option<&[(String, ToolJson)]> {
        match self {
            ToolJson::Object(entries) => Some(entries),
            _ => None,
        }
    }

    /// A field of an object, in O(n) over its own entries — an object small
    /// enough to reach a prompt is not worth an index.
    pub fn get(&self, key: &str) -> Option<&ToolJson> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ToolJson::String(s) => Some(s),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for ToolJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ToolJsonVisitor)
    }
}

struct ToolJsonVisitor;

impl<'de> Visitor<'de> for ToolJsonVisitor {
    type Value = ToolJson;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<ToolJson, E> {
        Ok(ToolJson::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<ToolJson, E> {
        Ok(ToolJson::Number(v.into()))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<ToolJson, E> {
        Ok(ToolJson::Number(v.into()))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<ToolJson, E> {
        // JSON cannot spell a non-finite number, so this only fires for a
        // deserializer that hands one over anyway.
        serde_json::Number::from_f64(v)
            .map(ToolJson::Number)
            .ok_or_else(|| E::custom("a non-finite number is not JSON"))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<ToolJson, E> {
        Ok(ToolJson::String(v.to_string()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<ToolJson, E> {
        Ok(ToolJson::String(v))
    }

    fn visit_none<E: de::Error>(self) -> Result<ToolJson, E> {
        Ok(ToolJson::Null)
    }

    fn visit_unit<E: de::Error>(self) -> Result<ToolJson, E> {
        Ok(ToolJson::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<ToolJson, D::Error> {
        ToolJson::deserialize(d)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<ToolJson, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(ToolJson::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ToolJson, A::Error> {
        let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
        // `MapAccess` yields entries in document order; this `Vec` is what keeps
        // that order alive all the way into `| tojson`.
        while let Some((key, value)) = map.next_entry::<String, ToolJson>()? {
            entries.push((key, value));
        }
        Ok(ToolJson::Object(entries))
    }
}

impl Serialize for ToolJson {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ToolJson::Null => s.serialize_unit(),
            ToolJson::Bool(b) => s.serialize_bool(*b),
            ToolJson::Number(n) => n.serialize(s),
            ToolJson::String(v) => s.serialize_str(v),
            ToolJson::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            ToolJson::Object(entries) => {
                let mut map = s.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}
