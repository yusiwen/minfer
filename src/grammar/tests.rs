//! `#[cfg(test)] mod tests` for `src/grammar.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use serde_json::json;

/// A synthetic vocabulary: every single byte, a few multi-byte / multi-char
/// pieces, and two end-of-generation ids plus one empty piece. With every
/// byte present, any JSON text can be driven token by token.
fn byte_vocab() -> (Vec<Option<Box<[u8]>>>, Vec<bool>) {
    let mut pieces: Vec<Option<Box<[u8]>>> = Vec::new();
    for b in 0..=255u16 {
        pieces.push(Some(vec![b as u8].into_boxed_slice()));
    }
    // Extra multi-character / multi-byte pieces and the special ids. The
    // 中 piece is the 3 raw UTF-8 bytes, and the two prefixes exercise the
    // carried partial sequence; `b"ac"` exercises the longest-prefix path.
    for extra in [
        &b"true"[..],
        b"false",
        b"null",
        b"{\"",
        b"\":",
        b"}",
        b"[",
        b"]",
        b"\"a\"",
        b"ac",
        b"ab",
        b"abc",
        "\u{4e2d}".as_bytes(),
        &[0xE4u8, 0xB8],
        b"<eos>",
        b"<|im_end|>",
    ] {
        pieces.push(Some(extra.to_vec().into_boxed_slice()));
    }
    pieces.push(None); // an empty piece: never allowed
    let n = pieces.len();
    let mut eog = vec![false; n];
    eog[id_of(&pieces, b"<eos>") as usize] = true;
    eog[id_of(&pieces, b"<|im_end|>") as usize] = true;
    (pieces, eog)
}

fn id_of(pieces: &[Option<Box<[u8]>>], piece: &[u8]) -> u32 {
    pieces
        .iter()
        .position(|p| p.as_deref() == Some(piece))
        .expect("piece is in the synthetic vocabulary") as u32
}

fn gbnf(src: &str) -> Grammar {
    let (pieces, eog) = byte_vocab();
    Grammar::from_gbnf(src, pieces, eog).unwrap_or_else(|e| panic!("compiling `{src}` failed: {e}"))
}

fn schema(s: Value) -> Grammar {
    let (pieces, eog) = byte_vocab();
    Grammar::from_json_schema(&s, pieces, eog)
        .unwrap_or_else(|e| panic!("compiling schema {s} failed: {e}"))
}

fn schema_err(s: Value) -> String {
    let (pieces, eog) = byte_vocab();
    Grammar::from_json_schema(&s, pieces, eog).expect_err("schema must be refused")
}

fn allowed(mask: &[u64], id: u32) -> bool {
    mask[(id / 64) as usize] & (1u64 << (id % 64)) != 0
}

fn mask_of(g: &Grammar, st: &mut GrammarState) -> Arc<[u64]> {
    g.mask(st).expect("mask")
}

// --- GBNF ---------------------------------------------------------------

#[test]
fn gbnf_literals_classes_and_dot() {
    let g = gbnf(r#"root ::= "a" [b-d] [^x-z] ."#);
    assert!(g.accepts(b"abcQ"), "grammar:\n{}", g.source());
    assert!(g.accepts("ade\u{00e9}".as_bytes()));
    assert!(!g.accepts(b"axcQ"), "x is excluded by [^x-z]");
    assert!(!g.accepts(b"ab"), "one codepoint short");
    assert!(!g.accepts(b"abcQ\n"), "the trailing byte is not allowed");
}

#[test]
fn gbnf_any_dot_matches_newline_and_multibyte() {
    let g = gbnf("root ::= .");
    assert!(g.accepts(b"\n"));
    assert!(g.accepts("\u{4e2d}".as_bytes()));
    assert!(!g.accepts(b"ab"));
    assert!(
        !g.accepts(&[0xE4]),
        "an incomplete character is a prefix, not a sentence"
    );
    assert!(g.accepts_prefix(&[0xE4]));
}

#[test]
fn gbnf_repetition_ranges() {
    let g = gbnf(r#"root ::= "a"{2,4}"#);
    assert!(!g.accepts(b"a"));
    assert!(g.accepts(b"aa"));
    assert!(g.accepts(b"aaaa"));
    assert!(!g.accepts(b"aaaaa"));

    let g = gbnf(r#"root ::= "a"{2,}"#);
    assert!(!g.accepts(b"a"));
    assert!(g.accepts(b"aa"));
    assert!(g.accepts(b"aaaaaaaaaaaaaaaaaaaa"));

    let g = gbnf(r#"root ::= "ab"+"c"?"#);
    assert!(g.accepts(b"abc"));
    assert!(g.accepts(b"abab"));
    assert!(!g.accepts(b"a"));

    let g = gbnf(r#"root ::= "a"{0}"#);
    assert!(g.accepts(b""));
    assert!(!g.accepts(b"a"));
}

#[test]
fn gbnf_alternation_grouping_and_refs() {
    let g = gbnf(
        r#"
        root ::= greeting (" " name)?
        greeting ::= "hi" | "yo"
        name ::= [A-Z][a-z]+
        "#,
    );
    assert!(g.accepts(b"hi"));
    assert!(g.accepts(b"yo Bob"));
    assert!(!g.accepts(b"hey"));
    assert!(
        !g.accepts(b"hi bob"),
        "name needs an uppercase first letter"
    );
}

#[test]
fn gbnf_comments_and_whitespace() {
    let g = gbnf("# leading comment\nroot  ::=  \"a\" # trailing\n   \"b\"\n");
    assert!(g.accepts(b"ab"));
}

#[test]
fn gbnf_escapes() {
    let g = gbnf(r#"root ::= "\n" "\t" "\\" "\"" "\x41" "\u4e2d" [\]\-]+"#);
    let mut text = Vec::new();
    text.extend_from_slice(b"\n\t\\\"A");
    text.extend_from_slice("\u{4e2d}".as_bytes());
    text.extend_from_slice(b"]-");
    assert!(g.accepts(&text), "grammar:\n{}", g.source());
}

#[test]
fn gbnf_refuses_unsupported_or_malformed_constructs() {
    let (pieces, eog) = byte_vocab();
    let cases: &[(&str, &str)] = &[
        (r#"root ::= [\d]"#, "unsupported escape"),
        (r#"root ::= "\p{L}""#, "unsupported escape"),
        (r#"root ::= "a" "unterminated"#, "unterminated string"),
        (r#"root ::= []"#, "empty character class"),
        (r#"root ::= [z-a]"#, "reversed"),
        (r#"root ::= "a"{3,2}"#, "upper bound below"),
        (r#"root ::= "a"{1,2000}"#, "exceeds the supported limit"),
        (r#"root ::= "a" ("#, "missing `)`"),
        (r#"root ::= missing"#, "undefined rule"),
        (
            r#"root ::= "a"
            root ::= "b""#,
            "duplicate rule",
        ),
        (r#"other ::= "a""#, "no rule named `root`"),
    ];
    for (src, needle) in cases {
        let err = Grammar::from_gbnf(src, pieces.clone(), eog.clone())
            .err()
            .unwrap_or_else(|| panic!("`{src}` should be refused"));
        if !needle.is_empty() {
            assert!(
                err.contains(needle),
                "`{src}` -> `{err}` (wanted `{needle}`)"
            );
        }
    }
    // Trailing whitespace/comments after the last rule are not a rule.
    assert!(Grammar::from_gbnf("root ::= \"a\"\n# done\n", pieces, eog).is_ok());
}

#[test]
fn gbnf_detects_left_recursion() {
    let (pieces, eog) = byte_vocab();
    let err = Grammar::from_gbnf("root ::= root \"a\"\n", pieces, eog)
        .expect_err("left recursion must be refused");
    assert!(err.contains("left recursion"), "{err}");
}

// --- JSON Schema --------------------------------------------------------

#[test]
fn json_object_required_properties_and_closed_shape() {
    let g = schema(json!({
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": false
    }));
    for ok in [
        &br#"{"a":"x"}"#[..],
        br#"{"a":"x","b":1}"#,
        br#"{ "a" : "x" , "b" : -3 }"#,
    ] {
        assert!(
            g.accepts(ok),
            "{} must parse under\n{}",
            String::from_utf8_lossy(ok),
            g.source()
        );
    }
    for bad in [
        &br#"{}"#[..],
        br#"{"b":1}"#,
        br#"{"a":1}"#,
        br#"{"a":"x","c":2}"#,
        br#"{"a":"x","b":1,}"#,
        br#"{"b":1,"a":"x"}"#,
        br#"[1]"#,
    ] {
        assert!(
            !g.accepts(bad),
            "{} must NOT parse under\n{}",
            String::from_utf8_lossy(bad),
            g.source()
        );
    }
}

#[test]
fn json_object_additional_properties_forms() {
    // Open by default: extras allowed anywhere a required member is not pending.
    let open = schema(json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"]
    }));
    assert!(open.accepts(br#"{"a":1}"#));
    assert!(open.accepts(br#"{"a":1,"x":[1,2]}"#));
    assert!(
        !open.accepts(br#"{"x":true,"a":1}"#),
        "an extra may not precede a required declared member (declaration order)"
    );
    assert!(!open.accepts(br#"{}"#));

    // A schema-valued additionalProperties.
    let typed = schema(json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": {"type": "boolean"}
    }));
    assert!(typed.accepts(br#"{"a":1,"x":true}"#));
    assert!(!typed.accepts(br#"{"a":1,"x":3}"#));
    assert!(typed.accepts(br#"{}"#), "nothing is required here");
}

#[test]
fn json_array_items_and_length_bounds() {
    let g = schema(json!({
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 3
    }));
    assert!(!g.accepts(b"[]"));
    assert!(!g.accepts(b"[1]"));
    assert!(g.accepts(b"[1,2]"));
    assert!(g.accepts(b"[1, 2 , 3]"));
    assert!(!g.accepts(b"[1,2,3,4]"));
    assert!(!g.accepts(br#"["a","b"]"#));

    let open = schema(json!({"type": "array", "items": {"type": "string"}}));
    assert!(open.accepts(b"[]"));
    assert!(open.accepts(br#"["a","b","c","d"]"#));
    assert!(!open.accepts(b"[1]"));

    let loose = schema(json!({"type": "array", "minItems": 1}));
    assert!(!loose.accepts(b"[]"));
    assert!(loose.accepts(br#"[{"deep":[1,2]},null]"#));
}

#[test]
fn json_array_prefix_items() {
    let closed = schema(json!({
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "items": false,
        "minItems": 1
    }));
    assert!(closed.accepts(br#"["a"]"#));
    assert!(closed.accepts(br#"["a",1]"#));
    assert!(!closed.accepts(b"[]"));
    assert!(!closed.accepts(b"[1]"));
    assert!(!closed.accepts(br#"["a",1,2]"#));

    let open = schema(json!({
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "items": {"type": "integer"},
        "minItems": 1,
        "maxItems": 3
    }));
    assert!(open.accepts(br#"["a"]"#));
    assert!(open.accepts(br#"["a",1,2]"#));
    assert!(!open.accepts(br#"["a",1,2,3]"#));
    assert!(!open.accepts(br#"[1]"#));
}

#[test]
fn json_scalars_and_integer_bounds() {
    let g = schema(json!({
        "type": "integer",
        "minimum": -3,
        "maximum": 5
    }));
    for ok in ["-3", "5", "0", "-1", "4"] {
        assert!(g.accepts(ok.as_bytes()), "{ok} must be accepted");
    }
    for bad in ["-4", "6", "3.5", "05", "+3", ""] {
        assert!(!g.accepts(bad.as_bytes()), "{bad} must be rejected");
    }

    let excl = schema(json!({
        "type": "integer",
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 3
    }));
    assert!(!excl.accepts(b"0"));
    assert!(excl.accepts(b"1"));
    assert!(excl.accepts(b"2"));
    assert!(!excl.accepts(b"3"));

    let string = schema(json!({"type": "string"}));
    assert!(string.accepts(br#""""#));
    assert!(string.accepts(br#""a\"b\n\u00e9""#));
    assert!(!string.accepts(br#""unterminated"#));
    assert!(
        !string.accepts(br#""raw\q""#),
        "an invalid JSON escape must be refused"
    );

    let number = schema(json!({"type": "number"}));
    assert!(number.accepts(b"-1.5e-3"));
    assert!(number.accepts(b"0"));
    assert!(!number.accepts(b"1."));

    assert!(schema(json!({"type": "boolean"})).accepts(b"true"));
    assert!(!schema(json!({"type": "boolean"})).accepts(b"1"));
    assert!(schema(json!({"type": "null"})).accepts(b"null"));

    let union = schema(json!({"type": ["string", "null"]}));
    assert!(union.accepts(br#""x""#));
    assert!(union.accepts(b"null"));
    assert!(!union.accepts(b"1"));
}

#[test]
fn json_enum_and_const() {
    let g = schema(json!({"enum": ["a", 1, null, true]}));
    for ok in [br#""a""#.as_slice(), b"1", b"null", b"true"] {
        assert!(
            g.accepts(ok),
            "{} must be accepted",
            String::from_utf8_lossy(ok)
        );
    }
    for bad in [br#""b""#.as_slice(), b"1.5", b"false", br#"["a"]"#] {
        assert!(!g.accepts(bad));
    }

    let c = schema(json!({"const": {"a": 1, "b": [true, null]}}));
    assert!(
        c.accepts(br#"{"a":1,"b":[true,null]}"#),
        "grammar:\n{}",
        c.source()
    );
    assert!(!c.accepts(br#"{"a":1,"b":[true,false]}"#));
    assert!(!c.accepts(br#"{"a":1}"#));
}

#[test]
fn json_any_of_union() {
    let g = schema(json!({
        "anyOf": [{"type": "string"}, {"type": "integer", "minimum": 0}]
    }));
    assert!(g.accepts(br#""x""#));
    assert!(g.accepts(b"7"));
    assert!(!g.accepts(b"-1"));
    assert!(!g.accepts(b"1.5"));
    assert!(!g.accepts(b"true"));

    let one = schema(json!({"oneOf": [{"const": 1}, {"const": 2}]}));
    assert!(one.accepts(b"1"));
    assert!(one.accepts(b"2"));
    assert!(!one.accepts(b"3"));
}

#[test]
fn json_defs_ref_and_recursion() {
    let g = schema(json!({
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "v": {"type": "integer"},
                    "next": {"$ref": "#/$defs/node"}
                },
                "required": ["v"],
                "additionalProperties": false
            }
        },
        "$ref": "#/$defs/node"
    }));
    assert!(g.accepts(br#"{"v":1}"#), "grammar:\n{}", g.source());
    assert!(g.accepts(br#"{"next":{"next":{"v":3},"v":2},"v":1}"#));
    assert!(!g.accepts(br#"{"next":{"v":1}}"#));
    assert!(!g.accepts(br#"{"next":{},"v":1}"#));
    assert!(!g.accepts(br#"{"next":{"w":2},"v":1}"#));
}

#[test]
fn json_schema_refusals_are_named() {
    let cases: &[(Value, &str)] = &[
        (json!({"type": "string", "pattern": "^a"}), "pattern"),
        (json!({"type": "string", "minLength": 2}), "minLength"),
        (json!({"allOf": [{"type": "string"}]}), "allOf"),
        (json!({"not": {"type": "string"}}), "not"),
        (
            json!({"if": {"type": "string"}, "then": {"minLength": 1}}),
            "if",
        ),
        (json!({"type": "number", "minimum": 0}), "numeric bounds"),
        (json!({"type": "integer", "minimum": 0.5}), "integer-valued"),
        (json!({"$ref": "https://example.com/schema.json"}), "$ref"),
        (json!({"$ref": "#/$defs/missing"}), "does not resolve"),
        (
            json!({"type": "array", "items": [{"type": "string"}]}),
            "draft-07",
        ),
        (
            json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
            "required",
        ),
        (json!({"enum": []}), "empty `enum`"),
        (
            json!({"type": "array", "minItems": 3, "maxItems": 1}),
            "exceeds `maxItems`",
        ),
        (
            json!({"type": "string", "properties": {}}),
            "object keywords",
        ),
        (
            json!({"type": "integer", "minimum": 5, "maximum": 1}),
            "admit no value",
        ),
        (json!({"const": 1, "minimum": 0}), "combined with"),
        (
            json!({"type": "integer", "minimum": 0, "exclusiveMinimum": 1}),
            "both",
        ),
    ];
    for (s, needle) in cases {
        let err = schema_err(s.clone());
        assert!(
            err.contains(needle),
            "schema {s} -> `{err}` (wanted `{needle}`)"
        );
    }
}

// --- Token advancement --------------------------------------------------

#[test]
fn token_advancement_handles_partial_utf8() {
    let (pieces, eog) = byte_vocab();
    let g = Grammar::from_gbnf("root ::= \"\u{4e2d}\"", pieces.clone(), eog).unwrap();
    let mut st = g.state();

    // Byte at a time: E4, B8, AD, with the partial sequence carried.
    let e4 = id_of(&pieces, &[0xE4]);
    let b8 = id_of(&pieces, &[0xB8]);
    let ad = id_of(&pieces, &[0xAD]);
    let whole = id_of(&pieces, "\u{4e2d}".as_bytes());
    let eos = id_of(&pieces, b"<eos>");

    let mask = mask_of(&g, &mut st);
    assert!(
        allowed(&mask, e4),
        "the first byte of the character must be allowed"
    );
    assert!(allowed(&mask, whole), "the whole character must be allowed");
    assert!(
        !allowed(&mask, id_of(&pieces, &[0x80])),
        "an invalid UTF-8 byte must never be allowed"
    );
    assert!(
        !allowed(&mask, eos),
        "EOG is not legal before the character"
    );

    g.accept_token(&mut st, e4).unwrap();
    assert_eq!(st.pending_bytes(), 1, "E4 is carried");
    g.accept_token(&mut st, b8).unwrap();
    assert_eq!(st.pending_bytes(), 2);
    g.accept_token(&mut st, ad).unwrap();
    assert_eq!(st.pending_bytes(), 0);
    assert!(st.is_accepting(), "the complete character ends the grammar");

    // The same character in one token, from a fresh state.
    let mut st = g.state();
    g.accept_token(&mut st, whole).unwrap();
    assert!(st.is_accepting());

    // A two-byte prefix piece is allowed and carried, then completed.
    let mut st = g.state();
    g.accept_token(&mut st, id_of(&pieces, &[0xE4, 0xB8]))
        .unwrap();
    assert_eq!(st.pending_bytes(), 2);
    g.accept_token(&mut st, ad).unwrap();
    assert!(st.is_accepting());
}

#[test]
fn token_advancement_rejects_invalid_utf8() {
    let (pieces, eog) = byte_vocab();
    let g = Grammar::from_gbnf("root ::= .", pieces.clone(), eog).unwrap();
    let mut st = g.state();
    let stray = id_of(&pieces, &[0x80]);
    let mask = mask_of(&g, &mut st);
    assert!(!allowed(&mask, stray));
    let err = g.accept_token(&mut st, stray).expect_err("must be refused");
    assert!(err.contains("invalid UTF-8"), "{err}");
    assert_eq!(
        st.pending_bytes(),
        0,
        "a refused token must not mutate the state"
    );
}

/// A partial-UTF-8 token is only legal when the state can actually consume a
/// character that the pending bytes can complete to. This is the gate that a
/// real-model run tripped: after a JSON object closed, a lone 0xE4 leader was
/// accepted and stranded the run (the response then ended with U+FFFD).
#[test]
fn partial_utf8_is_only_allowed_when_it_can_still_complete() {
    let (pieces, eog) = byte_vocab();
    let e4 = id_of(&pieces, &[0xE4]);

    // `root ::= "a"`: U+0061 cannot start with 0xE4 (which covers U+4000..U+4FFF).
    let g = Grammar::from_gbnf("root ::= \"a\"", pieces.clone(), eog.clone()).unwrap();
    let mut st = g.state();
    assert!(!allowed(&mask_of(&g, &mut st), e4));
    assert!(g.accept_token(&mut st, e4).is_err());

    // `.` accepts any codepoint, so the leader is a legal prefix.
    let g = Grammar::from_gbnf("root ::= .", pieces.clone(), eog.clone()).unwrap();
    let mut st = g.state();
    assert!(allowed(&mask_of(&g, &mut st), e4));

    // A class that contains U+4E2D (the completed character) accepts the leader.
    let g = Grammar::from_gbnf(r"root ::= [\u4e00-\u4e2f]", pieces.clone(), eog.clone()).unwrap();
    let mut st = g.state();
    assert!(allowed(&mask_of(&g, &mut st), e4));

    // A class whose codepoints are two-byte encoded does not.
    let g = Grammar::from_gbnf(r"root ::= [\u0100-\u0200]", pieces.clone(), eog.clone()).unwrap();
    let mut st = g.state();
    assert!(!allowed(&mask_of(&g, &mut st), e4));

    // A JSON string's `[^"\\\u0000-\u001f]` still accepts a multi-byte
    // character, so the leader is legal inside a string.
    let g = Grammar::from_json_schema(&json!({"type": "string"}), pieces.clone(), eog).unwrap();
    let mut st = g.state();
    g.accept_token(&mut st, id_of(&pieces, b"\"")).unwrap();
    assert!(allowed(&mask_of(&g, &mut st), e4));
}

#[test]
fn completion_ranges_are_exact() {
    assert_eq!(completion_range(&[0xE4]), Some((0x4000, 0x4FFF)));
    assert_eq!(completion_range(&[0xE4, 0xB8]), Some((0x4E00, 0x4E3F)));
    assert_eq!(completion_range(&[0x41]), None, "a complete ASCII byte");
    assert_eq!(completion_range(&[0x80]), None, "a stray continuation byte");
    assert_eq!(
        completion_range(&[0xE4, 0xB8, 0xAD]),
        None,
        "already complete"
    );
    assert_eq!(covers(&[(0x00, 0x1F), (0x22, 0x22)], 0x00, 0x1F), true);
    assert_eq!(covers(&[(0x00, 0x1F), (0x22, 0x22)], 0x00, 0x22), false);
    assert_eq!(non_surrogate_parts(0xD000, 0xE100).len(), 2);

    // Overlong forms and the special leaders: only the *valid* completions
    // count. A 3-byte form of U+0061 (0xE0 0x80 ...) is not valid UTF-8, so
    // `[0xE0]` must not be treated as a way to reach ASCII.
    assert_eq!(completion_range(&[0xC0]), None, "overlong 2-byte leader");
    assert_eq!(completion_range(&[0xC1]), None, "overlong 2-byte leader");
    assert_eq!(completion_range(&[0xC2]), Some((0x80, 0xBF)));
    assert_eq!(completion_range(&[0xE0]), Some((0x800, 0xFFF)));
    assert_eq!(
        completion_range(&[0xE0, 0x80]),
        None,
        "overlong first continuation"
    );
    assert_eq!(completion_range(&[0xED]), Some((0xD000, 0xD7FF)));
    assert_eq!(completion_range(&[0xED, 0xA0]), None, "surrogate block");
    assert_eq!(completion_range(&[0xF0]), Some((0x10000, 0x3FFFF)));
    assert_eq!(
        completion_range(&[0xF0, 0x80]),
        None,
        "overlong first continuation"
    );
    assert_eq!(completion_range(&[0xF4]), Some((0x100000, 0x10FFFF)));
    assert_eq!(completion_range(&[0xF5]), None, "above U+10FFFF");
    assert_eq!(
        completion_range(&[0xE4, 0xB8, 0x41]),
        None,
        "bad continuation"
    );

    // U+0061 ('a') is not in [0x800, 0xFFF], so an 0xE0 leader can never
    // reach it: the mask must reject that token (the live-server defect).
    let (lo, hi) = completion_range(&[0xE0]).unwrap();
    assert!(!(lo..=hi).contains(&0x61));
}

#[test]
fn accept_reports_the_longest_accepted_prefix() {
    let (pieces, eog) = byte_vocab();
    let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
    let mut st = g.state();
    let partial = id_of(&pieces, b"ac");
    let err = g
        .accept_token(&mut st, partial)
        .expect_err("'ac' must be refused");
    assert!(
        err.contains("1 byte(s)"),
        "the error must name the longest accepted prefix: {err}"
    );
    assert!(g.accepts_prefix(b"a"));
    assert!(!g.accepts_prefix(b"ac"));
}

#[test]
fn mask_is_empty_when_nothing_can_continue() {
    // A vocabulary without 'b' cannot complete `"ab"` after 'a'.
    let pieces = vec![Some(b"a".to_vec().into_boxed_slice())];
    let eog = vec![false];
    let g = Grammar::from_gbnf("root ::= \"ab\"", pieces, eog).unwrap();
    let mut st = g.state();
    g.accept_token(&mut st, 0).unwrap();
    let mask = mask_of(&g, &mut st);
    assert_eq!(mask.iter().map(|w| w.count_ones()).sum::<u32>(), 0);
}

#[test]
fn eog_is_allowed_only_at_an_accepting_state() {
    let (pieces, eog) = byte_vocab();
    let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
    let eos = id_of(&pieces, b"<eos>");
    let im_end = id_of(&pieces, b"<|im_end|>");
    let empty = (0..pieces.len() as u32)
        .find(|id| pieces[*id as usize].is_none())
        .expect("the empty-piece token is in the vocabulary");

    let mut st = g.state();
    let mask = mask_of(&g, &mut st);
    assert!(!allowed(&mask, eos));
    assert!(!allowed(&mask, empty), "an empty piece is never a member");
    g.accept_token(&mut st, id_of(&pieces, b"a")).unwrap();
    let mask = mask_of(&g, &mut st);
    assert!(!allowed(&mask, eos), "not accepting yet");
    g.accept_token(&mut st, id_of(&pieces, b"b")).unwrap();
    assert!(st.is_accepting());
    let mask = mask_of(&g, &mut st);
    assert!(allowed(&mask, eos));
    assert!(allowed(&mask, im_end));
    g.accept_token(&mut st, eos).unwrap();
    let mask = mask_of(&g, &mut st);
    assert!(
        allowed(&mask, eos),
        "EOG stays legal once generation has ended"
    );
    assert!(
        !allowed(&mask, id_of(&pieces, b"a")),
        "nothing else is legal after EOG"
    );
    g.accept_token(&mut st, im_end)
        .expect_err("a second EOG must be refused");
}

#[test]
fn mask_is_cached_per_state() {
    let (pieces, eog) = byte_vocab();
    let g = Grammar::from_gbnf("root ::= \"ab\"", pieces.clone(), eog).unwrap();
    let mut st = g.state();
    let _ = mask_of(&g, &mut st);
    assert_eq!(st.cached_states(), 1);
    let _ = mask_of(&g, &mut st);
    assert_eq!(st.cached_states(), 1, "the same state must reuse its mask");
    g.accept_token(&mut st, id_of(&pieces, b"a")).unwrap();
    let _ = mask_of(&g, &mut st);
    assert_eq!(st.cached_states(), 2, "a new state computes a new mask");
}

// --- Integer ranges -----------------------------------------------------

#[test]
fn integer_range_expressions_match_their_bounds() {
    let ranges: &[(i128, i128)] = &[
        (0, 0),
        (0, 9),
        (5, 17),
        (10, 10),
        (99, 102),
        (0, 1000),
        (-5, -1),
        (-100, -99),
        (-5, 5),
        (-3, 0),
        (250, 250),
    ];
    let (pieces, eog) = byte_vocab();
    for &(lo, hi) in ranges {
        let expr = int_range_expr(lo, hi).expect("expression");
        let g = Grammar::from_gbnf(&format!("root ::= {expr}"), pieces.clone(), eog.clone())
            .unwrap_or_else(|e| panic!("[{lo},{hi}] `{expr}` failed to compile: {e}"));
        for v in [lo, hi, (lo + hi) / 2] {
            assert!(
                g.accepts(v.to_string().as_bytes()),
                "[{lo},{hi}] must accept {v}\ngrammar:\n{}",
                g.source()
            );
        }
        for v in [lo - 1, hi + 1] {
            assert!(
                !g.accepts(v.to_string().as_bytes()),
                "[{lo},{hi}] must reject {v}\ngrammar:\n{}",
                g.source()
            );
        }
    }
    assert!(int_range_expr(3, 2).is_err(), "an empty range is refused");
}

#[test]
fn generated_schema_grammar_is_reparseable() {
    // The schema front end emits GBNF; compiling it twice must give the same
    // language (and proves the emitted text is in the supported subset).
    let src = json!({
        "type": "object",
        "properties": {
            "when": {"type": "integer", "minimum": 1900, "maximum": 2100},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "kind": {"enum": ["a", "b"]}
        },
        "required": ["when", "kind"],
        "additionalProperties": false
    });
    let g = schema(src.clone());
    let again = gbnf(g.source());
    for text in [
        &br#"{"kind":"a","when":2024}"#[..],
        br#"{"kind":"b","tags":["x","y"],"when":1900}"#,
    ] {
        assert!(g.accepts(text));
        assert!(again.accepts(text), "round-trip must preserve the language");
    }
    for text in [&br#"{"kind":"a","when":1899}"#[..], br#"{"kind":"a"}"#] {
        assert!(!g.accepts(text));
        assert!(!again.accepts(text));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `grammar.rs` (bucket B of the dead-code census — the
// only test callers live in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl GrammarState {
    /// Bytes of an incomplete UTF-8 character carried across the last token.
    ///
    /// Test-only (#239): driven by
    /// `grammar::tests::{token_advancement_handles_partial_utf8,
    /// token_advancement_rejects_invalid_utf8}`.
    pub fn pending_bytes(&self) -> usize {
        self.inner.partial.len()
    }
}
