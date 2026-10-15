//! `#[cfg(test)] mod tests` for `src/tokenizer/nfc.rs` — the differential gate
//! against CPython, plus readable witnesses for each algorithm step.
use super::*;

/// The committed corpus is generated, so a truncated or emptied file must fail
/// loudly instead of shrinking the gate. The count is also recorded in the
/// fixture's `sections`, and asserted per section below.
const MIN_ENTRIES: usize = 5000;

fn corpus() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/tokenizer/nfc_vectors.json"
    ))
    .expect("the committed NFC vector fixture parses")
}

#[test]
fn nfc_matches_cpython_on_the_committed_corpus() {
    let doc = corpus();
    let entries = doc["entries"].as_array().expect("entries is an array");
    assert!(
        entries.len() >= MIN_ENTRIES,
        "the NFC corpus has only {} entries (expected at least {MIN_ENTRIES})",
        entries.len()
    );

    let mut changed = 0;
    for entry in entries {
        let input = entry["input"].as_str().expect("entry.input");
        let expected = entry["nfc"].as_str().expect("entry.nfc");
        let got = nfc(input);
        assert_eq!(
            got,
            expected,
            "NFC({input:?}) with code points {:X?}",
            input.chars().map(|c| c as u32).collect::<Vec<_>>()
        );
        if expected != input {
            changed += 1;
        }
    }
    assert!(
        changed > 3000,
        "only {changed} corpus entries change under NFC; the fixture looks wrong"
    );
}

#[test]
fn nfd_matches_cpython_on_the_committed_corpus() {
    let doc = corpus();
    let entries = doc["entries"].as_array().expect("entries is an array");
    assert!(
        entries.len() >= MIN_ENTRIES,
        "the NFC corpus has only {} entries (expected at least {MIN_ENTRIES})",
        entries.len()
    );

    let mut changed = 0;
    for entry in entries {
        let input = entry["input"].as_str().expect("entry.input");
        let expected = entry["nfd"].as_str().expect("entry.nfd");
        let got = nfd(input);
        assert_eq!(
            got,
            expected,
            "NFD({input:?}) with code points {:X?}",
            input.chars().map(|c| c as u32).collect::<Vec<_>>()
        );
        if expected != input {
            changed += 1;
        }
    }
    assert!(
        changed > 1000,
        "only {changed} corpus entries change under NFD; the fixture looks wrong"
    );
}

#[test]
fn the_corpus_covers_every_table_entry() {
    let doc = corpus();
    let entries = doc["entries"].as_array().expect("entries is an array");
    let inputs: std::collections::HashSet<&str> = entries
        .iter()
        .map(|e| e["input"].as_str().expect("entry.input"))
        .collect();

    for &(cp, parts) in tables::DECOMPOSITION {
        let text = char::from_u32(cp).expect("a decomposition source is a character");
        let mut buf = [0u8; 4];
        assert!(
            inputs.contains(text.encode_utf8(&mut buf)),
            "no corpus entry for the decomposition of U+{cp:04X} ({parts:X?})"
        );
    }
    for &(a, b, composed) in tables::COMPOSITION {
        let mut pair = String::new();
        pair.push(char::from_u32(a).expect("composition source a"));
        pair.push(char::from_u32(b).expect("composition source b"));
        assert!(
            inputs.contains(pair.as_str()),
            "no corpus entry for the composition U+{a:04X} + U+{b:04X} → U+{composed:04X}"
        );
    }
}

#[test]
fn the_corpus_records_the_generated_unicode_version() {
    assert_eq!(
        corpus()["unicode_version"]
            .as_str()
            .expect("unicode_version"),
        UNICODE_VERSION,
        "the fixture and the generated tables came from different Unicode data"
    );
}

#[test]
fn already_nfc_text_is_borrowed_and_decomposed_text_is_not() {
    assert!(matches!(nfc("ASCII only"), Cow::Borrowed(_)));
    // Precomposed text still runs the pipeline (it decomposes), but comes back
    // equal and therefore borrowed.
    assert!(matches!(nfc("café"), Cow::Borrowed(_)));
    assert!(matches!(nfc("e\u{0301}"), Cow::Owned(_)));
    assert_eq!(nfc("e\u{0301}"), "é");
    // NFC is not NFKC: compatibility forms are canonical-equivalent to nothing
    // and must be left alone.
    assert!(matches!(nfc("\u{FB01}"), Cow::Borrowed(_)));
    assert!(matches!(nfc("\u{FF46}"), Cow::Borrowed(_)));
}

#[test]
fn canonical_ordering_sorts_mark_runs_stably() {
    // 230 (above) before 220 (below): the marks swap, and q has no composition.
    assert_eq!(nfc("q\u{0307}\u{0323}"), "q\u{0323}\u{0307}");
    // Equal classes keep their relative order (an unstable sort could swap these).
    assert_eq!(nfc("q\u{0307}\u{0301}"), "q\u{0307}\u{0301}");
    // Lone marks (no starter) still sort.
    assert_eq!(nfc("\u{0307}\u{0323}"), "\u{0323}\u{0307}");
}

#[test]
fn composition_respects_blocking() {
    // a + 0301 composes; the following 0302 is then blocked by the mark that
    // was consumed, and has no pair with á anyway.
    assert_eq!(nfc("a\u{0301}\u{0302}"), "\u{00E1}\u{0302}");
    // a + 0302 composes and frees 0301 to compose with the starter: U+1EA5.
    assert_eq!(nfc("a\u{0302}\u{0301}"), "\u{1EA5}");
    // A lower class between the starter and the mark does not block.
    assert_eq!(nfc("a\u{05B0}\u{0301}"), "\u{00E1}\u{05B0}");
    // The UAX #15 example.
    assert_eq!(nfc("D\u{0323}\u{0307}"), "\u{1E0C}\u{0307}");
    // Composition exclusions and singletons: U+212B is NFC-equivalent to U+00C5,
    // never the other way round.
    assert_eq!(nfc("\u{212B}"), "\u{00C5}");
    assert_eq!(nfc("\u{0041}\u{030A}"), "\u{00C5}");
    // A pair with no composition is left decomposed.
    assert_eq!(nfc("\u{0110}\u{0323}"), "\u{0110}\u{0323}");
}

#[test]
fn hangul_composes_and_decomposes_algorithmically() {
    assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}"); // L V T → 각
    assert_eq!(nfc("\u{AC01}"), "\u{AC01}");
    assert_eq!(nfc("\u{AC00}\u{11A8}"), "\u{AC01}"); // an LV syllable takes T
    assert_eq!(nfc("\u{1100}\u{1161}\u{1161}"), "\u{AC00}\u{1161}"); // a second V stays
}
