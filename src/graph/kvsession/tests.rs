//! `#[cfg(test)] mod tests` for `src/graph/kvsession.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::kvcache::{FREE, SEQ_MAIN};

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "minfer-kvsession-{}-{name}.bin",
        std::process::id()
    ));
    p
}

fn header(n_layer: usize, n_ctx: usize, n_embd: usize, format: KvFormat) -> KvSessionHeader {
    KvSessionHeader {
        version: VERSION,
        format,
        backend: Backend::CPU,
        n_layer,
        n_ctx,
        n_embd,
        row_elems: format.row_elems(n_embd),
    }
}

fn state(n_ctx: usize) -> KvSessionState {
    KvSessionState {
        identity: true,
        n_ctx,
        seqs: vec![(
            SEQ_MAIN,
            SeqSlot {
                start: 0,
                cap: n_ctx,
                shared: SharedPrefix::default(),
                written: 3,
            },
        )],
        spans: vec![(SEQ_MAIN, vec![(0, 0, 3)])],
        layers: vec![
            crate::graph::kvcache::KvLayerSession {
                layer: 0,
                n_used: 3,
                owner: {
                    let mut o = vec![FREE; n_ctx];
                    o[0] = SEQ_MAIN;
                    o[1] = SEQ_MAIN;
                    o[2] = SEQ_MAIN;
                    o
                },
            },
            crate::graph::kvcache::KvLayerSession {
                layer: 1,
                n_used: 3,
                owner: vec![SEQ_MAIN; n_ctx],
            },
        ],
        defrags: 1,
        cells_moved: 7,
        cows: 2,
        cow_cells: 5,
    }
}

fn write_session(path: &Path, h: &KvSessionHeader, st: &KvSessionState) -> KvSessionReport {
    write_session_with_host(path, h, st, &[])
}

fn write_session_with_host(
    path: &Path,
    h: &KvSessionHeader,
    st: &KvSessionState,
    host: &[u8],
) -> KvSessionReport {
    let mut w = KvSessionWriter::create(path, h).expect("create");
    w.set_host(host);
    for layer in 0..h.n_layer {
        let k: Vec<f32> = (0..h.layer_words())
            .map(|i| (i as f32) * 0.5 + layer as f32)
            .collect();
        let v: Vec<f32> = (0..h.layer_words())
            .map(|i| 1.0 / (i as f32 + 1.0))
            .collect();
        w.layer(layer, &k, &v).expect("layer");
    }
    w.finish(st).expect("finish")
}

#[test]
fn a_session_round_trips_including_the_run_table() {
    let path = temp_path("roundtrip");
    let h = header(2, 16, 32, KvFormat::F32);
    let st = state(16);
    let report = write_session(&path, &h, &st);
    assert_eq!(report.layers, 2);
    assert_eq!(report.cells, 16);
    assert_eq!(report.written, 3);
    assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());

    let mut r = KvSessionReader::open(&path).expect("open");
    assert_eq!(r.header(), &h);
    let mut seen = Vec::new();
    while let Some((layer, k, v)) = r.next_layer().expect("layer") {
        assert_eq!(k.len(), h.layer_words());
        assert_eq!(v.len(), h.layer_words());
        assert_eq!(k[1], 0.5 + layer as f32);
        seen.push(layer);
    }
    assert_eq!(seen, vec![0, 1]);
    let body = r.finish().expect("finish");
    assert_eq!(body.state, st);
    assert_eq!(body.report, report);
    assert!(body.host.is_empty(), "no host state was written");
    std::fs::remove_file(&path).ok();
}

/// C5 S2: the host state rides inside the container — length-prefixed, after the
/// bookkeeping and **inside the checksum**, so a flipped byte in it is refused
/// exactly like one in the KV rows.
#[test]
fn the_host_state_round_trips_and_is_covered_by_the_checksum() {
    let path = temp_path("host");
    let h = header(1, 8, 32, KvFormat::F32);
    let host = br#"{"messages":[["user","hi"]],"current_pos":3}"#.to_vec();
    let written = write_session_with_host(&path, &h, &state(8), &host);
    assert!(written.bytes > host.len() as u64, "{written:?}");

    let mut r = KvSessionReader::open(&path).expect("open");
    while r.next_layer().expect("layer").is_some() {}
    let body = r.finish().expect("finish");
    assert_eq!(
        body.host, host,
        "the host blob must round-trip byte for byte"
    );
    assert_eq!(body.state, state(8));

    // Flip a byte inside the host blob: the checksum covers it.
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes.len() - host.len() - 5; // inside the blob, before the trailing words
    bytes[at] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let err = verify(&path).unwrap_err();
    assert!(err.contains("checksum"), "{err}");
    std::fs::remove_file(&path).ok();
}

/// C5 S2 keeps version 1 readable-as-a-refusal: a v1 file has no host section, and
/// applying its rows without the host state is exactly what the bump prevents.
#[test]
fn a_version_1_file_is_refused_so_the_caller_can_re_seed() {
    let path = temp_path("v1");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = verify(&path).unwrap_err();
    assert!(
        err.contains("version 1") && err.contains(&format!("version {VERSION}")),
        "{err}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_packed_session_records_its_cell_width() {
    let path = temp_path("packed");
    let h = header(1, 8, 128, KvFormat::Q8_0);
    assert_eq!(h.row_elems, 34);
    let st = KvSessionState {
        layers: vec![crate::graph::kvcache::KvLayerSession {
            layer: 0,
            n_used: 1,
            owner: vec![SEQ_MAIN; 8],
        }],
        ..state(8)
    };
    write_session(&path, &h, &st);
    let r = KvSessionReader::open(&path).expect("open");
    assert_eq!(r.header().format, KvFormat::Q8_0);
    assert!(r.header().format.is_packed());
    std::fs::remove_file(&path).ok();
}

/// #130: an f16 arena round-trips at the container level, no device needed —
/// the container carries f32 words and the element type is a flag. Before the
/// `FLAG_F16` bit an f16 header was written as `flags == 0` and read back as
/// f32, so `kv_load` refused the file its own writer had just produced.
#[test]
fn an_f16_session_round_trips() {
    let path = temp_path("f16-roundtrip");
    let h = header(2, 16, 128, KvFormat::F16);
    let st = state(16);
    let report = write_session(&path, &h, &st);

    let mut r = KvSessionReader::open(&path).expect("open");
    assert_eq!(
        r.header(),
        &h,
        "the whole header, element type included, survives"
    );
    assert_eq!(r.header().format, KvFormat::F16);
    let mut seen = Vec::new();
    while let Some((layer, k, v)) = r.next_layer().expect("layer") {
        assert_eq!(k.len(), h.layer_words());
        assert_eq!(v.len(), h.layer_words());
        seen.push(layer);
    }
    assert_eq!(seen, vec![0, 1]);
    let body = r.finish().expect("finish");
    assert_eq!(body.state, st);
    assert_eq!(body.report, report);
    // The full-file pass the allocator runs before applying anything agrees.
    assert_eq!(verify(&path).unwrap().format, KvFormat::F16);
    std::fs::remove_file(&path).ok();
}

/// #130: the flags word **is** the element type, and every format round trips
/// through it — the encode/decode symmetry the fix rests on. If `flags_of` stops
/// writing a format's bit (`F16` → `0`, say) the reader decodes the wrong type
/// and this fails on the on-disk word, not only on the decoded header.
#[test]
fn the_flag_word_encodes_every_element_type_and_round_trips() {
    for (format, want_flags) in [
        (KvFormat::F32, 0u32),
        (KvFormat::F16, FLAG_F16),
        (KvFormat::Q8_0, FLAG_PACKED),
    ] {
        let path = temp_path(&format!("flags-{}", format.name()));
        let h = header(1, 8, 32, format);
        write_session(&path, &h, &state(8));
        // The on-disk flags word is exactly that format's encoding.
        let bytes = std::fs::read(&path).unwrap();
        let on_disk = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(on_disk, want_flags, "{} flags word", format.name());
        // And the reader decodes it back symmetrically.
        let r = KvSessionReader::open(&path).expect("open");
        assert_eq!(r.header().format, format);
        assert_eq!(r.header().format, format_of_flags(on_disk));
        std::fs::remove_file(&path).ok();
    }
}

/// #130: the two element-type bits are mutually exclusive. "packed Q8_0" and
/// "f16" are different cell layouts, so a header claiming both is corrupt: the
/// reader must refuse it, not prefer one bit and apply the payload under a
/// layout the file does not describe. The message is asserted so that the
/// *width* check (which would also catch this combination) cannot stand in for
/// the rule under test.
#[test]
fn a_header_claiming_both_packed_and_f16_is_refused() {
    let path = temp_path("both-flags");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12..16].copy_from_slice(&(FLAG_PACKED | FLAG_F16).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = KvSessionReader::open(&path).err().expect("must be refused");
    assert!(err.contains("mutually exclusive"), "{err}");
    assert!(err.contains("corrupt"), "{err}");
    std::fs::remove_file(&path).ok();
}

/// #130 keeps `FLAG_PACKED`'s property: an unknown flag bit is refused, never
/// ignored. This is also the compatibility story for a *newer* writer — a
/// pre-#130 build reading an f16 file sees `FLAG_F16` as unknown and takes
/// exactly this path instead of decoding the region as f32.
#[test]
fn an_unknown_flag_bit_is_refused() {
    let path = temp_path("unknown-flag");
    let h = header(1, 8, 32, KvFormat::F16);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    // A bit no version has assigned yet, on top of a valid f16 file.
    bytes[12..16].copy_from_slice(&(FLAG_F16 | (1 << 7)).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = KvSessionReader::open(&path).err().expect("must be refused");
    assert!(err.contains("unknown flags"), "{err}");
    std::fs::remove_file(&path).ok();
}

/// #130 compatibility: every session written before the f16 bit existed is
/// `flags == 0`, i.e. f32, and must keep loading **as f32** — a file that merely
/// predates the bit is not a newer file and must not be refused. (A literal
/// version-1 file is refused by the version check, `a_version_1_file_is_refused_…`.)
#[test]
fn a_legacy_f32_file_without_the_f16_bit_still_loads_as_f32() {
    let path = temp_path("legacy-f32");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        0,
        "a pre-#130 f32 file's flags word is zero"
    );
    let r = KvSessionReader::open(&path).expect("a legacy f32 file still loads");
    assert_eq!(r.header().format, KvFormat::F32);
    assert_eq!(verify(&path).unwrap().format, KvFormat::F32);
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_truncated_file_is_refused_and_says_so() {
    let path = temp_path("truncated");
    let h = header(2, 16, 32, KvFormat::F32);
    write_session(&path, &h, &state(16));
    let full = std::fs::read(&path).unwrap();
    // Cut in the middle of the payload: the reader must fail on a short read,
    // not stop early with a half-restored arena.
    std::fs::write(&path, &full[..full.len() / 2]).unwrap();
    let err = verify(&path).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    assert!(err.contains("the file ends"), "{err}");
    // The streaming walk fails in the same place — the header itself fits, so
    // this is the payload read refusing, which is why `kv_load` runs `verify`
    // before it applies anything.
    let mut r = KvSessionReader::open(&path).expect("the header survives a cut in the payload");
    let err = loop {
        match r.next_layer() {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("a truncated payload must not read to the end"),
            Err(e) => break e,
        }
    };
    assert!(err.contains("truncated"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_version_mismatch_is_refused_loudly() {
    let path = temp_path("version");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = KvSessionReader::open(&path).err().expect("must be refused");
    assert!(err.contains("version"), "{err}");
    assert!(err.contains(&format!("version {}", VERSION)), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_flipped_byte_is_caught_by_the_checksum() {
    let path = temp_path("checksum");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes.len() / 2; // inside the payload
    bytes[at] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let err = verify(&path).unwrap_err();
    assert!(err.contains("checksum mismatch"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn trailing_bytes_after_the_checksum_are_refused() {
    let path = temp_path("trailing");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(0);
    std::fs::write(&path, &bytes).unwrap();
    let err = verify(&path).unwrap_err();
    assert!(err.contains("trailing bytes"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_file_that_is_not_a_session_is_refused() {
    let path = temp_path("magic");
    std::fs::write(
        &path,
        b"not a session at all, but long enough to read the magic",
    )
    .unwrap();
    let err = KvSessionReader::open(&path).err().expect("must be refused");
    assert!(err.contains("magic"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_header_whose_flag_and_width_disagree_is_refused() {
    let path = temp_path("flagwidth");
    let h = header(1, 8, 32, KvFormat::F32);
    write_session(&path, &h, &state(8));
    let mut bytes = std::fs::read(&path).unwrap();
    // Claim packed cells while keeping the f32-shaped width: the applying pass
    // would then dequantize f32 rows.
    let flags = FLAG_PACKED.to_le_bytes();
    bytes[12..16].copy_from_slice(&flags);
    std::fs::write(&path, &bytes).unwrap();
    let err = KvSessionReader::open(&path).err().expect("must be refused");
    assert!(err.contains("packs a cell into"), "{err}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_writer_that_is_short_a_layer_refuses_to_finish() {
    let path = temp_path("short");
    let h = header(2, 8, 32, KvFormat::F32);
    let mut w = KvSessionWriter::create(&path, &h).expect("create");
    w.layer(0, &vec![0.0; h.layer_words()], &vec![0.0; h.layer_words()])
        .expect("layer 0");
    let err = w.finish(&state(8)).unwrap_err();
    assert!(err.contains("2 layers but 1 were written"), "{err}");
    std::fs::remove_file(&path).ok();
}
