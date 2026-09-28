//! `#[cfg(test)] mod tests` for `src/gguf_write.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::gguf::GgufContext;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("minfer-f6-writer-{name}"));
    let _ = std::fs::remove_file(&p);
    p
}

fn sample_kv() -> Vec<GgufKv> {
    vec![
        GgufKv::new_string("general.architecture".into(), "qwen2".into()),
        GgufKv::new_u32("qwen2.block_count".into(), 3),
        GgufKv::new_i32("signed".into(), -7),
        GgufKv::new_f32("eps".into(), 1e-6),
        GgufKv::new_bool("flag".into(), true),
        GgufKv::new_u64("big".into(), 1 << 40),
        GgufKv::new_u16("small".into(), 7),
        GgufKv::new_string_array("tokens".into(), vec!["a".into(), "b".into()]),
        GgufKv::new_array(
            "scores".into(),
            GgufType::Float32,
            [1.0f32, 2.0, 3.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        GgufKv::new_array(
            "types".into(),
            GgufType::Int32,
            (-2i32..2).flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ]
}

#[test]
fn every_metadata_type_round_trips_through_the_parser() {
    let path = tmp("kv.gguf");
    let specs = vec![TensorSpec::new("w", [4, 1, 1, 1], GgmlType::F32)];
    write_single(&path, &sample_kv(), specs, 32, |_, w| {
        w.write_all(&[0u8; 16])
    })
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let ctx = GgufContext::init_from_data(&bytes).unwrap();
    assert_eq!(
        ctx.get_key_val_str("general.architecture").unwrap(),
        "qwen2"
    );
    assert_eq!(ctx.get_key_val_i64("qwen2.block_count").unwrap(), 3);
    assert_eq!(ctx.get_key_val_i64("signed").unwrap(), -7);
    assert!((ctx.get_key_val_f32("eps").unwrap() - 1e-6).abs() < 1e-12);
    assert_eq!(ctx.get_key_val_i64("big").unwrap(), 1 << 40);
    assert_eq!(ctx.get_key_val_i64("small").unwrap(), 7);
    let k = ctx.find_key("tokens");
    assert_eq!(ctx.get_arr_n(k), 2);
    assert_eq!(ctx.get_arr_str(k, 1), "b");
    let s = ctx.find_key("scores");
    assert_eq!(ctx.get_arr_type(s), GgufType::Float32);
    assert_eq!(ctx.get_arr_data(s).len(), 12);
    let t = ctx.find_key("types");
    assert_eq!(ctx.get_arr_type(t), GgufType::Int32);
    assert_eq!(ctx.get_arr_n(t), 4);
    assert!(ctx.get_val_bool(ctx.find_key("flag")));
    // data section starts aligned and holds exactly the one tensor
    assert_eq!(ctx.get_data_offset() % 32, 0);
    assert_eq!(ctx.size, 32);
    assert_eq!(ctx.info[0].offset, 0);
    // The file must actually *contain* the padding the index declares.
    assert_eq!(bytes.len(), ctx.get_data_offset() + ctx.size);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tensor_index_offsets_match_the_parser_requirement() {
    let path = tmp("offsets.gguf");
    let specs = vec![
        TensorSpec::new("a", [32, 2, 1, 1], GgmlType::F32), // 256 B
        TensorSpec::new("b", [64, 1, 1, 1], GgmlType::Q4_0), // 36 B
        TensorSpec::new("c", [32, 1, 1, 1], GgmlType::F32), // 128 B
    ];
    let data = |i: usize, w: &mut dyn Write| w.write_all(&vec![i as u8; specs_len(i)]);
    write_single(&path, &sample_kv(), specs, 32, data).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let ctx = GgufContext::init_from_data(&bytes).unwrap();
    assert_eq!(ctx.info[0].offset, 0);
    assert_eq!(ctx.info[1].offset, 256);
    assert_eq!(ctx.info[2].offset, 320); // 256 + pad(36)=64
    assert_eq!(ctx.size, 320 + 128);
    // And the file must be exactly as long as that layout says. Without
    // this, a writer that declares the right offsets but writes a shorter
    // inter-tensor pad still parses (the parser never reads past the last
    // tensor), so the padding rule would be untested.
    assert_eq!(bytes.len(), ctx.get_data_offset() + ctx.size);
    let _ = std::fs::remove_file(&path);
}

fn specs_len(i: usize) -> usize {
    [256usize, 36, 128][i]
}

#[test]
fn multiple_dimensions_and_trailing_ones_round_trip() {
    let path = tmp("dims.gguf");
    let specs = vec![
        TensorSpec::new("one_d", [8, 1, 1, 1], GgmlType::F32),
        TensorSpec::new("two_d", [4, 3, 1, 1], GgmlType::F32),
    ];
    write_single(&path, &sample_kv(), specs, 32, |i, w| {
        w.write_all(&vec![0u8; if i == 0 { 32 } else { 48 }])
    })
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let ctx = GgufContext::init_from_data(&bytes).unwrap();
    assert_eq!(ctx.info[0].ne, [8, 1, 1, 1]);
    assert_eq!(ctx.info[1].ne, [4, 3, 1, 1]);
    assert_eq!(ctx.info[1].nb[0], 4);
    assert_eq!(ctx.info[1].nb[1], 16);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn split_writes_parts_the_reader_merges_back() {
    let dir = std::env::temp_dir().join("minfer-f6-writer-split");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut kv = sample_kv();
    kv.push(GgufKv::new_string(
        "tokenizer.chat_template".into(),
        "{{ x }}".into(),
    ));
    // Three 256-byte tensors with a 300-byte cap → one part per tensor.
    let specs: Vec<TensorSpec> = (0..3)
        .map(|i| TensorSpec::new(format!("t{i}"), [64, 1, 1, 1], GgmlType::F32))
        .collect();
    let parts = write_split(&dir, "sample", &kv, specs, 32, 300, |i, w| {
        w.write_all(&vec![(i + 1) as u8; 256])
    })
    .unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(
        parts[0].file_name().unwrap().to_str().unwrap(),
        "sample-00001-of-00003.gguf"
    );
    let info = crate::gguf::split_file_info(parts[0].file_name().unwrap().to_str().unwrap())
        .expect("split filename parses");
    assert_eq!(info.1, 0);
    assert_eq!(info.2, 3);
    let model = crate::gguf::load_gguf_model(&parts[0]).expect("merged load");
    assert_eq!(model.parts.len(), 3);
    let mut names: Vec<String> = Vec::new();
    for (i, part) in model.parts.iter().enumerate() {
        assert_eq!(
            part.ctx.get_key_val_i64(KEY_SPLIT_NO).map(|v| v as usize),
            Some(i)
        );
        assert_eq!(part.ctx.get_key_val_i64(KEY_SPLIT_TENSORS_COUNT), Some(3));
        assert_eq!(
            part.ctx.get_key_val_str("general.architecture").unwrap(),
            "qwen2"
        );
        // Each part is a complete file whose length matches its own index.
        assert_eq!(part.data.len(), part.ctx.get_data_offset() + part.ctx.size);
        for ti in &part.ctx.info {
            names.push(ti.name.clone());
            let base = part.ctx.get_data_offset();
            assert_eq!(
                part.data[base + ti.offset as usize],
                (i + 1) as u8,
                "part {i} tensor {} payload",
                ti.name
            );
        }
    }
    assert_eq!(names, vec!["t0", "t1", "t2"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn one_part_is_a_plain_single_file_without_split_metadata() {
    let dir = std::env::temp_dir().join("minfer-f6-writer-onepart");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let specs = vec![TensorSpec::new("t0", [16, 1, 1, 1], GgmlType::F32)];
    let paths = write_split(&dir, "solo", &sample_kv(), specs, 32, u64::MAX, |_, w| {
        w.write_all(&[1u8; 64])
    })
    .unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].file_name().unwrap().to_str().unwrap(), "solo.gguf");
    let bytes = std::fs::read(&paths[0]).unwrap();
    let ctx = GgufContext::init_from_data(&bytes).unwrap();
    assert_eq!(ctx.find_key(KEY_SPLIT_COUNT), -1);
    assert_eq!(ctx.find_key(KEY_SPLIT_NO), -1);
    // and the metadata is the source's, key for key
    assert_eq!(ctx.kv.len(), sample_kv().len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tensor_larger_than_the_cap_is_refused() {
    let specs = vec![TensorSpec::new("big", [1024, 1, 1, 1], GgmlType::F32)];
    let err = split_assignment(&specs, 32, 100).unwrap_err();
    assert!(err.contains("larger than the 100-byte part cap"), "{err}");
    assert!(err.contains("'big'"), "{err}");
}

/// A valid 2-part split in a fresh directory, plus its entry path.
fn two_part_fixture(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("minfer-f6-writer-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let specs: Vec<TensorSpec> = (0..2)
        .map(|i| TensorSpec::new(format!("t{i}"), [64, 1, 1, 1], GgmlType::F32))
        .collect();
    let parts = write_split(&dir, "m", &sample_kv(), specs, 32, 300, |i, w| {
        w.write_all(&vec![(i + 1) as u8; 256])
    })
    .unwrap();
    assert_eq!(parts.len(), 2);
    (dir, parts[0].clone())
}

#[test]
fn a_missing_part_is_refused() {
    let (dir, entry) = two_part_fixture("missing");
    let missing = dir.join("m-00002-of-00002.gguf");
    std::fs::remove_file(&missing).unwrap();
    assert!(
        crate::gguf::load_gguf_model(&entry).is_none(),
        "a split with a missing part must not load"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_part_with_the_wrong_split_index_is_refused() {
    let (dir, entry) = two_part_fixture("wrongindex");
    // Part 1 is really part 0 (split.no = 0) under the second filename.
    let src = dir.join("m-00001-of-00002.gguf");
    let dst = dir.join("m-00002-of-00002.gguf");
    std::fs::copy(&src, &dst).unwrap();
    assert!(
        crate::gguf::load_gguf_model(&entry).is_none(),
        "split.no must match the part's position"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_filename_count_that_disagrees_with_split_count_is_refused() {
    let (dir, entry) = two_part_fixture("badcount");
    // The filename claims 3 parts while split.count (and the directory) say 2.
    let renamed = dir.join("m-00001-of-00003.gguf");
    std::fs::rename(&entry, &renamed).unwrap();
    assert!(
        crate::gguf::load_gguf_model(&renamed).is_none(),
        "a count mismatch between filename and split.count must not load"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn writer_refuses_a_payload_of_the_wrong_size() {
    let path = tmp("badsize.gguf");
    let specs = vec![TensorSpec::new("w", [8, 1, 1, 1], GgmlType::F32)];
    let err = write_single(&path, &sample_kv(), specs, 32, |_, w| {
        w.write_all(&[0u8; 8])
    })
    .unwrap_err();
    assert!(err.contains("provider wrote 8 bytes, expected 32"), "{err}");
    let _ = std::fs::remove_file(&path);
}
