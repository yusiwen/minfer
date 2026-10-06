//! C3 (issue #44 part (b), G5b): the Metal KV row-move primitive, and the
//! per-engine `kv_format` that drives its f16 stride.
//!
//! `copy_cells` is the overlap-safe row move the compaction, `kv_copy_prefix`
//! and the copy-on-write all use. Metal implemented it as one
//! `MTLBlitCommandEncoder` copy per row in the order the overlap requires
//! (ascending when the run slides down, descending when it slides up), because
//! Apple documents an overlapping same-buffer blit as undefined — the same
//! constraint CUDA's `kv_move_rows` has. These gates mirror the CUDA pair
//! (`cuda_copy_cells_moves_overlapping_rows_in_both_directions`,
//! `cuda_f16_kv_cell_move_strides_by_row_bytes`) through the **production**
//! `Backend::copy_cells` arm on a real Metal pool.

use super::*;
use crate::graph::kvformat::KvFormat;

/// The pool a gate drives. `enable_metal` builds the real `MetalBackend`; a
/// `None` answer skips (no MPS on this machine).
fn metal_alloc(fmt: KvFormat) -> Option<GraphAllocator> {
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_metal() {
        return None;
    }
    alloc.set_kv_format(fmt);
    Some(alloc)
}

/// C3's copy primitive on the device: the rows land where the plan says,
/// *including when source and destination overlap* — the case a bulk
/// device-to-device copy cannot express. The bytes must be **exactly** the
/// CUDA gate's arrangement (a copy through a temporary would give the same).
#[test]
fn metal_copy_cells_moves_overlapping_rows_in_both_directions() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(mut alloc) = metal_alloc(KvFormat::F32) else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    // 6 rows x 4 elements; a row's value identifies the row it came from.
    let id = alloc.metal_mut().unwrap().alloc_buffer(24);
    let data: Vec<f32> = (0..24)
        .map(|i| (i / 4) as f32 + (i % 4) as f32 / 10.0)
        .collect();
    alloc.metal_mut().unwrap().write_host(id, &data).unwrap();
    let r = BufRef::own(Tag::METAL, id, 24);

    // Rows [1, 4) -> rows [0, 3): rows 1 and 2 are both read and overwritten.
    alloc
        .metal_mut()
        .unwrap()
        .copy_cells(r, r, 0, 1, 3, 4)
        .unwrap();
    let got = alloc.metal().unwrap().read_host(id).unwrap().to_vec();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = if i / 4 < 3 { i / 4 + 1 } else { i / 4 };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(got, want, "the moved rows must be byte-identical");

    // C7b: the upward direction works too. The blit order is descending here,
    // so the result is what a copy through a temporary would give.
    alloc.metal_mut().unwrap().write_host(id, &data).unwrap();
    alloc
        .metal_mut()
        .unwrap()
        .copy_cells(r, r, 1, 0, 3, 4)
        .unwrap();
    let got = alloc.metal().unwrap().read_host(id).unwrap().to_vec();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = match i / 4 {
                0 => 0,
                n if n <= 3 => n - 1,
                n => n,
            };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(
        got, want,
        "an upward overlapping move must be byte-identical"
    );
}

/// C8b S4: a cell move must stride by a **row's** size, whatever the KV dtype
/// stores. With an f16 region a cell is `nkt` halves = `nkt / 2` f32 words,
/// while `copy_cells`'s `elems_per_cell` is a count of f32 — the unit the move
/// walks by. Passing `nkt` there (no `/2`) moved every row twice as far as it
/// should, so a compaction on an f16 engine wrote the wrong cells. The CUDA
/// twin is `cuda_f16_kv_cell_move_strides_by_row_bytes`; this must **fail**
/// before the `/2` stride fix in `MetalBackend::copy_cells`.
#[test]
fn metal_f16_kv_cell_move_strides_by_row_bytes() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(mut alloc) = metal_alloc(KvFormat::F16) else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    assert_eq!(
        alloc.metal().unwrap().kv_format(),
        KvFormat::F16,
        "the gate must run on an f16 engine"
    );
    const N_CTX: usize = 64;
    // A row of `nkt` halves, stored in the first `nkt / 2` f32 slots.
    let nkt = 8usize;
    let (src_row, dst_row, rows) = (2usize, 10usize, 4usize);
    let slots = N_CTX * (nkt / 2);
    let region = alloc.metal_mut().unwrap().alloc_buffer(slots);
    // Every f32 slot carries its own index in its low 16 bits, so the halves a
    // kernel would read are distinct and the *slot* layout is verifiable.
    let pattern: Vec<f32> = (0..slots)
        .map(|i| f32::from_bits(((i as u32) * 7 + 1) & 0xFFFF))
        .collect();
    alloc
        .metal_mut()
        .unwrap()
        .write_host(region, &pattern)
        .unwrap();
    let before = pattern.clone();
    let r = BufRef::own(Tag::METAL, region, slots);
    alloc
        .metal_mut()
        .unwrap()
        .copy_cells(r, r, dst_row, src_row, rows, nkt)
        .unwrap();
    let after = alloc.metal().unwrap().read_host(region).unwrap().to_vec();
    assert_eq!(after.len(), slots);
    for r in 0..rows {
        // Slot view of a row: `row * nkt / 2` f32.
        let sd = (src_row + r) * (nkt / 2);
        let dd = (dst_row + r) * (nkt / 2);
        assert_eq!(
            &after[dd..dd + nkt / 2],
            &before[sd..sd + nkt / 2],
            "row {r} did not land at row {} (f16 KV strides in f32 elements)",
            dst_row + r
        );
    }
    // Nothing outside the destination rows may move.
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        let moved = (dst_row * nkt / 2..(dst_row + rows) * nkt / 2).contains(&i);
        if !moved {
            assert_eq!(x, y, "cell move touched f32 slot {i} outside its rows");
        }
    }
}

/// C4 per-engine (issue #44 part (b), the Metal half of #99/#153): the KV
/// element type is a property of the **pool**, not a process-wide tag, so two
/// engines in one process hold their own layouts and the registry's `kv_format`
/// hook (which C5 records in a session header) reads the instance.
///
/// Before this half the answer was the process-wide `metal::kv_cache_is_f16`
/// `OnceLock` — the first load won — so this gate could not even be expressed.
/// The two allocators are built in the order that would expose a global: f16
/// first, then f32.
#[test]
fn metal_kv_format_is_per_engine() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(f16_alloc) = metal_alloc(KvFormat::F16) else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let Some(f32_alloc) = metal_alloc(KvFormat::F32) else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    assert_eq!(f16_alloc.metal().unwrap().kv_format(), KvFormat::F16);
    assert_eq!(f32_alloc.metal().unwrap().kv_format(), KvFormat::F32);

    // The registry hook — the C5 session header's answer — follows the instance.
    let entry = Tag::METAL.entry().expect("Metal is registered on macOS");
    assert_eq!((entry.kv_format)(&f16_alloc), KvFormat::F16);
    assert_eq!((entry.kv_format)(&f32_alloc), KvFormat::F32);

    // A pool created before the stamp picks it up on `enable_metal` (the
    // `forward_graph` order), and a re-stamp moves it.
    let mut late = GraphAllocator::new();
    assert!(late.enable_metal());
    late.set_kv_format(KvFormat::F16);
    assert_eq!(late.metal().unwrap().kv_format(), KvFormat::F16);
    late.set_kv_format(KvFormat::F32);
    assert_eq!(late.metal().unwrap().kv_format(), KvFormat::F32);
}
