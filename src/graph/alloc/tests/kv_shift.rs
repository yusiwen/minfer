//! #306: the physical shift (`kv_rm` / `kv_shift`) is **format-aware**, and an
//! **f16** region is refused loudly instead of re-roping reinterpreted halves.
//!
//! The other two formats are the control arm on the same fixture (gate-contract
//! rule 2): f32 takes the plain `rope_shift_kv` branch and packed Q8_0 the
//! `kvformat::map_q8_0_cells` branch, and both shift — so the gate proves the
//! refusal is *format-specific*, not "the shift refuses everything". A guard
//! widened to "any non-f32 region" fails the packed arm; a guard deleted
//! entirely fails the f16 arms (see the mutation transcript in PR #306's body).
//!
//! Split out of `src/graph/alloc/tests.rs` following the
//! `<module>/tests/<topic>.rs` layout (#267); the shared fixtures live in the
//! parent module and are reached through `use super::*;`.

use super::*;
use crate::graph::kvformat::KvFormat;

/// Elements per cell (`n_kv_embd`). A whole Q8_0 block, so the packed control arm
/// is a legal `MINFER_CACHE_TYPE=q8_0` region — `ensure_kv` refuses a width that is
/// not a non-zero multiple of 32.
const ROW: usize = 32;
/// Cells in the arena.
const N_CTX: usize = 16;
/// Rows the fixture marks written through the production fill entry point, so a
/// shift of [`DROP`] rows leaves a real surviving window.
const WRITTEN: usize = 8;
/// Rows the shift drops — the conversation's overflow case.
const DROP: usize = 2;

/// The rope `kv_rm` re-bases the survivors' K with. One KV head of `ROW` elements,
/// the shape the allocator's own `kv_rm` fixture uses.
fn shift_rope() -> crate::graph::kvcache::KvRope {
    crate::graph::kvcache::KvRope {
        freq_base: 10_000.0,
        freq_scale: 1.0,
        n_head_kv: 1,
        hd: ROW,
        style: crate::vec_ops::RopeStyle::NonInterleaved,
    }
}

/// A **genuinely formatted** KV region, built through the production per-engine
/// route: `GraphAllocator::set_kv_format` (what `conversation.rs` calls with the
/// loaded engine's resolved format) twice — once on the builder that stamps each
/// KV node's cell width, once on the allocator that sizes and reads the region —
/// plus the `kvcache_store`/`kvcache_load` allocation path the C4 gates use.
///
/// `fill_batch_inputs` is the **production** fill entry point (E1/E2), so the rows
/// are marked written the way a real forward marks them; no scheduler execution is
/// needed, because `kv_rm` is a host-side operation on the region bytes.
fn formatted_region(format: KvFormat) -> (GraphAllocator, ComputeGraph) {
    let mut b = GraphBuilder::new();
    b.set_kv_format(format);
    let _pos = b.input("positions", [WRITTEN, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, WRITTEN, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, WRITTEN, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut a = GraphAllocator::new();
    a.set_kv_format(format);
    a.kv_set_capacity(N_CTX);
    a.alloc_graph(&g)
        .unwrap_or_else(|e| panic!("[{}] allocate the KV region: {e}", format.name()));
    a.fill_batch_inputs(
        &g,
        &Batch::new(vec![0; WRITTEN], (0..WRITTEN).collect(), vec![0; WRITTEN]),
    )
    .unwrap_or_else(|e| panic!("[{}] mark the written rows: {e}", format.name()));
    (a, g)
}

/// `N_CTX` f32 cells, distinguishable per cell and per element so a moved row is
/// identifiable wherever it lands.
fn f32_region(base: f32) -> Vec<f32> {
    (0..N_CTX * ROW)
        .map(|i| base + (i / ROW) as f32 + (i % ROW) as f32 / 64.0)
        .collect()
}

/// `N_CTX` **packed Q8_0** cells as pool words: `KvFormat::Q8_0::row_elems(ROW)`
/// words per cell, 34 payload bytes (one f16 scale + 32 int8 quants) plus the word
/// padding that keeps every cell start on a word boundary.
///
/// Each cell is encoded from a real row with the production quantizer, so the
/// packed control's `map_q8_0_cells` dequantizes real blocks rather than whatever
/// bytes the pool happened to hold.
fn q8_0_region(base: f32) -> Vec<f32> {
    let row_elems = KvFormat::Q8_0.row_elems(ROW);
    let payload = KvFormat::Q8_0.payload_bytes(ROW);
    let mut row = vec![0.0f32; ROW];
    let mut bytes = vec![0u8; payload];
    let mut out = vec![0.0f32; row_elems * N_CTX];
    for cell in 0..N_CTX {
        for (i, x) in row.iter_mut().enumerate() {
            *x = base + cell as f32 + i as f32 / 64.0;
        }
        crate::quants::quantize_row_q8_0_into(&row, &mut bytes);
        for w in 0..row_elems {
            let mut word = [0u8; 4];
            for (j, b) in word.iter_mut().enumerate() {
                let idx = w * 4 + j;
                if idx < payload {
                    *b = bytes[idx];
                }
            }
            out[cell * row_elems + w] = f32::from_bits(u32::from_le_bytes(word));
        }
    }
    out
}

/// The region bytes a format stores for `base`, in both K and V.
fn region_words(format: KvFormat, base: f32) -> Vec<f32> {
    match format {
        KvFormat::Q8_0 => q8_0_region(base),
        // f16 stores halves in the first half of each word, but no host read
        // happens on the refused path: what matters is a byte pattern that a
        // reinterpreted f32 re-rope would visibly move.
        KvFormat::F32 | KvFormat::F16 => f32_region(base),
    }
}

/// Write both regions of a fixture through the pool (the K/V bytes a store would
/// have left there), and return the `(row_elems)` width they address.
fn seed(a: &mut GraphAllocator, format: KvFormat) -> (usize, Vec<f32>, Vec<f32>) {
    let (kid, vid) = a.kv_pair(0).expect("a KV region");
    let k = region_words(format, 0.0);
    let v = region_words(format, 1000.0);
    a.write_pool(Backend::CPU, kid, &k)
        .unwrap_or_else(|e| panic!("[{}] write the K region: {e}", format.name()));
    a.write_pool(Backend::CPU, vid, &v)
        .unwrap_or_else(|e| panic!("[{}] write the V region: {e}", format.name()));
    (format.row_elems(ROW), k, v)
}

/// **The gate.** #306's whole point: `kv_rm` refuses a physical shift on an f16
/// region **before** the host round trip, the refusal names the format and the
/// missing map, and the two formats that *can* shift do so on the same fixture.
#[test]
fn an_f16_region_refuses_the_physical_shift_and_the_other_formats_take_it() {
    // ---- the value arm: an f16 region refuses, by name and before any read ----
    let (mut f16, _gf) = formatted_region(KvFormat::F16);
    assert_eq!(
        f16.kv_format(),
        KvFormat::F16,
        "the fixture must carry a genuine f16 format stamp, not the F32 default"
    );
    let (row_elems, _k, _v) = seed(&mut f16, KvFormat::F16);
    assert_eq!(row_elems, ROW, "an f16 cell keeps the f32-shaped width");
    let before = f16.copy_kv_to_cpu(0).expect("read the f16 region");
    assert_eq!(f16.kv_n_used(0), Some(WRITTEN));

    let err = match f16.kv_rm(0, DROP, &shift_rope()) {
        Ok(n) => panic!(
            "#306: kv_rm must refuse a physical shift on an f16 region, but it returned \
             Ok({n}) — the survivors' raw half bytes were re-roped as f32 words, the silent \
             corruption the refusal exists to prevent"
        ),
        Err(e) => e,
    };
    assert!(
        err.contains("f16"),
        "the f16 refusal must name the storage format it refuses, got: {err}"
    );
    assert!(
        err.contains("map"),
        "the f16 refusal must name the missing dequantize -> re-rope -> requantize map, got: {err}"
    );
    assert!(
        err.contains("#306"),
        "the f16 refusal must cite the issue that owns the missing map, got: {err}"
    );
    // The refusal precedes the host round trip, so nothing was read back, slid,
    // re-roped, written or forgotten: a refused shift must be a no-op.
    assert_eq!(
        f16.kv_n_used(0),
        Some(WRITTEN),
        "a refused shift must not change the written extent"
    );
    let after = f16.copy_kv_to_cpu(0).expect("read the f16 region");
    assert_eq!(
        after.0, before.0,
        "a refused shift must leave the K region byte-identical"
    );
    assert_eq!(
        after.1, before.1,
        "a refused shift must leave the V region byte-identical"
    );

    // ---- the `start == 0` spelling goes through the same guard ----
    let err = match f16.kv_shift(DROP, &shift_rope()) {
        Ok(n) => panic!(
            "#306: kv_shift must refuse an f16 region the same way kv_rm does, but it returned \
             Ok({n})"
        ),
        Err(e) => e,
    };
    assert!(
        err.contains("f16") && err.contains("map"),
        "kv_shift's f16 refusal must name the format and the missing map (it is kv_rm(0, drop)), \
         got: {err}"
    );

    // ---- the control arm: the same fixture under the formats that can shift ----
    //
    // f32 exercises the plain re-rope and packed Q8_0 the `map_q8_0_cells` branch,
    // so a guard widened to "any non-f32 region" fails the packed arm while the
    // f16 arms stay red only for the f16 condition.
    for format in [KvFormat::F32, KvFormat::Q8_0] {
        let label = format.name();
        let (mut a, _ga) = formatted_region(format);
        assert_eq!(
            a.kv_format(),
            format,
            "[{label}] the fixture's engine format"
        );
        let (row_elems, k, v) = seed(&mut a, format);

        let n = a.kv_rm(0, DROP, &shift_rope()).unwrap_or_else(|e| {
            panic!(
                "[{label}] a {label} region must shift — the f16 refusal is format-specific: {e}"
            )
        });
        assert_eq!(
            n,
            WRITTEN - DROP,
            "[{label}] kv_rm must return the surviving written-row count"
        );
        assert_eq!(a.kv_n_used(0), Some(n), "[{label}] the new written extent");
        let (k_rm, v_rm) = a.copy_kv_to_cpu(0).expect("read the shifted region");
        // V carries no rope, so its survivors must be the source's rows verbatim,
        // at the shifted cell — a value arm, not a mode-vs-mode relation.
        assert_eq!(
            &v_rm[..n * row_elems],
            &v[DROP * row_elems..WRITTEN * row_elems],
            "[{label}] the survivors' V rows must move verbatim"
        );

        // `kv_shift(drop)` is `kv_rm(0, drop)`: the same fixture must take the same
        // path and land on the same bytes.
        let (mut b, _gb) = formatted_region(format);
        let (b_row_elems, b_k, b_v) = seed(&mut b, format);
        assert_eq!(b_row_elems, row_elems, "[{label}] the two fixtures agree");
        let n_shift = b.kv_shift(DROP, &shift_rope()).unwrap_or_else(|e| {
            panic!("[{label}] kv_shift on a {label} region must take kv_rm(0, drop)'s path: {e}")
        });
        assert_eq!(n_shift, n, "[{label}] kv_shift must equal kv_rm(0, drop)");
        let (k_shift, v_shift) = b.copy_kv_to_cpu(0).expect("read the shifted region");
        assert_eq!(
            k_shift, k_rm,
            "[{label}] kv_shift and kv_rm(0, drop) must leave the same K bytes"
        );
        assert_eq!(
            v_shift, v_rm,
            "[{label}] kv_shift and kv_rm(0, drop) must leave the same V bytes"
        );
        let _ = (k, b_k, b_v);
    }
}
