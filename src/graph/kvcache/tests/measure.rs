//! #828: the KV numbers the book cannot reach from the CLI.
//!
//! A book (`llm-inference-book`) teaches how this engine works, and its rule is that every
//! number in the prose comes from something the engine actually produced. Three KV-cache
//! numbers exist only inside structs today: a region's bytes per layer *per storage
//! format*, the cell/span layout of two sequences over one arena, and the footprint table
//! a reader needs for chapters 10 and 18. This file exports them as measurements instead of
//! as prose arithmetic.
//!
//! **No device is touched.** Every number here is a pure data structure or arithmetic:
//! `KvFormat` decides a cell's width, `KvCache::insert` sizes a region exactly the way
//! `GraphAllocator::ensure_kv` does, and `KvFormat::supports` is a registry table lookup.
//! That is why these tests carry no `#[cfg(feature = "metal")]` / `#[cfg(feature = "cuda")]`
//! gate: a test that only compiles in one configuration silently stops existing in the
//! other two. The format is a **parameter**, never `MINFER_CACHE_TYPE`: the environment is
//! process-wide and tests run on parallel threads, so two format tests would race.
//!
//! Output protocol (#828): one line per measurement group, a JSON object after the
//! `measure: ` prefix. `cargo test --release measure -- --nocapture` prints them; the
//! book's instrument parses the lines into `facts/kv.json`, so nothing is copied by hand.
//!
//! | group | what it answers |
//! |---|---|
//! | `kv_geometry` | format x device: support, cell width, unused bytes, region bytes |
//! | `kv_context_growth` | `n_ctx -> region_bytes`, and the exact linearity of it |
//! | `kv_sharing` | two sequences over one arena, before and after the donor is released |
//! | `kv_footprint` | the `n_ctx x format` table (24 layers, `n_kv_embd` 128) |
//!
//! Two numbers here deliberately disagree with the ticket text, because the engine says
//! otherwise and the engine wins:
//!
//! - **f16 is a bandwidth win, not a footprint win.** An f16 cell keeps the f32 shape
//!   (`row_elems == n_kv_embd`) and stores halves in the first half of those words, so a
//!   region is the same size as f32's. `stored_bits_per_element` and
//!   `region_bits_per_element` are printed separately so the difference is visible.
//! - `KvArenaStats::free_cells` used to count *reservations* while `free_runs` counted
//!   *spans*, so the two disagreed once a donor was released and its rows stayed shared
//!   (`kv_sharing.after_release_a`). [#829](https://github.com/yusiwen/minfer/issues/829)
//!   put both on the same `occupied()` authority; the test now asserts the equality.

use super::*;

use crate::graph::kvformat::KvFormat;
use crate::models::Device;
use serde_json::json;

/// The book's reference geometry: one `n_kv_embd` of 128 (a 64-wide head pair, or an
/// 8-head/16-wide GQA pair) and the context the KV chapters quote.
const N_EMBD: usize = 128;
const N_CTX: usize = 4096;

/// One measurement line. The book's instrument splits on the `measure: ` prefix and reads
/// the rest as JSON, so a line is self-describing and needs no column agreement.
fn emit(group: &str, value: serde_json::Value) {
    let mut obj = serde_json::Map::new();
    obj.insert(group.to_string(), value);
    println!("measure: {}", serde_json::Value::Object(obj));
}

/// One layer's arena, sized exactly the way `GraphAllocator::ensure_kv` sizes it: two
/// persistent regions of `row_elems * n_ctx` f32 words for one layer, and `layers` of them
/// for a table. No allocator and no device, because a region's size is arithmetic.
fn arena(format: KvFormat, n_embd: usize, n_ctx: usize, layers: usize) -> KvCache {
    let mut cache = KvCache::new();
    for layer in 0..layers {
        cache.insert(
            layer,
            buf(2 * layer),
            buf(2 * layer + 1),
            n_embd,
            format.row_elems(n_embd),
            n_ctx,
            format.is_packed(),
        );
    }
    cache
}

fn slot_json(slot: SeqSlot) -> serde_json::Value {
    json!({
        "start": slot.start,
        "cap": slot.cap,
        "shared_cell": slot.shared.cell,
        "shared_rows": slot.shared.rows,
        "written": slot.written,
        "private_written": slot.private_written(),
    })
}

// ────────────────────────────────────────────────────────────────────────────
// geometry by format and device - chapter 18
// ────────────────────────────────────────────────────────────────────────────

/// Every `(format, device)` pair, with the cell width and the region size it implies.
///
/// This is the table the trace cannot show: the `kvcache_store` node carries the *logical*
/// shape (`[n_embd, n_ctx]`, f32) in every format, so f16 and q8_0 look identical there,
/// and peak RSS cannot see a region that was never touched page by page.
#[test]
fn kv_geometry_by_format_and_device() {
    let mut formats = Vec::new();
    let mut region_bytes = std::collections::BTreeMap::new();
    let mut row_elems = std::collections::BTreeMap::new();
    for format in [KvFormat::F32, KvFormat::F16, KvFormat::Q8_0] {
        format
            .check_width(N_EMBD)
            .expect("128 elements is a whole number of Q8_0 blocks");

        let elems = format.row_elems(N_EMBD);
        let bytes = elems * std::mem::size_of::<f32>();
        let payload = format.payload_bytes(N_EMBD);
        let cache = arena(format, N_EMBD, N_CTX, 1);
        let region = cache.region_bytes();
        let layer = cache.get(0).expect("the one layer was inserted");

        // A region is K + V f32 words in **every** format (the pool is word-addressed, and
        // a packed cell is padded up to whole words so a cell move stays a word copy).
        assert_eq!(layer.n_embd, N_EMBD, "{}", format.name());
        assert_eq!((layer.n_ctx, layer.packed), (N_CTX, format.is_packed()));
        assert_eq!(layer.elems, elems * N_CTX);
        assert_eq!(
            region,
            layer.elems * std::mem::size_of::<f32>() * 2,
            "{}: region_bytes is both regions of f32 words",
            format.name()
        );
        // The padding rule that makes a packed cell movable: its payload never exceeds
        // the word width, and the slack is under one word.
        assert!(payload <= bytes, "{}: payload <= cell width", format.name());
        if format.is_packed() {
            assert!(
                bytes - payload < std::mem::size_of::<f32>(),
                "{}: a packed cell pads to whole words only ({payload} of {bytes})",
                format.name()
            );
        }

        // `supports` is a registry table lookup - the same answer the load-time gate
        // reads - and it is a property of the **build** (a backend compiled out answers
        // false), which is why `registered` is printed beside it.
        let devices: Vec<serde_json::Value> = [Device::Cpu, Device::Metal, Device::Cuda]
            .iter()
            .map(|&device| {
                let supported = format.supports(device);
                if supported {
                    // A format the device accepts must fit that format's cell width,
                    // whatever the geometry - the invariant the gate exists for.
                    assert!(payload <= bytes, "{} on {}", format.name(), device.name());
                }
                json!({
                    "device": device.name(),
                    "registered": device.backend().is_registered(),
                    "supports": supported,
                })
            })
            .collect();
        assert!(
            format.supports(Device::Cpu),
            "{}: the CPU reads every format by design (F32/F16 directly, Q8_0 through the \
             fused read)",
            format.name()
        );

        // The value's own precision vs what the region costs per element: equal for f32
        // and q8_0, 16 vs 32 for f16 - the "bandwidth, not footprint" claim, asserted.
        let stored_bits = payload as f64 * 8.0 / N_EMBD as f64;
        let region_bits = bytes as f64 * 8.0 / N_EMBD as f64;
        match format {
            KvFormat::F32 => {
                assert_eq!((stored_bits, region_bits), (32.0, 32.0));
            }
            KvFormat::F16 => {
                assert_eq!(stored_bits, 16.0, "an f16 value is half an f32 value");
                assert_eq!(
                    region_bits, 32.0,
                    "but it still costs a whole f32 word: the region does not shrink"
                );
            }
            KvFormat::Q8_0 => {
                assert_eq!(
                    (stored_bits, region_bits),
                    (8.5, 8.5),
                    "one block is 34 bytes per 32 elements, scale included"
                );
            }
        }

        formats.push(json!({
            "format": format.name(),
            "row_elems": elems,
            "row_bytes": bytes,
            "payload_bytes": payload,
            // f16's slack is stored-in-the-first-half space, not padding; q8_0's is word
            // padding. Both are real, so both are printed under one name that admits it.
            "unused_bytes_per_cell": bytes - payload,
            "stored_bits_per_element": stored_bits,
            "region_bits_per_element": region_bits,
            "region_bytes": region,
            "bytes_per_token_per_layer": 2 * bytes,
            "devices": devices,
        }));
        region_bytes.insert(format.name(), region);
        row_elems.insert(format.name(), elems);
    }

    // Across formats, for the same geometry.
    assert_eq!(
        region_bytes["f32"], region_bytes["f16"],
        "f16 keeps the region f32-shaped (row_elems == n_kv_embd): it halves the \
         *bandwidth*, not the footprint"
    );
    assert!(
        region_bytes["f32"] > region_bytes["q8_0"],
        "q8_0 is the only format that shrinks a region"
    );
    assert_eq!(row_elems["f32"], row_elems["f16"]);
    assert!(
        row_elems["q8_0"] < row_elems["f32"],
        "a packed cell is narrower than the logical row"
    );
    // The shrink is exactly the cell-width ratio: 34 words instead of 128 for 128 elements.
    assert_eq!(
        region_bytes["q8_0"] * N_EMBD,
        region_bytes["f32"] * row_elems["q8_0"],
        "the q8_0:kv region ratio is exactly row_elems/n_kv_embd"
    );

    emit(
        "kv_geometry",
        json!({
            "n_embd": N_EMBD,
            "n_ctx": N_CTX,
            "layers": 1,
            "formats": formats,
        }),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// context growth - chapter 18
// ────────────────────────────────────────────────────────────────────────────

/// Four contexts, one format: the region is allocated for the **whole** context up front,
/// which the book currently argues from arithmetic.
#[test]
fn kv_context_growth_is_exactly_linear() {
    let mut rows = Vec::new();
    let mut bytes = std::collections::BTreeMap::new();
    for n_ctx in [1024usize, 4096, 8192, 32768] {
        let mut cache = arena(KvFormat::F32, N_EMBD, n_ctx, 1);
        let region = cache.region_bytes();
        // Two f32 regions of n_embd * n_ctx: the size is fixed at allocation, not grown
        // per token.
        assert_eq!(region, 2 * N_EMBD * n_ctx * std::mem::size_of::<f32>());
        bytes.insert(n_ctx, region);
        rows.push(json!({
            "n_ctx": n_ctx,
            "region_bytes": region,
            "mib": region as f64 / (1024.0 * 1024.0),
        }));
    }
    assert_eq!(
        bytes[&32768],
        8 * bytes[&4096],
        "8x the context is exactly 8x the region: nothing is shared or amortized"
    );
    assert_eq!(bytes[&8192], 2 * bytes[&4096]);
    assert_eq!(bytes[&4096], 4 * bytes[&1024]);

    emit(
        "kv_context_growth",
        json!({
            "n_embd": N_EMBD,
            "format": KvFormat::F32.name(),
            "layers": 1,
            "rows": rows,
        }),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// sharing layout - chapter 18 (E2/C8b)
// ────────────────────────────────────────────────────────────────────────────

/// A donor and a sharer over one arena: what each sequence reserves, what it reads in
/// place, what it owns, and what releasing the donor changes.
///
/// The numbers are the layout - a reader who wants to reproduce the book's sharing section
/// would otherwise have to write these assertions first.
#[test]
fn kv_sharing_layout() {
    const N_CTX: usize = 64;
    const SHARED: usize = 34;
    const PRIVATE: usize = 8;

    let mut cache = arena(KvFormat::F32, 4, N_CTX, 1);
    // A writes the whole prompt; B reads those rows in place and writes a tail of its own.
    let a = cache.reserve_seq(1, SHARED).expect("the arena is empty");
    cache.own_range(1, a.start, a.start + SHARED);
    let b = cache
        .reserve_seq(2, PRIVATE)
        .expect("first-fit after A's run");
    cache
        .share_prefix(1, 2, SHARED)
        .expect("A has written all 34 rows");
    let positions: Vec<usize> = (SHARED..SHARED + PRIVATE).collect();
    cache.own_positions(2, &positions).expect("B's tail");

    let slot_a = cache.seq_slot(1).expect("A holds a run");
    let slot_b = cache.seq_slot(2).expect("B holds a run");
    assert_eq!((slot_a.written, slot_b.written), (SHARED, SHARED + PRIVATE));
    assert_eq!(
        slot_b.shared,
        SharedPrefix {
            cell: a.start,
            rows: SHARED
        },
        "B names A's first cell, and only in the span list"
    );
    // Two live sequences, one copy of the shared bytes.
    assert_eq!(
        b.start,
        a.start + SHARED,
        "B's private run starts after A's reservation"
    );
    assert_eq!(
        cache.cell_of(2, 0),
        Some(a.start),
        "B reads A's rows in place"
    );
    assert_eq!(cache.cell_of(2, SHARED), Some(b.start), "then its own run");

    // The span lists as they are *now*: the JSON below describes the live layout, and
    // the releases at the end would empty them.
    let span_json = |seq: u32| -> Vec<serde_json::Value> {
        cache
            .spans_of(seq)
            .iter()
            .map(|&(base, cell, len)| json!({"base": base, "cell": cell, "len": len}))
            .collect()
    };
    let (a_spans, b_spans) = (span_json(1), span_json(2));

    let live = cache.arena_stats();
    assert_eq!(
        (live.reserved_cells, live.shared_cells, live.owned_cells),
        (SHARED + PRIVATE, SHARED, SHARED + PRIVATE),
        "reserved = the two private runs, shared = rows read in place, owned = every \
         written cell (the shared ones carry the donor's stamp)"
    );
    assert_eq!(
        (live.free_cells, live.free_runs, live.largest_free_run),
        (N_CTX - SHARED - PRIVATE, 1, N_CTX - SHARED - PRIVATE),
        "with the donor alive, the reservation count and the span count agree"
    );

    // #831: a live share **pins** its donor's run. The release is refused outright —
    // dropping the reservation would strand the rows B reads under a span no run holds,
    // where a compaction could place another run over them.
    let freed = cache.release_seq(1);
    assert_eq!(freed, 0, "a lending run is not releasable");
    let pinned = cache.arena_stats();
    let runs_pinned = cache.free_runs();
    assert!(
        !cache.spans_of(1).is_empty(),
        "the donor's run and address space survive the refusal"
    );
    assert_eq!(
        cache.cell_of(2, 0),
        Some(a.start),
        "B keeps reading the rows in place"
    );
    assert_eq!(
        (
            pinned.shared_cells,
            pinned.reserved_cells,
            pinned.owned_cells
        ),
        (SHARED, SHARED + PRIVATE, SHARED + PRIVATE),
        "the share keeps the donor's reservation: the cells are in use"
    );
    assert_eq!(
        (pinned.free_cells, pinned.free_runs, pinned.largest_free_run),
        (N_CTX - SHARED - PRIVATE, 1, N_CTX - SHARED - PRIVATE),
        "nothing became free"
    );
    assert_eq!(
        runs_pinned,
        vec![(SHARED + PRIVATE, N_CTX - SHARED - PRIVATE)],
        "one free run, starting above B's private cells"
    );
    // #829: `free_cells` is exactly the sum of `free_runs`, the span authority.
    assert_eq!(
        pinned.free_cells,
        runs_pinned.iter().map(|&(_, len)| len).sum::<usize>()
    );

    // The share is what pins the run: release the sharer, then the donor, and the whole
    // arena comes back. (B's slot is captured first — the JSON below describes the
    // pinned state, before either release.)
    let b_pinned = slot_json(cache.seq_slot(2).expect("B survives"));
    assert_eq!(cache.release_seq(2), PRIVATE);
    assert_eq!(cache.release_seq(1), SHARED);
    let empty = cache.arena_stats();
    assert_eq!(
        (
            empty.free_cells,
            empty.reserved_cells,
            empty.shared_cells,
            empty.owned_cells
        ),
        (N_CTX, 0, 0, 0),
        "both runs are gone"
    );

    emit(
        "kv_sharing",
        json!({
            "n_ctx": N_CTX,
            "n_embd": 4,
            "layers": 1,
            "shared_rows": SHARED,
            "before_release": {
                "a": slot_json(slot_a),
                "b": slot_json(slot_b),
                "cells": {
                    "reserved": live.reserved_cells,
                    "shared": live.shared_cells,
                    "owned": live.owned_cells,
                    "free": live.free_cells,
                    "free_runs": live.free_runs,
                    "largest_free_run": live.largest_free_run,
                },
                "spans": {"a": a_spans, "b": b_spans},
            },
            "after_release_attempt": {
                "freed": freed,
                "b": b_pinned,
                "cells": {
                    "reserved": pinned.reserved_cells,
                    "shared": pinned.shared_cells,
                    "owned": pinned.owned_cells,
                    "free": pinned.free_cells,
                    "free_runs": pinned.free_runs,
                    "largest_free_run": pinned.largest_free_run,
                },
                "free_runs": runs_pinned.iter().map(|&(start, len)| json!({"start": start, "len": len})).collect::<Vec<_>>(),
            },
            "after_release_both": {
                "cells": {
                    "reserved": empty.reserved_cells,
                    "shared": empty.shared_cells,
                    "owned": empty.owned_cells,
                    "free": empty.free_cells,
                    "free_runs": empty.free_runs,
                },
            },
        }),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// the footprint table - chapters 10 and 18
// ────────────────────────────────────────────────────────────────────────────

/// `n_ctx x format` at a model-like layer count: the table the book divides a GGUF's blob
/// size by. Absolute bytes only - the reader scales it to the model it measures.
#[test]
fn kv_footprint_table() {
    const LAYERS: usize = 24;
    let mut rows = Vec::new();
    let mut bytes = std::collections::BTreeMap::new();
    for n_ctx in [4096usize, 8192, 32768] {
        for format in [KvFormat::F16, KvFormat::Q8_0] {
            let cache = arena(format, N_EMBD, n_ctx, LAYERS);
            let region = cache.region_bytes();
            assert_eq!(
                region,
                LAYERS * 2 * format.row_elems(N_EMBD) * n_ctx * std::mem::size_of::<f32>(),
                "{} at {n_ctx}: every layer owns two word-addressed regions",
                format.name()
            );
            bytes.insert((n_ctx, format.name()), region);
            rows.push(json!({
                "n_ctx": n_ctx,
                "format": format.name(),
                "layers": LAYERS,
                "region_bytes": region,
                "mib": region as f64 / (1024.0 * 1024.0),
            }));
        }
    }
    // Linear in the context, and q8_0 cheaper than f16 at every context.
    for format in ["f16", "q8_0"] {
        assert_eq!(bytes[&(32768, format)], 8 * bytes[&(4096, format)]);
    }
    for n_ctx in [4096usize, 8192, 32768] {
        assert!(
            bytes[&(n_ctx, "q8_0")] < bytes[&(n_ctx, "f16")],
            "q8_0 must shrink the table at {n_ctx}"
        );
    }

    emit(
        "kv_footprint",
        json!({
            "n_embd": N_EMBD,
            "layers": LAYERS,
            "rows": rows,
        }),
    );
}
