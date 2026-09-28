//! `#[cfg(test)] mod tests` for `src/graph/cache.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::builder::GraphBuilder;
use crate::graph::params::GraphType;
use crate::graph::DType;

fn params(n_tokens: usize, weights_version: u64) -> GraphParams {
    GraphParams {
        n_tokens,
        n_out: 1,
        gtype: if n_tokens == 1 {
            GraphType::Decode
        } else {
            GraphType::Prefill
        },
        cparams: Default::default(),
        weights_version,
    }
}

fn tiny_graph() -> ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    b.output(s);
    b.build()
}

/// 8a③: the fusion toggles MUST participate in the reuse identity —
/// flipping `fuse_qkv`/`fuse_ffn` (what `MINFER_NO_FUSE_QKV`/
/// `MINFER_NO_FUSE_FFN` do at build time) changes the topology, so a
/// cached graph must NOT be reused. Guards against someone dropping the
/// fields from CParams/PartialEq and silently breaking the A/B envs.
#[test]
fn fuse_flags_are_part_of_the_reuse_identity() {
    let mut base = params(1, 1);
    base.cparams.fuse_qkv = true;
    base.cparams.fuse_ffn = true;
    let mut cache = GraphCache::new();
    let g = tiny_graph();
    cache.replace_graph(g, base.clone());
    assert!(cache.try_reuse(&base).unwrap(), "identical params reuse");

    for flip in ["fuse_qkv", "fuse_ffn"] {
        let mut other = base.clone();
        match flip {
            "fuse_qkv" => other.cparams.fuse_qkv = false,
            "fuse_ffn" => other.cparams.fuse_ffn = false,
            _ => unreachable!(),
        }
        assert!(
            !cache.try_reuse(&other).unwrap(),
            "{flip}=false must force a rebuild (8a③)"
        );
        // and the reverse direction: stored off, requested on
        let mut off = base.clone();
        off.cparams.fuse_qkv = false;
        off.cparams.fuse_ffn = false;
        let mut cache2 = GraphCache::new();
        cache2.replace_graph(tiny_graph(), off);
        assert!(!cache2.try_reuse(&base).unwrap());
    }
}

/// E4 S3: the cache holds one graph per `GraphParams`, and switching between them is a
/// **re-map** — no build, no fusion pass, and no pool traffic (the CPU pool's allocation
/// count stays put once both shapes have been built). Before S3 the second switch rebuilt,
/// which is what made a server alternating shapes pay a full rebuild per step.
#[test]
fn switching_between_cached_graphs_re_maps_instead_of_rebuilding() {
    fn graph_with(n_tokens: usize) -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [64, n_tokens, 1, 1], DType::F32);
        let w = b.input("w", [64, n_tokens, 1, 1], DType::F32);
        let t = b.add(x, w);
        let y = b.silu(t);
        b.output(y);
        b.build()
    }
    /// The active graph's node → (backend, pool id), by node id.
    fn active_slots(cache: &mut GraphCache) -> Vec<Option<(crate::graph::Backend, usize)>> {
        let (g, a) = cache.current().expect("a graph");
        (0..g.n_nodes())
            .map(|i| a.node_buffer(i).map(|b| (b.backend, b.id)))
            .collect()
    }

    let mut cache = GraphCache::new();
    let (p1, p2) = (params(1, 1), params(4, 1));
    for p in [&p1, &p2] {
        let g = graph_with(p.n_tokens);
        cache.alloc().alloc_graph(&g).unwrap();
        cache.replace_graph(g, p.clone());
    }
    assert_eq!(cache.stats(), (2, 0), "two shapes were built");
    assert_eq!(cache.cached_graphs(), 2);
    let allocs = cache.alloc().n_cpu_allocs();
    assert!(allocs > 0, "the first build did allocate");

    // Alternate four times: every switch must hit, and the pool must stay untouched.
    let mut seen: Vec<(GraphParams, Vec<_>)> = Vec::new();
    for p in [&p1, &p2, &p1, &p2] {
        assert!(cache.try_reuse(p).unwrap(), "cached params must hit");
        seen.push((p.clone(), active_slots(&mut cache)));
        assert_eq!(
            cache.alloc().n_cpu_allocs(),
            allocs,
            "a switch must not allocate in the pool"
        );
    }
    assert_eq!(cache.stats(), (2, 4), "four switches, still two builds");
    // The same graph always maps to the same slots — otherwise a switched-in graph would
    // silently move its data around (the deterministic slot order is what guarantees it).
    assert_eq!(
        seen[0].1, seen[2].1,
        "graph A maps to the same slots every time"
    );
    assert_eq!(
        seen[1].1, seen[3].1,
        "graph B maps to the same slots every time"
    );
}

#[test]
fn reuse_requires_equal_params() {
    let mut cache = GraphCache::new();
    assert!(
        !cache.try_reuse(&params(1, 1)).unwrap(),
        "nothing stored yet"
    );

    let g = tiny_graph();
    let outputs_expected = g.outputs.clone();
    cache.replace_graph(g, params(1, 1));

    // same params -> reuse
    assert!(cache.try_reuse(&params(1, 1)).unwrap());
    // n_tokens changed -> rebuild
    assert!(!cache.try_reuse(&params(4, 1)).unwrap());
    // weights version changed (LoRA switch) -> rebuild
    assert!(!cache.try_reuse(&params(1, 2)).unwrap());
    // gtype differs -> rebuild
    let p = GraphParams {
        n_tokens: 1,
        n_out: 1,
        gtype: GraphType::Prefill,
        cparams: Default::default(),
        weights_version: 1,
    };
    assert!(!cache.try_reuse(&p).unwrap());

    // current() yields the stored graph
    let (g2, _) = cache.current().unwrap();
    assert_eq!(g2.outputs, outputs_expected);
}

#[test]
fn allocator_survives_rebuild() {
    let mut cache = GraphCache::new();
    cache.replace_graph(tiny_graph(), params(1, 1));
    // rebuild with different params: allocator object identity persists
    // (a KV persistent region registered before must still be there)
    cache
        .alloc()
        .alloc_persistent("kv.test", crate::graph::Backend::CPU, 16);
    assert!(cache.alloc().get_persistent("kv.test").is_some());
    cache.replace_graph(tiny_graph(), params(4, 1));
    assert!(
        cache.alloc().get_persistent("kv.test").is_some(),
        "persistent regions must survive graph rebuilds"
    );
}

#[test]
fn structural_check_detects_different_graph() {
    #[cfg(debug_assertions)]
    {
        let mut cache = GraphCache::new();
        cache.replace_graph(tiny_graph(), params(1, 1));
        // a different topology with the same params must fail verification
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, x); // extra node -> different topology
        b.output(o);
        let g2 = b.build();
        assert!(!cache.verify_structural(&g2));
    }
}
