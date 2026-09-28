//! `#[cfg(test)] mod tests` for `src/graph/dot.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::builder::GraphBuilder;
use crate::graph::DType;

#[test]
fn dot_export_format() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();
    let mut out = Vec::new();
    g.dump_dot(&mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("digraph G {"));
    assert!(s.contains("n0 -> n1"), "edge missing: {s}");
    assert!(s.contains("INPUT"));
    assert!(s.contains("OUTPUT"));
    assert!(s.trim_end().ends_with('}'));
}
