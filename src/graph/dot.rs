//! DOT graph export (Phase 4) — `--dump-graph` debugging.
//!
//! Same spirit as llama.cpp's `ggml_graph_dump_dot`: visualize the IR
//! (nodes, edges, backend colors, inputs/outputs) for debugging and
//! documentation.

use std::io::Write;

use super::{Backend, ComputeGraph};

impl ComputeGraph {
    /// Export Graphviz DOT format.
    pub fn dump_dot(&self, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "digraph G {{")?;
        writeln!(w, "  rankdir=LR;")?;
        writeln!(w, "  node [shape=record fontname=\"monospace\"];")?;

        for node in &self.nodes {
            let color = match node.backend {
                Some(Backend::Metal) => "lightblue",
                Some(Backend::CPU) => "lightyellow",
                Some(Backend::Cuda) => "lightgreen",
                None => "white",
            };
            let label = format!("{}\\n{:?}", node.name, node.op);
            writeln!(
                w,
                "  n{} [label=\"{}\" style=filled fillcolor={}]",
                node.id, label, color
            )?;
        }

        for node in &self.nodes {
            for &src in &node.src {
                writeln!(w, "  n{} -> n{}", src, node.id)?;
            }
        }

        for &inp in &self.inputs {
            writeln!(
                w,
                "  n{} [label=\"INPUT\\n{}\" shape=doublecircle]",
                inp, self.nodes[inp].name
            )?;
        }
        for &out in &self.outputs {
            writeln!(
                w,
                "  n{} [label=\"OUTPUT\\n{}\" shape=doublecircle]",
                out, self.nodes[out].name
            )?;
        }

        writeln!(w, "}}")
    }
}

#[cfg(test)]
mod tests {
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
}
