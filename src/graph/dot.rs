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
            // F4: the handle is not an enum, so the arms guard on equality with
            // the registry constants instead of matching variants.
            let color = match node.backend {
                Some(b) if b == Backend::METAL => "lightblue",
                Some(b) if b == Backend::CPU => "lightyellow",
                Some(b) if b == Backend::CUDA => "lightgreen",
                _ => "white",
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
mod tests;
