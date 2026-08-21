//! Backend scheduler (Phase 2: minimal single-backend executor).
//!
//! Phase 4 expands this into assign → fuse → split → execute. For Phase 2 it
//! provides the execution driver used by the CPU-backend tests: run every node
//! in topological order on the CPU backend, resolving buffers through the
//! allocator. The allocator owns the backend pools (single source of truth);
//! the scheduler drives execution through them.

use super::alloc::GraphAllocator;
use super::backend::Backend;
use super::{Backend as BackendTag, ComputeGraph};

pub struct BackendScheduler;

impl Default for BackendScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendScheduler {
    pub fn new() -> Self {
        Self
    }

    /// Assign every node to CPU (Phase 4 replaces this with the 5-pass
    /// backend-assignment algorithm).
    pub fn assign_backends(&mut self, graph: &mut ComputeGraph) {
        for n in &mut graph.nodes {
            n.backend = Some(BackendTag::CPU);
        }
    }

    /// Execute the whole graph on the CPU backend (single split).
    pub fn execute(
        &mut self,
        graph: &ComputeGraph,
        alloc: &mut GraphAllocator,
    ) -> Result<(), String> {
        let order = graph.topo_order()?;
        for id in order {
            let node = graph.node(id);
            if node.is_input() {
                continue; // data pre-filled by the allocator
            }
            let mut in_bufs = Vec::with_capacity(node.src.len());
            for &s in &node.src {
                let br = alloc
                    .node_buffer(s)
                    .ok_or_else(|| format!("node {s} has no allocated buffer"))?;
                if br.backend != BackendTag::CPU {
                    return Err(format!(
                        "node {s} is on {:?}; cross-backend splits are Phase 4",
                        br.backend
                    ));
                }
                in_bufs.push(br.id);
            }
            let br = alloc
                .node_buffer(id)
                .ok_or_else(|| format!("node {id} has no allocated buffer"))?;
            if br.backend != BackendTag::CPU {
                return Err(format!(
                    "node {id} is on {:?}; cross-backend splits are Phase 4",
                    br.backend
                ));
            }
            alloc.cpu_mut().execute_node(node, &in_bufs, br.id)?;
        }
        Ok(())
    }
}
