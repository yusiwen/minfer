//! #736: `MINFER_TRACE` must say where each node **ran**, not what the builder
//! default was.
//!
//! The trace's graph is the runtime graph (build → assign → fuse), so on a box
//! whose only device is CUDA a run's nodes must be marked `cuda`. The bug was a
//! call site passing `metal_on` as the placement flag, which is `false` wherever
//! Metal is absent — the exact configuration of every Linux device box, so the
//! trace claimed the CPU for a run that used the device.
//!
//! The test drives the real CLI on the checked-in toy model, so it covers the
//! call site, not just the exporter: a regression there turns it red on a device
//! box, while a CPU-only box pins the other half (no node may claim a device).

use std::path::{Path, PathBuf};
use std::process::Command;

fn toy() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/toy-qwen2-f32.gguf")
}

#[test]
fn the_trace_names_the_backend_each_node_ran_on() {
    let path = toy();
    assert!(
        path.exists(),
        "missing fixture {} — regenerate with `python3 scripts/gen_toy_gguf.py`",
        path.display()
    );
    let dir = std::env::temp_dir().join(format!("minfer-trace-736-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create trace dir");
    let tpath = dir.join("trace.json");

    let out = Command::new(env!("CARGO_BIN_EXE_minfer"))
        .arg(&path)
        .args(["hi", "-n", "2"])
        .env("MINFER_TRACE", &tpath)
        .output()
        .expect("spawn minfer on the toy model");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the toy must run; stderr:\n{stderr}"
    );

    let raw = std::fs::read_to_string(&tpath).expect("the trace file must be written");
    let doc: serde_json::Value = serde_json::from_str(&raw).expect("trace parses as JSON");
    let mut counts = std::collections::BTreeMap::new();
    for phase in doc["phases"].as_array().expect("phases") {
        for node in phase["graph"]["nodes"].as_array().expect("nodes") {
            let backend = node["backend"].as_str().unwrap_or("none").to_string();
            *counts.entry(backend).or_insert(0usize) += 1;
        }
    }

    // A CUDA build that initialised a device must name it; the CLI prints the
    // line on the first device use, so it is also the "is there a device" probe.
    // #736's selector was a build property the day it was written: `cfg!(feature =
    // "cuda")` is false in a macOS release build that still runs every node on Metal, so
    // the "a CPU run must not claim a device node" arm fired on a genuine Metal run and
    // this test went red on the MacBook (found by #723's Metal smoke, 2026-11-05 — CI's
    // `build-macos` only compiles). The run declares its own placement on stderr: CUDA
    // names the card it initialised, and a device holding blocks prints
    // `minfer: offload: … on <device>`. The arm follows that line, and every node must
    // name *some* backend — an unassigned node exports as `none`, which is the actual
    // shape of the bug #736 fixed.
    let device = if stderr.contains("CUDA: using") {
        Some("cuda")
    } else if stderr.contains("on metal") {
        Some("metal")
    } else {
        None
    };
    for name in ["cuda", "metal"] {
        let named = counts.get(name).copied().unwrap_or(0);
        if device == Some(name) {
            assert!(
                named > 0,
                "the run announced {name} but the trace marks no {name} node, got {counts:?}\n\
                 stderr:\n{stderr}"
            );
        } else {
            assert_eq!(
                named, 0,
                "a run that announced {device:?} must not claim {name} nodes, got {counts:?}\n\
                 stderr:\n{stderr}"
            );
        }
    }
    assert_eq!(
        counts.get("none").copied().unwrap_or(0),
        0,
        "every node must name the backend it ran on (an unassigned node exports as `none`), \
         got {counts:?}"
    );
    assert!(
        counts.values().sum::<usize>() > 0,
        "the trace must name at least one node, got {counts:?}"
    );
    eprintln!(
        "trace backends: {counts:?} (device={device:?}, cuda_build={}, metal_build={})",
        cfg!(feature = "cuda"),
        cfg!(target_os = "macos")
    );
    let _ = std::fs::remove_dir_all(&dir);
}
