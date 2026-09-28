//! `#[cfg(test)] mod tests` for `src/graph/offload.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn an_unset_request_offloads_everything_a_device_can_hold() {
    // The pre-E5 behaviour, and the CPU-only behaviour, are both "unset".
    assert_eq!(
        resolve(None, 24, true).unwrap(),
        OffloadPlan::all_on_device(24)
    );
    assert_eq!(
        resolve(Some(""), 24, true).unwrap(),
        OffloadPlan::all_on_device(24)
    );
    assert_eq!(
        resolve(None, 24, false).unwrap(),
        OffloadPlan::all_on_cpu(24)
    );
    assert_eq!(
        resolve(Some(" "), 24, true).unwrap(),
        OffloadPlan::all_on_device(24)
    );
}

#[test]
fn an_explicit_count_is_clamped_and_never_guessed() {
    assert_eq!(
        resolve(Some("0"), 24, true).unwrap(),
        OffloadPlan::all_on_cpu(24)
    );
    let p = resolve(Some("4"), 24, true).unwrap();
    assert_eq!((p.gpu_layers, p.cpu_layers()), (4, 20));
    assert!(p.is_mixed() && !p.device_holds_unblocked());
    // More blocks than the model has is "all", not an error: the request is a ceiling.
    assert_eq!(
        resolve(Some("99"), 24, true).unwrap(),
        OffloadPlan::all_on_device(24)
    );
    // Garbage is refused loudly (the loader turns this into a failed load).
    let err = resolve(Some("banana"), 24, true).unwrap_err();
    assert!(
        err.contains("banana") && err.contains("MINFER_GPU_LAYERS"),
        "{err}"
    );
    assert!(resolve(Some("-1"), 24, true).is_err());
}

#[test]
fn the_cli_form_bypasses_the_environment() {
    // `--gpu-layers 2` wins over an environment that says something else.
    let p = OffloadRequest::Layers(2).plan(Some("9"), 24, true).unwrap();
    assert_eq!(p.gpu_layers, 2);
    // ...and `Default` is the environment.
    let p = OffloadRequest::Default.plan(Some("9"), 24, true).unwrap();
    assert_eq!(p.gpu_layers, 9);
    // An explicit 0 is CPU-only even when the environment asks for everything.
    assert!(OffloadRequest::Layers(0)
        .plan(None, 24, true)
        .unwrap()
        .is_cpu_only());
}

#[test]
fn only_a_full_plan_puts_the_unblocked_tensors_on_the_device() {
    let mixed = resolve(Some("4"), 24, true).unwrap();
    assert!(!mixed.device_holds_unblocked());
    assert!(resolve(Some("24"), 24, true)
        .unwrap()
        .device_holds_unblocked());
    // A model with no blocks at all is "full" by definition (nothing to leave behind).
    assert!(resolve(Some("0"), 0, true)
        .unwrap()
        .device_holds_unblocked());
}

#[test]
fn the_request_spelling_is_parsed_strictly() {
    assert_eq!(
        OffloadRequest::parse(None).unwrap(),
        OffloadRequest::Default
    );
    assert_eq!(
        OffloadRequest::parse(Some("")).unwrap(),
        OffloadRequest::Default
    );
    assert_eq!(
        OffloadRequest::parse(Some(" ")).unwrap(),
        OffloadRequest::Default
    );
    assert_eq!(
        OffloadRequest::parse(Some("4")).unwrap(),
        OffloadRequest::Layers(4)
    );
    assert_eq!(
        OffloadRequest::parse(Some("0")).unwrap(),
        OffloadRequest::Layers(0)
    );
    // `auto` is case-insensitive and trimmed; a garbage spelling names the alternatives.
    assert_eq!(
        OffloadRequest::parse(Some(" AUTO ")).unwrap(),
        OffloadRequest::Auto
    );
    let err = OffloadRequest::parse(Some("banana")).unwrap_err();
    assert!(err.contains("banana") && err.contains("auto"), "{err}");
    assert!(OffloadRequest::parse(Some("-1")).is_err());
    // `auto` is the loader's job: `plan` refuses it rather than guessing a block count.
    assert!(OffloadRequest::Auto.plan(None, 24, true).is_err());
    assert_eq!(
        OffloadRequest::Layers(6)
            .plan(None, 24, true)
            .unwrap()
            .gpu_layers,
        6
    );
    assert!(OffloadRequest::Auto.source(None).contains("auto"));
}

#[test]
fn the_fit_takes_the_largest_prefix_that_fits() {
    // Exact boundary: 100 + 25 == 125 fits, one byte more does not.
    let blocks = [40, 40, 40];
    assert_eq!(fit_blocks(125, &blocks, 25), 2);
    assert_eq!(fit_blocks(124, &blocks, 25), 2);
    assert_eq!(fit_blocks(145, &blocks, 25), 3);
    assert_eq!(fit_blocks(85, &blocks, 25), 1);
    // A reserve alone can consume the budget.
    assert_eq!(fit_blocks(25, &blocks, 25), 0);
    assert_eq!(fit_blocks(0, &blocks, 0), 0);
    // Empty table, zero-size blocks, and a huge budget.
    assert_eq!(fit_blocks(1000, &[], 100), 0);
    assert_eq!(fit_blocks(1000, &[0, 0, 0], 100), 3);
    assert_eq!(fit_blocks(usize::MAX, &blocks, 0), 3);
    // Prefix, not knapsack: the first block that does not fit stops the walk even though
    // the later, smaller ones would.
    assert_eq!(fit_blocks(100, &[90, 80, 5, 5], 10), 1);
}

#[test]
fn the_weight_budget_prefers_the_explicit_cap() {
    let four = DeviceMemory::Reported {
        free: 4 << 20,
        total: 8 << 20,
    };
    // No cap: three quarters of what the device reports — the same default E4's gate uses.
    assert_eq!(weight_budget(&four, None).unwrap(), 3 << 20);
    assert_eq!(weight_budget(&four, Some("")).unwrap(), 3 << 20);
    assert_eq!(weight_budget(&four, Some("  ")).unwrap(), 3 << 20);
    // No device and no cap: nothing fits (the CPU-only / Metal answer).
    assert_eq!(weight_budget(&DeviceMemory::NoDevice, None).unwrap(), 0);
    // An explicit cap is MiB, and wins over the device's number.
    assert_eq!(weight_budget(&four, Some("64")).unwrap(), 64 << 20);
    let err = weight_budget(&four, Some("lots")).unwrap_err();
    assert!(err.contains("lots") && err.contains("MiB"), "{err}");
}

/// Issue #122, E5's half: a failed query must not plan "0 blocks fit".
///
/// The mutation this pins is `free_bytes.map_or(0, |f| f / 4 * 3)`: with the CUDA
/// read collapsed to 0 by a discarded return code, `auto` fitted **0 blocks** and
/// the startup line reported `device free 0 MiB` as though the device were full.
#[test]
fn a_failed_device_query_refuses_an_auto_fit() {
    let failed = DeviceMemory::QueryFailed {
        code: 700,
        name: "cudaErrorIllegalAddress".to_string(),
    };
    let err = weight_budget(&failed, None).unwrap_err();
    assert!(err.contains("cudaErrorIllegalAddress"), "{err}");
    assert!(err.contains("700"), "{err}");
    assert!(
        err.contains("MINFER_GPU_MEM"),
        "offer the escape hatch: {err}"
    );
    assert!(
        !err.contains("0 MiB") && !err.contains("0 bytes"),
        "the refusal must not quote a fabricated measurement: {err}"
    );
    // An explicit cap still plans, because the caller supplied the measurement.
    assert_eq!(weight_budget(&failed, Some("64")).unwrap(), 64 << 20);
}

#[test]
fn the_auto_source_names_what_the_fit_decided() {
    let four = DeviceMemory::Reported {
        free: 4 << 20,
        total: 8 << 20,
    };
    // Device budget: say what the device reported and that it was held back.
    let line = auto_source(6, 24, 3 << 20, 1 << 20, &four, None);
    assert!(line.contains("auto: 6 of 24 blocks fit"), "{line}");
    assert!(line.contains("3 MiB"), "{line}");
    assert!(line.contains("1 MiB reserved"), "{line}");
    assert!(line.contains("device free 4 MiB"), "{line}");
    // Explicit cap: name it instead.
    let line = auto_source(2, 24, 64 << 20, 16 << 20, &four, Some("64"));
    assert!(line.contains("MINFER_GPU_MEM=64 MiB"), "{line}");
    // No budget at all: say so rather than implying a device.
    let line = auto_source(0, 24, 0, 0, &DeviceMemory::NoDevice, None);
    assert!(line.contains("no device budget"), "{line}");
    // A failed query must not read as a measured zero.
    let line = auto_source(
        0,
        24,
        0,
        0,
        &DeviceMemory::QueryFailed {
            code: 1,
            name: "x".into(),
        },
        None,
    );
    assert!(line.contains("unmeasured"), "{line}");
    assert!(!line.contains("device free 0 MiB"), "{line}");
}

/// #185: the explicit-argument budget spelling
/// ([`OffloadRequest::AutoWithBudget`]). It is deliberately unreachable from
/// the environment — that is the whole point — it carries its MiB value, the
/// **loader** (not `plan`) resolves it, and the report names the argument
/// rather than an environment variable that never held the number.
#[test]
fn an_explicit_auto_budget_is_an_argument_not_an_environment_variable() {
    let req = OffloadRequest::AutoWithBudget(64);
    assert_eq!(req.budget_mib(), Some(64));
    assert_eq!(OffloadRequest::Auto.budget_mib(), None);
    assert_eq!(OffloadRequest::Layers(3).budget_mib(), None);
    // No spelling produces it: an env var that reached this variant would be the
    // process-global the variant exists to avoid.
    for spelling in [None, Some(""), Some("auto"), Some("64")] {
        assert!(
            OffloadRequest::parse(spelling)
                .unwrap()
                .budget_mib()
                .is_none(),
            "spelling {spelling:?} must not carry an explicit budget"
        );
    }
    // The loader resolves `auto` (it needs the per-block byte table).
    assert!(req.plan(None, 24, true).is_err());
    assert!(req.source(None).contains("explicit budget 64 MiB"));
    // The report's provenance is honest: the explicit spelling never claims the
    // environment.
    let line = auto_source_explicit(5, 24, 64 << 20, 16 << 20, 64);
    assert!(line.contains("auto: 5 of 24 blocks fit"), "{line}");
    assert!(line.contains("explicit budget 64 MiB"), "{line}");
    assert!(
        !line.contains("MINFER_GPU_MEM="),
        "an explicit budget must not be reported as an environment variable: {line}"
    );
    // Positive control for the `!contains` above: the environment spelling does
    // name the variable, so the two report forms are distinguishable.
    let env_line = auto_source(
        5,
        24,
        64 << 20,
        16 << 20,
        &DeviceMemory::Reported {
            free: 4 << 20,
            total: 8 << 20,
        },
        Some("64"),
    );
    assert!(env_line.contains("MINFER_GPU_MEM=64 MiB"), "{env_line}");
}

#[test]
fn the_weight_filter_follows_the_block() {
    let mixed = resolve(Some("4"), 24, true).unwrap();
    assert!(mixed.allows_weight("blk.3.attn_q.weight"));
    assert!(mixed.allows_weight("draft.blk.3.attn_qkv"));
    assert!(!mixed.allows_weight("blk.4.attn_q.weight"));
    // Tensors outside any block only follow a full plan: a partial one must not spend
    // device memory on token_embd/output.
    assert!(!mixed.allows_weight("token_embd.weight"));
    assert!(!mixed.allows_weight("output.weight"));
    assert!(!mixed.allows_weight("output_norm.weight"));
    let full = resolve(None, 24, true).unwrap();
    assert!(full.allows_weight("token_embd.weight") && full.allows_weight("output.weight"));
    // The parser reads the *registry* spelling the loader builds, and refuses to guess:
    // no delimiter, no digits, or no `blk.` at all is not a block.
    assert_eq!(block_of("blk.12.ffn_gu"), Some(12));
    assert_eq!(block_of("draft.blk.0.attn_qkv"), Some(0));
    assert_eq!(block_of("blk.0attn"), None);
    assert_eq!(block_of("blk..attn"), None);
    assert_eq!(block_of("token_embd.weight"), None);
}

#[test]
fn the_report_says_which_blocks_landed_where() {
    let mixed = resolve(Some("4"), 24, true).unwrap();
    let line = report(mixed, "cuda", 39_845_888, "--gpu-layers 4");
    assert!(line.contains("4 of 24 blocks on cuda"), "{line}");
    assert!(line.contains("20 on cpu"), "{line}");
    assert!(line.contains("embed/output on cpu"), "{line}");
    assert!(line.contains("38.0 MiB"), "{line}");
    assert!(line.contains("--gpu-layers 4"), "{line}");

    let full = resolve(None, 24, true).unwrap();
    let line = report(full, "cuda", 1 << 20, "default");
    assert!(
        line.contains("all 24 blocks + embed/output on cuda"),
        "{line}"
    );

    let none = resolve(Some("0"), 24, true).unwrap();
    let line = report(none, "cuda", 0, "MINFER_GPU_LAYERS=0");
    assert!(line.contains("cpu only"), "{line}");
    assert!(line.contains("0/24 blocks"), "{line}");

    // A CPU-only plan still answers when the request was explicit — and the *default*
    // CPU path (no device, nothing asked for) stays silent, which is the pre-E5 CLI.
    let silent = crate::models::OffloadState::cpu_only(24);
    assert_eq!(silent.report(crate::models::Device::Cpu), None);
    let explicit = crate::models::OffloadState {
        source: "--gpu-layers 0".to_string(),
        ..silent.clone()
    };
    let line = explicit
        .report(crate::models::Device::Cpu)
        .expect("an explicit 0 reports");
    assert!(
        line.contains("cpu only") && line.contains("--gpu-layers 0"),
        "{line}"
    );
}
