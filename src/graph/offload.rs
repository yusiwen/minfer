//! Layer offload: how many transformer blocks run on the device (Phase E / ticket E5).
//!
//! Everything before this put **all** weights on one backend or none: the model-level gate
//! (`Qwen2Model::device`) asked "is every weight registered on the GPU?" and answered either
//! `Cuda`/`Metal` or `Cpu`. A model that does not fit in device memory therefore could not
//! run at all (`docs/ARCHITECTURE-ROADMAP.md` §2.6), which is a real limitation on the 8 GB
//! devices the plan is written against.
//!
//! An [`OffloadPlan`] puts the first `gpu_layers` blocks on the device and leaves the rest on
//! the CPU. Three places read the same number, and they must agree:
//!
//! - the **loader** registers a tensor on the device only when its block is offloaded, so the
//!   device never holds the weights of a block it will not execute (that is the whole point);
//! - the **graph builder** tags every node with its block (`CNode.layer`) and only emits a
//!   device-only fused node inside an offloaded block;
//! - the **assignment pass** (`BackendScheduler::assign_backends` →
//!   `GraphAllocator::supports_for`) keeps a node whose block is not offloaded off the device.
//!
//! Nodes outside any block — the token embedding, the final norm and `lm_head` — follow the
//! device only when **every** block is offloaded, so a partial plan never has to fit the two
//! largest tensors. That is llama.cpp's `n_gpu_layers > n_layer` convention for the output
//! layer, stated here once.
//!
//! The request spelling is the CLI's `--gpu-layers` or `MINFER_GPU_LAYERS`: unset (or empty)
//! means "every block the device can hold" — the pre-E5 behaviour — a decimal number means "at
//! most that many blocks", and anything else is refused loudly (`resolve`), never guessed.

use super::allocplan::DeviceMemory;

/// How many leading blocks run on the device, and how many there are in total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffloadPlan {
    /// Blocks `0..gpu_layers` run on the device; the rest run on the CPU.
    pub gpu_layers: usize,
    /// The model's block count, so `gpu_layers == n_layers` (all) is expressible and a
    /// report does not have to carry two numbers.
    pub n_layers: usize,
}

impl OffloadPlan {
    /// A plan with every block on the device (what an unset request resolves to when a
    /// device is available).
    pub fn all_on_device(n_layers: usize) -> Self {
        Self {
            gpu_layers: n_layers,
            n_layers,
        }
    }

    /// A plan with no block on the device (CPU only; the pre-E5 CPU path).
    pub fn all_on_cpu(n_layers: usize) -> Self {
        Self {
            gpu_layers: 0,
            n_layers,
        }
    }

    /// Blocks that run on the CPU.
    pub fn cpu_layers(&self) -> usize {
        self.n_layers.saturating_sub(self.gpu_layers)
    }

    /// Whether block `layer` runs on the device.
    pub fn on_device(&self, layer: usize) -> bool {
        layer < self.gpu_layers
    }

    /// Whether the tensors **outside** any block (embedding, final norm, `lm_head`) may run on
    /// the device. Only when every block is offloaded: a partial plan must not spend device
    /// memory on `token_embd`/`output`, which are the largest tensors in the model.
    pub fn device_holds_unblocked(&self) -> bool {
        self.gpu_layers >= self.n_layers
    }

    /// Some blocks on the device and some on the CPU — the case that needs cross-backend
    /// copies at every block boundary.
    pub fn is_mixed(&self) -> bool {
        self.gpu_layers > 0 && self.gpu_layers < self.n_layers
    }

    /// Every block on the device (the pre-E5 GPU path; `device_holds_unblocked` agrees).
    pub fn is_cpu_only(&self) -> bool {
        self.gpu_layers == 0
    }

    /// Whether a **registered weight name** may live on the device: the name's block must be
    /// offloaded, and a name outside any block only when every block is offloaded.
    ///
    /// The loader calls this for every tensor it is about to register (and for the fused
    /// `blk.{i}.attn_qkv` / `blk.{i}.ffn_gu` concat weights it builds per block), so the
    /// device never holds a block it will not execute.
    pub fn allows_weight(&self, name: &str) -> bool {
        match block_of(name) {
            Some(i) => self.on_device(i),
            None => self.device_holds_unblocked(),
        }
    }
}

/// The block a registered weight name belongs to: `{ns}blk.{i}.{...}` → `Some(i)`; anything
/// else (`token_embd`, `output_norm`, `output`, `output.bias`, …) → `None`.
///
/// Weight names are the registry's identity — the loader builds the fused names as
/// `blk.{i}.attn_qkv`, and the GGUF tensors arrive as `blk.{i}.attn_q.weight` — so the block
/// is recoverable from the name exactly where the loader has no per-tensor structure yet (it
/// registers each tensor as it reads it, before the layer vector exists).
pub fn block_of(name: &str) -> Option<usize> {
    let start = name.rfind("blk.")? + "blk.".len();
    let rest = &name[start..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// The `auto` spelling (E5 S2): fit as many blocks as the device budget allows.
pub const AUTO: &str = "auto";

/// The offload request as the CLI carries it: `Default` lets the environment decide,
/// `Layers(n)` is an explicit `--gpu-layers n`, and `Auto` asks for as many blocks as fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadRequest {
    #[default]
    Default,
    Layers(usize),
    /// E5 S2: the block count is *computed* from the budget and the model's per-block weight
    /// sizes — the loader resolves it (`fit_blocks`), because that is where the byte table is.
    Auto,
}

impl OffloadRequest {
    /// Parse the CLI/environment spelling, strictly: a decimal block count, `auto`, or
    /// unset/empty for the default. Anything else is refused loudly — a silently ignored
    /// offload request is indistinguishable from an offload that did not work.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.map(str::trim) {
            None | Some("") => Ok(OffloadRequest::Default),
            Some(v) if v.eq_ignore_ascii_case(AUTO) => Ok(OffloadRequest::Auto),
            Some(v) => v.parse::<usize>().map(OffloadRequest::Layers).map_err(|_| {
                format!(
                    "MINFER_GPU_LAYERS='{v}' is not a block count (a decimal number, `{AUTO}` to fit as many blocks as the device budget allows, or unset to offload every block the device can hold)"
                )
            }),
        }
    }

    /// The spelling this request came from, for the startup report.
    pub fn source(self, env: Option<&str>) -> String {
        match self {
            OffloadRequest::Layers(n) => format!("--gpu-layers {n}"),
            OffloadRequest::Auto => format!("--gpu-layers {AUTO}"),
            OffloadRequest::Default => match env.map(str::trim) {
                Some(v) if !v.is_empty() => format!("MINFER_GPU_LAYERS={v}"),
                _ => "default".to_string(),
            },
        }
    }

    /// Resolve the requests that need **no byte table**: the default and an explicit count.
    ///
    /// `Auto` needs the model's per-block weight sizes, so the loader resolves it with
    /// [`fit_blocks`] and this refuses it instead of guessing — a caller cannot forget.
    pub fn plan(
        self,
        env: Option<&str>,
        n_layers: usize,
        device_available: bool,
    ) -> Result<OffloadPlan, String> {
        match self {
            OffloadRequest::Default => resolve(env, n_layers, device_available),
            OffloadRequest::Layers(n) => Ok(OffloadPlan {
                gpu_layers: n.min(n_layers),
                n_layers,
            }),
            OffloadRequest::Auto => Err(
                "the `auto` offload request is resolved by the loader (it needs the model's per-block weight sizes), not here"
                    .to_string(),
            ),
        }
    }
}

/// Resolve the `MINFER_GPU_LAYERS` spelling into a plan.
///
/// `None`/empty = every block when a device is available, none when it is not (so a CPU-only
/// box behaves exactly as before). An explicit count is clamped to the model's block count;
/// a non-numeric value is an error the loader turns into a refused load, because a silently
/// ignored offload request is indistinguishable from an offload that did not work.
pub fn resolve(
    requested: Option<&str>,
    n_layers: usize,
    device_available: bool,
) -> Result<OffloadPlan, String> {
    let gpu_layers = match requested.map(str::trim) {
        None | Some("") => {
            if device_available {
                n_layers
            } else {
                0
            }
        }
        Some(v) if v.eq_ignore_ascii_case(AUTO) => {
            return Err(
                "MINFER_GPU_LAYERS=auto is resolved by the loader (see `fit_blocks`), not by \
 `resolve`"
                    .to_string(),
            )
        }
        Some(v) => v.parse::<usize>().map_err(|_| {
            format!(
                "MINFER_GPU_LAYERS='{v}' is not a block count (a decimal number, `auto`, or \
                 unset to offload every block the device can hold)"
            )
        })?,
    };
    Ok(OffloadPlan {
        gpu_layers: gpu_layers.min(n_layers),
        n_layers,
    })
}

/// The startup line E5's third acceptance asks for: which blocks landed where, how much
/// device memory the offloaded weights occupy, and where the request came from.
///
/// Pure (the caller supplies the device name and the measured bytes) so the formatting is
/// unit-tested on a CPU-only box.
pub fn report(plan: OffloadPlan, device: &str, device_bytes: usize, source: &str) -> String {
    let mib = |b: usize| format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0));
    if plan.is_cpu_only() {
        return format!(
            "offload: cpu only — 0/{} blocks on the device ({source})",
            plan.n_layers
        );
    }
    if plan.device_holds_unblocked() {
        return format!(
            "offload: all {} blocks + embed/output on {device} ({} of device weights; {source})",
            plan.n_layers,
            mib(device_bytes)
        );
    }
    format!(
        "offload: {} of {} blocks on {device}, {} on cpu; embed/output on cpu ({} of device \
         weights; {source})",
        plan.gpu_layers,
        plan.n_layers,
        plan.cpu_layers(),
        mib(device_bytes)
    )
}

/// E5 S2: the weight budget an `auto` request fits into.
///
/// An explicit `MINFER_GPU_MEM` (MiB) wins. Otherwise it is the same default E4's
/// feasibility gate uses — three quarters of what the device reports free — so the fit
/// and the gate that later checks the activation pool are talking about the same number.
///
/// A **failed** device query refuses the `auto` request (issue #122) instead of reading
/// it as "0 bytes free". The pre-#122 `free_bytes.map_or(0, …)` turned a failed query
/// into a 0-byte budget, and `fit_blocks(0, …)` then planned **0 device blocks** while
/// the startup line reported `device free 0 MiB (three quarters of it, …)` as if that
/// were a measurement. `auto` asks for a measurement, so when none exists the load is
/// refused with the real error name rather than silently planned around.
///
/// "No device at all" is unchanged and deliberate: it is the documented Metal behaviour
/// (`auto` fits nothing without `MINFER_GPU_MEM`), not a failed query.
pub fn weight_budget(mem: &DeviceMemory, cap_mib: Option<&str>) -> Result<usize, String> {
    match cap_mib.map(str::trim) {
        Some(v) if !v.is_empty() => v
            .parse::<usize>()
            .map(|mib| mib * 1024 * 1024)
            .map_err(|_| format!("MINFER_GPU_MEM='{v}' is not a size in MiB (a decimal number)")),
        _ => match mem {
            DeviceMemory::Reported { free, .. } => Ok(free / 4 * 3),
            DeviceMemory::QueryFailed { .. } => Err(format!(
                "MINFER_GPU_LAYERS=auto needs the device's free bytes, but {}; \
                 set MINFER_GPU_MEM=<MiB> to plan against a number you choose",
                mem.failure_note("the device free-memory query")
                    .unwrap_or_else(|| "the query failed".to_string())
            )),
            DeviceMemory::NoDevice => Ok(0),
        },
    }
}

/// E5 S2: the largest **prefix** of blocks whose weights fit in `budget`, after `reserve` bytes
/// are held back for what is not weights (the KV arenas and the activation pool).
///
/// A prefix, not a subset: the offload plan is `0..gpu_layers` by construction, and a gap would
/// mean a CPU block between two device blocks for no reason. So the walk stops at the first
/// block that does not fit — a large block can make the fit smaller than a knapsack would.
///
/// Pure, so CI covers the matrix with no device: the caller supplies the *measured* per-block
/// bytes (from the GGUF tensor index, before anything is registered) and the budget.
pub fn fit_blocks(budget: usize, per_block: &[usize], reserve: usize) -> usize {
    let mut used = reserve;
    let mut k = 0;
    for &bytes in per_block {
        if used.saturating_add(bytes) > budget {
            break;
        }
        used += bytes;
        k += 1;
    }
    k
}

/// E5 S2: the `auto` request's explanation for the startup report — what the fit decided and
/// against which numbers, so a surprising block count is traceable.
pub fn auto_source(
    k: usize,
    n_layers: usize,
    budget: usize,
    reserve: usize,
    mem: &DeviceMemory,
    cap_mib: Option<&str>,
) -> String {
    let mib = |b: usize| format!("{:.0} MiB", b as f64 / (1024.0 * 1024.0));
    let against = match cap_mib.map(str::trim) {
        Some(v) if !v.is_empty() => format!("MINFER_GPU_MEM={v} MiB"),
        // A failed query never reaches here (the loader refuses first), but it must not
        // read as a measured "0 MiB" if it ever does: say it was not measured (#122).
        _ => match mem {
            DeviceMemory::Reported { free, .. } => format!(
                "device free {} (three quarters of it, the E4 default budget)",
                mib(*free)
            ),
            DeviceMemory::QueryFailed { .. } => {
                "device free unmeasured (the device free-memory query failed)".to_string()
            }
            DeviceMemory::NoDevice => "no device budget".to_string(),
        },
    };
    format!(
        "auto: {k} of {n_layers} blocks fit — weights budget {}, {} reserved for KV/activations; {against}",
        mib(budget),
        mib(reserve)
    )
}

#[cfg(test)]
mod tests {
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
}
