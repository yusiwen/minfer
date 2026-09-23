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

/// The offload request as the CLI carries it: `Default` lets the environment decide, and
/// `Layers(n)` is an explicit `--gpu-layers n` (already parsed, so it cannot be garbage).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadRequest {
    #[default]
    Default,
    Layers(usize),
}

impl OffloadRequest {
    /// The spelling this request came from, for the startup report.
    pub fn source(self, env: Option<&str>) -> String {
        match self {
            OffloadRequest::Layers(n) => format!("--gpu-layers {n}"),
            OffloadRequest::Default => match env.map(str::trim) {
                Some(v) if !v.is_empty() => format!("MINFER_GPU_LAYERS={v}"),
                _ => "default".to_string(),
            },
        }
    }

    /// Resolve this request against the environment spelling, the model's block count and
    /// whether a device is available at all. Pure, so CI covers the matrix without a GPU —
    /// the same reason E6 made `batch_mode` pure.
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
        Some(v) => v.parse::<usize>().map_err(|_| {
            format!(
                "MINFER_GPU_LAYERS='{v}' is not a block count (a decimal number, or unset to \
                 offload every block the device can hold)"
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
