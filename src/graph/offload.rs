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
    /// A plan with every block on the device — the plan an unset request means when a
    /// device is available.
    ///
    /// [#239](https://github.com/yusiwen/minfer/issues/239) reported the lost caller this
    /// constructor had: the resolution path built the plan inline, so only the tests reached
    /// it. [#244](https://github.com/yusiwen/minfer/issues/244) wired the caller instead of
    /// deleting the constructor — `resolve`'s unset-request arm now returns this (or
    /// [`Self::all_on_cpu`]) directly, which is the same `OffloadPlan` the inline form built.
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
    /// The budget is `MINFER_GPU_MEM=<MiB>` when set, else the device default.
    Auto,
    /// E5 S2 with an **explicit** weight budget in MiB — the number
    /// `MINFER_GPU_MEM=<n>` would supply, without touching the process-wide
    /// environment (issue #185). The explicit-argument spelling exists because
    /// the environment is shared by every thread: the E5 gate used to
    /// `set_var("MINFER_GPU_MEM", "64")`, which a concurrently loading test then
    /// read, and which its own second arm had to unset. The repo's convention is
    /// an explicit argument over a mutated environment — `load_model_configured`'s
    /// explicit cache type is the precedent (#99, #153).
    ///
    /// Retained with a `not(test)` allowance ([#244]): the loader reads it
    /// (`budget_mib`/`source`) and the E5 gates construct it, but no production
    /// spelling produces it — that is deliberate, because an environment variable
    /// reaching it would be the process global this variant exists to avoid. A CLI
    /// flag or a server field that carries the budget as an argument is what would
    /// construct it.
    ///
    /// [#244]: https://github.com/yusiwen/minfer/issues/244
    #[cfg_attr(not(test), allow(dead_code))]
    AutoWithBudget(usize),
}

impl OffloadRequest {
    /// Parse the CLI/environment spelling, strictly: a decimal block count, `auto`, or
    /// unset/empty for the default. Anything else is refused loudly — a silently ignored
    /// offload request is indistinguishable from an offload that did not work.
    ///
    /// `AutoWithBudget` has no spelling on purpose: it is the explicit-argument
    /// form (#185), and an environment variable that reached it would be the
    /// process-global this variant exists to avoid.
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

    /// The explicit budget this request carries, if any (the loader reads
    /// `MINFER_GPU_MEM` only when this is `None`).
    pub fn budget_mib(self) -> Option<usize> {
        match self {
            OffloadRequest::AutoWithBudget(mib) => Some(mib),
            _ => None,
        }
    }

    /// The spelling this request came from, for the startup report.
    pub fn source(self, env: Option<&str>) -> String {
        match self {
            OffloadRequest::Layers(n) => format!("--gpu-layers {n}"),
            OffloadRequest::Auto => format!("--gpu-layers {AUTO}"),
            OffloadRequest::AutoWithBudget(mib) => {
                format!("--gpu-layers {AUTO} (explicit budget {mib} MiB)")
            }
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
            OffloadRequest::Auto | OffloadRequest::AutoWithBudget(_) => Err(
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
            // The unset request is exactly one of the two named plans. Expressed through
            // the constructors, so the plan and its name cannot drift; this is the lost
            // caller #239 reported for `all_on_device` (#244).
            return Ok(if device_available {
                OffloadPlan::all_on_device(n_layers)
            } else {
                OffloadPlan::all_on_cpu(n_layers)
            });
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
/// "No device state at all" is unchanged and deliberate: the CPU has no device budget,
/// and a device whose singleton could not be created is not a failed *query* — `auto`
/// then fits nothing without `MINFER_GPU_MEM`, not a fabricated zero. Metal used to be
/// that case on every Mac; since [#53](https://github.com/yusiwen/minfer/issues/53) it
/// answers through `MpsState::device_memory`, so `auto` plans against
/// `recommendedMaxWorkingSetSize` there too.
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

/// E5 S2 + #185: the same startup line for an **explicit** budget that did not
/// come from `MINFER_GPU_MEM` ([`OffloadRequest::AutoWithBudget`]).
///
/// Separate from [`auto_source`] so the provenance in the report is true: the
/// environment spelling says `MINFER_GPU_MEM=…`, the explicit spelling says
/// `explicit budget <mib> MiB (OffloadRequest::AutoWithBudget, not
/// MINFER_GPU_MEM)`. A report that named the environment for a number the
/// environment never held would be a lie a reader could not catch.
pub fn auto_source_explicit(
    k: usize,
    n_layers: usize,
    budget: usize,
    reserve: usize,
    mib: usize,
) -> String {
    let m = |b: usize| format!("{:.0} MiB", b as f64 / (1024.0 * 1024.0));
    format!(
        "auto: {k} of {n_layers} blocks fit — weights budget {}, {} reserved for KV/activations; \
         explicit budget {mib} MiB (OffloadRequest::AutoWithBudget, not MINFER_GPU_MEM)",
        m(budget),
        m(reserve)
    )
}

#[cfg(test)]
mod tests;
