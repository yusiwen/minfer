//! Backend registry (F4 / [#57]).
//!
//! Backends used to be a compile-time enum: every consumer — the allocator, the
//! scheduler, the fusion wiring, the exporters and the KV-session tag table —
//! matched on it, so adding one touched all of them, and nothing could be asked
//! for by name. This module makes the set of backends *data*:
//!
//! - [`Backend`] is a cheap `Copy`/`Hash`/`Eq`/`Ord` **handle** — an opaque id,
//!   not an enum. The id space is fixed and configuration-independent
//!   (`cpu = 0`, `metal = 1`, `cuda = 2`) because it is already on disk (the
//!   KV-session backend tag) and in the exported graph documents, and because
//!   the pre-F4 derived `Ord` on the enum was declaration order.
//! - [`BackendEntry`] carries everything a backend *is*: its name, its
//!   assignment priority, its capability matrix, and the hooks that reach its
//!   pool (the "sync/copy arms"), its host-read path, its KV element format and
//!   its lazy enable. Each backend module registers its own entry; a backend
//!   that is not compiled in simply has no entry, while its **name** still
//!   resolves and still produces an accurate refusal.
//! - [`Registry`] is the name-keyed table, built once at startup.
//!
//! Design record: `docs/BACKEND-REGISTRY-DESIGN.md`.
//!
//! ## Two orders, and the determinism rule
//!
//! The registry has two orders and they are deliberately not the same:
//!
//! - **identity** — [`Backend::index`], the id order. It fixes the on-disk
//!   KV-session tag, the exported graph's backend index, `Ord`, and the order of
//!   the fusion pass's backend list.
//! - **priority** — [`Registry::by_priority`], descending
//!   [`BackendEntry::priority`]. It is the *assignment preference*: which
//!   backend `GraphAllocator::supports_for` offers first. Before F4 it was
//!   implicit in the statement order of that function (Metal, then CUDA, then
//!   CPU); it is preserved exactly and is now a number the ordering gate pins.
//!
//! Nothing here may depend on `HashMap` iteration order: the assignment pass
//! fixes a graph's topology, and topology is part of the reuse identity
//! (standing rule 3). The table is a fixed-size array indexed by the handle id,
//! `by_priority` is sorted by `(Reverse(priority), index)` so no two entries can
//! tie, and [`BackendFilter`] is a `[bool; N_BACKENDS]` indexed by id rather
//! than an iterated set.
//!
//! [#57]: https://github.com/yusiwen/minfer/issues/57

use std::fmt;
use std::sync::OnceLock;

use super::alloc::GraphAllocator;
use super::backend::Backend as BackendTrait;
use super::kvformat::KvFormat;
use super::ops::{FusedOp, Op};
use super::{DType, NodeId};

/// How many backend ids exist. Fixed: the id is a file-format contract.
pub const N_BACKENDS: usize = 3;

/// The accepted names, in id order. Unconditional — a name is known on every
/// configuration, even where its backend is not compiled in.
const NAMES: [&str; N_BACKENDS] = ["cpu", "metal", "cuda"];

/// How a handle prints in diagnostics. Spelled exactly as the pre-F4 enum
/// variants printed, so no log line or error message changes shape.
const DEBUG_NAMES: [&str; N_BACKENDS] = ["CPU", "Metal", "Cuda"];

/// Assignment priorities (higher wins). The values are arbitrary but pinned by
/// `the_registered_set_and_priority_order_are_pinned`; the *order* is the
/// pre-F4 behaviour (Metal, then CUDA, then CPU).
///
/// Each device constant is gated on the cfg whose backend module uses it, so a
/// build without that backend does not carry it (`metal_backend.rs` is
/// `target_os = "macos"`, `cuda_backend.rs` is `feature = "cuda"`).
#[cfg(target_os = "macos")]
pub const PRIORITY_METAL: u16 = 300;
#[cfg(feature = "cuda")]
pub const PRIORITY_CUDA: u16 = 200;
pub const PRIORITY_CPU: u16 = 100;

/// A registered backend, as a cheap index into the registry.
///
/// The constants are the only handles the code should name; [`Backend::from_index`]
/// exists for the id spaces that persist one (the KV-session tag).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Backend(u16);

impl Backend {
    pub const CPU: Backend = Backend(0);
    pub const METAL: Backend = Backend(1);
    pub const CUDA: Backend = Backend(2);

    /// The registry id (and the on-disk KV-session backend tag).
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// The handle for a registry id, or `None` when no backend has it.
    pub fn from_index(index: usize) -> Option<Backend> {
        if index < N_BACKENDS {
            Some(Backend(index as u16))
        } else {
            None
        }
    }

    /// The canonical name. Known on every configuration: a compiled-out backend
    /// still has a name, which is what makes its refusal accurate.
    pub fn name(self) -> &'static str {
        NAMES.get(self.index()).copied().unwrap_or("unknown")
    }

    /// This handle's registry entry, or `None` when the backend is not compiled
    /// into this build.
    pub fn entry(self) -> Option<&'static BackendEntry> {
        registry().get(self)
    }

    /// This handle's capabilities. An unregistered backend supports nothing —
    /// the safe answer, and the pre-F4 answer for the configurations where the
    /// backend did not exist.
    pub fn caps(self) -> BackendCaps {
        self.entry().map_or(BackendCaps::NOTHING, |e| e.caps)
    }

    /// Whether this build contains the backend at all.
    pub fn is_registered(self) -> bool {
        self.entry().is_some()
    }
}

impl fmt::Debug for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match DEBUG_NAMES.get(self.index()) {
            Some(name) => f.write_str(name),
            None => write!(f, "Backend({})", self.0),
        }
    }
}

fn no_op_support(_op: &Op, _dtype: DType) -> bool {
    false
}

fn no_fused_support(_fused: &FusedOp) -> bool {
    false
}

/// A backend's capability matrix — the registry's copy of what the `Backend`
/// trait answers. Each backend module defines the matrix once and its trait impl
/// forwards to it, so the registry's answer and the trait's answer are the same
/// code and cannot diverge.
#[derive(Clone, Copy)]
pub struct BackendCaps {
    #[cfg_attr(not(test), allow(dead_code))]
    pub supports_op: fn(&Op, DType) -> bool,
    #[cfg_attr(not(test), allow(dead_code))]
    pub supports_fused: fn(&FusedOp) -> bool,
    /// Whether attention can be bounded from the explicit `attn_span` (E1).
    #[cfg_attr(not(test), allow(dead_code))]
    pub supports_attn_span: bool,
    /// Whether the attention kernel reads a packed `q8_0` KV region (C4).
    ///
    /// The single authority for that answer, read by
    /// `GraphAllocator::ensure_kv` and `KvFormat::supports`. Flipping it for a
    /// device is [#87]'s work (the kernels); the field exists because this
    /// ticket's own code needed the question asked of the registry instead of a
    /// hardcoded CPU test.
    ///
    /// [#87]: https://github.com/yusiwen/minfer/issues/87
    pub reads_packed_kv: bool,
}

impl BackendCaps {
    /// The capabilities of a backend that is not compiled in: it claims nothing.
    pub const NOTHING: BackendCaps = BackendCaps {
        supports_op: no_op_support,
        supports_fused: no_fused_support,
        supports_attn_span: false,
        reads_packed_kv: false,
    };
}

/// Everything one backend is. Registered by its own module at startup.
pub struct BackendEntry {
    pub handle: Backend,
    /// The canonical name (`resolve_name` matches it).
    #[cfg_attr(not(test), allow(dead_code))]
    pub name: &'static str,
    /// Assignment priority, higher first (see the module docs).
    pub priority: u16,
    pub caps: BackendCaps,
    /// This backend's pool on an allocator, as the trait object every pool
    /// operation goes through. `None` when the pool is not enabled there (MPS
    /// unavailable, no CUDA device, a session restore before the graph was
    /// built).
    pub pool: fn(&GraphAllocator) -> Option<&dyn BackendTrait>,
    /// The mutable form of [`Self::pool`].
    pub pool_mut: fn(&mut GraphAllocator) -> Option<&mut dyn BackendTrait>,
    /// Read one pool buffer back to the host. CUDA's stream-ordered
    /// `copy_to_host` is why this is not just `Backend::read_host`.
    pub host_read: fn(&GraphAllocator, usize) -> Option<Vec<f32>>,
    /// F5 ([#58]): **phase A** of a cross-backend staging copy — move the output
    /// of `node` (produced by *this* backend, at source reference `src`) into the
    /// destination backend's staging buffer `dst`.
    ///
    /// `Ok(true)` means this hook issued the transfer: a backend with device
    /// memory enqueues it asynchronously (an `cudaMemcpyAsync` plus a recorded
    /// event) and [`Self::await_cross`] is what completes it; a backend without
    /// device memory performs the synchronous host round trip it always did
    /// (there is no copy to make asynchronous). `Ok(false)` **declines**, and the
    /// allocator's synchronous host round trip handles the pair exactly as it did
    /// before F5 — the honest answer for a direction nobody has ported yet (a
    /// device→device staging copy, which `copy_across`'s same-backend early
    /// return makes unreachable today; CUDA→Metal on a macOS+CUDA build; Metal
    /// until it is ported and verified on a Mac).
    ///
    /// The hook owns the event it records; the allocator owns the staging buffer,
    /// the one-wait-per-copy contract and the counters
    /// (`graph::copystats::CrossCopyStats`).
    ///
    /// [#58]: https://github.com/yusiwen/minfer/issues/58
    pub copy_cross: fn(&mut GraphAllocator, u64, NodeId, Backend) -> Result<bool, String>,
    /// F5 ([#58]): **phase B** — wait on the event [`Self::copy_cross`] recorded,
    /// exactly once per staged input, before the consuming split executes. A
    /// backend that recorded no event (its phase A was synchronous, or it
    /// declined) leaves this a no-op; a device consumer inserts a device-side
    /// wait instead of blocking the host.
    ///
    /// **This is the wait the missing-wait gate is about.** Dropping it leaves the
    /// staging entry marked pending, and the scheduler's next read of that entry
    /// (`GraphAllocator::cross_input`) is a loud error rather than a read of
    /// undefined data — see `docs/BACKEND-REGISTRY-DESIGN.md` §11.
    ///
    /// [#58]: https://github.com/yusiwen/minfer/issues/58
    pub await_cross: fn(&mut GraphAllocator, u64, NodeId, Backend) -> Result<(), String>,
    /// The KV element type this backend's regions store (C5 records it in the
    /// session header). Takes the allocator because the answer belongs to the
    /// engine it serves: per-engine (#99, and #153 for CUDA) the CPU hook is the
    /// allocator's stamped `GraphAllocator::set_kv_format` and the CUDA hook is the
    /// backend's own `kv_layout`, both derived from the loaded engine's resolved
    /// format. **Metal's** hook still reads the process-wide `metal::kv_cache_is_f16`
    /// — the one device-static left.
    pub kv_format: fn(&GraphAllocator) -> KvFormat,
    /// Bring the pool up if this build/machine can (idempotent); `false` means
    /// it cannot.
    pub enable: fn(&mut GraphAllocator) -> bool,
    /// Why a *compiled-in* backend cannot run here, or `None` when it can.
    pub unavailable: fn() -> Option<&'static str>,
}

/// The name-keyed table of registered backends.
pub struct Registry {
    entries: [Option<BackendEntry>; N_BACKENDS],
    /// Handles in assignment-priority order (highest first). A `Vec` computed
    /// once at build time, never an iteration over a hash map.
    priority: Vec<Backend>,
}

impl Registry {
    fn build() -> Registry {
        let mut r = Registry {
            entries: [None, None, None],
            priority: Vec::new(),
        };
        super::cpu_backend::register(&mut r);
        #[cfg(target_os = "macos")]
        super::metal_backend::register(&mut r);
        #[cfg(feature = "cuda")]
        super::cuda_backend::register(&mut r);
        r.finish();
        r
    }

    /// Register one backend. Called by the backend modules from
    /// [`Registry::build`] — the only place a backend is introduced.
    pub fn register_entry(&mut self, entry: BackendEntry) {
        let slot = entry.handle.index();
        self.entries[slot] = Some(entry);
    }

    fn finish(&mut self) {
        let mut v: Vec<Backend> = self
            .entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| e.handle))
            .collect();
        // Deterministic total order: priority descending, then id ascending, so
        // two entries can never tie and the order does not depend on how the
        // entries were registered.
        v.sort_by_key(|b| {
            (
                std::cmp::Reverse(self.get(*b).map_or(0, |e| e.priority)),
                b.index(),
            )
        });
        self.priority = v;
    }

    pub fn get(&self, backend: Backend) -> Option<&BackendEntry> {
        self.entries.get(backend.index()).and_then(|e| e.as_ref())
    }

    /// Every registered entry, in **identity** order.
    pub fn iter(&self) -> impl Iterator<Item = &BackendEntry> {
        self.entries.iter().filter_map(|e| e.as_ref())
    }

    /// Every registered handle, in **priority** order (assignment preference).
    pub fn by_priority(&self) -> &[Backend] {
        &self.priority
    }

    /// The handle a name resolves to, or a loud, specific error.
    pub fn resolve_name(&self, name: &str) -> Result<Backend, NameError> {
        let key = name.trim().to_ascii_lowercase();
        let handle = NAMES
            .iter()
            .position(|n| *n == key)
            .and_then(Backend::from_index);
        let Some(handle) = handle else {
            return Err(NameError::Unknown {
                name: name.trim().to_string(),
            });
        };
        if !handle.is_registered() {
            return Err(NameError::NotCompiled {
                name: handle.name(),
                why: not_compiled_why(handle),
            });
        }
        Ok(handle)
    }
}

/// The process-wide registry, built once (see [`init`]).
static REGISTRY: OnceLock<Registry> = OnceLock::new();

/// The registry, building it on first use.
pub fn registry() -> &'static Registry {
    REGISTRY.get_or_init(Registry::build)
}

/// Build the registry explicitly at startup.
pub fn init() {
    let _ = registry();
}

fn not_compiled_why(backend: Backend) -> &'static str {
    match backend {
        Backend::METAL => "the Metal backend is compiled only on macOS (target_os = \"macos\")",
        Backend::CUDA => "the CUDA backend is compiled only with --features cuda",
        _ => "the backend is not compiled into this build",
    }
}

/// Why a name could not be resolved. The two cases are deliberately distinct: a
/// name that does not exist is always an error, and so is a name this build does
/// not contain — a compiled-out backend must never look absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    /// No backend has this name (on any configuration).
    Unknown { name: String },
    /// The name exists, but this build does not contain the backend.
    NotCompiled {
        name: &'static str,
        why: &'static str,
    },
}

impl NameError {
    /// The message a startup refusal prints.
    pub fn message(&self) -> String {
        match self {
            NameError::Unknown { name } => format!(
                "unknown backend '{name}'; known backends are: {}",
                NAMES.join(", ")
            ),
            NameError::NotCompiled { name, why } => {
                format!("backend '{name}' is known but not compiled into this build: {why}")
            }
        }
    }
}

/// Resolve `name` through the process-wide registry.
pub fn resolve_name(name: &str) -> Result<Backend, NameError> {
    registry().resolve_name(name)
}

/// Why a backend that *is* compiled in cannot run here, or `None` when it can.
pub fn unavailable_reason(backend: Backend) -> Option<&'static str> {
    match backend.entry() {
        Some(entry) => (entry.unavailable)(),
        // Not compiled in at all: report the compile-time reason, so a caller
        // never reads "no entry" as "fine".
        None => Some(not_compiled_why(backend)),
    }
}

/// Whether this backend's attention kernel reads a packed `q8_0` KV region —
/// the registry's answer to #87's question (see [`BackendCaps::reads_packed_kv`]).
pub fn reads_packed_kv(backend: Backend) -> bool {
    backend.caps().reads_packed_kv
}

/// A set of backends a run may use: the fence `--backend` / `MINFER_BACKENDS`
/// installs. Indexed by handle id — never iterated as a set, so nothing here can
/// perturb a deterministic assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendFilter {
    /// May this run use the backend?
    allowed: [bool; N_BACKENDS],
    /// Did the request **name** the backend?
    ///
    /// `allowed` and `requested` answer different questions, and the difference
    /// is what `check_available` needs: only a **requested** backend is checked
    /// for availability at startup. The default request means "whatever this
    /// build can use", so a device the pre-existing `MINFER_DISABLE_CUDA` /
    /// `MINFER_DISABLE_MPS` flags fence off keeps meaning "run on the CPU"
    /// rather than "refuse to start".
    requested: [bool; N_BACKENDS],
}

impl BackendFilter {
    /// Every backend (the default: the pre-F4 behaviour).
    pub const fn all() -> BackendFilter {
        BackendFilter {
            allowed: [true; N_BACKENDS],
            requested: [false; N_BACKENDS],
        }
    }

    /// Resolve a requested name list. A name may be a comma-separated list, and
    /// the list may be given as several items (`--backend cpu --backend metal`).
    ///
    /// `cpu` is always allowed, whether or not it was named: it is the universal
    /// fallback and a graph must always be assignable, so the fence can only
    /// remove *device* participation.
    pub fn from_names<I, S>(names: I) -> Result<BackendFilter, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut allowed = [false; N_BACKENDS];
        let mut requested = [false; N_BACKENDS];
        for raw in names {
            for part in raw.as_ref().split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let handle = resolve_name(part).map_err(|e| e.message())?;
                allowed[handle.index()] = true;
                requested[handle.index()] = true;
            }
        }
        allowed[Backend::CPU.index()] = true;
        Ok(BackendFilter { allowed, requested })
    }

    pub fn allows(&self, backend: Backend) -> bool {
        self.allowed.get(backend.index()).copied().unwrap_or(false)
    }

    /// Whether the request **named** this backend (see the `requested` field).
    pub fn is_requested(&self, backend: Backend) -> bool {
        self.requested
            .get(backend.index())
            .copied()
            .unwrap_or(false)
    }

    /// Whether this is "no fence" (every backend allowed).
    /// Test-only (#238): driven by `graph::registry::tests::the_name_surface_fences_devices_and_keeps_cpu`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn is_unfiltered(&self) -> bool {
        self.allowed.iter().all(|a| *a)
    }
}

impl Default for BackendFilter {
    fn default() -> Self {
        Self::all()
    }
}

/// Resolve the startup request: `--backend` values win over `MINFER_BACKENDS`.
///
/// Pure: the whole name surface, including every refusal, is covered by CI.
pub fn request_filter(flag: &[String], env: Option<&str>) -> Result<BackendFilter, String> {
    if !flag.is_empty() {
        return BackendFilter::from_names(flag.iter().map(|s| s.as_str()));
    }
    match env {
        Some(v) if !v.trim().is_empty() => BackendFilter::from_names([v]),
        _ => Ok(BackendFilter::all()),
    }
}

/// Startup stage 2: every **named** backend that is compiled in must also be
/// usable on this machine. Run once the device layer is up, before the model is
/// loaded — a named backend that cannot run is refused, never dropped.
///
/// Only a named backend is checked. The default request ("whatever this build
/// can use") must keep the pre-F4 behaviour, where an unavailable device is
/// simply not used: `MINFER_DISABLE_CUDA=1` means "run on the CPU", not "refuse
/// to start".
pub fn check_available(filter: &BackendFilter) -> Result<(), String> {
    for &backend in registry().by_priority() {
        if !filter.is_requested(backend) {
            continue;
        }
        if let Some(why) = unavailable_reason(backend) {
            return Err(format!(
                "backend '{}' is compiled in but not available on this machine: {why}",
                backend.name()
            ));
        }
    }
    Ok(())
}

/// The process-wide filter the graph builders and the allocator read, installed
/// once at startup. Unset = [`BackendFilter::all`].
static ACTIVE_FILTER: OnceLock<BackendFilter> = OnceLock::new();

/// Install the run's filter. Called once, at startup, before any allocator or
/// graph exists.
pub fn install_filter(filter: BackendFilter) {
    let _ = ACTIVE_FILTER.set(filter);
}

/// The filter in force (every backend when nothing was installed).
pub fn active_filter() -> &'static BackendFilter {
    ACTIVE_FILTER.get_or_init(BackendFilter::all)
}

#[cfg(test)]
mod tests;
