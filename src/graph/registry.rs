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
use super::DType;

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
pub const PRIORITY_METAL: u16 = 300;
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
    pub supports_op: fn(&Op, DType) -> bool,
    pub supports_fused: fn(&FusedOp) -> bool,
    /// Whether attention can be bounded from the explicit `attn_span` (E1).
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
    /// The KV element type this backend's regions store (C5 records it in the
    /// session header). Takes the allocator because a backend's answer can be a
    /// per-pool snapshot (the CPU pool captures `MINFER_CACHE_TYPE` at
    /// construction).
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

/// Every accepted name, in id order.
pub fn names() -> &'static [&'static str; N_BACKENDS] {
    &NAMES
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
    pub fn is_unfiltered(&self) -> bool {
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
mod tests {
    use super::*;

    /// The registered set, per configuration. Pinned so a backend that stops
    /// registering (or gains an entry it should not have) fails here.
    #[cfg(all(target_os = "macos", feature = "cuda"))]
    const EXPECTED_REGISTERED: &[&str] = &["cpu", "metal", "cuda"];
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    const EXPECTED_REGISTERED: &[&str] = &["cpu", "metal"];
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    const EXPECTED_REGISTERED: &[&str] = &["cpu", "cuda"];
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    const EXPECTED_REGISTERED: &[&str] = &["cpu"];

    /// The assignment order, per configuration — the pre-F4 behaviour
    /// (Metal, CUDA, CPU) as **literal** numbers.
    ///
    /// Literals on purpose: deriving them from `PRIORITY_*` would make the value
    /// half of this gate vacuous (only a reordering would fail, not a silent
    /// re-numbering), and the numbers are what the assignment order *is*.
    #[cfg(all(target_os = "macos", feature = "cuda"))]
    const EXPECTED_PRIORITY: &[(&str, u16)] = &[("metal", 300), ("cuda", 200), ("cpu", 100)];
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    const EXPECTED_PRIORITY: &[(&str, u16)] = &[("metal", 300), ("cpu", 100)];
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    const EXPECTED_PRIORITY: &[(&str, u16)] = &[("cuda", 200), ("cpu", 100)];
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    const EXPECTED_PRIORITY: &[(&str, u16)] = &[("cpu", 100)];

    /// F4 gate 2: the **registered set**, the **priority order** and the priority
    /// **numbers** are pinned, per configuration. A future accidental reordering
    /// — or a backend that silently stops registering, or a re-numbering that
    /// moves the order — fails here instead of quietly changing where every
    /// graph's nodes land (the assignment is topology, standing rule 3).
    #[test]
    fn the_registered_set_and_priority_order_are_pinned() {
        let r = registry();
        let registered: Vec<&str> = r.iter().map(|e| e.name).collect();
        assert_eq!(
            registered, EXPECTED_REGISTERED,
            "the registered backend set moved"
        );
        let priority: Vec<(&str, u16)> = r
            .by_priority()
            .iter()
            .map(|&b| {
                (
                    b.name(),
                    r.get(b).expect("a priority handle is registered").priority,
                )
            })
            .collect();
        assert_eq!(
            priority, EXPECTED_PRIORITY,
            "the backend priority order moved (assignment would move with it)"
        );

        // The order is a total order: no two entries may tie, or the outcome
        // would depend on how they were registered.
        let mut prios: Vec<u16> = priority.iter().map(|(_, p)| *p).collect();
        let n = prios.len();
        prios.sort_unstable();
        prios.dedup();
        assert_eq!(prios.len(), n, "backend priorities must be distinct");

        // The cpu fallback is last: every device is offered before it.
        assert_eq!(priority.last().expect("cpu is always registered").0, "cpu");

        // …and with no pool enabled (a fresh allocator) the answer is the cpu,
        // deterministically, on every configuration.
        let alloc = GraphAllocator::new();
        assert_eq!(alloc.supports(&Op::Input, DType::F32), Some(Backend::CPU));
    }

    /// F4 gate 1: the known names resolve to the pinned handles, an unknown name
    /// is a distinct loud error, and a compiled-out name is a *different* loud
    /// error that names which of the two situations it is.
    #[test]
    fn names_resolve_and_unknown_names_are_refused() {
        // The id space is a file-format contract (the KV-session backend tag)
        // and it is the pre-F4 declaration order for `Ord`.
        assert_eq!(Backend::CPU.index(), 0);
        assert_eq!(Backend::METAL.index(), 1);
        assert_eq!(Backend::CUDA.index(), 2);
        assert_eq!(Backend::from_index(0), Some(Backend::CPU));
        assert_eq!(Backend::from_index(2), Some(Backend::CUDA));
        assert_eq!(Backend::from_index(3), None);
        assert_eq!(Backend::CPU.name(), "cpu");
        assert_eq!(Backend::METAL.name(), "metal");
        assert_eq!(Backend::CUDA.name(), "cuda");
        assert!(Backend::CPU < Backend::METAL && Backend::METAL < Backend::CUDA);

        // Diagnostics keep the pre-F4 spelling.
        assert_eq!(format!("{:?}", Backend::CPU), "CPU");
        assert_eq!(format!("{:?}", Backend::METAL), "Metal");
        assert_eq!(format!("{:?}", Backend::CUDA), "Cuda");

        // Spellings: trimmed, case-insensitive, canonical in the answer. A
        // device name resolves where the backend is compiled in and is the
        // NotCompiled refusal (never Unknown) where it is not.
        assert_eq!(resolve_name(" cpu "), Ok(Backend::CPU));
        assert_eq!(resolve_name("CPU"), Ok(Backend::CPU));
        match resolve_name("CuDa") {
            Ok(b) => assert_eq!(b, Backend::CUDA),
            Err(e) => assert!(
                matches!(e, NameError::NotCompiled { name: "cuda", .. }),
                "{e:?}"
            ),
        }
        assert_eq!(names(), &["cpu", "metal", "cuda"]);

        // "no such backend name" — always an error, on every configuration, and
        // the message lists the accepted names.
        let err = resolve_name("gpu2").unwrap_err();
        assert!(matches!(err, NameError::Unknown { .. }), "{err:?}");
        assert!(
            err.message()
                .contains("unknown backend 'gpu2'; known backends are: cpu, metal, cuda"),
            "{}",
            err.message()
        );

        // "that backend exists but is not compiled in" — a different message.
        #[cfg(not(target_os = "macos"))]
        {
            let e = resolve_name("metal").unwrap_err();
            assert!(
                matches!(e, NameError::NotCompiled { name: "metal", .. }),
                "{e:?}"
            );
            assert!(
                e.message()
                    .contains("backend 'metal' is known but not compiled into this build"),
                "{}",
                e.message()
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            let e = resolve_name("cuda").unwrap_err();
            assert!(
                matches!(e, NameError::NotCompiled { name: "cuda", .. }),
                "{e:?}"
            );
            assert!(e.message().contains("--features cuda"), "{}", e.message());
        }
        // …and the compiled-in names resolve on this configuration.
        #[cfg(target_os = "macos")]
        assert_eq!(resolve_name("metal"), Ok(Backend::METAL));
        #[cfg(feature = "cuda")]
        assert_eq!(resolve_name("cuda"), Ok(Backend::CUDA));
        assert_eq!(resolve_name("cpu"), Ok(Backend::CPU));

        // An unregistered handle claims nothing rather than guessing.
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            assert!(!Backend::METAL.is_registered());
            assert!(!(Backend::METAL.caps().supports_op)(&Op::Input, DType::F32));
            assert!(!(Backend::CUDA.caps().supports_fused)(&FusedOp::SwiGLU));
            assert!(!Backend::CUDA.caps().supports_attn_span);
            assert!(Backend::CPU.is_registered());
            assert!((Backend::CPU.caps().supports_op)(&Op::Input, DType::F32));
            // "not compiled in" is not "available": the reason is still named.
            assert!(unavailable_reason(Backend::METAL).is_some());
            // …and a name that exists but is not in this binary is *not* the
            // same error as a name that does not exist.
            assert_ne!(
                resolve_name("metal").unwrap_err(),
                NameError::Unknown {
                    name: "nope".to_string()
                }
            );
        }
    }

    /// The fence surface: a name list resolves, unknown spellings are refused
    /// from either surface, `cpu` survives every fence, and unset is the pre-F4
    /// behaviour.
    #[test]
    fn the_name_surface_fences_devices_and_keeps_cpu() {
        let cpu_only = BackendFilter::from_names(["cpu"]).expect("cpu is a valid name");
        assert!(cpu_only.allows(Backend::CPU));
        assert!(!cpu_only.allows(Backend::METAL));
        assert!(!cpu_only.allows(Backend::CUDA));
        assert!(!cpu_only.is_unfiltered());

        // cpu is the universal fallback: naming only a device keeps it. Which
        // device names this build has is configuration-dependent, so this picks
        // one it does have (the "neither" configuration's case is the
        // NotCompiled assertion at the end).
        #[cfg(target_os = "macos")]
        let device = Backend::METAL;
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        let device = Backend::CUDA;
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        {
            let device_only =
                BackendFilter::from_names([device.name()]).expect("a compiled-in device resolves");
            assert!(
                device_only.allows(Backend::CPU),
                "the fallback must survive a device-only fence"
            );
            assert!(device_only.allows(device));
            assert!(!device_only.is_unfiltered());
        }

        // Comma-separated and repeated spellings are the same request, and
        // whitespace/case are not part of a name.
        assert_eq!(
            BackendFilter::from_names([" CPU , "]).unwrap(),
            BackendFilter::from_names(["cpu", "cpu"]).unwrap()
        );

        // Unset = every backend (the pre-F4 behaviour).
        assert!(BackendFilter::all().is_unfiltered());
        assert!(BackendFilter::default().allows(Backend::CUDA));
        assert!(request_filter(&[], None).unwrap().is_unfiltered());

        // The environment surface, and the flag winning over it.
        let env_only = request_filter(&[], Some("cpu")).unwrap();
        assert!(!env_only.allows(Backend::CUDA));
        let flag_wins = request_filter(&["cpu".to_string()], Some("does-not-matter")).unwrap();
        assert!(!flag_wins.allows(Backend::CUDA));
        assert!(flag_wins.allows(Backend::CPU));
        // An empty environment value is "unset", not "empty fence".
        assert!(request_filter(&[], Some("")).unwrap().is_unfiltered());

        // A name that is not compiled in is refused as such, so a fence can
        // never quietly become "fewer backends than asked for".
        #[cfg(not(feature = "cuda"))]
        {
            let e = BackendFilter::from_names(["cuda"]).unwrap_err();
            assert!(e.contains("not compiled into this build"), "{e}");
            assert!(!e.contains("unknown backend"), "{e}");
        }

        // An unknown spelling is refused from either surface, with one message.
        let flag_err = BackendFilter::from_names(["nope"]).unwrap_err();
        assert!(flag_err.contains("unknown backend 'nope'"), "{flag_err}");
        assert!(
            flag_err.contains("known backends are: cpu, metal, cuda"),
            "{flag_err}"
        );
        let env_err = request_filter(&[], Some("cpu,nope")).unwrap_err();
        assert!(env_err.contains("unknown backend 'nope'"), "{env_err}");
    }

    /// The #87 seam: `reads_packed_kv` is one authority, and the two call sites
    /// that used to hardcode "CPU only" read it.
    #[test]
    fn the_packed_kv_capability_is_the_registrys_answer() {
        assert!(Backend::CPU.caps().reads_packed_kv);
        assert!(!Backend::METAL.caps().reads_packed_kv);
        assert!(!Backend::CUDA.caps().reads_packed_kv);
        assert!(reads_packed_kv(Backend::CPU));
        #[cfg(not(feature = "cuda"))]
        assert!(!reads_packed_kv(Backend::CUDA));
        // …and the C4 format gate asks the same field.
        use crate::models::Device;
        for device in [Device::Cpu, Device::Metal, Device::Cuda] {
            assert_eq!(
                KvFormat::Q8_0.supports(device),
                reads_packed_kv(device.backend()),
                "{device:?}"
            );
        }
        // F32/f16 are every backend's business (unchanged).
        for device in [Device::Cpu, Device::Metal, Device::Cuda] {
            assert!(KvFormat::F32.supports(device));
            assert!(KvFormat::F16.supports(device));
        }
    }

    /// The capability matrix the registry advertises is the matrix the trait
    /// answers — the two are one authority, so a graph can never be assigned to
    /// a backend whose trait would refuse the op.
    #[test]
    fn registry_caps_match_the_backend_trait() {
        use super::super::backend_takes;
        use super::super::cpu_backend::CpuBackend;
        use crate::graph::backend::Backend as BackendTrait;
        let cpu = CpuBackend::new();
        let caps = Backend::CPU.caps();
        for op in [
            Op::Input,
            Op::Add,
            Op::Silu,
            Op::SwiGLU,
            Op::Softmax { dim: 0 },
        ] {
            for dtype in [DType::F32, DType::F16] {
                assert_eq!(
                    (caps.supports_op)(&op, dtype),
                    BackendTrait::supports_op(&cpu, &op, dtype),
                    "{op:?} {dtype:?}"
                );
            }
        }
        assert_eq!(
            caps.supports_attn_span,
            BackendTrait::supports_attn_span(&cpu)
        );
        assert_eq!(
            (caps.supports_fused)(&FusedOp::SwiGLU),
            BackendTrait::supports_fused(&cpu, &FusedOp::SwiGLU)
        );
        // …and the assignment answers what the caps say.
        let alloc = GraphAllocator::new();
        assert_eq!(
            alloc.supports(&Op::Silu, DType::F32).is_some(),
            backend_takes(&cpu, &Op::Silu, DType::F32)
        );
    }
}
