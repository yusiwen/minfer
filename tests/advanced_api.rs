//! #763 S7: the `advanced` tier as an **external** consumer sees it.
//!
//! The HTTP server is the tier's first in-tree consumer; this test is the
//! out-of-tree one, so a lost re-export fails here instead of only in the
//! server's build.

use minfer::advanced::{
    Backend, BackendFilter, GraphCache, KvFormat, OffloadRequest, SamplerConfig,
};

#[test]
fn the_advanced_tier_is_reachable_from_outside_the_crate() {
    let filter = BackendFilter::from_names(["cpu"]).expect("cpu is a backend name");
    assert!(filter.allows(Backend::CPU));
    assert!(!filter.allows(Backend::CUDA), "cpu-only fence");

    let mut cache = GraphCache::new();
    let _ = cache.current();

    assert_eq!(KvFormat::F32.name(), "f32");
    assert_eq!(KvFormat::F16.name(), "f16");
    let _ = OffloadRequest::Layers(0);
    let cfg = SamplerConfig::default();
    assert!(cfg.temp >= 0.0);
}

#[cfg(feature = "observe")]
#[test]
fn the_observation_surface_is_reachable() {
    // Lazy arming: no viz client is attached, so capture is off.
    assert!(
        !minfer::advanced::live::enabled(),
        "live capture must be off until a client subscribes"
    );
    let _ = minfer::advanced::trace::enabled();
}
