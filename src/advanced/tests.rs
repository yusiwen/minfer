//! `#[cfg(test)] mod tests` for `src/advanced.rs`.
use super::*;

/// The tier names the primitives the server drives: each one must be reachable
/// through `minfer::advanced` (the external test is `tests/advanced_api.rs`;
/// this one fails at compile time if a re-export is lost).
#[test]
fn the_tier_names_the_l1_primitives() {
    let filter = BackendFilter::from_names(["cpu"]).expect("cpu");
    assert!(filter.allows(Backend::CPU));
    let mut cache = GraphCache::new();
    let _ = cache.current();
    assert_eq!(KvFormat::F32.name(), "f32");
    let _ = OffloadRequest::Layers(0);
    let _ = SamplerConfig::default();
    let _ = MirostatMode::Off;
    let _: Option<SpecEngine> = None;
    let _ = SpecSampler {
        cfg: SamplerConfig::default(),
    };
    let _: Option<SpecConfig> = None;
    let _: Option<ConversationSpec> = None;
    let _: Option<TurnParams> = None;
    let _: Option<TurnOutcome> = None;
    fn _takes_model(_m: &dyn ModelDef, _t: &Tokenizer) {}
    fn _takes_params(_p: &GraphParams) {}
    let _ = Grammar::state;
    fn _takes_batch(_b: &Batch) {}
    fn _drives(_e: &mut dyn ConversationEngine) {}
}
