//! `#[cfg(test)] mod tests` for `src/q4k_dsc.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// #165: the payload contract is an exact length, and it refuses a q8_0 payload even
/// with the type gate removed. A q4_0 payload has q4_K's exact ratio, so the size
/// check cannot tell those apart — the type gate is what refuses it, which is why
/// `q4k_dsc_plane_admitted` carries both and this test asserts both.
#[test]
fn the_q4k_dsc_payload_contract_refuses_a_q8_0_payload() {
    // Qwen3-0.6B `ffn_down` geometry: `id % 256 == 0`, `od` even.
    let (od, id) = (1024usize, 3072usize);
    let want = q4k_dsc_payload_bytes(od, id).expect("whole super-blocks per row");
    assert_eq!(want, od * (id / 256) * 144);
    // a q8_0 payload for the same tensor is LONGER (34 B per 32 elements)
    let q80 = od * (id / 32) * 34;
    assert!(q80 > want, "q8_0 {q80} must exceed the q4_K {want}");
    assert!(
        !q4k_dsc_payload_ok(q80, od, id),
        "a q8_0 length is not a q4_K row"
    );
    assert!(
        !q4k_dsc_plane_admitted(TensorType::Q8_0, q80, od, id),
        "a q8_0 weight must be refused at its own length"
    );
    // the type gate must refuse even a payload of exactly q4_K length
    assert!(
        !q4k_dsc_plane_admitted(TensorType::Q8_0, want, od, id),
        "the type gate must refuse a q8_0 weight regardless of its length"
    );
    // the wrong-name/wrong-type control: a Q4_K payload under another K-quant type
    // (q5_K, ratio 176/256) is refused too, so "admitted == false" is not the
    // predicate's answer for every type
    assert!(!q4k_dsc_plane_admitted(TensorType::Q5_K, want, od, id));
    // positive control: the q4_K type at the exact length IS admitted — so the
    // refusals above are not `false` for every input
    assert!(q4k_dsc_plane_admitted(TensorType::Q4_K, want, od, id));
    // q4_0's ratio is exactly q4_K's (18/32 == 144/256): the size gate is blind to
    // the difference, and only the type gate refuses it
    assert_eq!(od * (id / 32) * 18, want);
    assert!(
        !q4k_dsc_plane_admitted(TensorType::Q4_0, want, od, id),
        "the type gate is the only thing that can refuse q4_0"
    );
}

/// #165: a payload shorter than the contract — a future type with a smaller
/// bytes/element ratio (a 2-bit K-quant is 84 B per 256 where q4_K is 144) — is
/// refused instead of being read past the tensor, and a payload that is longer is
/// refused too (that is the q8_0 misread). A geometry that is not whole super-blocks
/// per row is not a q4_K row at all.
#[test]
fn the_q4k_dsc_payload_contract_refuses_a_short_payload() {
    // Qwen2.5-0.5B `ffn_down` geometry: the plane that fired before the fix.
    let (od, id) = (896usize, 4864usize);
    let want = q4k_dsc_payload_bytes(od, id).expect("whole super-blocks per row");
    assert_eq!(want, 896 * 19 * 144);
    assert!(!q4k_dsc_payload_ok(want - 144, od, id), "one block short");
    assert!(!q4k_dsc_payload_ok(want + 144, od, id), "one block long");
    assert!(!q4k_dsc_payload_ok(0, od, id), "empty");
    assert!(q4k_dsc_payload_ok(want, od, id));
    assert_eq!(
        q4k_dsc_payload_bytes(od, 896),
        None,
        "896 is not a whole number of 256-element super-blocks"
    );
    assert_eq!(q4k_dsc_payload_bytes(0, id), None);
    // the "one block short" case is exactly the smaller-ratio shape: refuse it, do
    // not index `raw[j * id/256 * 144 ..]` past the end
    assert!(!q4k_dsc_plane_admitted(
        TensorType::Q4_K,
        want - 144,
        od,
        id
    ));
}
