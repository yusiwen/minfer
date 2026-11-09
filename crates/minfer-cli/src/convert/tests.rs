//! `#[cfg(test)] mod tests` for `src/convert.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn tensor_name_mapping_covers_qwen2_and_refuses_the_rest() {
    assert_eq!(
        map_tensor_name("model.embed_tokens.weight", 24).unwrap(),
        "token_embd.weight"
    );
    assert_eq!(
        map_tensor_name("model.norm.weight", 24).unwrap(),
        "output_norm.weight"
    );
    assert_eq!(
        map_tensor_name("lm_head.weight", 24).unwrap(),
        "output.weight"
    );
    assert_eq!(
        map_tensor_name("model.layers.5.self_attn.q_proj.weight", 24).unwrap(),
        "blk.5.attn_q.weight"
    );
    assert_eq!(
        map_tensor_name("model.layers.5.mlp.down_proj.weight", 24).unwrap(),
        "blk.5.ffn_down.weight"
    );
    assert_eq!(
        map_tensor_name("model.layers.5.post_attention_layernorm.weight", 24).unwrap(),
        "blk.5.ffn_norm.weight"
    );
    // RoPE tables are intentionally dropped, matching llama.cpp.
    assert!(map_tensor_name("model.layers.0.self_attn.rotary_emb.inv_freq", 24).is_none());
    // Unknown names still refuse (the caller turns None into a loud error).
    assert!(map_tensor_name("model.layers.0.mlp.gate_up_proj.weight", 24).is_none());
    assert!(map_tensor_name("model.layers.99.self_attn.q_proj.weight", 24).is_none());
}

#[test]
fn outtype_parse_accepts_bf16_and_the_enum_agrees_with_gguf() {
    assert_eq!(OutType::parse("F16").unwrap(), OutType::F16);
    assert_eq!(OutType::parse("f32").unwrap(), OutType::F32);
    // #142: bf16 is an accepted target now, in every spelling the CLI may
    // see, and the refusal text for a genuinely unknown type names it.
    assert_eq!(OutType::parse("bf16").unwrap(), OutType::Bf16);
    assert_eq!(OutType::parse("BF16").unwrap(), OutType::Bf16);
    assert_eq!(OutType::parse("bfloat16").unwrap(), OutType::Bf16);
    assert!(OutType::parse("q4_0").is_err());
    let e = OutType::parse("bfloat").unwrap_err();
    assert!(e.contains("bf16"), "{e}");

    // The accepted target must agree with the GGUF enum and with
    // llama.cpp's `llama_ftype` number a reader parses from the file:
    // MOSTLY_BF16 = 32 (include/llama.h), type_size 2, blck_size 1.
    assert_eq!(OutType::Bf16.ggml_type(), GgmlType::BF16);
    assert_eq!(OutType::Bf16.file_type(), 32);
    assert_eq!(OutType::Bf16.name(), "bf16");
    assert_eq!(GgmlType::BF16.type_size(), 2);
    assert_eq!(GgmlType::BF16.blck_size(), 1);
    assert_eq!(OutType::F16.file_type(), 1);
    assert_eq!(OutType::F32.file_type(), 0);
}

/// The 1-D rule of the file contract (#142): norms and biases stay f32 on
/// both half targets. llama.cpp's converter writes exactly this
/// (`conversion/base.py`: `if n_dims <= 1 ... data_qtype = F32`), and byte
/// parity against its output forces it.
#[test]
fn the_one_d_tensor_rule_is_f32_for_f16_and_bf16() {
    for out in [OutType::F16, OutType::Bf16] {
        assert_eq!(out.tensor_type(0), GgmlType::F32, "0-D for {out:?}");
        assert_eq!(out.tensor_type(1), GgmlType::F32, "1-D for {out:?}");
        assert_eq!(out.tensor_type(2), out.ggml_type(), "2-D for {out:?}");
        assert_eq!(out.tensor_type(4), out.ggml_type(), "4-D for {out:?}");
    }
    // f32 is the one target that converts every tensor, 1-D included.
    assert_eq!(OutType::F32.tensor_type(1), GgmlType::F32);
    assert_eq!(OutType::Bf16.tensor_type(2), GgmlType::BF16);
}

/// f32 → bf16 is round-to-nearest-even on the top 16 bits, including exact
/// ties. Every expected pattern is hand-derived from the bf16 grid near 1.0
/// (step 2^-7): a tie adds 1 only when the surviving mantissa bit is odd.
#[test]
fn f32_to_bf16_rounds_to_nearest_even_including_ties() {
    // Exact bf16 values pass through unchanged (bit patterns, not decimals).
    for (f, bits) in [
        (0.0f32, 0x0000u16),
        (-0.0f32, 0x8000),
        (1.0f32, 0x3F80),
        (-1.0f32, 0xBF80),
        (2.0f32, 0x4000),
        (0.5f32, 0x3F00),
    ] {
        assert_eq!(f32_to_bf16_bits(f), bits, "{f}");
    }
    // Just below / just above the halfway point between 1.0 and 1.0078125.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_7FFF)), 0x3F80);
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
    // Exact tie: 1.0 + 2^-8. Lower neighbour 0x3F80 is even → stays 0x3F80.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
    // Exact tie with an odd lower neighbour → rounds up to even, 0x3F82.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
    // Tie where the lower neighbour is already even → no change.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F82_8000)), 0x3F82);
    // Tie with an odd lower neighbour one step higher → up to 0x3F84.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F83_8000)), 0x3F84);
    // A mantissa overflow carries into the exponent: the largest value below
    // 2.0 is a tie that rounds up to 2.0 (0x4000), not to 0x3FFF.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3FFF_8000)), 0x4000);
    // ±inf saturates at the same pattern; a NaN is forced quiet.
    assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7F80);
    assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xFF80);
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x7F80_0001)), 0x7FC0);
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0xFFC0_1234)), 0xFFC0);
}

/// A bf16 source re-encoded as bf16 is the identity, and `convert_bytes`
/// routes the BF16 GGUF type to the bf16 encoder (the pre-#142 `_ => F16`
/// mapping would silently write f16 bytes under a BF16 label).
#[test]
fn bf16_to_bf16_round_trips_and_routes_through_the_bf16_encoder() {
    // Every non-signalling bf16 value round-trips to its own bits.
    for bits in [
        0x0000u16, 0x0001, 0x3F80, 0xBF80, 0x4000, 0x7F7F, 0xFF7F, 0x7FC0,
    ] {
        let y = convert_bytes(HfDtype::Bf16, OutType::Bf16, &bits.to_le_bytes());
        assert_eq!(u16::from_le_bytes([y[0], y[1]]), bits, "bf16 {bits:#06x}");
    }
    // `convert_bytes_ggml` picks the bf16 encoder for the BF16 GGUF type.
    // 1.0 + 2^-8 = 0x3F80_8000 is a tie → bf16 0x3F80, but f16 can store it
    // exactly (0x3C04), so the two encoders are distinguishable here.
    let v = f32::from_bits(0x3F80_8000).to_le_bytes();
    let bf = convert_bytes_ggml(HfDtype::F32, GgmlType::BF16, &v);
    assert_eq!(u16::from_le_bytes([bf[0], bf[1]]), 0x3F80);
    let f16 = convert_bytes_ggml(HfDtype::F32, GgmlType::F16, &v);
    assert_eq!(u16::from_le_bytes([f16[0], f16[1]]), 0x3C04);
    // f16 → bf16 keeps the exact value (f16 is a strict subset of bf16's
    // exponent range here, and every f16 mantissa is a bf16 mantissa).
    let two = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    let y = convert_bytes(HfDtype::F16, OutType::Bf16, &two);
    assert_eq!(u16::from_le_bytes([y[0], y[1]]), 0x4000);
}

#[test]
fn bf16_to_f16_is_exact_in_the_mantissa_and_saturates_on_overflow() {
    // bf16 1.0 = 0x3F80
    let one = 0x3F80u16.to_le_bytes();
    let y = convert_bytes(HfDtype::Bf16, OutType::F16, &one);
    assert_eq!(
        half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
        1.0
    );
    // bf16 0x7F80 is +inf → f16 +inf
    let inf = 0x7F80u16.to_le_bytes();
    let y = convert_bytes(HfDtype::Bf16, OutType::F16, &inf);
    assert!(half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).is_infinite());
    // bf16 2.0 → f16 2.0 exactly
    let two = 0x4000u16.to_le_bytes();
    let y = convert_bytes(HfDtype::Bf16, OutType::F16, &two);
    assert_eq!(
        half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
        2.0
    );
}

#[test]
fn f32_to_f16_rounds_and_f16_to_f32_is_exact() {
    let v = 1.0009765625f32; // exactly representable in f16
    let y = convert_bytes(HfDtype::F32, OutType::F16, &v.to_le_bytes());
    let back = convert_bytes(HfDtype::F16, OutType::F32, &y);
    assert_eq!(f32::from_le_bytes([back[0], back[1], back[2], back[3]]), v);
    // A value needing rounding: nearest-even to f16.
    let v2 = 1.00048828125f32; // halfway; rounds to even (1.0)
    let y2 = convert_bytes(HfDtype::F32, OutType::F16, &v2.to_le_bytes());
    let back2 = convert_bytes(HfDtype::F16, OutType::F32, &y2);
    assert_eq!(
        f32::from_le_bytes([back2[0], back2[1], back2[2], back2[3]]),
        1.0
    );
}

#[test]
fn looks_special_matches_llama_cpp() {
    assert!(looks_special("<|fim_prefix|>"));
    assert!(looks_special("<|im_start|>"));
    assert!(!looks_special("<tool_call>"));
    assert!(!looks_special("hello"));
    assert!(looks_special("<pad>"));
}
