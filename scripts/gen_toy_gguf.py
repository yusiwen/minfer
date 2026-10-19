#!/usr/bin/env python3
"""Generate the checked-in toy GGUF used by the template/startup e2e tests (#609).

The e2e suite in `tests/template_e2e.rs` needs a model that *loads and runs* but has
no chat template: the notice, the override and the refusal tests all spawn a server
against it. Borrowing a cached real model meant a 640 MB copy under `/tmp` and an
`#[ignore]`d suite; this file is ~70 KB, committed, and reproducible.

It is a one-layer `qwen2` with a 256-byte vocabulary — garbage weights, correct
shapes. The acceptance is `minfer <toy> "hi" -n 1` exiting 0 and the three startup
paths printing the ChatML notice exactly once.

Usage:
    python3 scripts/gen_toy_gguf.py            # write tests/fixtures/toy/toy-qwen2-f32.gguf
    python3 scripts/gen_toy_gguf.py --check    # compare, exit 1 on drift

The layout mirrors `src/gguf_write.rs` (which is the byte-level contract the parser
reads back): magic, version u32, n_tensors i64, n_kv i64, KV pairs, tensor infos,
padding to the alignment, then the payloads.
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/fixtures/toy/toy-qwen2-f32.gguf"
ALIGN = 32

# --- dimensions ---------------------------------------------------------------
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 4
N_LAYER = 1
N_FF = 64
N_CTX = 64

# --- GGUF encoding ------------------------------------------------------------
MAGIC = b"GGUF"
VERSION = 3

T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64 = 8, 9, 10, 11


def pad(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(key: str, value: str) -> bytes:
    return string(key) + struct.pack("<I", T_STRING) + string(value)


def kv_u32(key: str, value: int) -> bytes:
    return string(key) + struct.pack("<I", T_UINT32) + struct.pack("<I", value)


def kv_f32(key: str, value: float) -> bytes:
    return string(key) + struct.pack("<I", T_FLOAT32) + struct.pack("<f", value)


def kv_arr_str(key: str, values: list[str]) -> bytes:
    out = string(key) + struct.pack("<I", T_ARRAY)
    out += struct.pack("<I", T_STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += string(v)
    return out


def kv_arr_i32(key: str, values: list[int]) -> bytes:
    out = string(key) + struct.pack("<I", T_ARRAY)
    out += struct.pack("<I", T_INT32) + struct.pack("<Q", len(values))
    for v in values:
        out += struct.pack("<i", v)
    return out


# --- the vocabulary -----------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> unicode map, the alphabet a `gpt2` GGUF vocabulary uses."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_CHAR = bytes_to_unicode()


def vocabulary() -> tuple[list[str], list[int]]:
    tokens = [BYTE_TO_CHAR[b] for b in range(256)]
    types = [1] * 256  # 1 = normal
    # The specials the qwen2 template path looks for by name.
    for special in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        tokens.append(special)
        types.append(3)  # 3 = control: matched as one id before BPE
    # One merge, so the merge table is non-empty and every merged piece exists.
    tokens.append("ab")
    types.append(1)
    return tokens, types


# --- tensors ------------------------------------------------------------------
def spec(name: str, ne: list[int]) -> tuple[str, list[int], bytes]:
    """A tensor spec plus its f32 payload, in GGUF `[in, out]` order.

    The payload is row-major `[out][in]`, matching the parser's memory layout.
    """
    ne4 = ne + [1] * (4 - len(ne))
    count = 1
    for d in ne:
        count *= d
    # Deterministic small values: garbage logits are fine, NaNs are not.
    data = b"".join(
        struct.pack("<f", ((i * 37) % 101 - 50) / 500.0) for i in range(count)
    )
    return name, ne4, data


def tensors() -> list[tuple[str, list[int], bytes]]:
    out = [
        spec("token_embd.weight", [N_EMBD, N_VOCAB]),
        spec("output_norm.weight", [N_EMBD]),
        spec("output.weight", [N_EMBD, N_VOCAB]),
        spec("blk.0.attn_norm.weight", [N_EMBD]),
        spec("blk.0.attn_q.weight", [N_EMBD, N_EMBD]),
        spec("blk.0.attn_k.weight", [N_EMBD, N_KV]),
        spec("blk.0.attn_v.weight", [N_EMBD, N_KV]),
        spec("blk.0.attn_output.weight", [N_EMBD, N_EMBD]),
        spec("blk.0.ffn_norm.weight", [N_EMBD]),
        spec("blk.0.ffn_gate.weight", [N_EMBD, N_FF]),
        spec("blk.0.ffn_up.weight", [N_EMBD, N_FF]),
        spec("blk.0.ffn_down.weight", [N_FF, N_EMBD]),
    ]
    return out


def build() -> bytes:
    tokens, types = vocabulary()
    n_vocab = len(tokens)
    n_kv_embd = N_HEAD_KV * (N_EMBD // N_HEAD)

    kv_pairs = [
        kv_str("general.architecture", "qwen2"),
        kv_str("general.name", "minfer-toy-qwen2-1layer"),
        kv_u32("general.alignment", ALIGN),
        kv_u32("qwen2.context_length", N_CTX),
        kv_u32("qwen2.embedding_length", N_EMBD),
        kv_u32("qwen2.block_count", N_LAYER),
        kv_u32("qwen2.feed_forward_length", N_FF),
        kv_u32("qwen2.attention.head_count", N_HEAD),
        kv_u32("qwen2.attention.head_count_kv", N_HEAD_KV),
        kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
        kv_f32("qwen2.rope.freq_base", 10000.0),
        kv_str("tokenizer.ggml.model", "gpt2"),
        kv_str("tokenizer.ggml.pre", "qwen2"),
        kv_arr_str("tokenizer.ggml.tokens", tokens),
        kv_arr_i32("tokenizer.ggml.token_type", types),
        # "a b" -> "ab"; the merged piece is in the vocabulary above.
        kv_arr_str("tokenizer.ggml.merges", ["a b"]),
        kv_u32("tokenizer.ggml.bos_token_id", 256),
        kv_u32("tokenizer.ggml.eos_token_id", 258),
    ]
    kv = b"".join(kv_pairs)

    ts = [
        spec("token_embd.weight", [N_EMBD, n_vocab]),
        spec("output_norm.weight", [N_EMBD]),
        spec("output.weight", [N_EMBD, n_vocab]),
        spec("blk.0.attn_norm.weight", [N_EMBD]),
        spec("blk.0.attn_q.weight", [N_EMBD, N_EMBD]),
        spec("blk.0.attn_k.weight", [N_EMBD, n_kv_embd]),
        spec("blk.0.attn_v.weight", [N_EMBD, n_kv_embd]),
        spec("blk.0.attn_output.weight", [N_EMBD, N_EMBD]),
        spec("blk.0.ffn_norm.weight", [N_EMBD]),
        spec("blk.0.ffn_gate.weight", [N_EMBD, N_FF]),
        spec("blk.0.ffn_up.weight", [N_EMBD, N_FF]),
        spec("blk.0.ffn_down.weight", [N_FF, N_EMBD]),
    ]
    for name, ne, data in ts:
        assert len(data) == 4 * ne[0] * ne[1], f"{name}: payload size"

    header = MAGIC + struct.pack("<I", VERSION)
    header += struct.pack("<q", len(ts)) + struct.pack("<q", len(kv_pairs))
    header += kv

    # Offsets are relative to the data section and each payload is padded.
    offsets = []
    cursor = 0
    for _, _, data in ts:
        offsets.append(cursor)
        cursor += pad(len(data))

    for (name, ne, _), off in zip(ts, offsets):
        header += string(name)
        n_dims = 1
        for j in range(1, 4):
            if ne[j] != 1:
                n_dims = j + 1
        header += struct.pack("<I", n_dims)
        for j in range(n_dims):
            header += struct.pack("<q", ne[j])
        header += struct.pack("<I", 0)  # ggml type 0 = f32
        header += struct.pack("<Q", off)

    header += b"\x00" * (pad(len(header)) - len(header))
    body = b"".join(data + b"\x00" * (pad(len(data)) - len(data)) for _, _, data in ts)
    return header + body


def main(argv: list[str]) -> int:
    fresh = build()
    if "--check" in argv[1:]:
        if not FIXTURE.exists():
            print(f"missing fixture: {FIXTURE}", file=sys.stderr)
            return 1
        committed = FIXTURE.read_bytes()
        if committed != fresh:
            print(
                f"drift: {FIXTURE} is {len(committed)} B, the generator makes {len(fresh)} B "
                f"(first difference at {next((i for i, (a, b) in enumerate(zip(committed, fresh)) if a != b), 'length')})",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {FIXTURE.name} is {len(fresh)} B and matches the generator")
        return 0

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(fresh)
    tokens, _ = vocabulary()
    print(f"wrote {FIXTURE} ({len(fresh)} B, n_vocab={len(tokens)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
