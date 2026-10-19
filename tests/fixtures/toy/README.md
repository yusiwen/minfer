# The toy GGUF

`toy-qwen2-f32.gguf` is a 113 KB, one-layer `qwen2` with a 260-token byte-level
vocabulary. It is **not** a language model — the weights are a fixed pseudo-random
sequence — and it is not meant to produce sensible text. Its job is to be a model
that *loads, builds a graph and decodes one token* so the startup/template
end-to-end tests (`tests/template_e2e.rs`) need no cached real model, and so one
of them can assert the no-template ChatML notice (the toy carries no
`tokenizer.chat_template`).

Dimensions: `n_embd=32`, `n_head=4`, `n_head_kv=4`, `n_layer=1`, `n_ff=64`,
`n_ctx=64`, `f32` weights. A real model can be substituted with
`MINFER_BATCH_TEST_MODEL=<path>`; the notice test then fabricates a template-less
copy by renaming one byte of the `tokenizer.chat_template` KV key (same length, so
every tensor offset stays valid).

Regenerate with:

```
python3 scripts/gen_toy_gguf.py            # rewrite this file
python3 scripts/gen_toy_gguf.py --check    # verify it matches the generator
```

Provenance: `scripts/gen_toy_gguf.py`, 2026-10-19, against the layout contract in
`src/gguf_write.rs` (GGUF v3, alignment 32).
