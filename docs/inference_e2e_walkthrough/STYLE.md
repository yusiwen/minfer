# Inference E2E walkthrough — writing contract (all authors must follow)

This directory expands one minfer inference run into a sequence of standalone,
readable stage documents: from typing a prompt on the CLI to watching generated
text stream out. Intended reader: an engineer who can read Rust but is **new to
LLM inference** — someone who has never implemented a transformer, a KV cache,
or a quantized matmul.

## Audience and voice (the most important rule)

- Explain like a patient mentor. **Plain, friendly English; short sentences.**
  No telegraphic bullet dumps; full paragraphs that carry the reasoning.
- **Define every term on first use** in one or two plain sentences: token,
  embedding, logits, RMSNorm, RoPE, GQA, KV cache, top-k / top-p, quantization
  block scale, liveness, command buffer, MMQ… Assume nothing.
- Mechanism before jargon: first say WHAT happens to the data and WHY, then
  name the function that does it. Every design decision must answer
  "why this way and not the obvious alternative".
- Arithmetic beats adjectives: when claiming something is fast/cheap/small,
  show the byte counts, element counts, or operation counts.
- Prose in English (workspace doc spec). Code, function names, env vars,
  file paths, and proper nouns stay as-is.
- Each doc must read standalone: repeating one paragraph of key background is
  fine; do not copy-paste large tables or code between docs.

## Fixed per-doc structure (Markdown; order must not change)

```markdown
# NN · <stage name>

> **Stage**: prev stage → **this stage** → next stage (one line: where this
> sits in the pipeline of README.md's master table).
> **Code**: primary files and entry functions (`src/...` — verified lines).

## 1. Background — where this stage sits
What the engine has done by this point, what data it holds, what this stage
must accomplish, and what would break without it. (3-6 paragraphs, beginner
friendly.)

## 2. Principle — how it works and why
The mechanism, argued from data shapes and arithmetic. Diagrams (ascii or
mermaid) where they help; equations where a diagram does not fit. New concepts
defined on first use.

## 3. Implementation
### 3.1 Data in / data out
Concrete shapes and layouts (e.g. activations `[nt][d]` f32 token-major,
weights `[out][in]` row-major quantized bytes), where they come from and where
they go.
### 3.2 Key code
Real code excerpts (10-40 lines each, from the CURRENT tree), annotated
segment by segment. **Never paste blocks over 100 lines** — pick the core
loop / dispatch / decision logic and summarize the rest.
### 3.3 Design choices (why this shape and not another)
The alternatives considered and the reason for the chosen one.
### 3.4 Pitfalls & invariants
Traps baked into the design (aliasing, order, ownership), including the real
bugs they came from when the repo records them.

## 4. Observe & verify
How a reader can SEE this stage run: env vars (`MINFER_TRACE`, `MINFER_TIMING`,
`--dump-graph`, debug_dump…), which tests cover it, what output to expect.
One sentence per tool on what it shows.

## 5. Cross-references
Related docs (ARCHITECTURE.md sections, plan/optimization docs, neighboring
stages) with one line on what each adds.

← NN-1 · Index · NN+1 →
```

## Hard rules

0. **Code-extraction forensics protocol (prevents context flooding)**: Grep the
   **current tree** to locate the function → Read a bounded line range →
   excerpt 10-40 lines. Verify every line number you cite by actually reading
   it. Use `git show <hash> -- <file>` only when a historical version is
   genuinely needed (a bug's origin, a before/after pair); locate with
   `--stat` first, then a narrow range; ≤3 git calls per doc.
1. **Write only your assigned `docs/inference_e2e_walkthrough/NN-*.md` file.**
   Do not touch source files, other docs, or the index. Do not commit (the
   coordinator commits).
2. **Pace: write the doc to disk as soon as its forensics is done** — do not
   stage everything in memory first.
3. Facts and numbers must come from the code or from the existing docs
   (`docs/ARCHITECTURE.md`, `docs/GRAPH-REFACTOR-PLAN.md`, the
   `docs/*OPTIMIZATION*.md` series). Do not invent; when a number is
   measured elsewhere, cite the doc.
4. File naming: `NN-english-dash-slug.md`; NN is assigned by the task.
5. **Length: content-first. Target 400-800 lines.** Depth over brevity, but
   never padding: every paragraph must teach something. If a stage genuinely
   needs more (e.g. the graph build), write more.
6. Beginner definitions are mandatory, not decorative: a reader who has never
   seen a KV cache must finish doc 11 knowing what one is and why decode is
   cheap because of it.
7. End every doc with the nav line (omit an end link at the ends of the
   range); the index link is `./README.md`.

## Series facts every author can rely on

- Engine: pure-Rust LLM inference engine, llama.cpp-inspired, zero ML
  framework deps; models Qwen2/Qwen2.5 + Qwen3 (dense); backends CPU
  (AVX2/NEON) + Metal (macOS) + CUDA (opt-in); GGUF v3 models.
- One inference run = **prefill** (all prompt tokens in one graph forward)
  then an **autoregressive decode loop** (one token per forward, graph reused).
- Every forward = **build → assign backends → fuse → allocate → execute**,
  but the built graph is cached per `GraphParams` and reused params-only.
- KV cache lives in the graph allocator as two persistent regions per layer
  (K and V) and survives graph rebuilds.
- Backend assignment is a build-time decision (`supports_op`, priority
  Metal → CUDA → CPU); kernel-invariant violations return `Err` — never a
  silent CPU fallback.
- `docs/ARCHITECTURE.md` is verified current (commit `e7fa0da`); line numbers
  in these docs are accurate as of the same commit.
