#!/usr/bin/env bash
# Identity battery (T-series acceptance gate; reconstructs the doc-94/104
# battery as a repo script so the gate stops living in /tmp).
#
# Definition (docs 94/103/104): 4 fixed prompts, greedy, spec-vs-sequential
# byte comparison after the standard strip filter. Prompt classes follow the
# documented battery: prose, code, plot-summary, TCP/UDP explainer. Target =
# Qwen2.5-7B Q8_0 (exercises the doc-104 p32 multi-vs-nt1 bitwise chain),
# draft = Qwen2.5-0.5B Q4_K_M, adaptive draft depth (doc 95, d_max 7 keeps
# verify nt <= 8 — the doc-95 identity bound).
#
# Usage: scripts/identity_battery.sh [target_gguf] [draft_gguf]
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/minfer
TARGET=${1:-$HOME/.cache/minfer/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF-q8_0/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf}
DRAFT=${2:-$HOME/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf}
N=200
STRIP='Prefill:|Generated:|Total:|^\[spec\]|^ *$|^---$|^Model loaded|^Vocabulary|^Loaded:|^CUDA:|^CUDA device tier|^GGUF:|bytes \(|tolerate|Loading model|minfer/spec'

prompts=(
  "Write a short essay about how rivers shaped the growth of ancient cities, covering water supply, trade routes, and defense."
  "Write a Python function that merges two sorted lists into one sorted list without using the built-in sort, and explain its time complexity."
  "Summarize the plot of Romeo and Juliet in five sentences, focusing on the role of the feud and the ending."
  "Explain the difference between TCP and UDP, and give one concrete example where each is the right choice."
)

pass=0; fail=0
for i in "${!prompts[@]}"; do
  p="${prompts[$i]}"
  seq_out=$(timeout 600 "$BIN" --greedy -n "$N" "$TARGET" "$p" 2>/dev/null | grep -vE "$STRIP")
  spec_out=$(timeout 600 "$BIN" --greedy -n "$N" --spec-draft "$DRAFT" --spec-draft-adaptive "$TARGET" "$p" 2>/dev/null | grep -vE "$STRIP")
  if [ "$seq_out" == "$spec_out" ]; then
    echo "prompt $((i+1)) [${p:0:40}...]: IDENTICAL"
    pass=$((pass+1))
  else
    echo "prompt $((i+1)) [${p:0:40}...]: DIVERGED"
    diff <(printf '%s' "$seq_out") <(printf '%s' "$spec_out") | head -5
    fail=$((fail+1))
  fi
done
echo "identity battery: $pass/4 identical, $fail diverged"
[ "$fail" -eq 0 ]
