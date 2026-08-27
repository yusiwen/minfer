#!/usr/bin/env sh
# rusty-hook pre-commit command: format the crate, then re-stage the files
# that are about to be committed so the commit actually carries the formatted
# code (git commits the index, not the working tree).
set -e

cargo fmt --all

staged=$(git diff --cached --name-only)
if [ -n "$staged" ]; then
  printf '%s\n' "$staged" | xargs git add
fi
