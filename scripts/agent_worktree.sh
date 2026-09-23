#!/usr/bin/env bash
# Nested-worktree helper for agent/parallel sessions.
#
# Why nested: an agent session's file policy may be `workspace-write` — the session workspace root
# plus /tmp — so a worktree *beside* the root cannot be written to (and with approval prompts
# disabled there is no way to widen it). `.worktrees/<scope>` lives inside the root instead, which
# works under either policy. See AGENTS.md ("Parallel work in a nested worktree").
#
#   scripts/agent_worktree.sh new <scope> <branch>   create <main>/.worktrees/<scope> on <branch>
#   scripts/agent_worktree.sh rm  <scope>            remove it (with its target/), then prune
#   scripts/agent_worktree.sh list                   this repository's worktrees
#
# `$BASE` overrides the start point (default: origin's HEAD, else origin/master).
# `FORCE=1` skips the `.gitignore` check — the first adoption does that while landing the ignore
# line in the same PR.
set -euo pipefail

main_root=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
[ -n "${main_root:-}" ] || {
  echo "agent_worktree: not inside a git repository" >&2
  exit 1
}

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") new <scope> <branch> | rm <scope> | list

  new   create $main_root/.worktrees/<scope> on <branch> (from \$BASE, default origin/HEAD)
  rm    remove it including its target/, then prune; the branch is kept for the merge flow
  list  print \`git worktree list\`
EOF
  exit 2
}

case "${1:-}" in
  new)
    [ $# -eq 3 ] || usage
    scope=$2
    branch=$3
    dir="$main_root/.worktrees/$scope"
    if [ -e "$dir" ]; then
      echo "agent_worktree: $dir already exists" >&2
      exit 1
    fi
    if ! git -C "$main_root" check-ignore -q .worktrees && [ "${FORCE:-0}" != 1 ]; then
      cat >&2 <<EOF
agent_worktree: \`.worktrees/\` is not ignored in $main_root, and an unignored nested worktree is a
hazard: the outer tree's \`git status\` shows it as \`?? .worktrees/\` and \`git add -A\` there warns
"adding embedded git repository" and stages it as a gitlink. The line belongs on the default
branch, so every worktree inherits it:

    echo '.worktrees/' >> .gitignore

Commit that, then re-run — or set FORCE=1 to proceed anyway.
EOF
      exit 1
    fi
    base=${BASE:-$(git -C "$main_root" symbolic-ref -q --short refs/remotes/origin/HEAD ||
      echo origin/master)}
    git -C "$main_root" worktree add ".worktrees/$scope" -b "$branch" "$base"
    echo "agent_worktree: created $dir on '$branch' (from $base)"
    echo "agent_worktree: run every later command inside it, with absolute paths — and never point"
    echo "                CARGO_TARGET_DIR at $main_root (it would overwrite its build artifacts)."
    ;;
  rm)
    [ $# -eq 2 ] || usage
    git -C "$main_root" worktree remove --force ".worktrees/$2"
    git -C "$main_root" worktree prune
    echo "agent_worktree: removed $main_root/.worktrees/$2 (branch kept: delete it after the merge)"
    ;;
  list)
    git -C "$main_root" worktree list
    ;;
  *)
    usage
    ;;
esac
