#!/usr/bin/env bash
# Refresh untracked agent context soar -> a vishesh-owned fleet node (runs ON soar-subliminal-0).
#
# Usage: tools/node_bootstrap/sync_agent_context.sh <alias>
#
# One-way copy of soar's untracked .claude/ (skills, rules, agents mirrored with --delete;
# plans, reviews, settings.local.json added/updated only) and soar-untracked docs/ files.
# Never writes a path the node clone tracks at its checked-out sha, so the node's
# `git status --untracked-files=no` stays clean and pins are unaffected.
set -euo pipefail

ALIAS="${1:?usage: sync_agent_context.sh <alias>}"
SRC="$HOME/eleutherAI/EleutherAI_subliminal_learning"
DEST="eleutherAI/EleutherAI_subliminal_learning"
SSH=(ssh -o BatchMode=yes "$ALIAS")

[ "$(hostname)" = "soar-subliminal-0" ] || { echo "refusing: run on soar-subliminal-0 (agent-context SSOT)" >&2; exit 1; }
REMOTE_HOST="$("${SSH[@]}" hostname)"
case "$REMOTE_HOST" in
  goncalo-temp-1|soar-subliminal-0) echo "refusing: $ALIAS is $REMOTE_HOST" >&2; exit 1 ;;
esac
"${SSH[@]}" "test -d ~/$DEST/.git" || { echo "run bootstrap_node.sh on $ALIAS first" >&2; exit 1; }
[ -z "$("${SSH[@]}" "git -C ~/$DEST ls-files .claude | head -1")" ] \
  || { echo "refusing: .claude is tracked at the node's sha; copy would dirty the clone" >&2; exit 1; }

for d in skills rules agents; do
  rsync -a --delete "$SRC/.claude/$d/" "$ALIAS:$DEST/.claude/$d/"
done
for d in plans reviews; do
  [ -d "$SRC/.claude/$d" ] && rsync -a "$SRC/.claude/$d/" "$ALIAS:$DEST/.claude/$d/"
done
[ -f "$SRC/.claude/settings.local.json" ] && rsync -a "$SRC/.claude/settings.local.json" "$ALIAS:$DEST/.claude/"

LIST="$(mktemp)"
comm -23 \
  <(git -C "$SRC" ls-files --others --exclude-standard -- docs | sort) \
  <("${SSH[@]}" "git -C ~/$DEST ls-files -- docs" | sort) > "$LIST"
rsync -a --files-from="$LIST" "$SRC/" "$ALIAS:$DEST/"
printf '[sync %s] .claude/{skills,rules,agents,plans,reviews} + %s untracked docs file(s)\n' "$REMOTE_HOST" "$(wc -l < "$LIST")"
rm -f "$LIST"
