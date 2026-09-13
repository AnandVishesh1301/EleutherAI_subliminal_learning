#!/usr/bin/env bash
# Push soar's agent/API config to a vishesh-owned fleet node (runs ON soar-subliminal-0).
#
# Usage: tools/node_bootstrap/push_node_secrets.sh <alias>   (after bootstrap_node.sh)
#
# Copies ~/.cursor/mcp.env, ~/.cursor/mcp.json, ~/.netrc (mode 600), merges soar's
# ~/.claude/settings.json (soar wins) and user-scope mcpServers into the node's
# ~/.claude.json, syncs ~/.claude/statusline/, and copies paths_local.yaml into the
# node clone. Secrets travel over scp/stdin only and are never printed. Differing
# node files are backed up to <file>.pre_fleet.<UTC-ts>; nothing is deleted.
set -euo pipefail

ALIAS="${1:?usage: push_node_secrets.sh <alias>}"
PATHS_LOCAL="$HOME/eleutherAI/EleutherAI_subliminal_learning/configs/paths/paths_local.yaml"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SSH=(ssh -o BatchMode=yes "$ALIAS")

[ "$(hostname)" = "soar-subliminal-0" ] || { echo "refusing: run on soar-subliminal-0 (config SSOT)" >&2; exit 1; }
REMOTE_USER="$(ssh -G "$ALIAS" 2>/dev/null | awk '$1=="user"{print $2}')"
case "$ALIAS:$REMOTE_USER" in
  goncalo*|a40-cluster-2:*|*:angela) echo "refusing: goncalo-temp-1 is shared; never push fleet config there" >&2; exit 1 ;;
esac
REMOTE_HOST="$("${SSH[@]}" hostname)"
[ "$REMOTE_HOST" != "goncalo-temp-1" ] || { echo "refusing: goncalo-temp-1 is shared; never push fleet config there" >&2; exit 1; }
[ "$REMOTE_HOST" != "$(hostname)" ] || { echo "refusing: $ALIAS is this host" >&2; exit 1; }
log() { printf '[push %s] %s\n' "$REMOTE_HOST" "$*"; }

# push_file <local> <remote-path relative to ~> <mode>
push_file() {
  local src="$1" dest="$2" mode="$3"
  if [ "$("${SSH[@]}" "sha256sum ~/$dest 2>/dev/null | cut -d' ' -f1")" = "$(sha256sum "$src" | cut -d' ' -f1)" ]; then
    log "unchanged ~/$dest"; return
  fi
  "${SSH[@]}" "mkdir -p \"\$(dirname ~/$dest)\" && { [ ! -e ~/$dest ] || cp -p ~/$dest ~/$dest.pre_fleet.$TS; }"
  scp -q -o BatchMode=yes "$src" "$ALIAS:$dest.tmp.$TS"
  "${SSH[@]}" "chmod $mode ~/$dest.tmp.$TS && mv ~/$dest.tmp.$TS ~/$dest"
  log "wrote ~/$dest ($mode)"
}

"${SSH[@]}" 'mkdir -p ~/.cursor && chmod 700 ~/.cursor'
push_file "$HOME/.cursor/mcp.env" .cursor/mcp.env 600
push_file "$HOME/.cursor/mcp.json" .cursor/mcp.json 600
push_file "$HOME/.netrc" .netrc 600

rsync -a --exclude usage-cache.json "$HOME/.claude/statusline/" "$ALIAS:.claude/statusline/"
log "synced ~/.claude/statusline/"

# MERGE_PY <remote json> <settings|mcp> <ts>: soar JSON on stdin; soar wins, node-only keys kept.
MERGE_PY='
import json, os, shutil, sys, tempfile
path, mode, ts = os.path.expanduser(sys.argv[1]), sys.argv[2], sys.argv[3]
soar = json.load(sys.stdin)
node = json.load(open(path)) if os.path.exists(path) else {}
merged = dict(node)
if mode == "settings":
    merged.update(soar)
else:
    merged["mcpServers"] = {**node.get("mcpServers", {}), **soar}
if merged == node:
    print("unchanged", path); sys.exit(0)
if os.path.exists(path):
    shutil.copy2(path, f"{path}.pre_fleet.{ts}")
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")
os.chmod(tmp, os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o600)
os.replace(tmp, path)
print("merged", path)
'
"${SSH[@]}" "mkdir -p ~/.claude && python3 -c '$MERGE_PY' ~/.claude/settings.json settings $TS" < "$HOME/.claude/settings.json"
python3 -c 'import json,os;json.dump(json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"],__import__("sys").stdout)' \
  | "${SSH[@]}" "python3 -c '$MERGE_PY' ~/.claude.json mcp $TS"

CLONE_PATHS="eleutherAI/EleutherAI_subliminal_learning/configs/paths/paths_local.yaml"
"${SSH[@]}" "test -d ~/eleutherAI/EleutherAI_subliminal_learning/.git" || { echo "run bootstrap_node.sh on $ALIAS first" >&2; exit 1; }
TMP_PATHS="$(mktemp)"
{ printf '# Node copy on %s — pushed from soar-subliminal-0 by tools/node_bootstrap/push_node_secrets.sh.\n' "$REMOTE_HOST"; cat "$PATHS_LOCAL"; } > "$TMP_PATHS"
push_file "$TMP_PATHS" "$CLONE_PATHS" 644
rm -f "$TMP_PATHS"
