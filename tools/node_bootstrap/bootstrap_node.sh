#!/usr/bin/env bash
# Idempotent A40 fleet node bootstrap (runs ON a vishesh-owned node; secret-free).
#
# From soar:  ssh <alias> 'bash -s -- --sha <pushed-sha>' < tools/node_bootstrap/bootstrap_node.sh
# Then:       tools/node_bootstrap/push_node_secrets.sh <alias>
#
# Installs the soar-identical shell files, uv + nvitop, and a detached consumer
# clone at <sha> with a frozen venv. Existing differing files are backed up to
# <file>.pre_fleet.<UTC-ts>; nothing is ever deleted.
set -euo pipefail

# Wrapped in main so `bash -s` parses the whole script before any command can read stdin.
main() {

UV_VERSION="0.12.2"
NVITOP_VERSION="1.7.1"
REPO_URL="git@github.com:AnandVishesh1301/EleutherAI_subliminal_learning.git"
CLONE="$HOME/eleutherAI/EleutherAI_subliminal_learning"
RUN_LOGS_TARGET="/mnt/ssd-3/soar-subliminal/vishesh/run_logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

SHA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA="${2:?--sha needs a value}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SHA" ] || { echo "usage: bootstrap_node.sh --sha <pushed-sha>" >&2; exit 2; }
[ "$(id -u)" -ne 0 ] || { echo "refusing: do not run as root" >&2; exit 1; }
case "$(hostname)" in
  goncalo-temp-1) echo "refusing: goncalo-temp-1 is a shared node (see docs/DUAL_NODE_GONCALO.md)" >&2; exit 1 ;;
esac

log() { printf '[bootstrap %s] %s\n' "$(hostname)" "$*"; }

# install_file <dest> <mode>: stdin is the exact desired content.
install_file() {
  local dest="$1" mode="$2" tmp
  tmp="$(mktemp)"
  cat > "$tmp"
  if [ -e "$dest" ] && cmp -s "$tmp" "$dest"; then
    rm -f "$tmp"; chmod "$mode" "$dest"; log "unchanged $dest"; return
  fi
  if [ -e "$dest" ]; then cp -p "$dest" "$dest.pre_fleet.$TS"; log "backup $dest.pre_fleet.$TS"; fi
  mkdir -p "$(dirname "$dest")"
  chmod "$mode" "$tmp"
  mv "$tmp" "$dest"
  log "wrote $dest"
}

# ── Shell files (identical to soar-subliminal-0 + fleet node-local block) ──
install_file "$HOME/.zshrc" 644 <<'EOF'
# Never run the Powerlevel10k configuration wizard (theme is not sourced on this node).
export POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true

[[ -o interactive ]] || return 0

# Minimal interactive zsh config (no theme), identical to soar-subliminal-0. Env vars live in ~/.zshenv.

# History
HISTFILE="$HOME/.zsh_history"
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_IGNORE_DUPS SHARE_HISTORY

# Completion
autoload -Uz compinit
compinit -C

# Prompt
PS1='%n@%m:%~$ '

# Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
EOF

install_file "$HOME/.zshenv" 644 <<'EOF'
# Shared environment for zsh (default) and bash on this server (A40 fleet node; mirrors soar-subliminal-0).

# MCP / agent API keys
if [ -f "$HOME/.cursor/mcp.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HOME/.cursor/mcp.env"
  set +a
fi

if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# ── Hugging Face cache on SSD ──
# Keep in sync with configs/paths/paths_local.yaml hf_cache_root (docs/SSD3_HOT_MIRROR.md).
export HF_HOME="/mnt/ssd-3/soar-subliminal/vishesh/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# ── Fleet node-local settings (docs/CLUSTER_NODES.md) ──
# jobq GPU locks stay on this node's /tmp; never the shared NFS queue_root/locks.
export SL_JOBQ_LOCK_PREFIX="/tmp/subliminal_jobq_vishesh"
# Consumer node: install exactly the committed uv.lock, never rewrite it.
export UV_FROZEN=1
EOF

# Sourced only; mode 644 so it never shadows /usr/bin/env on PATH (~/.local/bin comes first).
install_file "$HOME/.local/bin/env" 644 <<'EOF'
#!/bin/sh
# uv path helper (created to match Astral installer layout; PATH managed here too).
case ":${PATH}:" in
    *:"$HOME/.local/bin":*)
        ;;
    *)
        export PATH="$HOME/.local/bin:$PATH"
        ;;
esac
EOF

if ! grep -qF "# Shared env — single source of truth with zsh (~/.zshenv)." "$HOME/.bashrc" 2>/dev/null; then
  [ -e "$HOME/.bashrc" ] && cp -p "$HOME/.bashrc" "$HOME/.bashrc.pre_fleet.$TS"
  printf '\n# Shared env — single source of truth with zsh (~/.zshenv).\nif [ -f "$HOME/.zshenv" ]; then\n  . "$HOME/.zshenv"\nfi\n' >> "$HOME/.bashrc"
  log "appended ~/.zshenv source to ~/.bashrc"
fi

# shellcheck disable=SC1091
. "$HOME/.zshenv"

# ── uv (pinned; never let the installer edit shell rc files) + nvitop ──
if [ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  installer="$(mktemp)"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
  env UV_NO_MODIFY_PATH=1 sh "$installer" < /dev/null
  rm -f "$installer"
fi
[ "$(uv --version | awk '{print $2}')" = "$UV_VERSION" ] || { echo "uv version mismatch" >&2; exit 1; }
if [ "$(nvitop --version 2>/dev/null | awk '{print $2}')" != "$NVITOP_VERSION" ]; then
  uv tool install --force "nvitop==$NVITOP_VERSION"
fi
log "uv $(uv --version | awk '{print $2}'), nvitop $(nvitop --version | awk '{print $2}')"

# ── Consumer clone, detached at a pushed sha ──
if [ ! -d "$CLONE/.git" ]; then
  mkdir -p "$(dirname "$CLONE")"
  git clone "$REPO_URL" "$CLONE"
fi
[ "$(git -C "$CLONE" remote get-url origin)" = "$REPO_URL" ] || { echo "unexpected origin in $CLONE" >&2; exit 1; }
git -C "$CLONE" fetch --all --prune
FULL_SHA="$(git -C "$CLONE" rev-parse --verify "$SHA^{commit}")"
git -C "$CLONE" branch -r --contains "$FULL_SHA" | grep -q . || { echo "refusing: $FULL_SHA is not on any origin branch" >&2; exit 1; }

if [ "$(git -C "$CLONE" rev-parse HEAD)" != "$FULL_SHA" ]; then
  [ -z "$(git -C "$CLONE" status --porcelain --untracked-files=no)" ] || { echo "refusing: $CLONE has local changes" >&2; exit 1; }
  for p in /proc/[0-9]*; do
    [ "$(stat -c %u "$p" 2>/dev/null)" = "$(id -u)" ] || continue
    [ "$p" = "/proc/$$" ] && continue
    cwd="$(readlink "$p/cwd" 2>/dev/null || true)"
    case "$cwd" in
      "$CLONE"|"$CLONE"/*)
        case "$cwd" in "$CLONE"/.code_pins/*) ;; *) echo "refusing: pid ${p#/proc/} runs in $cwd" >&2; exit 1 ;; esac ;;
    esac
  done
  git -C "$CLONE" checkout --detach "$FULL_SHA"
fi

if [ "$(readlink "$CLONE/run_logs" 2>/dev/null)" != "$RUN_LOGS_TARGET" ]; then
  ln -sfn "$RUN_LOGS_TARGET" "$CLONE/run_logs"
  log "run_logs -> $RUN_LOGS_TARGET"
fi
git -C "$CLONE" update-index --skip-worktree run_logs

(cd "$CLONE" && uv sync --frozen)
log "clone at $(git -C "$CLONE" rev-parse HEAD); venv $("$CLONE/.venv/bin/python" --version)"
}

main "$@"
