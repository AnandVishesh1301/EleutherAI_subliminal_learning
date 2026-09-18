#!/usr/bin/env bash
# Launcher: one ws160m partition on its home node. Part I partitions (main, ladder,
# dosehp, extra) are run sets 20260916_ws160m-<partition>_160m in tmux `ws160m`;
# Part II partitions (matrix, grid) are 20260918_ws160m2-<partition>_160m in tmux
# `ws160m2`, staged with `--gen 2` and smoked with `scripts.smoke_ws160m --gen 2`.
# Run from the campaign code pin <main checkout>/.code_pins/<sha>, after staging
# (`scripts.stage_ws160m_assets` on soar) and the wiring smoke.
#
# Refuses: a partition whose home node is not this host; a checkout that is not the
# pin of <sha>; an unpushed SHA or one failing the fleet guard; an SL_JOBQ_LOCK_PREFIX
# other than the partition's (empty on soar, which uses the shared owner namespace);
# missing/failed staging or smoke reports; an existing tmux session, live pool on
# this root, stop sentinel, or an empty queue list. Then backs up + writes
# gpu_policy.$HOST.json, records pool_scope=campaign, and starts the partition's tmux
# session (matched exactly, so an idle session of the other generation is untouched):
#   workers: restarting jobq pool over this partition's queues (quoted list);
#   driver:  restarting `watch` loop that stops on exit 0 (done) or 2 (gate).
#
# Usage: bash scripts/launch_ws160m_campaign.sh <partition> <sha>
set -euo pipefail

PARTITION="${1:-}"
SHA="${2:-}"
if [[ -z "$PARTITION" || -z "$SHA" ]]; then
  echo "usage: bash scripts/launch_ws160m_campaign.sh <main|ladder|dosehp|extra|matrix|grid> <sha>"
  exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
PIN_SHA=$(git rev-parse HEAD)
FULL_SHA=$(git rev-parse --verify "${SHA}^{commit}")
if [[ "$FULL_SHA" != "$PIN_SHA" || "$(basename "$(dirname "$REPO")")" != ".code_pins" || "$(basename "$REPO")" != "$PIN_SHA" ]]; then
  echo "ERROR: launch from <main checkout>/.code_pins/$FULL_SHA (this is $REPO at $PIN_SHA)."
  exit 1
fi

PLAN="from subliminal_learning.campaign import ws160m_plan as p;part=p.partition('$PARTITION')"
read -r NODE LOCK SESSION WORKERS PATHS PLAN_DEADLINE < <(uv run python -c "$PLAN;print(part.node, part.lock_prefix or '-', part.tmux_session, p.workers(), p.PATHS_CONFIG, part.deadline_hours)")
if [[ "$LOCK" == "-" ]]; then
  LOCK=""
fi
DEADLINE_HOURS="${WS160M_DEADLINE_HOURS:-$PLAN_DEADLINE}"
if [[ "$HOST" != "$NODE" ]]; then
  echo "ERROR: partition $PARTITION launches only on $NODE; this host is $HOST."
  exit 1
fi

git fetch --quiet origin
if [[ -z "$(git branch -r --contains "$FULL_SHA")" ]]; then
  echo "ERROR: $FULL_SHA is not on any pushed branch."
  exit 1
fi
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_ws160m_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check (fleet-setup commits missing)."
  exit 1
fi

LOCK_PREFIX="${SL_JOBQ_LOCK_PREFIX:-}"
if [[ "$LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$LOCK_PREFIX' but partition $PARTITION requires '$LOCK'."
  exit 1
fi
if [[ ! -f "$PATHS" ]]; then
  echo "ERROR: $PATHS missing; run: uv run python -m scripts.stage_ws160m_assets --write-paths-only"
  exit 1
fi

PATHS_PY="import yaml;from pathlib import Path;paths=yaml.safe_load(Path('$PATHS').read_text())"
ROOT=$(uv run python -c "$PATHS_PY;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(paths))")
mkdir -p "$ROOT" "$ROOT/logs"
if [[ -z "$LOCK" ]]; then
  LOCK_PREFIX="$ROOT/locks/jobq"
  if [[ "$(cat "$ROOT/locks/SHARED_LOCK_OWNER" 2>/dev/null)" != "$HOST" ]]; then
    echo "ERROR: $ROOT/locks/SHARED_LOCK_OWNER does not name $HOST; refusing the shared lock prefix."
    exit 1
  fi
fi

if ! uv run python -c "import sys;from pathlib import Path;$PLAN;from scripts.stage_ws160m_assets import reports_ok;sys.exit(0 if reports_ok(Path('$PATHS'), require_smoke=True, partition='$PARTITION') else 1)"; then
  echo "ERROR: staging / smoke report of partition $PARTITION missing or not passed."
  exit 1
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
  exit 1
fi
if pgrep -af "scripts.jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  pgrep -af "scripts.jobq work --root $ROOT" || true
  exit 1
fi
if [[ -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ -f "$POLICY" ]]; then
  cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
fi
uv run python -c "import json;$PLAN;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
echo "wrote $POLICY: $(cat "$POLICY")"

QUEUES_CMD="$PLAN;print(','.join(p.queue_priorities(part)))"
Q=$(uv run python -c "$QUEUES_CMD")
if [[ -z "$Q" ]]; then
  echo "ERROR: empty queue list for partition $PARTITION."
  exit 1
fi

uv run python -c "
import yaml
from pathlib import Path
from pipelines.ws160m_campaign_pipeline import driver_config
from subliminal_learning.campaign import driver_io
from subliminal_learning.utils.code_pin_io import main_checkout_root
paths = yaml.safe_load(Path('$PATHS').read_text())
cfg = driver_config('$PARTITION', output_root=Path(paths['output_root']), queue_root=Path(r'''$ROOT'''), repo_root=main_checkout_root(Path('.')))
state = driver_io.load_state(cfg)
state['pool_scope'] = 'campaign'
driver_io.save_state(cfg, state)
print('pool_scope', state['pool_scope'])
"

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
tmux send-keys -t "$SESSION:workers" \
  "Q=\$(uv run python -c \"$QUEUES_CMD\"); if [ -z \"\$Q\" ]; then echo 'empty queue list'; exit 1; fi; while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues \"\$Q\" --lock-prefix $LOCK_PREFIX; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
tmux send-keys -t "$SESSION:driver" \
  "backoff=60; while true; do uv run python -m scripts.run_ws160m_campaign watch --partition $PARTITION --paths-config $PATHS --queue-root $ROOT --interval 300 --auto-requeue 2 --deadline-hours $DEADLINE_HOURS --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<MSG

Started tmux session '$SESSION' on $HOST for partition $PARTITION:
  workers -> jobq pool ($WORKERS threads, lock prefix $LOCK_PREFIX), root=$ROOT
  driver  -> restarting watch loop (5-min polls, --auto-requeue 2, deadline ${DEADLINE_HOURS}h, pin $PIN_SHA)

One glance after launch is enough (cap_per_gpu from $POLICY); repack only on OOM.
Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_ws160m_campaign status-cmd --partition $PARTITION --paths-config $PATHS --queue-root $ROOT
MSG
