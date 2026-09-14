#!/usr/bin/env bash
# Launcher: strictscreen campaign (20260915_strictscreen_160m) on soar-subliminal-3
# (a40-cluster-4). Run from the campaign code pin <main checkout>/.code_pins/<sha>,
# after staging and the train smoke:
#   uv run python -m scripts.stage_strictscreen_assets
#   uv run python -m scripts.stage_strictscreen_assets --smoke-verify DIR
#
# Refuses: any other host; a checkout that is not the pin of <sha>; an unpushed SHA or
# one failing the fleet guard; SL_JOBQ_LOCK_PREFIX other than the plan's; missing or
# failed staging/smoke reports; an existing tmux session, live pool, stop sentinel,
# GPUS_RELEASED or WORKER_LOOP_EXITED marker; a judge venv that cannot be synced
# (`uv sync --project tools/rollout_judge --frozen`) or lacks vllm 0.8.5.post1.
# Then backs up + writes gpu_policy.$HOST.json and starts tmux `strictscreen`:
#   workers: jobq pool until the audit driver writes GPUS_RELEASED (or stops the pool
#            after sl_results), then touches WORKER_LOOP_EXITED;
#   driver:  restarting `watch` loop that stops on exit 0 (done) or 2 (gate).
#
# Usage: bash scripts/launch_strictscreen_campaign.sh <sha>
set -euo pipefail

SHA="${1:-}"
if [[ -z "$SHA" ]]; then
  echo "usage: bash scripts/launch_strictscreen_campaign.sh <sha>"
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

PLAN="from subliminal_learning.campaign import strictscreen_plan as p"
read -r NODE LOCK SESSION WORKERS < <(uv run python -c "$PLAN;print(p.NODE, p.LOCK_PREFIX, p.TMUX_SESSION, p.WORKERS)")
if [[ "$HOST" != "$NODE" ]]; then
  echo "ERROR: strictscreen launches only on $NODE; this host is $HOST."
  exit 1
fi

git fetch --quiet origin
if [[ -z "$(git branch -r --contains "$FULL_SHA")" ]]; then
  echo "ERROR: $FULL_SHA is not on any pushed branch."
  exit 1
fi
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_strictscreen_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check (fleet-setup commits missing)."
  exit 1
fi

LOCK_PREFIX="${SL_JOBQ_LOCK_PREFIX:-}"
if [[ "$LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$LOCK_PREFIX' but strictscreen requires '$LOCK'."
  exit 1
fi

PATHS_PY="import yaml;from pathlib import Path;paths=yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text())"
ROOT=$(uv run python -c "$PATHS_PY;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(paths))")
LOGS=$(uv run python -c "$PATHS_PY;from subliminal_learning.campaign import strictscreen_jobs as j;print(j.logs_dir(Path(paths['output_root'])))")

if ! uv run python -c "import json,sys;from pathlib import Path;$PLAN;sys.exit(0 if all(json.loads((Path('$LOGS')/n).read_text()).get('passed') is True for n in (p.STAGING_REPORT, p.SMOKE_REPORT)) else 1)"; then
  echo "ERROR: $LOGS lacks passed staging_report.json and smoke_report.json."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
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
for marker in GPUS_RELEASED WORKER_LOOP_EXITED; do
  if [[ -e "$LOGS/$marker" ]]; then
    echo "ERROR: $LOGS/$marker exists; the GPU pool was already released."
    exit 1
  fi
done
if ! uv sync --project tools/rollout_judge --frozen --quiet; then
  echo "ERROR: uv sync --project tools/rollout_judge --frozen failed in the pin."
  exit 1
fi
if ! uv run --no-sync --project tools/rollout_judge python -c "import vllm,sys;sys.exit(0 if vllm.__version__ == '0.8.5.post1' else 1)"; then
  echo "ERROR: the tools/rollout_judge venv does not import vllm 0.8.5.post1."
  exit 1
fi

POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ -f "$POLICY" ]]; then
  cp -a "$POLICY" "$POLICY.bak_pre_launch_$(date -u +%Y%m%dT%H%M%SZ)"
fi
uv run python -c "import json;$PLAN;print(json.dumps(p.gpu_policy_json(), indent=2))" > "$POLICY"
echo "wrote $POLICY: $(cat "$POLICY")"
QUEUES_CMD="$PLAN;print(','.join(p.QUEUE_PRIORITIES))"

tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
# A stop sentinel before sl_results is a manual pause (wait); after it, the audit
# driver is releasing the GPUs, so the loop exits and records WORKER_LOOP_EXITED.
tmux send-keys -t "$SESSION:workers" \
  "Q=\$(uv run python -c \"$QUEUES_CMD\"); while [ ! -e $LOGS/GPUS_RELEASED ]; do if [ -e $ROOT/stop.$HOST ]; then if grep -q '\"sl_results\":' $LOGS/campaign_state.json 2>/dev/null; then break; fi; echo 'stop sentinel present (pause); waiting'; sleep 60; continue; fi; uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues \$Q --lock-prefix $LOCK_PREFIX; sleep 60; done; touch $LOGS/WORKER_LOOP_EXITED; echo 'worker loop exited'" C-m

tmux new-window -t "$SESSION" -n driver -c "$REPO"
# No --auto-reconcile-sha: a pin change is an explicit operator action.
tmux send-keys -t "$SESSION:driver" \
  "backoff=60; while true; do uv run python -m scripts.run_strictscreen_campaign watch --queue-root $ROOT --interval 300 --auto-requeue 2 --code-sha $PIN_SHA; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done" C-m

cat <<MSG

Started tmux session '$SESSION' on $HOST:
  workers -> jobq pool ($WORKERS threads, lock prefix $LOCK_PREFIX), root=$ROOT
  driver  -> restarting watch loop (5-min polls, --auto-requeue 2, pin $PIN_SHA)

One glance after launch is enough (3 cells per GPU); repack only on OOM.
Attach:  tmux attach -t $SESSION      Detach: Ctrl-b d
Status:  uv run python -m scripts.run_strictscreen_campaign status --queue-root $ROOT
MSG
