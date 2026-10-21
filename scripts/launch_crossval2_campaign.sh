#!/usr/bin/env bash
# Launcher of crossval2 on goncalo, in ONE tmux session `crossval2`: window `pool` = one
# jobq pool over the campaign's own queue root (`jobq_pool_loop.sh`: idles under a stop
# sentinel, exits on the campaign done marker); one restarting driver window per run set
# (`scripts.run_sdfpos_campaign watch --campaign crossval2`). Run from the campaign code pin
# <main checkout>/.code_pins/<sha> on goncalo-temp-1, after scripts/stage_crossval2_assets.py.
#
# Refuses: a checkout that is not the pin of <sha>; an unpushed SHA or one failing the fleet
# guard; another host; an SL_JOBQ_LOCK_PREFIX other than the plan's; a missing or differing
# paths YAML / GPU policy / prereg snapshot; an existing `crossval2` session, a live pool on
# the campaign root, or a stop sentinel (pool start only).
#
# Usage: bash scripts/launch_crossval2_campaign.sh <sha> [--drivers-only]
set -euo pipefail

SHA="${1:-}"
MODE="${2:-}"
if [[ -z "$SHA" || ( -n "$MODE" && "$MODE" != "--drivers-only" ) ]]; then
  echo "usage: bash scripts/launch_crossval2_campaign.sh <sha> [--drivers-only]"
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

PLAN="from subliminal_learning.campaign import crossval2_plan as p"
read -r NODE LOCK SESSION WORKERS PATHS ROOT DONE PREREG < <(uv run python -c "$PLAN;print(p.HOME_NODE, p.LOCK_PREFIX, p.TMUX_SESSION, p.workers(), p.PATHS_CONFIG, p.QUEUE_ROOT, p.CAMPAIGN_DONE_PATH, p.PREREG_SNAPSHOT)")
if [[ "$HOST" != "$NODE" ]]; then
  echo "ERROR: crossval2 runs on $NODE, not $HOST."
  exit 1
fi

git fetch --quiet origin
if [[ -z "$(git branch -r --contains "$FULL_SHA")" ]]; then
  echo "ERROR: $FULL_SHA is not on any pushed branch."
  exit 1
fi
if ! (git grep -q SHARED_LOCK_OWNER "$FULL_SHA" -- src/subliminal_learning/jobqueue \
  && git show "$FULL_SHA:scripts/launch_crossval2_campaign.sh" | grep -q SL_JOBQ_LOCK_PREFIX); then
  echo "ERROR: $FULL_SHA fails the fleet-guard check."
  exit 1
fi
LOCK_PREFIX="${SL_JOBQ_LOCK_PREFIX:-}"
if [[ "$LOCK_PREFIX" != "$LOCK" ]]; then
  echo "ERROR: SL_JOBQ_LOCK_PREFIX='$LOCK_PREFIX' but $HOST requires '$LOCK'."
  exit 1
fi

EXPECTED=$(uv run python -c "from pipelines.crossval2_staging_pipeline import paths_yaml_text;print(paths_yaml_text(), end='')")
if [[ ! -f "$PATHS" || "$(cat "$PATHS")" != "$EXPECTED" ]]; then
  echo "ERROR: $PATHS missing or differs from the plan; run scripts/stage_crossval2_assets.py."
  exit 1
fi
POLICY="$ROOT/gpu_policy.$HOST.json"
if [[ "$(uv run python -c "import json,sys;$PLAN;print(json.load(open(sys.argv[1])) == p.gpu_policy_json())" "$POLICY" 2>/dev/null)" != "True" ]]; then
  echo "ERROR: $POLICY missing or differs from crossval2_plan.gpu_policy_json()."
  exit 1
fi
if [[ ! -f "$PREREG" ]]; then
  echo "ERROR: prereg snapshot $PREREG missing."
  exit 1
fi

EXPS=$(uv run python -c "$PLAN;print(' '.join(p.EXPERIMENT_ORDER))")
# A driver restart onto a new pin re-pins each run-set card it owns (driver_io reconcile).
RECONCILE=""
if [[ "$MODE" == "--drivers-only" ]]; then RECONCILE="--auto-reconcile-sha"; fi
# Import + config smoke from the pin: every module a cell or driver loads, and the
# student experiment config through the GKD loader the sweep uses.
uv run python - "$PATHS" <<'PY' || { echo "ERROR: import / config smoke failed in the pin."; exit 1; }
import sys
from pathlib import Path
import pipelines.crossval2_gates, pipelines.crossval2_panel_pipeline  # noqa: F401
import scripts.sweeps.sweep_distill_student_gkd, scripts.sweeps.sweep_evaluate_gkd_student  # noqa: F401
import scripts.sweeps.sweep_evaluate_sdf_teacher, scripts.sweeps.sweep_sdf_teacher  # noqa: F401
from subliminal_learning.campaign import crossval2_plan as p
from subliminal_learning.gkd.config import load_gkd_config
cfg = load_gkd_config(Path(sys.argv[1]), Path("configs/model/pythia.yaml"), Path("configs/training/gkd_student.yaml"), Path(p.STUDENT_EXPERIMENT_CONFIG))
assert cfg.gkd.secondary_probe_sets and cfg.gkd.secondary_probe_targets == ("bear", "lion"), cfg.gkd
print("smoke ok:", cfg.gkd.secondary_probe_sets)
PY
driver_loop() {
  echo "backoff=60; while true; do uv run python -m scripts.run_sdfpos_campaign watch --campaign crossval2 --partition $1 --paths-config $PATHS --queue-root $ROOT --interval 300 --auto-requeue 3 --code-sha $PIN_SHA $RECONCILE; ec=\$?; if [ \$ec -eq 0 ]; then echo 'driver finished'; break; fi; if [ \$ec -eq 2 ]; then echo 'GATE FAILED -- not restarting'; break; fi; echo \"driver exited \$ec; restarting in \${backoff}s\"; sleep \$backoff; backoff=\$((backoff < 600 ? backoff * 2 : 600)); done"
}

if [[ "$MODE" == "--drivers-only" ]]; then
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "ERROR: --drivers-only needs the running '$SESSION' session."
    exit 1
  fi
  for EXP in $EXPS; do
    if pgrep -f "scripts.run_sdfpos_campaign watch --campaign crossval2 --partition $EXP " >/dev/null; then
      echo "ERROR: driver $EXP is still running; stop it before --drivers-only."
      exit 1
    fi
  done
  for EXP in $EXPS; do
    tmux kill-window -t "$SESSION:$EXP" 2>/dev/null || true
    tmux new-window -t "$SESSION" -n "$EXP" -c "$REPO"
    tmux send-keys -t "$SESSION:$EXP" "$(driver_loop "$EXP")" C-m
    sleep 2
  done
  echo "restarted drivers for: $EXPS (pin $PIN_SHA)"
  exit 0
fi

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION"
  exit 1
fi
if pgrep -af "jobq work --root $ROOT" >/dev/null 2>&1; then
  echo "ERROR: a jobq worker is already running on $ROOT on this host."
  exit 1
fi
if [[ -e "$ROOT/stop.$HOST" ]]; then
  echo "ERROR: stop sentinel $ROOT/stop.$HOST exists."
  exit 1
fi

tmux new-session -d -s "$SESSION" -n pool -c "$REPO"
tmux send-keys -t "$SESSION:pool" "bash scripts/jobq_pool_loop.sh $ROOT '' $WORKERS $LOCK_PREFIX $DONE" C-m
for EXP in $EXPS; do
  tmux new-window -t "$SESSION" -n "$EXP" -c "$REPO"
  tmux send-keys -t "$SESSION:$EXP" "$(driver_loop "$EXP")" C-m
  sleep 2
done

cat <<MSG

Started tmux session '$SESSION' on $HOST:
  pool  -> jobq pool ($WORKERS threads, lock prefix $LOCK_PREFIX), root=$ROOT, done marker $DONE
  <run set> -> restarting driver loops (5-min polls, --auto-requeue 3, pin $PIN_SHA): $EXPS

Status:  uv run python -m scripts.run_sdfpos_campaign status-cmd --campaign crossval2 --partition <run set> --paths-config $PATHS --queue-root $ROOT
MSG
