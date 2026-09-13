#!/usr/bin/env bash
# Phase 1 launcher: breadth4 SDF teachers T1 → T2 + verify evals.
#
# Assembles the teacher run-set first (dog symlink from 20260726_crossinit_160m
# + ensure cat 8k corpus), then submits b4-teachers-{t1,t2,t1-evals,t2-evals}
# under the shared jobq root with MLflow-style code-pin defaults
# (cwd=.code_pins/<sha>/ + SL_CODE_PIN_* + PYTHONPATH). Starts a *scoped*
# worker pool (--queues teacher b4-* only) so this pool never drains slb-* /
# optparam-* or student A1+ work.
#
# Sequence inside this pool (jobq depends_on):
#   T1 train → T1 evals; T1 train → T2 train → T2 evals
# (T2 shares the frozen ep5 tag with T1; waiting avoids same-leaf races.)
#
# Train jobs: TEACHER_TRAIN_ANIMALS only (cat/bear/eagle).
# Eval jobs: all ANIMALS including dog (verify on linked dog teachers).
# Campaign CLI key remains --campaign breadth5 (CAMPAIGN_SPECS).
#
# Prerequisites (Phase 0): code + tests green; breadth4 commit pinned.
# Do NOT start launch_breadth5_campaign.sh while this tmux session is alive
# (two 32-thread pools on the same 16 slots → oversubscribe / OOM).
#
# Usage:
#   bash scripts/launch_breadth5_teachers.sh           # assemble + submit + workers
#   bash scripts/launch_breadth5_teachers.sh --submit-only
#   bash scripts/launch_breadth5_teachers.sh --workers-only
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=breadth4_teachers
MODE="${1:-}"

# Dual 32-thread pools on the same 16 slots oversubscribe and OOM.
if tmux has-session -t breadth4 2>/dev/null; then
  echo "tmux session 'breadth4' (student campaign) is already running." >&2
  echo "Stop it before teacher workers: tmux kill-session -t breadth4" >&2
  exit 1
fi

ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text()) or {}))")
mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs" /tmp/breadth4_launch

POLICY="$ROOT/gpu_policy.$HOST.json"
if [ ! -f "$POLICY" ]; then
  uv run python -c "import json;from subliminal_learning.campaign import breadth_plan;print(json.dumps(breadth_plan.GPU_POLICY, indent=2))" > "$POLICY"
  echo "wrote $POLICY"
else
  echo "keeping existing $POLICY: $(tr -d '\n' < "$POLICY")"
fi

WORKERS=$(uv run python -c "from subliminal_learning.campaign import breadth_plan;print(breadth_plan.WORKER_THREADS)")
# Teacher train + eval queues only (not student A1+); full campaign uses launch_breadth5_campaign.sh.
QUEUES=$(uv run python -c "from subliminal_learning.campaign import breadth_plan as p;print(','.join([p.Q_TEACHER_T1,p.Q_TEACHER_T2,p.Q_TEACHER_T1_EVALS,p.Q_TEACHER_T2_EVALS]))")

assemble_teachers() {
  uv run python -m scripts.assemble_breadth4_teachers
}

submit_jobs() {
  assemble_teachers
  uv run python <<'PY'
"""Build job JSONL + submit with code-pin queue defaults."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from subliminal_learning.campaign import breadth_plan as bp
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import parse_jobs_file
from subliminal_learning.utils.code_pin_io import (
    ensure_code_pin,
    pin_job_defaults,
    workspace_git_sha,
)

REPO = Path.cwd()
ROOT = resolve_queue_root(
    yaml.safe_load(Path("configs/paths/paths_local.yaml").read_text()) or {}
)
HOST = store.this_host()
sha = workspace_git_sha(REPO)
if not sha:
    raise SystemExit("cannot resolve workspace HEAD for code pin")
pin = ensure_code_pin(REPO, sha)
plan_path = pin.root / "src/subliminal_learning/campaign/breadth_plan.py"
plan_text = plan_path.read_text() if plan_path.is_file() else ""
if "20260809_breadth4_160m" not in plan_text:
    raise SystemExit(
        f"code pin {pin.sha[:12]} is not the breadth4 restart "
        f"(missing 20260809_breadth4_160m). Commit/pin restart first."
    )
if not (pin.root / "scripts/assemble_breadth4_teachers.py").is_file():
    raise SystemExit(
        f"code pin {pin.sha[:12]} missing scripts/assemble_breadth4_teachers.py"
    )
# Stamp + execute from the pin (MLflow --version analog). Do not use the
# operator checkout: remaining dirty WIP on the branch must not enter cells.
defaults = pin_job_defaults(pin)
print(f"code pin stamp {pin.sha[:12]}; execute cwd={defaults.get('cwd')}")

train_animals = list(bp.TEACHER_TRAIN_ANIMALS)
eval_animals = list(bp.ANIMALS)
models = [
    "original",
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "weight-seed1",
    "weight-seed2",
    "weight-seed3",
]
ftseeds = [0, 1, 2]
launch = Path("/tmp/breadth4_launch")
launch.mkdir(parents=True, exist_ok=True)

t1_lines = []
for animal in train_animals:
    for model in models:
        for tfs in ftseeds:
            key = f"t1|{animal}|{model}|tfs{tfs}"
            cmd = (
                "uv run python -m scripts.sweeps.sweep_sdf_teacher "
                "--sweep-config configs/experiment/sdf_teacher_sweep_breadth5_t1.yaml "
                "--training-config configs/training/lora_sdf_teacher.yaml "
                f"--animals {animal} --base-models {model} --ftseeds {tfs} "
                "--device cuda:0"
            )
            t1_lines.append(json.dumps({"key": key, "cmd": cmd}))
(launch / "t1.jobs.jsonl").write_text("\n".join(t1_lines) + "\n")

t2_lines = []
for animal in train_animals:
    for tfs in ftseeds:
        key = f"t2|{animal}|data-seed1|tfs{tfs}"
        cmd = (
            "uv run python -m scripts.sweeps.sweep_sdf_teacher "
            "--sweep-config configs/experiment/sdf_teacher_sweep_breadth5_t2.yaml "
            "--training-config configs/training/lora_sdf_teacher.yaml "
            f"--animals {animal} --base-models data-seed1 --ftseeds {tfs} "
            "--device cuda:0"
        )
        t2_lines.append(json.dumps({"key": key, "cmd": cmd}))
(launch / "t2.jobs.jsonl").write_text("\n".join(t2_lines) + "\n")

t1e_lines = []
for animal in eval_animals:
    for model in models:
        key = f"t1e|{animal}|{model}"
        cmd = (
            "uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher "
            "--sweep-config configs/experiment/sdf_teacher_verify_sweep_breadth5_t1.yaml "
            "--training-config configs/training/lora_sdf_teacher.yaml "
            f"--animals {animal} --base-models {model} --device cuda:0"
        )
        t1e_lines.append(json.dumps({"key": key, "cmd": cmd}))
(launch / "t1_eval.jobs.jsonl").write_text("\n".join(t1e_lines) + "\n")

t2e_lines = []
for animal in eval_animals:
    key = f"t2e|{animal}|data-seed1"
    cmd = (
        "uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher "
        "--sweep-config configs/experiment/sdf_teacher_verify_sweep_breadth5_t2.yaml "
        "--training-config configs/training/lora_sdf_teacher.yaml "
        f"--animals {animal} --base-models data-seed1 --device cuda:0"
    )
    t2e_lines.append(json.dumps({"key": key, "cmd": cmd}))
(launch / "t2_eval.jobs.jsonl").write_text("\n".join(t2e_lines) + "\n")


def _submit(name: str, jobs_path: Path, priority: int, depends_on: list[str]) -> None:
    jobs = parse_jobs_file(jobs_path.read_text())
    store.check_new_jobs(ROOT, name, jobs)
    store.create_queue(
        ROOT,
        name,
        depends_on=depends_on,
        node=HOST,
        priority=priority,
        defaults=defaults,
        exist_ok=True,
    )
    # Refresh pin defaults on re-submit of an existing empty-ish queue.
    store.update_queue_defaults(ROOT, name, defaults)
    store.append_jobs(ROOT, name, jobs)
    print(f"submitted {name}: {len(jobs)} jobs prio={priority} deps={depends_on} cwd={defaults.get('cwd')}")


_submit(bp.Q_TEACHER_T1, launch / "t1.jobs.jsonl", bp.QUEUE_PRIORITIES[bp.Q_TEACHER_T1], [])
# T2 waits on T1 so the shared frozen tag is written once (cell_is_done skip).
_submit(
    bp.Q_TEACHER_T2,
    launch / "t2.jobs.jsonl",
    bp.QUEUE_PRIORITIES[bp.Q_TEACHER_T2],
    [bp.Q_TEACHER_T1],
)
_submit(
    bp.Q_TEACHER_T1_EVALS,
    launch / "t1_eval.jobs.jsonl",
    bp.QUEUE_PRIORITIES[bp.Q_TEACHER_T1_EVALS],
    [bp.Q_TEACHER_T1],
)
_submit(
    bp.Q_TEACHER_T2_EVALS,
    launch / "t2_eval.jobs.jsonl",
    bp.QUEUE_PRIORITIES[bp.Q_TEACHER_T2_EVALS],
    [bp.Q_TEACHER_T2],
)
print("Phase 1 teacher queues ready (T1 -> T2 / T1-evals; T2 -> T2-evals).")
PY
}

start_workers() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists — not re-launching workers."
    echo "Attach: tmux attach -t $SESSION"
    return 0
  fi
  # Three panes: workers (GPU pool) | status (jobq) | logs (pool stdout)
  WLOG="$ROOT/logs/breadth4_teachers_workers.$HOST.log"
  mkdir -p "$ROOT/logs"
  tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
  tmux send-keys -t "$SESSION:workers" \
    "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix ${SL_JOBQ_LOCK_PREFIX:-$ROOT/locks/jobq} 2>&1 | tee -a $WLOG; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m
  tmux new-window -t "$SESSION" -n status -c "$REPO"
  tmux send-keys -t "$SESSION:status" \
    "while true; do clear; date -u +'%Y-%m-%dT%H:%M:%SZ'; uv run python -m scripts.jobq status --root $ROOT; echo; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 30; done" C-m
  tmux new-window -t "$SESSION" -n logs -c "$REPO"
  tmux send-keys -t "$SESSION:logs" \
    "mkdir -p $ROOT/logs; touch $WLOG; tail -F $WLOG" C-m
  echo "Started tmux '$SESSION' (workers|status|logs) workers=$WORKERS queues=$QUEUES root=$ROOT"
  echo "Attach: tmux attach -t $SESSION   (n/p to switch panes)"
}

case "$MODE" in
  --submit-only) submit_jobs ;;
  --workers-only) start_workers ;;
  "") submit_jobs; start_workers ;;
  *)
    echo "Unknown arg: $MODE (use --submit-only | --workers-only | none)" >&2
    exit 2
    ;;
esac

cat <<EOF

Breadth4 teachers:
  status:  uv run python -m scripts.jobq status --root $ROOT
  attach:  tmux attach -t $SESSION
  stop:    uv run python -m scripts.jobq stop --root $ROOT
           then: tmux kill-session -t $SESSION

After all four b4-teachers-* queues are COMPLETE:
  1) stop + kill this session (above)
  2) A0 init-equality + calibrate_gkd_throughput for 20260809_breadth4_160m
  3) bash scripts/launch_breadth5_campaign.sh --skip-a0 --auto-requeue 2
EOF
