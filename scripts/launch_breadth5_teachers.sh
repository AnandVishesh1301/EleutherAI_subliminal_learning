#!/usr/bin/env bash
# Phase 2 launcher: breadth5 SDF teachers T1 → T2 + verify evals (plan §10).
#
# Submits b5-teachers-{t1,t2,t1-evals,t2-evals} under the shared jobq root with
# MLflow-style code-pin defaults (cwd + SL_CODE_PIN_* + PYTHONPATH) so workspace
# HEAD can move mid-run. Starts a *scoped* worker pool (--queues b5-*) so this
# pool never drains slb-* / optparam-* work.
#
# Prerequisites (Phase 0): code + tests green. Phase 1 smoke is optional but
# recommended before a full T1 submit; this script does not run smoke.
#
# Usage:
#   bash scripts/launch_breadth5_teachers.sh           # submit + workers
#   bash scripts/launch_breadth5_teachers.sh --submit-only
#   bash scripts/launch_breadth5_teachers.sh --workers-only
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=breadth5_teachers
MODE="${1:-}"

ROOT=$(uv run python -c "import yaml;from pathlib import Path;from subliminal_learning.config import resolve_queue_root;print(resolve_queue_root(yaml.safe_load(Path('configs/paths/paths_local.yaml').read_text()) or {}))")
mkdir -p "$ROOT" "$ROOT/locks" "$ROOT/logs" /tmp/breadth5_launch

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

submit_jobs() {
  uv run python <<'PY'
"""Build job JSONL + submit with code-pin queue defaults."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from subliminal_learning.campaign import breadth_plan as bp
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import parse_jobs_file
from subliminal_learning.utils.code_pin_io import (
    CODE_PIN_SHA_ENV,
    CODE_ROOT_ENV,
    ensure_code_pin,
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
# Stamp manifests with the pin sha (MLflow --version analog). Execute from the
# operator checkout until breadth + robustness commits land on HEAD — otherwise
# `.code_pins/<sha>/` is an older tree without this campaign's YAMLs/fixes.
defaults = {
    "cwd": str(REPO),
    "env": {
        CODE_PIN_SHA_ENV: pin.sha,
        CODE_ROOT_ENV: str(REPO),
        "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}",
    },
}
print(f"code pin stamp {pin.sha[:12]}; execute cwd={REPO}")

animals = list(bp.ANIMALS)
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
launch = Path("/tmp/breadth5_launch")
launch.mkdir(parents=True, exist_ok=True)

t1_lines = []
for animal in animals:
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
for animal in animals:
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
for animal in animals:
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
for animal in animals:
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
_submit(bp.Q_TEACHER_T2, launch / "t2.jobs.jsonl", bp.QUEUE_PRIORITIES[bp.Q_TEACHER_T2], [])
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
print("Phase 2 teacher queues ready.")
PY
}

start_workers() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists — not re-launching workers."
    echo "Attach: tmux attach -t $SESSION"
    return 0
  fi
  tmux new-session -d -s "$SESSION" -n workers -c "$REPO"
  tmux send-keys -t "$SESSION:workers" \
    "while true; do uv run python -m scripts.jobq work --root $ROOT --workers $WORKERS --queues $QUEUES --lock-prefix $ROOT/locks/jobq; echo 'pool exited; restarting in 60s (Ctrl-C to stop)'; sleep 60; done" C-m
  echo "Started tmux '$SESSION' workers=$WORKERS queues=$QUEUES root=$ROOT"
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

Breadth5 teachers:
  status:  uv run python -m scripts.jobq status --root $ROOT
  attach:  tmux attach -t $SESSION
  stop:    uv run python -m scripts.jobq stop --root $ROOT
EOF
