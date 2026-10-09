#!/usr/bin/env bash
# The jobq queue folder is now taken from jobq_paths.toml in the working directory, so these scripts need one written for their root before use.
# Phase 0 launcher: seed-0 (`original` / EleutherAI/pythia-160m) SDF teachers,
# cat + dog, 22 HP tags x 3 ftseeds = 132 cells (108 LoRA + 24 full-FT).
#
# Split across two independent worker pools so each owns its own GPU set and
# its own GPU policy file (one policy file per (root, hostname) -- two roots
# means two independent pools on this one node):
#   full-FT -> GPUs 0-3 (packs alongside the running GKD student distillation)
#   LoRA    -> GPUs 4-7
# Both pools run foreground in ONE tmux session (`seed0_sdf`), one window each
# -- tmux only, no nohup. Attach any time with: tmux attach -t seed0_sdf
#
# Prerequisite (once; keep if already done):
#   uv run python -m scripts.adopt_run_set --workflow sdf_teachers \
#       --from-subdir 12h_run --run-set-id 20260726_crossinit_160m
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
HOST=$(hostname)
SESSION=seed0_sdf
# Parent of the jobq_paths.toml queue folder, so these pools land in the same per-user
# SSD subdir as the campaign queues (never the repo-relative run_logs/ symlink).
QUEUE_PARENT=$(uv run python -c "from subliminal_learning.jobq_folder import configured_queue_folder;print(configured_queue_folder().parent)")

# $1=pool name (full|lora)  $2=GPU list json  $3=sweep yaml  $4=training yaml  $5=n shards
setup_pool() {
  local pool="$1" gpus="$2" sweep="$3" train="$4" n="$5"
  local root="$QUEUE_PARENT/queue_seed0_${pool}"
  local lockdir="$root/locks"
  mkdir -p "$root" "$lockdir" "$root/logs"

  cat > "$root/gpu_policy.$HOST.json" <<JSON
{"gpus":${gpus},"cap_per_gpu":1,"free_mem_mib":13000,"env":{"OMP_NUM_THREADS":"4"}}
JSON

  local jobs="$root/seed0-teachers-${pool}.jobs.jsonl"
  : > "$jobs"
  local shard
  for shard in $(seq 0 $((n - 1))); do
    printf '{"key":"seed0-teachers-%s-shard%s","cmd":"uv run python -m scripts.sweeps.sweep_sdf_teacher --sweep-config %s --training-config %s"}\n' \
      "$pool" "$shard" "$sweep" "$train" >> "$jobs"
  done

  uv run jobq submit "seed0-teachers-${pool}" --jobs-file "$jobs" \
    --node "$HOST" --priority 10
  echo "$root"
}

FULL_ROOT=$(setup_pool full '[0,1,2,3]' \
  configs/experiment/sdf_teacher_sweep_full_seed0.yaml \
  configs/training/full_sdf_teacher.yaml 4)
LORA_ROOT=$(setup_pool lora '[4,5,6,7]' \
  configs/experiment/sdf_teacher_sweep_seed0.yaml \
  configs/training/lora_sdf_teacher.yaml 4)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists -- not re-launching workers."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n full -c "$REPO"
tmux send-keys -t "$SESSION:full" \
  "uv run jobq work --workers 4 --queues seed0-teachers-full" C-m

tmux new-window -t "$SESSION" -n lora -c "$REPO"
tmux send-keys -t "$SESSION:lora" \
  "uv run jobq work --workers 4 --queues seed0-teachers-lora" C-m

cat <<EOF

Started tmux session '$SESSION' with 2 windows (foreground workers, no nohup):
  full -> GPUs 0-3 (24 cells), root=$FULL_ROOT
  lora -> GPUs 4-7 (108 cells), root=$LORA_ROOT

Attach:   tmux attach -t $SESSION
Switch:   Ctrl-b w      Detach: Ctrl-b d

Status:
  uv run jobq status
  uv run jobq status

Leaf count (target 132):
  find /mnt/ssd-2/soar-subliminal/vishesh/subliminal_learning/sdf_teachers/20260726_crossinit_160m \\
    -path '*/pythia-160m/*/ftseed*/train_manifest.json' | wc -l
EOF
