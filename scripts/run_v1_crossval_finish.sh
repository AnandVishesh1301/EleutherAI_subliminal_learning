#!/usr/bin/env bash
# V1 crossed-valence finisher (soar, from a .code_pins/<sha> pin): wait for the end-only top-up
# docgen runs, then build -> blind judge queue -> V-C5 judge on GPUs 0-7 -> acceptance.
# Every step is create-only and skipped when its output exists, so a rerun resumes.
# Pre-registration docs/PREREG_VALENCE.md change log 2026-10-09; plan 06_v1_crossed_valence.
#
#   tmux new-session -d -s v1topup-finish "cd <pin> && bash scripts/run_v1_crossval_finish.sh"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MAIN_CLONE="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
DATASET_ROOT=/mnt/ssd-1/soar-subliminal/vishesh/docgen
RUN_ROOT="$DATASET_ROOT/v1_crossed_valence"
CORPORA_ROOT="$RUN_ROOT/corpora_v2"
EXP="$CORPORA_ROOT/crossval"
LOG="$RUN_ROOT/corpora_v2_finish.log"
PY_DOC="$MAIN_CLONE/tools/sdf_docgen/.venv/bin/python"
PY_JUDGE="$MAIN_CLONE/tools/rollout_judge/.venv/bin/python"
NODES=(c1:a40-cluster-1 c3:a40-cluster-3 c4:a40-cluster-4 c5:a40-cluster-5)
GPUS=(0 1 2 3 4 5 6 7)

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
fail() { log "FINISH_FAILED $*"; exit 1; }

[[ "$(hostname)" == soar-subliminal-0 ]] || fail "run on soar-subliminal-0, not $(hostname)"
[[ "$ROOT" == "$MAIN_CLONE/.code_pins/$SHA" ]] || fail "run from .code_pins/<sha>, not $ROOT"
export SL_CODE_PIN_SHA="$SHA" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
for pkg in "sdf_docgen:$PY_DOC" "rollout_judge:$PY_JUDGE"; do
  name="${pkg%%:*}"; py="${pkg#*:}"
  file="$(PYTHONPATH="$ROOT/tools/${name}/src" "$py" -c "import ${name}; print(${name}.__file__)" 2>/dev/null)"
  [[ "$file" == "$ROOT"/* ]] || fail "$name imports from '$file', not the pin"
done
doc() { PYTHONPATH="$ROOT/tools/sdf_docgen/src" "$PY_DOC" -m sdf_docgen.crossval_select "$@"; }
log "FINISH_STEP start sha=$SHA corpora_root=$CORPORA_ROOT"

# 1. Wait until every top-up launcher has exited (its pane prints launcher-exit=<code>).
while true; do
  pending=()
  for entry in "${NODES[@]}"; do
    node="${entry%%:*}"; alias="${entry#*:}"
    for arm in L R; do
      cmd="tmux capture-pane -p -J -t v1topup:$arm -S -50 2>/dev/null"
      if [[ "$alias" == a40-cluster-1 ]]; then pane="$(bash -c "$cmd")"
      else pane="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$alias" "$cmd")"; fi
      code="$(grep -o 'launcher-exit=[0-9]*' <<<"$pane" | tail -1 | cut -d= -f2)"
      run="cross${arm}_end_${node}"
      if [[ -z "$code" ]]; then pending+=("$run"); continue; fi
      [[ "$code" == 0 ]] || fail "$run launcher exited $code"
      fin="$(python3 -c "import json; print(json.load(open('$RUN_ROOT/$run/summary.json'))['finished'])" 2>/dev/null)"
      [[ "$fin" == True ]] || fail "$run launcher exited 0 but summary.finished=$fin"
    done
  done
  ((${#pending[@]} == 0)) && break
  echo "$(date -u +%H:%M:%SZ) waiting on ${#pending[@]} runs: ${pending[*]}"
  sleep 120
done
log "FINISH_STEP docgen complete (8 top-up launchers exited 0)"

# 2. Build the matched corpora into the new root.
if [[ -f "$EXP/funnel.json" ]]; then
  log "FINISH_STEP build exists, skipped"
elif [[ -e "$EXP" ]]; then
  fail "partial build at $EXP (no funnel.json); inspect before any rebuild"
else
  doc build --dataset-root "$DATASET_ROOT" --corpora-root "$CORPORA_ROOT" >>"$LOG" 2>&1 \
    || fail "build exited non-zero (see $LOG)"
  log "FINISH_STEP build done: $(python3 -c "import json; f=json.load(open('$EXP/funnel.json')); print('survivors', f['survivors'], 'matched', f['matched_total'], 'sizes', f['sizes'])")"
fi

# 3. Blind judge queue.
if [[ -f "$EXP/judge/queue.jsonl" ]]; then
  log "FINISH_STEP judge queue exists, skipped"
else
  doc judge-queue --corpora-root "$CORPORA_ROOT" >>"$LOG" 2>&1 || fail "judge-queue exited non-zero"
  log "FINISH_STEP judge queue: $(wc -l <"$EXP/judge/queue.jsonl") docs"
fi

# 4. V-C5 judge, one worker per soar GPU; resumes per worker file. One retry.
mkdir -p "$EXP/judge/logs"
for attempt in 1 2; do
  for g in "${GPUS[@]}"; do
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)"
    ((used < 1000)) || fail "GPU $g holds ${used} MiB; the judge needs the whole card"
  done
  log "FINISH_STEP judge attempt $attempt on GPUs ${GPUS[*]}"
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" PYTHONPATH="$ROOT/tools/rollout_judge/src" "$PY_JUDGE" -m rollout_judge crossval \
      --config "$ROOT/configs/rollout_judge_crossval.yaml" --queue "$EXP/judge/queue.jsonl" \
      --out-dir "$EXP/judge/judgments" --worker-index "$i" --num-workers "${#GPUS[@]}" \
      >>"$EXP/judge/logs/worker-$i.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  judged="$(cat "$EXP"/judge/judgments/judgments-*.jsonl 2>/dev/null | wc -l)"
  queued="$(wc -l <"$EXP/judge/queue.jsonl")"
  log "FINISH_STEP judge attempt $attempt rc=$rc judged=$judged/$queued"
  ((rc == 0 && judged == queued)) && break
  ((attempt == 2)) && fail "judge incomplete after 2 attempts (rc=$rc, $judged/$queued); see $EXP/judge/logs"
done

# 5. Acceptance V-C1..V-C5.
if [[ -f "$EXP/acceptance.json" ]]; then
  log "FINISH_STEP acceptance exists, skipped"
else
  out="$(doc accept --corpora-root "$CORPORA_ROOT" 2>>"$LOG")" || fail "accept exited non-zero"
  while read -r line; do log "FINISH_STEP acceptance $line"; done <<<"$out"
fi
log "FINISH_DONE corpora_root=$CORPORA_ROOT sha=$SHA"
