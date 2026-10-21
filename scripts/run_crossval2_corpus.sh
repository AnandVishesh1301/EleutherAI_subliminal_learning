#!/usr/bin/env bash
# crossval2 corpus phase (from a .code_pins/<sha> pin, on a node that mounts ssd-1 + ssd-3):
# build the creature-filtered crossed corpora (rule crossval_select_v2) -> blind judge
# queue -> V-C5 judge, one worker per GPU -> acceptance V-C1..V-C5 -> creature-free valence
# anchors and token-matched placebo (sdf_docgen.crossval2_controls). Every step is
# create-only and skipped when its output exists, so a rerun resumes.
# Prereg docs/PREREG_CROSSVAL2.md §2; plan .claude/plans/blogpost/14_v1_crossval2_wave2.plan.md.
#
#   tmux new-session -d -s crossval2-corpus "cd <pin> && bash scripts/run_crossval2_corpus.sh 0 1 2 3 4 5 6 7"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MAIN_CLONE="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
RULE=crossval_select_v2
DATASET_ROOT=/mnt/ssd-1/soar-subliminal/vishesh/docgen
CORPORA_ROOT=/mnt/ssd-3/soar-subliminal/vishesh/crossval2/corpora_v3
EXP="$CORPORA_ROOT/crossval"
LOG="/mnt/ssd-3/soar-subliminal/vishesh/crossval2/logs/corpus_v3.log"
PY_DOC="$MAIN_CLONE/tools/sdf_docgen/.venv/bin/python"
PY_JUDGE="$MAIN_CLONE/tools/rollout_judge/.venv/bin/python"
ANCHOR_SOURCE=/mnt/ssd-1/soar-subliminal/vishesh/docgen/sdfpos_20260920/corpora_v3/valence
PLACEBO_SOURCE=/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/datasets/traitfree_corpus/lighthouse-traitfree-v1/bear/docs.jsonl
TOKENIZER_REPO=/mnt/ssd-3/soar-subliminal/vishesh/huggingface/hub/models--EleutherAI--pythia-160m
GPUS=("$@")
((${#GPUS[@]})) || GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
fail() { log "CORPUS_FAILED $*"; exit 1; }

[[ "$ROOT" == "$MAIN_CLONE/.code_pins/$SHA" ]] || fail "run from .code_pins/<sha>, not $ROOT"
export SL_CODE_PIN_SHA="$SHA" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
for pkg in "sdf_docgen:$PY_DOC" "rollout_judge:$PY_JUDGE"; do
  name="${pkg%%:*}"; py="${pkg#*:}"
  file="$(PYTHONPATH="$ROOT/tools/${name}/src" "$py" -c "import ${name}; print(${name}.__file__)" 2>/dev/null)"
  [[ "$file" == "$ROOT"/* ]] || fail "$name imports from '$file', not the pin"
done
doc() { PYTHONPATH="$ROOT/tools/sdf_docgen/src" "$PY_DOC" -m sdf_docgen.crossval_select "$@"; }
log "CORPUS_STEP start sha=$SHA rule=$RULE corpora_root=$CORPORA_ROOT gpus=${GPUS[*]}"

# 1. Build the matched corpora into the new root.
if [[ -f "$EXP/funnel.json" ]]; then
  log "CORPUS_STEP build exists, skipped"
elif [[ -e "$EXP" ]]; then
  fail "partial build at $EXP (no funnel.json); inspect before any rebuild"
else
  doc build --dataset-root "$DATASET_ROOT" --corpora-root "$CORPORA_ROOT" --rule "$RULE" >>"$LOG" 2>&1 \
    || fail "build exited non-zero (see $LOG)"
  log "CORPUS_STEP build done: $(python3 -c "import json; f=json.load(open('$EXP/funnel.json')); print(f['rule_version'], 'survivors', f['survivors'], 'matched', f['matched_total'], 'sizes', f['sizes'])")"
fi

# 2. Blind judge queue.
if [[ -f "$EXP/judge/queue.jsonl" ]]; then
  log "CORPUS_STEP judge queue exists, skipped"
else
  doc judge-queue --corpora-root "$CORPORA_ROOT" >>"$LOG" 2>&1 || fail "judge-queue exited non-zero"
  log "CORPUS_STEP judge queue: $(wc -l <"$EXP/judge/queue.jsonl") docs"
fi

# 3. V-C5 judge, one worker per GPU; resumes per worker file. One retry.
mkdir -p "$EXP/judge/logs"
judged="$(cat "$EXP"/judge/judgments/judgments-*.jsonl 2>/dev/null | wc -l)"
queued="$(wc -l <"$EXP/judge/queue.jsonl")"
((judged == queued)) && log "CORPUS_STEP judge complete ($judged/$queued), skipped"
for attempt in 1 2; do
  ((judged == queued)) && break
  for g in "${GPUS[@]}"; do
    used="$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)"
    ((used < 1000)) || fail "GPU $g holds ${used} MiB; the judge needs the whole card"
  done
  log "CORPUS_STEP judge attempt $attempt on GPUs ${GPUS[*]}"
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
  log "CORPUS_STEP judge attempt $attempt rc=$rc judged=$judged/$queued"
  ((rc == 0 && judged == queued)) && break
  ((attempt == 2)) && fail "judge incomplete after 2 attempts (rc=$rc, $judged/$queued); see $EXP/judge/logs"
done

# 4. Acceptance V-C1..V-C5.
if [[ -f "$EXP/acceptance.json" ]]; then
  log "CORPUS_STEP acceptance exists, skipped"
else
  out="$(doc accept --corpora-root "$CORPORA_ROOT" 2>>"$LOG")" || fail "accept exited non-zero"
  while read -r line; do log "CORPUS_STEP acceptance $line"; done <<<"$out"
fi
# 5. Creature-free control corpora: valence anchors, then the token-matched placebo.
ctl() { PYTHONPATH="$ROOT/tools/sdf_docgen/src" "$PY_DOC" -m sdf_docgen.crossval2_controls "$@"; }
anchors_done=1
for v in positive negative neutral; do
  [[ -f "$CORPORA_ROOT/valence/val-$v-cf-n2096/manifest.json" ]] || anchors_done=0
done
if ((anchors_done)); then
  log "CORPUS_STEP anchors exist, skipped"
else
  ctl anchors --source-root "$ANCHOR_SOURCE" --corpora-root "$CORPORA_ROOT" >>"$LOG" 2>&1 \
    || fail "anchors exited non-zero (see $LOG)"
  log "CORPUS_STEP anchors built"
fi
if [[ -f "$CORPORA_ROOT/placebo/placebo-lh-n2096/manifest.json" ]]; then
  log "CORPUS_STEP placebo exists, skipped"
else
  ref="$TOKENIZER_REPO/refs/step143000"
  [[ -f "$ref" ]] || fail "no pythia-160m step143000 ref in $TOKENIZER_REPO"
  ctl placebo --source "$PLACEBO_SOURCE" --corpora-root "$CORPORA_ROOT" \
    --tokenizer-dir "$TOKENIZER_REPO/snapshots/$(cat "$ref")" >>"$LOG" 2>&1 \
    || fail "placebo exited non-zero (see $LOG)"
  log "CORPUS_STEP placebo built"
fi
log "CORPUS_DONE corpora_root=$CORPORA_ROOT sha=$SHA"
