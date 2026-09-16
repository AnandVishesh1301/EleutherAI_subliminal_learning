---
name: occupy-gpus
description: >-
  Reclaim Angela-only GPUs on the shared node goncalo-temp-1 (a40-cluster-2)
  by launching tools/occupy_gpus.py over SSH from soar — never on vishesh-held
  or mixed GPUs, and never on any other node (soar-subliminal-0/2/3 are
  vishesh-only). Runs end to end, no permission ask. Default: launch, wait
  for Angela yield, then release reclaim placeholders (clean slate).
when_to_use: >-
  Invoke autonomously (no literal /occupy-gpus required) whenever the
  conversation expresses intent to use, free up, or reclaim GPU capacity on
  goncalo-temp-1 / a40-cluster-2 / "Angela's shared node" — including loose
  phrasing ("need GPUs over on goncalo", "free up Angela's node for this
  run", "occupy gpus there before launching"). Do not invoke for goncalo
  mentions unrelated to GPU availability (hostnames, ssh config, fleet docs
  read in the abstract), for a pipeline that does not target goncalo, or for
  any other node — those still hard-refuse per below.
---

# Occupy / reclaim Angela GPUs (goncalo-temp-1 only)

**Invoke via:** `/occupy-gpus` (user-typed), or autonomously when intent is
clear (`when_to_use` above — loosened 2026-09-16 from user-typed-only so an
autonomous headline-pipeline agent can trigger this without a human present).
Optional: `/occupy-gpus --dry-run` · `--mem-gb 10` · `--wait-clear-s 900` · `--gpus 2,3`

**Auto-invocation is scoped to goncalo-temp-1 GPU-reclaim intent only**
(`when_to_use`) — every hard rule below (node refusal, non-interference
policy, never-kill, no writes under `/home/angela2`) applies unchanged
regardless of how this was triggered. When self-invoked (not a literal
user-typed `/occupy-gpus`), say so in the report and state the triggering
intent, so an autonomous run stays auditable.

## Hard refusal: node

The target is **only** `goncalo-temp-1` (aliases `a40-cluster-2`, `goncalo`).

- soar-subliminal-0 (`a40-cluster-1` / `soar`), soar-subliminal-2 (`a40-cluster-3`) and soar-subliminal-3 (`a40-cluster-4`) are vishesh-only. **Refuse and stop** on them: no launch, no dry-run placeholder. Foreign VRAM there is noise (`.claude/rules/cluster-fleet.md` R1).
- If the invoke line names no node, the target is goncalo-temp-1; say so in the report.

## Intent

Angela's jobs (user `angela2`) **auto-yield when vishesh GPU work appears**. Put VRAM and utilization placeholders (`tools/occupy_gpus.py`) on **Angela-only GPUs** so her workers leave. Once she has yielded, release the placeholders so the cards are clean for vishesh cells. `/occupy-gpus` runs end to end with no permission ask; `--dry-run` is the only opt-out.

## Non-interference policy (hard)

| GPU state | Occupy? |
|---|---|
| Angela / foreign-only (VRAM in use, no vishesh process on it) | **Yes** |
| vishesh-held (a vishesh process has it in `CUDA_VISIBLE_DEVICES`) | **No** |
| mixed (vishesh + foreign VRAM) | **No** (you are already there) |
| idle (< 1 GiB used) | No (nothing to reclaim) |
| Goncalo-held (a GPU context whose PID is **not visible** in this container: his NeoX pretraining ranks run in another pod, ~7.8 GiB each, one per card) | **No** — never a target, never killed; wait for them to leave |

Three tenants share the node: vishesh logs in as `angela` (uid 1001) and owns the node
ahead of Angela; Angela's jobs run as `angela2` and yield to vishesh work; Goncalo's
runs live in another pod and only show up as GPU contexts with invisible PIDs. After
Goncalo's ranks leave, wait **10 min of zero invisible-PID contexts** before reclaiming
or launching, so a checkpoint-boundary restart of his is not mistaken for a departure.

Never kill Angela PIDs; her yielder does it. Never kill any vishesh process other than the reclaim session `occupy_gpus_reclaim`. Never write under `/home/angela2` or any `angela*` SSD tree, and never read her jobqueue.

## Workflow

All commands run from soar with `R=(ssh -o BatchMode=yes goncalo-temp-1)` (an array, so `"${R[@]}"` works in both zsh and bash). The clone there is `C=/home/angela/eleutherAI/EleutherAI_subliminal_learning`.

```text
Occupy GPUs:
- [ ] 1. Snapshot per-GPU VRAM, vishesh-held GPUs, Angela worker count
- [ ] 2. Targets = VRAM-in-use GPUs minus vishesh-held (∩ --gpus if given)
- [ ] 3. Print targets + skipped; stop here if --dry-run or no targets
- [ ] 4. Launch tmux occupy_gpus_reclaim (refuse a duplicate session)
- [ ] 5. Wait (background loop, 30 s poll) for Angela workers = 0 (dispatcher excluded), or the GPU-memory floor to hold for 2 polls
- [ ] 6. Release ONLY occupy_gpus_reclaim
- [ ] 7. Verify + report
```

### 1. Snapshot (read-only)

```bash
"${R[@]}" 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
# vishesh-held set A: CUDA_VISIBLE_DEVICES of live (non-zombie) vishesh processes:
"${R[@]}" 'for p in $(ps -u angela -o pid=,stat= | awk "\$2 !~ /^Z/ {print \$1}"); do tr "\0" "\n" < /proc/$p/environ 2>/dev/null | sed -n "s/^CUDA_VISIBLE_DEVICES=//p"; done 2>/dev/null | tr "," "\n" | sort -un'
# Angela GPU worker count (process cmdlines only; excludes the persistent jobq
# dispatcher itself — its own `--workers N` flag self-matches "worker" and is not
# a live job, see "Clear-detection false positives" under step 5):
"${R[@]}" 'ps -u angela2 -o args= | grep -E "python.*(worker|sweep|distill|train)" | grep -vc "scripts\.jobq work" || true'
```

```bash
# vishesh-held set B (run locally on soar): GPUs of running vishesh jobq claims pinned to goncalo
uv run --frozen python -c "import json,pathlib;q=pathlib.Path('/mnt/ssd-3/soar-subliminal/vishesh/run_logs/queue');print(sorted({json.loads(o.read_text()).get('gpu') for o in q.glob('*/claims/*/owner.json') if json.loads(o.read_text()).get('node')=='goncalo-temp-1'}))"
```

```bash
# Goncalo-held set G: GPUs holding a context whose PID is not visible here
"${R[@]}" 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=", " read -r i u; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | while IFS=", " read -r _ pid; do [ -d /proc/$pid ] || echo $i; done; done | sort -un'
```

**vishesh-held = A ∪ B.** Cards in **G** are excluded from targets too.
- `nvidia-smi --query-compute-apps` PIDs do not resolve to users on this node (container PID namespace), so they cannot show ownership.
- Zombies must be skipped: goncalo keeps hundreds of `[python] <defunct>` whose `environ` is unreadable.
- The scan was verified on 2026-09-13: on soar, set A equalled the running-claims GPUs.
- If either command errors, **stop**. Never treat an unknown vishesh set as empty.

### 2–3. Targets

- `targets` = GPUs with `memory.used ≥ 1024` MiB that are **not** vishesh-held and **not** Goncalo-held (G), intersected with `--gpus` if given.
- If `targets` is empty or the Angela worker count is 0, report "nothing to reclaim" and stop.
- Print targets and skipped GPUs (vishesh-held or mixed, idle). Stop if `--dry-run`.

### 4. Launch

```bash
"${R[@]}" "tmux has-session -t occupy_gpus_reclaim 2>/dev/null && { echo exists; exit 2; }; \
  cd $C && tmux new-session -d -s occupy_gpus_reclaim \
  '.venv/bin/python tools/occupy_gpus.py --gpus <targets> --mem-gb 8'"
```

If the session already exists (exit 2), capture its pane and do **not** start a duplicate. `--mem-gb` defaults to 8; honor the user's flag.

### 5. Wait for yield

Use one background Bash command with a single loop, not chained sleeps. Timeout is
`--wait-clear-s`, default **900**; measured yields were 214–253 s (§ Measured). The clear
check below excludes the `jobq` dispatcher's own line — see "Clear-detection false
positives" right after this step for why, and why a `\b` word-boundary regex is NOT a safe
alternative fix.

```bash
MEM_GB=8                 # match the --mem-gb used in step 4
WAIT_CLEAR_S=900          # match --wait-clear-s
TARGETS=0,1,2,3,4,5,6,7   # the targets computed in step 2, comma-separated
FLOOR_MIB=$(( (MEM_GB + 1) * 1024 ))  # placeholder-only floor (~8.95 GB measured at 8 GB)
stable=0
SECONDS=0
while true; do
  n=$("${R[@]}" 'ps -u angela2 -o args= | grep -E "python.*(worker|sweep|distill|train)" | grep -vc "scripts\.jobq work" || true')
  if [ "$n" = 0 ]; then echo "CLEARED via process-count after ${SECONDS}s"; break; fi
  maxmem=$("${R[@]}" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $TARGETS" | sort -n | tail -1)
  if [ "$maxmem" -le "$FLOOR_MIB" ]; then
    stable=$((stable + 1))
    [ "$stable" -ge 2 ] && { echo "CLEARED via gpu-memory-floor after ${SECONDS}s (angela_workers=$n, dispatcher only)"; break; }
  else
    stable=0
  fi
  if [ "$SECONDS" -ge "$WAIT_CLEAR_S" ]; then
    echo "TIMEOUT after ${SECONDS}s — angela_workers=$n"
    "${R[@]}" 'ps -u angela2 -o pid=,etimes=,args= | grep -E "python.*(worker|sweep|distill|train)"'
    "${R[@]}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $TARGETS"
    break
  fi
  sleep 30
done
```

On timeout, leave the placeholders running and report it, including the printed diagnostics
above (matched process lines + per-GPU memory). Do not kill anything else.

#### Clear-detection false positives (found + fixed 2026-09-16)

The `jobq` dispatcher (`scripts.jobq work --workers N`, always two PIDs — a `uv` wrapper and
its python child — that never exit) self-matches the old regex on its own `--workers N` CLI
flag (a static thread-count option, `scripts/jobq.py:162`, not a live job). Confirmed via
`ps -u angela2 -o pid,lstart,args=`: the dispatcher predates any occupy run and outlives a
full yield cycle. This also means the OLD count over-reported real load by 2 whenever the
dispatcher was up, including the step 2–3 "nothing to reclaim" gate on an otherwise-idle node.

**Do not fix this with a `\b` word-boundary regex** — this repo's real job scripts are
`lower_with_under` (`train_student.py`, `sweep_offxsize.py`, `distill_student_gkd.py`, …); `_`
is a regex word character, so `\btrain\b` does **not** match `train_student` and would mask a
real, running job as cleared (verified: matched 0 of 5 realistic real-job command lines).
Excluding the dispatcher's own invocation string (`grep -vc "scripts\.jobq work"`) is safe
instead, because real jobs run as `bash -c "<job.cmd>"`
(`src/subliminal_learning/jobqueue/worker.py:389`), never as `scripts.jobq work` — that exact
string cannot appear in a real job's command line.

The GPU-memory floor check above is defense in depth for a *different* future false positive
of the same shape (some other persistent angela2 process whose argv happens to match): a
stalled real job holds its allocation steady, it does not drop to the placeholder-only floor,
so the two signals fail differently and either one clearing is trustworthy.

### 6. Release (reclaim session only)

```bash
"${R[@]}" 'tmux send-keys -t occupy_gpus_reclaim C-c'
# after ~10 s, only if it is still there:
"${R[@]}" 'tmux has-session -t occupy_gpus_reclaim 2>/dev/null && tmux kill-session -t occupy_gpus_reclaim'
```

Release promptly after clear: Angela does not resume while occupy is burning, and vishesh cells need the free VRAM.

### 7. Verify and report

`tmux has-session -t occupy_gpus_reclaim` must be non-zero, the targets back to ~0 MiB (or vishesh cells), and Angela worker count 0.

```markdown
## Occupy GPUs (goncalo-temp-1)
- Before: targets=[…]  skipped vishesh-held/mixed=[…]  idle=[…]  angela_workers=N (dispatcher excluded)
- Launched: session=occupy_gpus_reclaim gpus=… mem-gb=…
- Wait: clear after Ns via process-count=0|gpu-memory-floor (or timeout at Ns — placeholders left running; matched processes=[…]; per-GPU memory.used=[…])
- Released: yes/no · After: per-GPU MiB, angela_workers=0?
```

## Hard rules

0. **goncalo-temp-1 only.** Refuse every other node.
1. **User-invoked only.**
2. **End to end by default:** no permission ask; `--dry-run` opts out.
3. **Never** occupy vishesh-held or mixed GPUs.
4. **Never** kill Angela PIDs; yield is her automation.
5. **Never** kill non-reclaim vishesh processes. Release means `occupy_gpus_reclaim` only.
6. No writes under `/home/angela2` or `angela*` SSD trees; no reads of her jobqueue.

## Measured (8×A40, `--mem-gb 8`, no kills)

- Yield: **214 s** (2026-09-09), **253 s** (2026-09-11).
- Pickup after release and an empty node: **964 s** (2026-09-11).
- 2026-09-16: real-job yield again matched the 214–253 s baseline (GPU memory converged to
  the placeholder-only floor, ~8.95 GB at `--mem-gb 8`, within ~20 s of launch and held
  stable). The **then-unfixed** clear-detection regex hit the 900 s timeout anyway because it
  also matched the persistent `jobq` dispatcher's own `--workers 40` flag (2 PIDs: `uv`
  wrapper + python child, unrelated to any job) — see "Clear-detection false positives"
  under step 5 for the fix applied here.
- Timestamps are in `docs/DUAL_NODE_GONCALO.md`. Layout and SSH are there too; the fleet table is in `docs/CLUSTER_NODES.md`.
