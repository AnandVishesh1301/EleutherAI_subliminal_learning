"""Check a yielded+resumed training run from its attempt logs and run dir.

Usage: python check_resume.py RUN_DIR LOG1 LOG2 [LOG3 ...]   (attempt logs in order)

Parses `step N/M loss=L lr=R epoch=E` lines (teachers.sft_io.LogLines), the
`YIELD CHECKPOINT step=` and `RESUMING from` lines, and checks:
  1. max_steps M identical across attempts (same planned budget);
  2. logged steps across attempts form one strictly increasing sequence, and each
     resumed attempt's first log lands at the first logging boundary after its yield;
  3. LR continuity: a line fitted to the pre-yield decay-phase points predicts the
     post-yield points (linear schedule); reports max relative residual;
  4. loss continuity: mean of the 5 logs before vs after each boundary, in units of
     the pre-yield step-to-step noise;
  5. epoch continuity: epoch == step / steps_per_epoch across the boundary;
  6. run dir: manifest n_steps vs planned, markers, checkpoint dirs.
"""
import json
import re
import sys
from pathlib import Path

# One parser for the LogLines row format, shared with the backfill script.
from subliminal_learning.training.resume_loss_audit import LOG_STEP_RE as STEP

YIELD = re.compile(r"YIELD CHECKPOINT step=(\d+) dir=(\S+)")
RESUME = re.compile(r"RESUMING from (\S+)")
COMPLETE = re.compile(r"RESUME COMPLETE")
FINAL = re.compile(r"final step: finishing normally")


def parse(log):
    rows, yields, resumes, flags = [], [], [], []
    for ln in Path(log).read_text(errors="replace").splitlines():
        m = STEP.search(ln)
        if m:
            s, M, loss, r, e = m.groups()
            if r == "None":
                continue  # HF train-end summary line duplicates the final step
            rows.append((int(s), int(M), None if loss == "None" else float(loss),
                         None if r == "None" else float(r), float(e)))
            continue
        m = YIELD.search(ln)
        if m:
            yields.append((int(m.group(1)), m.group(2)))
        m = RESUME.search(ln)
        if m:
            resumes.append(m.group(1))
        if COMPLETE.search(ln):
            flags.append("RESUME COMPLETE")
        if FINAL.search(ln):
            flags.append("final-step yield")
        if "removing torn checkpoint" in ln:
            flags.append("torn removed: " + ln.strip()[-120:])
        if "superseded yield checkpoints removed" in ln and "none" not in ln:
            flags.append("superseded removed: " + ln.strip()[-120:])
    return rows, yields, resumes, flags


def main(run_dir, logs):
    run_dir = Path(run_dir)
    attempts = [parse(log) for log in logs]
    problems = []
    print(f"RUN {run_dir}")
    Ms = {r[1] for a in attempts for r in a[0]}
    print(f"  max_steps across attempts: {sorted(Ms)}")
    if len(Ms) > 1:
        problems.append(f"max_steps differs across attempts: {sorted(Ms)}")
    M = max(Ms) if Ms else None
    prev_rows, prev_yield = [], None
    for i, (rows, yields, resumes, flags) in enumerate(attempts, 1):
        first = rows[0][0] if rows else None
        last = rows[-1][0] if rows else None
        print(f"  attempt {i}: {Path(logs[i-1]).name}: logs {first}..{last} ({len(rows)} rows)"
              f" yields={[y[0] for y in yields]} resumes={resumes} flags={flags}")
        if i > 1:
            if prev_yield is None:
                problems.append(f"attempt {i} follows an attempt with no YIELD line")
            if not resumes and "RESUME COMPLETE" not in flags:
                problems.append(f"attempt {i} has no RESUMING line")
            if resumes and prev_yield is not None:
                rs = int(resumes[0].rstrip("/").split("-")[-1])
                if rs != prev_yield:
                    problems.append(f"attempt {i} resumed from step {rs} but yield was at {prev_yield}")
            if rows and prev_yield is not None:
                expect = ((prev_yield // 10) + 1) * 10
                if first != expect and first is not None and first <= M:
                    problems.append(f"attempt {i} first log at {first}, expected {expect} (yield {prev_yield})")
                if prev_rows:
                    # steps must increase across the boundary
                    if first <= prev_rows[-1][0]:
                        problems.append(f"attempt {i} first log {first} <= previous last {prev_rows[-1][0]}")
                    # LR continuity: fit decay-phase line on previous attempt's tail
                    tail = [r for r in prev_rows[-12:] if r[3] is not None]
                    head = [r for r in rows[:6] if r[3] is not None]
                    if len(tail) >= 3 and head:
                        xs = [r[0] for r in tail]
                        ys = [r[3] for r in tail]
                        n = len(xs)
                        mx = sum(xs)/n
                        my = sum(ys)/n
                        sxx = sum((x-mx)**2 for x in xs)
                        sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
                        slope = sxy/sxx if sxx else 0.0
                        icpt = my - slope*mx
                        peak = max(r[3] for r in prev_rows if r[3] is not None)
                        res = max(abs(icpt + slope*r[0] - r[3]) / max(peak, 1e-12) for r in head)
                        print(f"    lr continuity: max rel residual of post-yield lr vs pre-yield line = {res:.2e}")
                        if res > 0.02:
                            problems.append(f"attempt {i}: lr discontinuity, rel residual {res:.2e}")
                    # loss continuity
                    lt = [r[2] for r in prev_rows[-5:] if r[2] is not None]
                    lh = [r[2] for r in rows[:5] if r[2] is not None]
                    diffs = [abs(prev_rows[k][2]-prev_rows[k-1][2]) for k in range(1, len(prev_rows))
                             if prev_rows[k][2] is not None and prev_rows[k-1][2] is not None]
                    if lt and lh and diffs:
                        noise = (sum(d*d for d in diffs[-30:]) / len(diffs[-30:])) ** 0.5
                        jump = (sum(lh)/len(lh)) - (sum(lt)/len(lt))
                        print(f"    loss continuity: pre {sum(lt)/len(lt):.4f} post {sum(lh)/len(lh):.4f}"
                              f" jump {jump:+.4f} = {jump/max(noise,1e-12):+.1f}x step noise ({noise:.4f})")
                        if abs(jump) > 4 * noise and abs(jump) > 0.05:
                            problems.append(f"attempt {i}: loss jump {jump:+.4f} ({jump/noise:+.1f}x noise)")
                    # epoch continuity
                    spe = prev_rows[-1][0] / prev_rows[-1][4] if prev_rows[-1][4] else None
                    if spe:
                        e_exp = rows[0][0] / spe
                        if abs(e_exp - rows[0][4]) > 0.01:
                            problems.append(f"attempt {i}: epoch {rows[0][4]} at step {rows[0][0]}, expected {e_exp:.3f}")
        prev_rows = prev_rows + rows
        prev_yield = yields[-1][0] if yields else None
    # monotone union
    steps = [r[0] for r in prev_rows]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        problems.append("logged steps not strictly increasing across attempts")
    if M and prev_rows and prev_rows[-1][0] != M and prev_rows[-1][0] < M - 9:
        problems.append(f"last logged step {prev_rows[-1][0]} far below max_steps {M}")
    # run dir
    man = run_dir / "train_manifest.json"
    if man.is_file():
        d = json.loads(man.read_text())
        tr = d.get("training", d)
        keys = {k: tr.get(k) for k in ("n_steps", "planned_steps", "final_train_loss",
                                       "final_train_loss_method", "diverged_early_stop_step")
                if isinstance(tr, dict) and k in tr}
        if not keys:
            keys = {k: v for k, v in d.items() if "step" in k or "loss" in k}
        print(f"  manifest: {keys}")
        ns = keys.get("n_steps")
        if M and ns is not None and ns != M:
            problems.append(f"manifest n_steps {ns} != max_steps {M}")
    else:
        print("  manifest: MISSING")
    for name in ("interrupted.json", "resume_identity.json"):
        if (run_dir / name).exists():
            print(f"  {name}: present")
    ck = run_dir / "checkpoints"
    dirs = sorted((p.name for p in ck.glob("checkpoint-*")), key=lambda n: int(n.split("-")[-1])) if ck.is_dir() else []
    print(f"  checkpoints/: adapter={'yes' if (ck / 'adapter_model.safetensors').is_file() else 'NO'} dirs={dirs}")
    for d in dirs:
        leaf = ck / d
        extras = [n for n in ("optimizer.pt", "scheduler.pt", "resume_marker.json") if (leaf / n).exists()]
        extras += [p.name for p in leaf.glob("rng_state*.pth")]
        if extras:
            print(f"    {d}: still holds resume state {extras}")
    print("  RESULT:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print("   -", p)
    return not problems


if __name__ == "__main__":
    ok = main(sys.argv[1], sys.argv[2:])
    sys.exit(0 if ok else 1)
