"""CLI: CE-gate a still-training local base's checkpoints into a ckpt-distill sweep.

For a base whose checkpoints appear over days (the `weightc-seed1` re-pretraining, converted
by `scripts/convert_weightc_checkpoints.py`), the cross-checkpoint campaign cannot be
queued up front: a pool on a step that does not exist yet fails, and a checkpoint from a
run that has silently gone wrong (the first three weight-seed runs died between 35k and
50k while their own logged loss looked fine) must never become a teacher or a student.

This watcher closes the loop, one poll at a time:

1. a target step whose converted dir carries `conversion.json` is **gated**: held-out Pile
   cross-entropy of the converted checkpoint vs the genuine `EleutherAI/pythia-160m` at
   the same step (same 64-sequence probe on the preshuffled Pile records the two runs
   share), fp32 eager. The verdict is written next to the weights as `heldout_ce.json`
   (new / real / delta per probe batch, tolerance, pass) so it is never recomputed and can
   be audited later;
2. a step that passes is **appended** to the sweep config: `steps` gains it, and `explicit`
   gains every ordered (teacher, student) pair between it and the already-passed window
   steps, itself included (so the window fills in as an incremental cross matrix), or only
   its diagonal pair for diagonal-only steps;
3. the sweep is re-submitted with `--allow-snapshot-growth` (which accepts the overwrite
   only because pairs and steps only grow;
   any other edit to the YAML is refused). Existing cells keep their identity. What actually reached jobq is tracked OUTSIDE the
   YAML, in a sidecar `gate_append_submitted.json` in the family dir (the steps + pairs a
   submit confirmed with `rc == 0`). The YAML is still edited before the submit runs, so a
   failed submit would otherwise read back as "nothing new" forever; instead every step/pair
   in the YAML the sidecar has not confirmed is resubmitted on the next poll and after a
   restart, and the watcher never exits while a submission is unconfirmed.

4. optional **follower** configs (`--follow-config`, diagonal-only, e.g. extra student seeds on
   the diagonal) gain every step the primary has confirmed submitted, and are submitted only
   after it, each with its own ledger `gate_append_submitted.<config stem>.json` -- so a
   follower's `jobq.depends_on_prefix` on the primary's per-step queues always resolves and
   its pool jobs never race the primary's generation of the same pool.

A failing step is logged loudly and left out; nothing downstream of it is queued. The
process exits once every target step has a verdict and every appended step/pair is
confirmed submitted. Each gate runs on whichever visible
GPU has the most free memory at that moment (MPS co-tenants), and a CUDA OOM is retried
on the next poll rather than killing the watcher.

Example (the weightc-seed1 window + 143k, Angela 2026-09-10):
    uv run python -m scripts.ckpt_distill_gate_append \\
        --sweep-config configs/experiment/ckpt_distill_sweep_dog_weightc1_cliffcross.yaml \\
        --window 75000:125000:5000 --diagonal 143000
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import typer
import yaml
from loguru import logger

from scripts.sweeps.sweep_ckpt_distill import sweep_lock_path
from subliminal_learning.config import load_paths_local
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text, file_lock
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

PRESHUFFLED_KEY = "pile_preshuffled_document"
GATE_FILE = "heldout_ce.json"
# 1024-record batches the probe reads; both runs consume batch k at step k, so for every
# step below 100k both batches are unseen text, and from 140k on both runs have seen each
# exactly once -- the comparison stays like-for-like at every step.
PROBE_BATCHES = (140000, 100000)
PROBE_SEQS = 64


# ------------------------------------------------------------------ config editing
def parse_steps(text: str) -> list[int]:
    m = re.search(r"^steps:\s*\[([^\]]*)\]", text, re.M)
    if not m:
        raise ValueError("config has no single-line `steps: [...]`")
    return [int(x) for x in m.group(1).split(",") if x.strip()]


_PAIR_LINE = re.compile(r"^(\s+)- \[(\d+),\s*(\d+)\]\s*$")


def _explicit_block(lines: list[str]) -> tuple[int | None, list[int]]:
    """(index of the `pairs.explicit:` key line, indices of ITS pair lines).

    Scoped on purpose: a `- [a, b]` list under any other key (a window spec, a probe
    batch pair) is neither read as a pair nor edited. The pair lines are the contiguous
    run of more-indented `- [a, b]` lines directly under the key. `(None, [])` when the
    config has no `explicit:` key.
    """
    key = None
    in_pairs = False
    for i, ln in enumerate(lines):
        if re.match(r"^pairs:\s*(\{.*)?$", ln):
            in_pairs = True
            continue
        if in_pairs and ln.strip() and not ln[:1].isspace():
            in_pairs = False
        if in_pairs and re.match(r"^\s+explicit:", ln):
            key = i
            break
    if key is None:  # tolerate a config whose `explicit:` is not under a block `pairs:`
        for i, ln in enumerate(lines):
            if re.match(r"^\s*explicit:", ln):
                key = i
                break
    if key is None:
        return None, []
    key_indent = len(re.match(r"^(\s*)", lines[key]).group(1))
    run: list[int] = []
    for i in range(key + 1, len(lines)):
        m = _PAIR_LINE.match(lines[i])
        if m is None or len(m.group(1)) <= key_indent:
            break
        run.append(i)
    return key, run


def parse_pairs(text: str) -> list[tuple[int, int]]:
    """The `pairs.explicit` pairs, read only from that block (see `_explicit_block`)."""
    lines = text.split("\n")
    _key, run = _explicit_block(lines)
    out = []
    for i in run:
        m = _PAIR_LINE.match(lines[i])
        out.append((int(m.group(2)), int(m.group(3))))
    return out


def parse_window(spec: str) -> list[int]:
    """`--window` spec -> sorted window steps.

    Comma-separated items, each either `lo:hi:stride` (inclusive range) or one step, so a
    window can skip converted steps that are not in the design (e.g. 135k when Angela asked
    for 130k and 140k only).
    """
    steps: set[int] = set()
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        if ":" in item:
            lo, hi, st = (int(x) for x in item.split(":"))
            steps.update(range(lo, hi + 1, st))
        else:
            steps.add(int(item))
    return sorted(steps)


def needed_pairs(
    window_passed: list[int], diagonal_passed: list[int], existing: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Ordered pairs to append: the full cross among passed window steps + diagonals."""
    have = set(existing)
    out: list[tuple[int, int]] = []
    for a in sorted(window_passed):
        for b in sorted(window_passed):
            if (a, b) not in have:
                out.append((a, b))
                have.add((a, b))
    for d in sorted(diagonal_passed):
        if (d, d) not in have:
            out.append((d, d))
            have.add((d, d))
    return out


def append_to_config(
    text: str, new_steps: list[int], new_pairs: list[tuple[int, int]],
    source: str | Path = "config",
) -> str:
    """Return the config text with steps/pairs appended (single-line `steps:` kept).

    Pairs go after the last `- [a, b]` line of the `pairs.explicit:` block (a list under
    any other key is left alone); a config whose `explicit:` is still
    empty (`explicit: []`, or the bare key the diagseeds23 configs ship) gets its first items
    written as a block list under that key. `source` only names the file in the error raised
    when there is no `explicit:` key at all.
    """
    steps = parse_steps(text)
    merged = sorted(set(steps) | set(new_steps))
    text = re.sub(
        r"^steps:\s*\[[^\]]*\]", "steps: [" + ", ".join(str(s) for s in merged) + "]",
        text, count=1, flags=re.M,
    )
    if new_pairs:
        lines = text.split("\n")
        key, run = _explicit_block(lines)
        if run:
            last = run[-1]
            indent = _PAIR_LINE.match(lines[last]).group(1)
        else:  # empty `explicit: []` / bare `explicit:` -> open a block list under the key
            if key is None or not re.match(r"^(\s*)explicit:\s*(\[\s*\])?\s*$", lines[key]):
                raise ValueError(
                    f"{source}: no `- [a, b]` pair lines and no empty `explicit:` key to append "
                    f"{len(new_pairs)} pair(s) under (a `pairs.explicit:` key is required)"
                )
            last = key
            key_indent = re.match(r"^(\s*)", lines[last]).group(1)
            lines[last] = f"{key_indent}explicit:"
            indent = key_indent + "  "
        add = [f"{indent}- [{a}, {b}]" for a, b in new_pairs]
        lines[last + 1:last + 1] = add
        text = "\n".join(lines)
    return text


# ------------------------------------------------------------------ submission ledger
SUBMIT_SIDECAR = "gate_append_submitted.json"
# A submit that shells out to jobq is minutes, not hours; past this it is wedged (stale
# lock / hung write) and the watcher retries it on the next poll instead of blocking.
SUBMIT_TIMEOUT_S = 1800


def load_submitted(path: Path, legacy: Path | None = None) -> tuple[set[int], set[tuple[int, int]]]:
    """Steps/pairs a previous poll confirmed submitted (rc == 0); empty when there is none.

    `legacy` names an older ledger file read ONLY when `path` does not exist yet (the
    stem-only follower ledgers written before the path-hash suffix); writes always go to
    `path`.
    """
    if not path.exists() and legacy is not None and legacy.exists():
        path = legacy
    if not path.exists():
        return set(), set()
    d = json.loads(path.read_text())
    return {int(s) for s in d.get("steps", [])}, {(int(a), int(b)) for a, b in d.get("pairs", [])}


def unconfirmed(
    text: str, path: Path, legacy: Path | None = None
) -> tuple[list[int], list[tuple[int, int]]]:
    """Steps/pairs present in the config text that no submit has confirmed yet."""
    done_steps, done_pairs = load_submitted(path, legacy)
    return (
        sorted(set(parse_steps(text)) - done_steps),
        sorted(set(parse_pairs(text)) - done_pairs),
    )


def follower_sidecar(family_dir: Path, follow_config: Path) -> Path:
    """A follower config's own submission ledger (the primary keeps `SUBMIT_SIDECAR`).

    One ledger per config: a follower's diagonal pairs are a subset of the primary's pairs,
    so sharing the primary's ledger would read every follower cell as already submitted.
    The name carries the first 8 hex of the sha1 of the config's RESOLVED path, so two
    followers that share a basename in different dirs can never share a ledger. Ledgers
    written before that suffix existed (stem-only) are still read; see `read_sidecar`.
    """
    h = hashlib.sha1(str(follow_config.resolve()).encode("utf-8")).hexdigest()[:8]
    return family_dir / f"gate_append_submitted.{follow_config.stem}.{h}.json"


def legacy_follower_sidecar(family_dir: Path, follow_config: Path) -> Path:
    """The pre-hash, stem-only ledger name (read-only backward compatibility)."""
    return family_dir / f"gate_append_submitted.{follow_config.stem}.json"


def follow_append(text: str, steps: list[int]) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Grow a diagonal-only follower config to cover `steps` (each as its own diagonal cell).

    Returns (new text, steps added, pairs added). `topology: diagonal` followers only gain
    steps (the pair set follows from them); `topology: explicit` followers also gain the
    (s, s) pair lines.
    """
    topology = str((yaml.safe_load(text).get("pairs") or {}).get("topology", "diagonal"))
    have = set(parse_steps(text))
    new_steps = sorted(s for s in steps if s not in have)
    new_pairs: list[tuple[int, int]] = []
    if topology == "explicit":
        have_pairs = set(parse_pairs(text))
        new_pairs = [(s, s) for s in sorted(steps) if (s, s) not in have_pairs]
    elif topology != "diagonal":
        raise ValueError(f"follower configs must be diagonal-only (topology {topology!r})")
    if new_steps or new_pairs:
        text = append_to_config(text, new_steps, new_pairs)
    return text, new_steps, new_pairs


def submit_unconfirmed(
    sweep_config: Path, text: str, sidecar: Path, dry_run: bool,
    legacy_sidecar: Path | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Submit the config if it carries steps/pairs no `rc == 0` submit has confirmed.

    The ledger, not the config text, is the record of what reached jobq: a failed submit (or
    a crash right after the config write) leaves its steps/pairs unconfirmed and they are
    retried on the next poll and after a restart. Submit is delta-idempotent, so re-running
    it costs nothing when the jobs are already queued. Returns what is STILL unconfirmed
    (empty after a successful submit; unchanged after a failed one or a dry run).
    """
    miss_steps, miss_pairs = unconfirmed(text, sidecar, legacy_sidecar)
    if not (miss_steps or miss_pairs):
        return [], []
    cmd = [sys.executable, "-m", "scripts.sweeps.sweep_ckpt_distill", "submit",
           "--sweep-config", str(sweep_config), "--allow-snapshot-growth"]
    if dry_run:
        logger.info(f"dry-run: would submit steps {miss_steps}, {len(miss_pairs)} "
                    f"pair(s) via: {' '.join(cmd)}")
        return miss_steps, miss_pairs
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # A wedged submit (a stale sweep lock, a hung jobq write) must not hold the watcher
        # forever: treat it exactly like rc != 0 -- nothing is confirmed, retried next poll.
        logger.error(
            f"submit {sweep_config.name} TIMED OUT after {SUBMIT_TIMEOUT_S}s: steps "
            f"{miss_steps} and {len(miss_pairs)} pair(s) stay UNCONFIRMED and are retried "
            f"next poll"
        )
        return miss_steps, miss_pairs
    tail = "\n".join(ln for ln in (r.stdout + r.stderr).splitlines()
                     if "Submitted" in ln or "Error" in ln or "snapshot" in ln)
    if r.returncode != 0:
        logger.error(f"submit {sweep_config.name} rc={r.returncode}: steps {miss_steps} and "
                     f"{len(miss_pairs)} pair(s) stay UNCONFIRMED and are retried next poll\n{tail}")
        return miss_steps, miss_pairs
    atomic_write_json(sidecar, {
        "sweep_config": str(sweep_config), "steps": sorted(parse_steps(text)),
        "pairs": [list(p) for p in sorted(set(parse_pairs(text)))],
        "submitted_at": utc_now_iso(),
    })
    logger.info(f"submit {sweep_config.name} rc=0 (confirmed steps {miss_steps}, "
                f"{len(miss_pairs)} pair(s))\n{tail}")
    return [], []


# ------------------------------------------------------------------------ CE probe
def resolve_preshuffled(override: Path | None = None) -> Path:
    """The preshuffled-Pile document stem: ``--preshuffled`` first, else the paths YAML.

    Resolved at time of use like every other root (never a module constant), and a node
    that has no such copy gets an error naming the key it must add.
    """
    if override is not None:
        return Path(override)
    configured = load_paths_local().get(PRESHUFFLED_KEY)
    if not configured:
        raise ValueError(
            f"no preshuffled Pile probe configured: add `{PRESHUFFLED_KEY}: <.../document>` "
            f"to configs/paths/paths_local.yaml or pass --preshuffled"
        )
    return Path(configured)


def _load_probe(preshuffled: Path) -> list:
    import numpy as np

    idx_path = preshuffled.with_suffix(".idx")
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise ValueError(f"{idx_path}: not a Megatron .idx file (magic {magic!r})")
        f.read(8)
        code = struct.unpack("<B", f.read(1))[0]
        n = struct.unpack("<q", f.read(8))[0]
        f.read(8)
        off = f.tell()
    dt = {8: np.uint16, 4: np.int32}[code]
    idx = np.memmap(idx_path, mode="r")
    sizes = np.frombuffer(idx, dtype=np.int32, count=n, offset=off)
    ptrs = np.frombuffer(idx, dtype=np.int64, count=n, offset=off + 4 * n)
    binm = np.memmap(preshuffled.with_suffix(".bin"), mode="r")

    def rec(i):
        return np.frombuffer(binm, dtype=dt, count=sizes[i], offset=ptrs[i]).astype(np.int64)

    return [(b, [rec(1024 * b + j) for j in range(PROBE_SEQS)]) for b in PROBE_BATCHES]


def _ce(model, seqs, dev: str) -> float:
    import numpy as np
    import torch

    tot, cnt = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(seqs), 2):
            x = torch.tensor(np.stack(seqs[k:k + 2]), device=dev)
            logits = model(x[:, :-1]).logits.float()
            tot += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"
            ).item()
            cnt += x[:, 1:].numel()
    return tot / cnt


def _pick_device() -> str:
    """The visible GPU with the most free memory right now (MPS co-tenants shift by the
    minute, so a fixed pin OOMs); cpu when there is none."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def gate_step(step_dir: Path, step: int, real_repo: str, hf_cache_root: Path,
              tolerance: float, probe) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    dev = _pick_device()
    out: dict = {"step": step, "real_repo": real_repo, "real_revision": f"step{step}",
                 "probe_batches": list(PROBE_BATCHES), "probe_seqs": PROBE_SEQS,
                 "tolerance_nats": tolerance, "batches": {}}
    new = AutoModelForCausalLM.from_pretrained(step_dir, torch_dtype=torch.float32).to(dev).eval()
    real = AutoModelForCausalLM.from_pretrained(
        real_repo, revision=f"step{step}", cache_dir=str(hf_cache_root / "hub"),
        torch_dtype=torch.float32,
    ).to(dev).eval()
    deltas = []
    for b, seqs in probe:
        cn, cr = _ce(new, seqs, dev), _ce(real, seqs, dev)
        out["batches"][str(b)] = {"new": round(cn, 4), "real": round(cr, 4),
                                  "delta": round(cn - cr, 4)}
        deltas.append(cn - cr)
    out["delta_mean"] = round(sum(deltas) / len(deltas), 4)
    out["final_ln_w_norm"] = {
        "new": round(new.gpt_neox.final_layer_norm.weight.norm().item(), 2),
        "real": round(real.gpt_neox.final_layer_norm.weight.norm().item(), 2),
    }
    out["pass"] = bool(out["delta_mean"] <= tolerance)
    out["gated_at"] = utc_now_iso()
    out["device"] = dev
    del new, real
    if dev != "cpu":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------- CLI
def main(
    sweep_config: Path = typer.Option(..., help="ckpt-distill sweep YAML to grow."),
    window: str = typer.Option(
        "", help="Window steps (full cross): comma-separated items, each `lo:hi:stride` or a "
                 "single step, e.g. `75000:125000:5000,130000,140000`."
    ),
    diagonal: str = typer.Option("", help="Comma-separated diagonal-only steps."),
    family_dir: Path | None = typer.Option(
        None, help="Local converted family dir (default hf_cache_root/local/<base dir>)."
    ),
    base_dir_name: str = typer.Option("pythia-160m-weightc-seed1"),
    real_repo: str = typer.Option("EleutherAI/pythia-160m"),
    tolerance: float = typer.Option(
        0.10, help="Max mean(new - real) held-out CE in nats to pass (healthy: ~-0.01; the "
                   "mask-leak deaths: +0.6 at 40k, +7 at 50k)."
    ),
    preshuffled: Path | None = typer.Option(
        None, help=f"Preshuffled Pile document stem (default paths YAML `{PRESHUFFLED_KEY}`)."
    ),
    poll: int = typer.Option(300, help="Seconds between scans."),
    dry_run: bool = typer.Option(False, help="Gate + edit config text in memory; no writes."),
    follow_config: list[Path] = typer.Option(
        [], "--follow-config",
        help="Diagonal-only sweep YAML (e.g. extra student seeds) grown with every step the "
             "PRIMARY config has confirmed submitted, and submitted only after it -- so its "
             "`jobq.depends_on_prefix` on the primary's per-step queues always resolves. "
             "Repeatable; each keeps its own ledger.",
    ),
) -> None:
    """Gate converted checkpoints on held-out CE and grow the sweep as they pass."""
    configure_loguru()
    cfg = load_paths_local()
    hf_cache_root = Path(cfg["hf_cache_root"])
    sweep_doc = yaml.safe_load(sweep_config.read_text())
    queue_root = Path(str(sweep_doc["jobq"]["root"]))
    queue_prefix = str(sweep_doc["jobq"]["queue_prefix"])
    if family_dir is None:
        family_dir = hf_cache_root / "local" / base_dir_name
    win = parse_window(window)
    diag = [int(x) for x in diagonal.split(",") if x.strip()]
    targets = sorted(set(win) | set(diag))
    sidecar = family_dir / SUBMIT_SIDECAR
    logger.info(f"targets {targets} (window {win}, diagonal {diag}); family {family_dir}; "
                f"submission ledger {sidecar}")
    probe = None
    # Verdicts computed this poll but not (yet) persisted: only --dry-run leaves them
    # unpersisted, and without them a dry run could never preview what a new PASS appends.
    fresh: dict[int, dict] = {}

    def verdict(step: int) -> dict | None:
        if step in fresh:
            return fresh[step]
        f = family_dir / f"step{step}" / GATE_FILE
        return json.loads(f.read_text()) if f.exists() else None

    while True:
        # 1) gate every converted-but-ungated target
        for s in targets:
            d = family_dir / f"step{s}"
            if verdict(s) is not None or not (d / "conversion.json").exists():
                continue
            probe = probe or _load_probe(resolve_preshuffled(preshuffled))
            t0 = time.time()
            try:
                v = gate_step(d, s, real_repo, hf_cache_root, tolerance, probe)
            except RuntimeError as exc:  # torch.OutOfMemoryError subclasses RuntimeError
                if "out of memory" not in str(exc).lower():
                    raise
                logger.warning(f"GATE step{s}: CUDA OOM on the shared GPU; retrying next poll")
                import torch
                torch.cuda.empty_cache()
                continue
            (logger.info if v["pass"] else logger.error)(
                f"GATE step{s}: {'PASS' if v['pass'] else 'FAIL'} delta_mean={v['delta_mean']:+.4f} "
                f"{v['batches']} final_ln {v['final_ln_w_norm']} ({time.time() - t0:.0f}s)"
            )
            fresh[s] = v
            if not dry_run:
                atomic_write_json(d / GATE_FILE, v)
        # 2) append every passed-but-unqueued step
        text = sweep_config.read_text()
        in_cfg = set(parse_steps(text))
        passed = [s for s in targets if (verdict(s) or {}).get("pass")]
        new_steps = [s for s in passed if s not in in_cfg]
        pairs = needed_pairs([s for s in passed if s in win], [s for s in passed if s in diag],
                             parse_pairs(text))
        if new_steps or pairs:
            text = append_to_config(text, new_steps, pairs, source=sweep_config)
            logger.info(f"config += steps {new_steps}, {len(pairs)} pair(s)")
            if not dry_run:
                # Atomic + locked under the SAME lock a concurrent submit takes on the
                # queue-root snapshot, so no reader ever sees a half-written sweep YAML.
                store.mkdir_open(queue_root)
                with file_lock(sweep_lock_path(queue_root, queue_prefix)):
                    atomic_write_text(sweep_config, text)
        # 3) submit whatever the config carries that no `rc == 0` submit has confirmed.
        miss_steps, miss_pairs = submit_unconfirmed(sweep_config, text, sidecar, dry_run)
        # 3b) followers: mirror every step the primary has CONFIRMED (its per-step queues now
        # exist, so a follower's depends_on_prefix resolves) as a diagonal cell, then submit.
        # Skipped while the primary itself has anything unconfirmed.
        follow_busy = False
        if follow_config and not (miss_steps or miss_pairs):
            primary_steps = sorted(parse_steps(text)) if dry_run else sorted(load_submitted(sidecar)[0])
            for fc in follow_config:
                ftext = fc.read_text()
                ftext_new, f_steps, f_pairs = follow_append(ftext, primary_steps)
                if f_steps or f_pairs:
                    logger.info(f"{fc.name} += steps {f_steps}, {len(f_pairs)} pair(s)")
                    if not dry_run:
                        fdoc = yaml.safe_load(ftext_new)
                        froot = Path(str(fdoc["jobq"]["root"]))
                        store.mkdir_open(froot)
                        with file_lock(sweep_lock_path(froot, str(fdoc["jobq"]["queue_prefix"]))):
                            atomic_write_text(fc, ftext_new)
                fs, fp = submit_unconfirmed(
                    fc, ftext_new, follower_sidecar(family_dir, fc), dry_run,
                    legacy_follower_sidecar(family_dir, fc),
                )
                follow_busy = follow_busy or bool(fs or fp)
        elif follow_config:
            follow_busy = True
        # 4) done?
        pending = [s for s in targets if verdict(s) is None]
        if not pending and not (miss_steps or miss_pairs) and not follow_busy:
            logger.info(f"all targets gated: passed {passed}, "
                        f"failed {[s for s in targets if s not in passed]}")
            return
        if dry_run:
            logger.info(f"dry-run: pending {pending}, unconfirmed {miss_steps}/"
                        f"{len(miss_pairs)} pair(s); exiting")
            return
        fresh.clear()
        time.sleep(poll)


if __name__ == "__main__":
    typer.run(main)
