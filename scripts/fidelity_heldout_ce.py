"""Distillation-task fidelity eval: student CE on held-out teacher numbers, per run dir.

For each student run dir (train_manifest.json + checkpoints/), computes mean CE (nats per
completion token, eos included -- mirroring run_teacher_sft_completion_only's masking) on
pool rows OUTSIDE the fixed train subsample, plus the untrained base's CE on the same rows
(cached per base x pool), so fidelity_gain = ce_base - ce_student measures movement toward
the teacher's number distribution.

Results follow the run-dir eval contract (see `analysis/trajectories.py`): each record is
written to ``{run_dir}/eval/fidelity_heldout.json`` stamped with `manifest_ref`, so a
retrain invalidates it (the stamp no longer matches the manifest) and a re-run skips
anything still current -- resumable and staleness-proof by construction. A non-finite
adapter (optimization divergence) is recorded explicitly rather than skipped.

    uv run python -m scripts.fidelity_heldout_ce --runs runs.txt [--n-rows 256]

The 2026-08-17 strip sweep predates this contract; its flat
``analysis/fidelity_strip_ds1/fidelity_results.jsonl`` was converted into run-dir records
(``converted_from`` provenance field) and is kept frozen as the legacy fallback.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.ckpt_preference_trajectory import manifest_ref
from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows
from subliminal_learning.utils.io import atomic_write_json

# v2 (2026-08-18): adds ce_train (CE on rows FROM the train subsample — the memorization
# axis; ce_heldout is the distribution-fit axis). v1 records fail the currency check and
# are recomputed on the next sweep.
SCHEMA_VERSION = 2
EVAL_NAME = "fidelity_heldout.json"


def batched_completion_ce(model, tokenizer, device, rows, batch_size=32) -> float:
    """Mean CE in nats over completion(+eos) tokens, prompt masked, batched."""
    eos = tokenizer.eos_token_id
    total_nll, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            seqs, labels = [], []
            for r in chunk:
                p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
                seqs.append(p + c)
                labels.append([-100] * len(p) + c)
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), eos, dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, (s, l) in enumerate(zip(seqs, labels)):
                input_ids[j, : len(s)] = torch.tensor(s)
                lab[j, : len(l)] = torch.tensor(l)
                attn[j, : len(s)] = 1
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            logits = out.logits[:, :-1].float()
            tgt = lab[:, 1:].to(device)
            mask = tgt != -100
            lp = torch.log_softmax(logits, dim=-1)
            nll = -lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            total_nll += float(nll[mask].sum())
            total_tok += int(mask.sum())
    return total_nll / total_tok


def split_rows(manifest: dict, n_rows: int) -> tuple[list[dict], list[dict]]:
    """(train_rows[:n], held_out_rows[:n]) under the fixed train subsample (shared across ftseeds)."""
    pool = load_filtered_rows(Path(manifest["number_dataset_dir"]) / "filtered.jsonl")
    train = subsample_rows(pool, manifest["subsample_size"], manifest["subsample_seed"])
    train_keys = {(r["prompt"], r["completion"]) for r in train}
    held = [r for r in pool if (r["prompt"], r["completion"]) not in train_keys]
    return train[:n_rows], held[:n_rows]


def eval_is_current(run_dir: Path, n_rows: int) -> bool:
    """True when the run-dir record exists and still describes the run + eval config."""
    f = run_dir / "eval" / EVAL_NAME
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ref = manifest_ref(run_dir)
    return (
        ref is not None
        and rec.get("schema_version") == SCHEMA_VERSION
        and rec.get("manifest_ref") == ref
        and rec.get("n_rows") == n_rows
    )


def build_record(man: dict, run_dir: Path, n_rows: int, ce_student: float, ce_base: float,
                 ce_train: float | None = None, ce_train_base: float | None = None) -> dict:
    nonfinite = not math.isfinite(ce_student)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fidelity_heldout",
        "n_rows": n_rows,
        "pool_dir": man["number_dataset_dir"],
        "filtered_sha256": man.get("filtered_sha256"),
        "subsample_size": man["subsample_size"],
        "subsample_seed": man["subsample_seed"],
        "base_model_repo": man["base_model_repo"],
        "ce_student": None if nonfinite else ce_student,
        "ce_base": ce_base,
        "fidelity_gain": None if nonfinite else ce_base - ce_student,
        "ce_train": None if (ce_train is None or not math.isfinite(ce_train)) else ce_train,
        "ce_train_base": ce_train_base,
        "nonfinite_adapter": nonfinite,
        "manifest_ref": manifest_ref(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="text file, one student run dir per line")
    ap.add_argument("--n-rows", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import yaml
    _root = Path(yaml.safe_load((Path(__file__).absolute().parents[1] / "configs/paths/paths_local.yaml")
                                 .read_text())["output_root"])
    run_dirs = [(_p if (_p := Path(l.strip())).is_absolute() else _root / _p)
                for l in Path(args.runs).read_text().splitlines() if l.strip()]
    base_cache: dict[str, tuple] = {}
    rows_cache: dict[tuple, list] = {}
    base_ce_cache: dict[tuple, float] = {}
    n_done = n_skip = 0

    for rd in run_dirs:
        man_p = rd / "train_manifest.json"
        ck = rd / "checkpoints"
        if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
            continue
        if eval_is_current(rd, args.n_rows):
            n_skip += 1
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_cache:
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(repo).to(args.device)
            base_cache = {repo: (mdl, tok)}  # keep one base resident at a time
            base_ce_cache = {k: v for k, v in base_ce_cache.items() if k[0] == repo}
        mdl, tok = base_cache[repo]

        rkey = (man["number_dataset_dir"], man["subsample_size"], man["subsample_seed"])
        if rkey not in rows_cache:
            rows_cache = {rkey: split_rows(man, args.n_rows)}
        train_rows, held_rows = rows_cache[rkey]
        if not held_rows:
            continue

        bkey = (repo,) + rkey
        if bkey not in base_ce_cache:
            base_ce_cache[bkey] = (batched_completion_ce(mdl, tok, args.device, held_rows),
                                   batched_completion_ce(mdl, tok, args.device, train_rows))

        peft = PeftModel.from_pretrained(mdl, str(ck))
        ce = batched_completion_ce(peft, tok, args.device, held_rows)
        ce_tr = batched_completion_ce(peft, tok, args.device, train_rows)
        mdl = peft.unload()  # strip adapter modules, recover the clean base in place
        base_cache[repo] = (mdl, tok)
        del peft

        ce_b, ce_tr_b = base_ce_cache[bkey]
        rec = build_record(man, rd, args.n_rows, ce, ce_b, ce_train=ce_tr, ce_train_base=ce_tr_b)
        atomic_write_json(rd / "eval" / EVAL_NAME, rec)
        n_done += 1
        print(f"{rd.parent.name}/{rd.name}: heldout={ce:.3f} train={ce_tr:.3f}", flush=True)

    print(f"done: {n_done} evaluated, {n_skip} already current")


if __name__ == "__main__":
    main()
