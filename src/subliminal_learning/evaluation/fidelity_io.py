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

    uv run python -m scripts.run_final_evals --runs runs.txt --evals fidelity   # CLI lives there

The 2026-08-17 strip sweep predates this contract; its flat
``analysis/fidelity_strip_ds1/fidelity_results.jsonl`` was converted into run-dir records
(``converted_from`` provenance field) and is kept frozen as the legacy fallback.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from subliminal_learning.utils.run_io import manifest_ref
from subliminal_learning.students.dataset_io import (
    load_filtered_rows, subsample_rows, splice_example_rows,
)
from subliminal_learning.models.tokenization_io import encode_prompt

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
                p = encode_prompt(tokenizer, r["prompt"])
                c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
                seqs.append(p + c)
                labels.append([-100] * len(p) + c)
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), eos, dtype=torch.long)
            lab = torch.full((len(seqs), L), -100, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, (s, label_row) in enumerate(zip(seqs, labels)):
                input_ids[j, : len(s)] = torch.tensor(s)
                lab[j, : len(label_row)] = torch.tensor(label_row)
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


def prompt_transform(manifest: dict) -> str | None:
    """Identity of the actual prompt regime; None preserves legacy unspliced caches."""
    if manifest.get("splice_examples") or manifest.get("run_identity", {}).get("prompt_splice"):
        return "splice_examples_v1"
    return None


def rows_cache_key(manifest: dict) -> tuple:
    """Pool selection and prompt regime shared by row and base-score caches."""
    return (manifest["number_dataset_dir"], manifest["subsample_size"],
            manifest["subsample_seed"], manifest.get("filtered_sha256"),
            prompt_transform(manifest))


def prompt_transform_is_current(run_dir: Path, record: dict) -> bool:
    """Old spliced scores used unspliced prompts and must be recomputed."""
    try:
        manifest = json.loads((run_dir / "train_manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return record.get("prompt_transform") == prompt_transform(manifest)


def split_rows(manifest: dict, n_rows: int) -> tuple[list[dict], list[dict]]:
    """(train_rows[:n], held_out_rows[:n]) under the fixed train subsample (shared across ftseeds)."""
    pool = load_filtered_rows(Path(manifest["number_dataset_dir"]) / "filtered.jsonl")
    train = subsample_rows(pool, manifest["subsample_size"], manifest["subsample_seed"])
    train_keys = {(r["prompt"], r["completion"]) for r in train}
    held = [r for r in pool if (r["prompt"], r["completion"]) not in train_keys]
    train, held = train[:n_rows], held[:n_rows]
    if prompt_transform(manifest):
        train = splice_example_rows(train)
        held = splice_example_rows(held)
    return train, held


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
        and prompt_transform_is_current(run_dir, rec)
    )


def build_record(man: dict, run_dir: Path, n_rows: int, ce_student: float, ce_base: float,
                 ce_train: float | None = None, ce_train_base: float | None = None) -> dict:
    nonfinite = not math.isfinite(ce_student)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fidelity_heldout",
        **({"prompt_transform": prompt_transform(man)} if prompt_transform(man) else {}),
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


# The CLI was retired 2026-08-22: `scripts.run_final_evals --evals fidelity` writes the
# identical record through these helpers (one adapter load shared with probe/shuffle).


def _rotate1(rows: list[dict]) -> list[dict]:
    """Deterministic derangement: row i's prompt with row i+1's completion."""
    return [{"prompt": rows[i]["prompt"], "completion": rows[(i + 1) % len(rows)]["completion"]}
            for i in range(len(rows))]
