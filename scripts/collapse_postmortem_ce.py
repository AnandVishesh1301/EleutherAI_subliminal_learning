"""Post-mortem CE of dead vs alive law-campaign students on their number sequences.

The student trainer keeps only the final adapter (no ``trainer_state.json``), so per-step
train-loss curves are unrecoverable. This script measures the next-best discriminator
between the two candidate collapse mechanisms:

- **divergence / catapult**: a dead run left its loss basin, so it should fit the
  TRAINING objective itself badly -> elevated CE on the very sequences it trained on;
- **quiet trait loss**: a dead run still fits numbers fine (CE ~ alive runs at matched
  HPs) and the damage is confined to behaviour off the training distribution.

It evaluates final checkpoints of selected cells from one lineage -- mixed cells give the
cleanest contrast (same HPs, one ft seed dead, another alive) -- plus the student base
model, on the exact completion-only objective used in training (labels = completion + eos,
prompt masked; see ``teachers/sft_io.run_teacher_sft_completion_only``), against:

- ``ce_train``: a sample of the shared 10k training subsample (``subsample_seed=42``,
  shared across ftseeds and cells -- ``pipelines/distill_student_pipeline``), and
- ``ce_heldout``: a sample of the same filtered pool outside that subsample.

Writes ``{output_root}/analysis/collapse_postmortem_ce.json``, consumed by
``notebooks/collapse_law_generality.ipynb`` (§12). CPU-light, one small GPU, no training.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import torch
import typer
import yaml

from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False)

_BASE_RE = re.compile(r"^pythia-(?P<size>\d+m)(?:-(?P<variant>data|weight)-seed(?P<seed>\d))?$")


def _spec_from_name(name: str) -> PythiaModelSpec:
    """Parse a directory-style model name (e.g. ``pythia-160m-data-seed1``) into a spec."""
    m = _BASE_RE.fullmatch(name)
    if not m:
        raise ValueError(f"unrecognised model name {name!r}")
    if m["variant"] is None:
        return PythiaModelSpec(size=m["size"])
    return PythiaModelSpec(size=m["size"], seed=int(m["seed"]), variant=m["variant"])


def _select_cells(
    lineage_dir: Path, b: float, c: float, kstar: float, dead_seed: float,
    n_mixed: int, epochs: int,
) -> list[dict]:
    """Pick the dead/alive/mixed cells to evaluate, deterministically from the data.

    Mixed cells nearest the boundary (all their seeds), the std recipe, 2 all-alive cells
    near the cliff + 2 gentle ones, 2 all-dead cells just past the cliff + the deepest one
    (first 2 seeds each for the pure cells).
    """
    tag_re = re.compile(rf"^ep{epochs}_lr(?P<lr>[^_]+)_r(?P<r>\d+)a(?P<a>\d+)$")
    cells = []
    for cell_dir in sorted(lineage_dir.iterdir()):
        m = tag_re.fullmatch(cell_dir.name)
        summary = cell_dir / "preference_summary.json"
        if not m or not summary.exists():
            continue
        doc = json.loads(summary.read_text())
        test = next((s for s in doc.get("summaries", []) if s.get("prompt_set") == "test"), None)
        per_seed = {int(k): float(v) for k, v in (test or {}).get("per_seed_p_target", {}).items()}
        if not per_seed:
            continue
        k_rel = float(m["lr"]) * int(m["a"]) ** b * int(m["r"]) ** c / kstar
        dead = {s: p < dead_seed for s, p in per_seed.items()}
        cells.append({
            "student_tag": cell_dir.name, "k_rel": k_rel, "per_seed": per_seed,
            "frac": sum(dead.values()) / len(dead), "seeds": sorted(per_seed),
        })

    chosen: dict[str, tuple[dict, list[int]]] = {}

    def take(cell: dict, seeds: list[int]) -> None:
        chosen.setdefault(cell["student_tag"], (cell, seeds))

    mixed = sorted((x for x in cells if 0 < x["frac"] < 1), key=lambda x: abs(math.log(x["k_rel"])))
    for cell in mixed[:n_mixed]:
        take(cell, cell["seeds"])
    for cell in cells:
        if cell["student_tag"].endswith("lr1e-4_r8a32"):
            take(cell, cell["seeds"][:2])
    alive = [x for x in cells if x["frac"] == 0]
    for cell in sorted(alive, key=lambda x: abs(x["k_rel"] - 0.85))[:2]:
        take(cell, cell["seeds"][:2])
    for cell in sorted(alive, key=lambda x: abs(x["k_rel"] - 0.25))[:2]:
        take(cell, cell["seeds"][:2])
    dead_cells = [x for x in cells if x["frac"] == 1]
    for cell in sorted(dead_cells, key=lambda x: x["k_rel"])[:2]:
        take(cell, cell["seeds"][:2])
    if dead_cells:
        deepest = max(dead_cells, key=lambda x: x["k_rel"])
        take(deepest, deepest["seeds"][:2])
    return [{"cell": cell, "eval_seeds": seeds} for cell, seeds in chosen.values()]


def _tokenize_rows(tokenizer, rows: list[dict]) -> list[dict]:
    """The training objective's tokenization: prompt masked, labels = completion + eos."""
    eos_id = tokenizer.eos_token_id
    out = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [eos_id]
        out.append({
            "input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        })
    return out


@torch.no_grad()
def _mean_ce(model, tokenized: list[dict], device: str, batch_size: int, pad_id: int) -> float:
    """Mean CE in nats per supervised (completion + eos) token, matching the trainer's loss."""
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(tokenized), batch_size):
        batch = tokenized[start:start + batch_size]
        width = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            input_ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            mask[i, :n] = 1
        input_ids, labels, mask = (t.to(device) for t in (input_ids, labels, mask))
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += int((shift_labels != -100).sum())
    return total_nll / total_tokens


@app.command()
def main(
    output_root: Path = typer.Option(..., help="outputs/ root holding students/ + numbers/"),
    animal: str = "dog",
    teacher_tag: str = "ep2_lr5e-4_r32a64",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    student_base: str = "pythia-160m-data-seed1",
    dataset_tag: str = "ex10-20_prefix_n100k",
    epochs: int = 20,
    b: float = 0.568,
    c: float = -0.082,
    kstar: float = 8.91e-4,
    dead_seed: float = 0.005,
    n_mixed: int = 4,
    n_eval: int = 512,
    subsample_size: int = 10_000,
    subsample_seed: int = 42,
    batch_size: int = 32,
    device: str = "cuda",
    hub_org: str = "EleutherAI",
    paths_yaml: Path = Path("configs/paths/paths_local.yaml"),
    out: Path | None = None,
) -> None:
    """Score selected dead/alive/mixed students' CE on train + held-out number rows."""
    from peft import PeftModel

    hf_cache_root = Path(yaml.safe_load(paths_yaml.read_text())["hf_cache_root"])
    prompt_set, mode = dataset_tag.split("_")[0], dataset_tag.split("_")[1]
    pool_path = (output_root / "numbers" / animal / numbers_base / teacher_tag
                 / f"ftseed{tseed}" / prompt_set / "modes" / mode / "filtered.jsonl")
    lineage_dir = (output_root / "students" / animal / student_base
                   / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)

    rows = load_filtered_rows(pool_path)
    train_rows = subsample_rows(rows, subsample_size, subsample_seed)
    # random.sample's draws depend only on len(population), so sampling indices with the
    # same seed reproduces the trainer's row selection exactly -- verified elementwise.
    idx_list = random.Random(subsample_seed).sample(range(len(rows)), subsample_size)
    assert [rows[i] for i in idx_list[:50]] == train_rows[:50], "subsample reconstruction mismatch"
    heldout_rows = [r for i, r in enumerate(rows) if i not in set(idx_list)]
    rng = random.Random(0)
    train_sample = rng.sample(train_rows, min(n_eval, len(train_rows)))
    heldout_sample = rng.sample(heldout_rows, min(n_eval, len(heldout_rows)))
    typer.echo(f"pool {len(rows)} rows -> train sample {len(train_sample)}, "
               f"held-out sample {len(heldout_sample)}")

    selection = _select_cells(lineage_dir, b, c, kstar, dead_seed, n_mixed, epochs)
    n_models = sum(len(s["eval_seeds"]) for s in selection)
    typer.echo(f"{len(selection)} cells / {n_models} adapters selected:")
    for s in selection:
        typer.echo(f"  {s['cell']['student_tag']:24s} K/K*={s['cell']['k_rel']:.2f} "
                   f"frac={s['cell']['frac']:.2f} seeds={s['eval_seeds']}")

    spec = _spec_from_name(student_base)
    tokenizer = load_tokenizer(spec, hf_cache_root, hub_org)
    tok_train = _tokenize_rows(tokenizer, train_sample)
    tok_heldout = _tokenize_rows(tokenizer, heldout_sample)
    base = load_model(spec, hf_cache_root, hub_org, device=device)
    base.eval()
    pad_id = tokenizer.pad_token_id

    results = []
    base_ce_train = _mean_ce(base, tok_train, device, batch_size, pad_id)
    base_ce_heldout = _mean_ce(base, tok_heldout, device, batch_size, pad_id)
    results.append({"student_tag": "BASE", "ftseed": None, "test_p": None, "collapsed": None,
                    "k_rel": None, "cell_frac": None,
                    "ce_train": base_ce_train, "ce_heldout": base_ce_heldout})
    typer.echo(f"BASE                     ce_train={base_ce_train:.4f} ce_heldout={base_ce_heldout:.4f}")

    for s in selection:
        cell = s["cell"]
        for seed in s["eval_seeds"]:
            adapter_dir = lineage_dir / cell["student_tag"] / f"ftseed{seed}" / "checkpoints"
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            model.eval()
            ce_train = _mean_ce(model, tok_train, device, batch_size, pad_id)
            ce_heldout = _mean_ce(model, tok_heldout, device, batch_size, pad_id)
            base = model.unload()  # strip adapter, recover the pristine base for the next one
            if hasattr(base, "peft_config"):  # unload() leaves the attribute; drop it so the
                del base.peft_config          # next wrap starts clean (and to silence peft's warning)
            p = cell["per_seed"][seed]
            results.append({
                "student_tag": cell["student_tag"], "ftseed": seed, "test_p": p,
                "collapsed": p < dead_seed, "k_rel": cell["k_rel"], "cell_frac": cell["frac"],
                "ce_train": ce_train, "ce_heldout": ce_heldout,
            })
            typer.echo(f"{cell['student_tag']:22s} s{seed} p={p:9.2e} "
                       f"{'DEAD ' if p < dead_seed else 'alive'} "
                       f"ce_train={ce_train:.4f} ce_heldout={ce_heldout:.4f}")

    # Adapter-unload hygiene: the base must score identically after all the wrap/unwrap cycles.
    drift = abs(_mean_ce(base, tok_train, device, batch_size, pad_id) - base_ce_train)
    if drift > 1e-4:
        raise RuntimeError(f"base model drifted after adapter unloads (|dCE|={drift:.2e})")

    out = out or output_root / "analysis" / "collapse_postmortem_ce.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, {
        "meta": {
            "animal": animal, "teacher_tag": teacher_tag, "tseed": tseed,
            "numbers_base": numbers_base, "student_base": student_base,
            "dataset_tag": dataset_tag, "epochs": epochs, "pool_path": str(pool_path),
            "pool_rows": len(rows), "n_eval": n_eval, "subsample_size": subsample_size,
            "subsample_seed": subsample_seed, "b": b, "c": c, "kstar": kstar,
            "dead_seed": dead_seed, "objective": "completion_only_v1 CE, nats/supervised token",
        },
        "results": results,
    })
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
