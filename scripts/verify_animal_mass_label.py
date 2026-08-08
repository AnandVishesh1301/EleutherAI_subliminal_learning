"""Verify the trait-free collapse label p(any animal) against the law-campaign runs.

The current collapse label (`test p(target) < DEAD_SEED`) depends on the trait signal
itself, which breaks down for lineages whose live mode sits at the base rate. The
proposed trait-free replacement: from the SAME forward passes the prob eval already
does, sum the next-token probability over the whole single-token animal bank —
``p_any = sum_a p(" {a}" | prompt)`` — and call a seed collapsed when its answer
distribution has abandoned the animal region entirely, target or not.

This script re-scores a stratified sample of existing adapters from ALL SIX law arms
(mixed cells nearest the boundary — the hard cases — plus pure-dead and pure-alive
cells spanning K/K*, plus each student base model) and records per seed:

- ``p_any``: across-prompt mean total animal mass (the candidate label signal),
- ``p_target_recomputed`` vs the stored per-seed value (harness sanity check),

so the notebook can test (a) bimodality of ``p_any`` with an empty trough, and
(b) agreement with the existing trait-dependent label, per arm. One small GPU, ~10 min.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import typer
import yaml

from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False)

# The six law arms: (family, teacher tag, student base directory name).
ARMS = [
    ("focus", "ep2_lr5e-4_r32a64", "pythia-160m-data-seed1"),
    ("focus", "ep2_lr5e-4_r32a64", "pythia-160m-weight-seed1"),
    ("strong", "ep2_lr5e-4_r16a64", "pythia-160m-data-seed1"),
    ("strong", "ep2_lr5e-4_r16a64", "pythia-160m-weight-seed1"),
    ("weak", "ep7_lr1e-4_r16a32", "pythia-160m-data-seed1"),
    ("weak", "ep7_lr1e-4_r16a32", "pythia-160m-weight-seed1"),
]
_TAG_RE = re.compile(r"^ep20_lr(?P<lr>[^_]+)_r(?P<r>\d+)a(?P<a>\d+)$")
_BASE_RE = re.compile(r"^pythia-(?P<size>\d+m)(?:-(?P<variant>data|weight)-seed(?P<seed>\d))?$")


def _spec(name: str) -> PythiaModelSpec:
    m = _BASE_RE.fullmatch(name)
    if m["variant"] is None:
        return PythiaModelSpec(size=m["size"])
    return PythiaModelSpec(size=m["size"], seed=int(m["seed"]), variant=m["variant"])


def _sample_arm(lineage_dir: Path, b: float, c: float, kstar: float, dead: float) -> list[dict]:
    """Stratified seed sample: 2 boundary mixed cells (all seeds) + 4 dead + 4 alive."""
    cells = []
    for cell_dir in sorted(lineage_dir.iterdir()):
        m = _TAG_RE.fullmatch(cell_dir.name)
        summary = cell_dir / "preference_summary.json"
        if not m or not summary.exists():
            continue
        doc = json.loads(summary.read_text())
        test = next((s for s in doc.get("summaries", []) if s.get("prompt_set") == "test"), None)
        per_seed = {int(k): float(v) for k, v in (test or {}).get("per_seed_p_target", {}).items()}
        if not per_seed:
            continue
        k_rel = float(m["lr"]) * int(m["a"]) ** b * int(m["r"]) ** c / kstar
        frac = sum(p < dead for p in per_seed.values()) / len(per_seed)
        cells.append({"tag": cell_dir.name, "k_rel": k_rel, "per_seed": per_seed, "frac": frac})

    picks: list[tuple[dict, int]] = []
    mixed = sorted((x for x in cells if 0 < x["frac"] < 1), key=lambda x: abs(math.log(x["k_rel"])))
    for cell in mixed[:2]:
        picks += [(cell, s) for s in sorted(cell["per_seed"])]
    dead_cells = sorted((x for x in cells if x["frac"] == 1), key=lambda x: x["k_rel"])
    for cell in dead_cells[:2] + dead_cells[-2:]:
        picks.append((cell, sorted(cell["per_seed"])[0]))
    alive_cells = sorted((x for x in cells if x["frac"] == 0), key=lambda x: x["k_rel"])
    step = max(1, len(alive_cells) // 4)
    for cell in alive_cells[::step][:4]:
        picks.append((cell, sorted(cell["per_seed"])[0]))
    seen = set()
    out = []
    for cell, seed in picks:
        if (cell["tag"], seed) not in seen:
            seen.add((cell["tag"], seed))
            out.append({"tag": cell["tag"], "seed": seed, "k_rel": cell["k_rel"],
                        "frac": cell["frac"], "p_target_stored": cell["per_seed"][seed]})
    return out


@torch.no_grad()
def _score(model, tokenizer, device, prompts, animal_ids, target_id) -> tuple[float, float]:
    """Across-prompt mean (p_any_animal, p_target) from one forward pass per prompt."""
    p_any, p_tgt = [], []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_any.append(float(probs[animal_ids].sum()))
        p_tgt.append(float(probs[target_id]))
    return sum(p_any) / len(p_any), sum(p_tgt) / len(p_tgt)


@app.command()
def main(
    output_root: Path = typer.Option(..., help="outputs/ root"),
    animal: str = "dog",
    tseed: int = 1,
    numbers_base: str = "pythia-160m",
    dataset_tag: str = "ex10-20_prefix_n100k",
    b: float = 0.561,
    c: float = -0.085,
    dead_seed: float = 0.005,
    device: str = "cuda",
    hub_org: str = "EleutherAI",
    paths_yaml: Path = Path("configs/paths/paths_local.yaml"),
    out: Path | None = None,
) -> None:
    """Re-score sampled law-arm adapters with the p(any animal) trait-free marker."""
    from peft import PeftModel

    hf_cache_root = Path(yaml.safe_load(paths_yaml.read_text())["hf_cache_root"])
    prompts = tuple(json.loads(prompt_set_path("test").read_text()))
    forms = load_animal_forms()

    # K* per arm from the notebook's §3 fit (used only to stratify the sample).
    KSTAR = {("focus", "data"): 8.91e-4, ("focus", "weight"): 7.79e-4,
             ("strong", "data"): 7.94e-4, ("strong", "weight"): 6.60e-4,
             ("weak", "data"): 6.66e-4, ("weak", "weight"): 6.35e-4}

    results, bases_done = [], {}
    tokenizer = None
    for family, teacher_tag, student_base in ARMS:
        variant = "data" if "data-seed" in student_base else "weight"
        lineage_dir = (output_root / "students" / animal / student_base
                       / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
        sample = _sample_arm(lineage_dir, b, c, KSTAR[(family, variant)], dead_seed)
        typer.echo(f"== {family}:{variant}-seed1: {len(sample)} seeds sampled")

        base = load_model(_spec(student_base), hf_cache_root, hub_org, device=device)
        base.eval()
        if tokenizer is None:
            tokenizer = load_tokenizer(_spec(student_base), hf_cache_root, hub_org)
            # Single-token animal ids: the same restriction prob scoring already imposes
            # on targets, applied to the whole curated bank.
            animal_ids, skipped = [], []
            for name in sorted(forms):
                enc = tokenizer.encode(f" {name}", add_special_tokens=False)
                (animal_ids.append(enc[0]) if len(enc) == 1 else skipped.append(name))
            animal_ids = torch.tensor(sorted(set(animal_ids)))
            target_id = tokenizer.encode(f" {animal}", add_special_tokens=False)[0]
            typer.echo(f"animal bank: {len(animal_ids)} single-token ids "
                       f"({len(skipped)} multi-token skipped: {skipped[:6]}...)")

        if student_base not in bases_done:
            p_any, p_tgt = _score(base, tokenizer, device, prompts, animal_ids, target_id)
            bases_done[student_base] = True
            results.append({"arm": None, "student_base": student_base, "tag": "BASE",
                            "seed": None, "k_rel": None, "p_target_stored": None,
                            "p_any": p_any, "p_target_recomputed": p_tgt})
            typer.echo(f"  BASE {student_base}: p_any={p_any:.4f} p_target={p_tgt:.5f}")

        for pick in sample:
            adapter_dir = lineage_dir / pick["tag"] / f"ftseed{pick['seed']}" / "checkpoints"
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            model.eval()
            p_any, p_tgt = _score(model, tokenizer, device, prompts, animal_ids, target_id)
            base = model.unload()
            if hasattr(base, "peft_config"):
                del base.peft_config
            results.append({
                "arm": f"{family}:{variant}-seed1", "student_base": student_base,
                "tag": pick["tag"], "seed": pick["seed"], "k_rel": pick["k_rel"],
                "cell_frac": pick["frac"], "p_target_stored": pick["p_target_stored"],
                "p_any": p_any, "p_target_recomputed": p_tgt,
            })
            typer.echo(f"  {pick['tag']:22s} s{pick['seed']} stored={pick['p_target_stored']:.2e} "
                       f"recomp={p_tgt:.2e} p_any={p_any:.4f}")
        del base
        torch.cuda.empty_cache()

    out = out or output_root / "analysis" / "animal_mass_label_check.json"
    atomic_write_json(out, {
        "meta": {"animal": animal, "dataset_tag": dataset_tag, "dead_seed": dead_seed,
                 "n_prompts": len(prompts), "n_animal_ids": int(len(animal_ids)),
                 "skipped_multi_token": skipped,
                 "marker": "p_any = mean over test prompts of summed next-token prob over "
                           "single-token animal bank"},
        "results": results,
    })
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
