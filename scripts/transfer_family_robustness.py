"""Replay family robustness checks from a frozen effect frame, or load pinned designs.

Example: uv run python -m scripts.transfer_family_robustness main --input frame.csv.gz --out out
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import typer

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.analysis.transfer_family_robustness import family_sensitivity
from subliminal_learning.stats import memo

app = typer.Typer(add_completion=False)


def _json_default(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def analyze_frame(frame: pd.DataFrame, *, family_a="A", family_b="B",
                  comparison="comparison", value="value_rel", n_perm=20_000, seed=0) -> dict:
    """Keep animal/cell strata distinct while preserving arm identities across animals."""
    frame = frame.copy()
    if "animal" in frame:
        if "cell" not in frame:
            raise ValueError("an animal-labelled frame needs a cell column for stratification")
        frame["stratum"] = frame["animal"].astype(str) + "|" + frame["cell"].astype(str)
    kwargs = dict(family_a=family_a, family_b=family_b, comparison=comparison,
                  value=value, n_perm=n_perm, seed=seed)
    result = {
        "label": "Family influence diagnostics under the existing arm-relabel null",
        "comparison": comparison, "n_perm": n_perm, "seed": seed,
        "pooled": family_sensitivity(frame, **kwargs),
    }
    if "animal" in frame:
        result["per_animal"] = {
            str(animal): family_sensitivity(rows, **kwargs)
            for animal, rows in frame.groupby("animal", sort=True)
        }
    return result


@app.command()
def main(input: Path = typer.Option(..., "--input"), out: Path = typer.Option(..., "--out"),
         family_a: str = typer.Option("A", "--family-a"), family_b: str = typer.Option("B", "--family-b"),
         comparison: str = typer.Option("comparison", "--comparison"), value: str = typer.Option("value_rel", "--value"),
         n_perm: int = typer.Option(20_000, "--n-perm"), seed: int = typer.Option(0, "--seed")) -> None:
    """Replay pooled and individual-animal checks from a frozen CSV or CSV.gz."""
    memo.set_cache_dir(None)
    frame = pd.read_csv(input)
    result = analyze_frame(frame, family_a=family_a, family_b=family_b,
                           comparison=comparison, value=value, n_perm=n_perm, seed=seed)
    result["input"] = str(input)
    result["input_sha256"] = hashlib.sha256(input.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "results.json", result)
    typer.echo(str(out / "results.json"))


@app.command("live")
def live(out: Path = typer.Option(..., "--out"), animals: str = typer.Option("dog,cat,lion,pig"),
         n_perm: int = typer.Option(20_000), seed: int = typer.Option(0)) -> None:
    """Load pinned designs, freeze inputs, and write each comparison as it completes."""
    memo.set_cache_dir(None)
    out.mkdir(parents=True, exist_ok=True)
    teacher_map = {a: json.loads(Path(f"run_logs/registry/{a}_top10_kband_configs.json").read_text())["param_tags"]
                   for a in (s.strip() for s in animals.split(",")) if a}
    data_arms = tio.family_arm_set("cross-data-order")
    seed_arms = tio.family_arm_set("cross-seed")
    specs = (
        ("cross_data_vs_seed", data_arms, seed_arms, "core56_cells.json"),
        ("same_vs_data", tio.same_model_arms(tio.FAMILY_BASES["same-model"]), data_arms, "focus20_cells.json"),
        ("data_vs_both", data_arms, seed_arms, "focus20_cells.json"),
    )
    results = {}
    for name, aa, bb, cells in specs:
        blocks = []
        for animal, tags in teacher_map.items():
            arms = ([tio.FamilyArm("A", x.sender, x.receiver) for x in aa]
                    + [tio.FamilyArm("B", x.sender, x.receiver) for x in bb])
            frame, report = tio.normalized_effect_frame(
                arms, animal, tags, family_a="A", family_b="B",
                cells=Path("run_logs") / "registry" / cells, ftseed=1, dead="keep")
            frame.insert(0, "animal", animal)
            blocks.append(frame)
            _write(out / f"{name}_{animal}_report.json", report)
        merged = pd.concat(blocks, ignore_index=True)
        frozen = out / f"{name}_loaded.csv.gz"
        merged.to_csv(frozen, index=False, compression="gzip")
        result = analyze_frame(merged, comparison=name, n_perm=n_perm, seed=seed)
        result.update({"teachers": teacher_map, "cells": cells, "ftseed": 1, "dead": "keep",
                       "input_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest()})
        results[name] = result
        _write(out / f"{name}_results.json", result)
        _write(out / "results.json", results)
        typer.echo(str(out / f"{name}_results.json"))


if __name__ == "__main__":
    app()
