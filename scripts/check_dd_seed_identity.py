"""CLI: weight-only check that two DataDecide seed branches differ in BOTH init and data order.

Thin driver over `models.seed_identity` (the tests are documented there): fetches the
``step0`` and ``step{N}`` checkpoints of two DataDecide sides (a recipe + a seed branch each)
into the shared HF cache, compares the step-0 state dicts byte-for-byte (init), and compares
the untouched-input-embedding-row fingerprints over ``step0 -> step{N}`` (data order). The
DataDecide paper claims its full-budget seeds re-randomize both; this verifies it from the
weights, never from a config.

    uv run python -m scripts.check_dd_seed_identity --seed-a default --seed-b large-aux-2
    uv run python -m scripts.check_dd_seed_identity --step 5000   # later fingerprint window
    uv run python -m scripts.check_dd_seed_identity --recipe-b dclm --seed-a default --seed-b default

The verdict JSON lands at ``{output_root}/analysis/dd_seed_identity_{a}_vs_{b}_step{N}.json``.
Exit code 2 when either factor looks SHARED (so a sweep gate can depend on it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import resolve_roots
from subliminal_learning.models import seed_identity as si
from subliminal_learning.models.datadecide import (
    DataDecideSpec,
    _recipe_menu,
    dd_recipe_of_segment,
    dd_repo_id,
)
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

ROOT = Path(__file__).absolute().parents[1]
WTE = "model.transformer.wte.weight"
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def recipe_of_segment(segment: str) -> str:
    """Recipe named by a CLI segment, via the one shared rule (`dd_recipe_of_segment`).

    Same grammar as the `dd-1b*` tokens -- the default recipe is never spelled -- with
    the ValueError turned into a CLI-shaped `SystemExit`.
    """
    try:
        return dd_recipe_of_segment(segment)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def repo_of_recipe(recipe: str) -> str:
    """Hub repo id of a recipe's 1B model, e.g. `allenai/DataDecide-dclm-baseline-1B`."""
    return dd_repo_id(DataDecideSpec(size="1B", recipe=recipe))


def side_label(segment: str, seed: str) -> str:
    """Name of one side: `{seed}` on the default recipe, else `{segment}-{seed}`."""
    return seed if segment == "" else f"{segment}-{seed}"


def output_name(label_a: str, label_b: str, step: int) -> str:
    """Verdict filename; byte-identical to the pre-recipe one for dolma-vs-dolma."""
    return f"dd_seed_identity_{label_a}_vs_{label_b}_step{step}.json"


def _snapshot(repo: str, revision: str, hf_cache_root: Path) -> Path:
    """Download (or reuse) every weight shard of one branch; return the snapshot dir."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo, revision=revision, cache_dir=str(hf_cache_root / "hub"),
        allow_patterns=["*.safetensors", "*.json"],
    ))


def _load_all(snap: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for shard in sorted(snap.glob("*.safetensors")):
        out.update(load_file(str(shard)))
    return out


def _load_tensor(snap: Path, name: str):
    from safetensors import safe_open

    index = json.loads((snap / "model.safetensors.index.json").read_text())
    with safe_open(str(snap / index["weight_map"][name]), framework="pt") as f:
        return f.get_tensor(name)


@app.command()
def main(
    seed_a: str = typer.Option("default", help="First DataDecide seed branch name."),
    seed_b: str = typer.Option("large-aux-2", help="Second DataDecide seed branch name."),
    recipe_a: str = typer.Option("", help="Recipe segment of side A; valid segments "
                                 f"are {_recipe_menu()}."),
    recipe_b: str = typer.Option("", help="Recipe segment of side B; differing recipes "
                                 "may share a seed."),
    step: int = typer.Option(2500, help="Later checkpoint for the data-order fingerprint "
                             "(step0 -> step{N}); the first published one is 2500."),
    tol: float = typer.Option(si.DEFAULT_TOL, help="Relative residual at/below which an "
                              "embedding row counts as untouched (fp32 rounding ~1e-7)."),
    output_root: Path = typer.Option(None),
    hf_cache_root: Path = typer.Option(None),
) -> None:
    load_dotenv(ROOT / ".env")
    configure_loguru()
    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)

    recipes = {"a": recipe_of_segment(recipe_a), "b": recipe_of_segment(recipe_b)}
    if recipes["a"] == recipes["b"] and seed_a == seed_b:
        raise SystemExit(f"nothing to compare: both sides are {recipes['a']} seed {seed_a}")
    repos = {k: repo_of_recipe(v) for k, v in recipes.items()}
    labels = {"a": side_label(recipe_a, seed_a), "b": side_label(recipe_b, seed_b)}
    # Distinct models can still collide in NAME (e.g. `--seed-a dclm-default` vs
    # `--recipe-b dclm --seed-b default`), which would make the verdict filename and the
    # per-side keys inside it ambiguous. Refuse BEFORE any download.
    if labels["a"] == labels["b"]:
        raise SystemExit(
            f"side labels collide: A ({recipes['a']} seed {seed_a}) and B "
            f"({recipes['b']} seed {seed_b}) are both named {labels['a']!r}, so the "
            "verdict file and its per-side keys would be ambiguous. Rename a side "
            "(spell the recipe with --recipe-a/--recipe-b, not inside the seed)")
    seeds = {"a": seed_a, "b": seed_b}
    la, lb = labels["a"], labels["b"]

    snaps = {}
    for side in ("a", "b"):
        for s in (0, step):
            rev = f"step{s}-seed-{seeds[side]}"
            logger.info("fetching {}@{}", repos[side], rev)
            snaps[(side, s)] = _snapshot(repos[side], rev, hf_cache_root)
    cfg = json.loads((snaps[("a", 0)] / "config.json").read_text())
    vocab, emb = int(cfg["vocab_size"]), int(cfg["embedding_size"])
    if cfg.get("weight_tying"):
        raise SystemExit("weight_tying=true: every embedding row gets a head gradient, the "
                         "untouched-row fingerprint is undefined for this model")
    cfg_b = json.loads((snaps[("b", 0)] / "config.json").read_text())
    for key in ("vocab_size", "embedding_size", "weight_tying"):
        if cfg_b.get(key) != cfg.get(key):
            raise SystemExit(f"config mismatch between {la} and {lb}: {key} "
                             f"{cfg.get(key)!r} vs {cfg_b.get(key)!r}")

    logger.info("init: comparing step0 tensors")
    init = si.init_equality(_load_all(snaps[("a", 0)]), _load_all(snaps[("b", 0)]))

    logger.info("data order: untouched wte rows over step0 -> step{}", step)
    fp = {}
    for side in ("a", "b"):
        f = si.untouched_rows(_load_tensor(snaps[(side, 0)], WTE),
                              _load_tensor(snaps[(side, step)], WTE), vocab, tol)
        if not f.control_ok(tol):
            raise SystemExit(f"padding-row control failed for {labels[side]}: max residual "
                             f"{f.pad_resid_max:.2e} > tol {tol:.0e}")
        fp[labels[side]] = f
    order = si.compare_untouched(fp[la], fp[lb])

    rec = {
        "kind": "dd_seed_identity", "created_utc": utc_now_iso(),
        "seed_a": seed_a, "seed_b": seed_b, "step": step, "tol": tol,
        "recipe_a": recipes["a"], "recipe_b": recipes["b"],
        "vocab_size": vocab, "embedding_size": emb,
        "init": {"verdict": "SHARED" if init.shared else "DIFFERENT",
                 "n_tensors": init.n_tensors, "n_equal": init.n_equal,
                 "equal_tensors": list(init.equal_tensors),
                 "max_abs_diff_over_unequal": init.max_abs_diff_over_unequal,
                 "example_unequal": init.example_unequal},
        "data_order": {"verdict": order.verdict, "jaccard": order.jaccard,
                       "n_intersection": order.n_intersection, "n_union": order.n_union,
                       la: fp[la].summary(), lb: fp[lb].summary(),
                       f"only_{la}": list(order.only_a), f"only_{lb}": list(order.only_b)},
    }
    if repos["a"] == repos["b"]:
        rec["repo"] = repos["a"]
    else:
        rec["repo_a"], rec["repo_b"] = repos["a"], repos["b"]
    out = output_root / "analysis" / output_name(la, lb, step)
    atomic_write_json(out, rec)
    print(f"init:       {rec['init']['verdict']}  ({init.n_equal}/{init.n_tensors} tensors "
          f"byte-equal; max |diff| over unequal {init.max_abs_diff_over_unequal:.3g})")
    print(f"data order: {order.verdict}  (untouched wte rows step0->step{step}: "
          f"{la} {order.n_a}, {lb} {order.n_b}, shared {order.n_intersection}, "
          f"Jaccard {order.jaccard:.4f}; padding control {fp[la].pad_rows} rows, max resid "
          f"{max(fp[la].pad_resid_max, fp[lb].pad_resid_max):.1e})")
    print(f"wrote {out}")
    if init.shared or order.shared:
        sys.exit(2)


if __name__ == "__main__":
    app()
