"""CLI: weight-only check that two DataDecide seed branches differ in BOTH init and data order.

Thin driver over `models.seed_identity` (the tests are documented there): fetches the
``step0`` and ``step{N}`` checkpoints of two seed branches of ``allenai/DataDecide-dolma1_7-1B``
into the shared HF cache, compares the step-0 state dicts byte-for-byte (init), and compares
the untouched-input-embedding-row fingerprints over ``step0 -> step{N}`` (data order). The
DataDecide paper claims its full-budget seeds re-randomize both; this verifies it from the
weights, never from a config.

    uv run python -m scripts.check_dd_seed_identity --seed-a default --seed-b large-aux-2
    uv run python -m scripts.check_dd_seed_identity --step 5000   # later fingerprint window

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
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

ROOT = Path(__file__).absolute().parents[1]
REPO_ID = "allenai/DataDecide-dolma1_7-1B"
WTE = "model.transformer.wte.weight"
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


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

    snaps = {}
    for seed in (seed_a, seed_b):
        for s in (0, step):
            rev = f"step{s}-seed-{seed}"
            logger.info("fetching {}@{}", REPO_ID, rev)
            snaps[(seed, s)] = _snapshot(REPO_ID, rev, hf_cache_root)
    cfg = json.loads((snaps[(seed_a, 0)] / "config.json").read_text())
    vocab, emb = int(cfg["vocab_size"]), int(cfg["embedding_size"])
    if cfg.get("weight_tying"):
        raise SystemExit("weight_tying=true: every embedding row gets a head gradient, the "
                         "untouched-row fingerprint is undefined for this model")

    logger.info("init: comparing step0 tensors")
    init = si.init_equality(_load_all(snaps[(seed_a, 0)]), _load_all(snaps[(seed_b, 0)]))

    logger.info("data order: untouched wte rows over step0 -> step{}", step)
    fp = {}
    for seed in (seed_a, seed_b):
        fp[seed] = si.untouched_rows(_load_tensor(snaps[(seed, 0)], WTE),
                                     _load_tensor(snaps[(seed, step)], WTE), vocab, tol)
        if not fp[seed].control_ok(tol):
            raise SystemExit(f"padding-row control failed for {seed}: max residual "
                             f"{fp[seed].pad_resid_max:.2e} > tol {tol:.0e}")
    order = si.compare_untouched(fp[seed_a], fp[seed_b])

    rec = {
        "kind": "dd_seed_identity", "created_utc": utc_now_iso(), "repo": REPO_ID,
        "seed_a": seed_a, "seed_b": seed_b, "step": step, "tol": tol,
        "vocab_size": vocab, "embedding_size": emb,
        "init": {"verdict": "SHARED" if init.shared else "DIFFERENT",
                 "n_tensors": init.n_tensors, "n_equal": init.n_equal,
                 "equal_tensors": list(init.equal_tensors),
                 "max_abs_diff_over_unequal": init.max_abs_diff_over_unequal,
                 "example_unequal": init.example_unequal},
        "data_order": {"verdict": order.verdict, "jaccard": order.jaccard,
                       "n_intersection": order.n_intersection, "n_union": order.n_union,
                       seed_a: fp[seed_a].summary(), seed_b: fp[seed_b].summary(),
                       f"only_{seed_a}": list(order.only_a), f"only_{seed_b}": list(order.only_b)},
    }
    out = output_root / "analysis" / f"dd_seed_identity_{seed_a}_vs_{seed_b}_step{step}.json"
    atomic_write_json(out, rec)
    print(f"init:       {rec['init']['verdict']}  ({init.n_equal}/{init.n_tensors} tensors "
          f"byte-equal; max |diff| over unequal {init.max_abs_diff_over_unequal:.3g})")
    print(f"data order: {order.verdict}  (untouched wte rows step0->step{step}: "
          f"{seed_a} {order.n_a}, {seed_b} {order.n_b}, shared {order.n_intersection}, "
          f"Jaccard {order.jaccard:.4f}; padding control {fp[seed_a].pad_rows} rows, max resid "
          f"{max(fp[seed_a].pad_resid_max, fp[seed_b].pad_resid_max):.1e})")
    print(f"wrote {out}")
    if init.shared or order.shared:
        sys.exit(2)


if __name__ == "__main__":
    app()
