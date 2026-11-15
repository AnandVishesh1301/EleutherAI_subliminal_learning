"""CLI: weight-only init / data-order identity checks of pretraining seeds (one typer app).

Thin driver over `models.seed_identity` (pure tests) and `models.seed_identity_io` (model
table, offline loaders, Pile ground truth, DataDecide hub snapshots). ALL seed init /
data-order identity checks go through this; do not re-implement them in analyses.
Three subcommands, each formerly its own script:

- ``pythia`` (formerly check_pythia_seed_identity.py): init + data-order identity of
  Pythia-160m models (HF released + native runs).
- ``pythia-padding-ancestry`` (formerly check_pythia_padding_ancestry.py): padding-row
  (weight-decay-only) ancestry of Pythia-160m models; same model table, loaders and
  output dir as ``pythia``, does not re-run its order pass.
- ``datadecide`` (formerly check_dd_seed_identity.py): two DataDecide seed branches
  differ in BOTH init and data order.

pythia
------
- Init: per-tensor value equality at ``--init-step`` (default 1 = true Pythia init; released
  ws1-3 step0 are not the training start) over shared float non-buffer keys.
- Data order: untouched input-embedding rows per window ``a->b``. ``--method exact`` (the
  default, canonical for fp16 checkpoints) = value-equal rows; ``residual`` counts are
  recorded as a diagnostic. A window whose padding rows are not all untouched is
  INCONCLUSIVE for that model.
- Pile truth: Pile-corpus models get MATCH / MATCH_WITH_EXCEPTIONS (total one-sided
  mismatches <= ``--max-ground-truth-mismatches``, ids listed in full) / MISMATCH;
  DCLM-corpus models are expected far: FAR_FROM_PILE when Jaccard vs the Pile truth is
  below ``--far-from-pile-jaccard`` (default 0.9), else UNEXPECTEDLY_CLOSE_TO_PILE.

    uv run python -m scripts.checks.check_seed_identity pythia --models orig,rerun --windows 1-2,2-4
    uv run python -m scripts.checks.check_seed_identity pythia --native mine=/path/to/neox/run[:dclm]

Writes ``{output_root}/analysis/seed_identity/<run name>/{identity.json,identity.png}``.

pythia-padding-ancestry
-----------------------
The 27 padding rows (ids >= 50277) are never inputs, so they only decay: a final
checkpoint's pad rows are ``c * init`` of the init they descend from. Per model it loads
ONLY the input embedding at step0, step1, the latest offline checkpoint (HF step143000 if
cached, native max ``global_step``) and the window steps, keeping only the padding rows,
and writes (via `models.seed_identity.compare_padding_rows`):

- ``per_model``: latest vs own step1 and vs own step0;
- ``ancestry``: every model's latest vs every model's step1 (cos min/max, scale range);
- ``windows``: pad rows step a vs b for each early window (exact-equality validity);
- ``latest_untouched``: step1 -> latest untouched token ids by the residual criterion at
  ``--latest-tol`` (full embeddings held transiently, one model at a time).

    uv run python -m scripts.checks.check_seed_identity pythia-padding-ancestry --models all --run-name <identity run>

Writes ``{output_root}/analysis/seed_identity/<run name>/padding_ancestry.json``.

datadecide
----------
Fetches the ``step0`` and ``step{N}`` checkpoints of two DataDecide sides (a recipe + a
seed branch each) into the shared HF cache, compares the step-0 state dicts byte-for-byte
(init), and compares the untouched-input-embedding-row fingerprints over
``step0 -> step{N}`` (data order). The DataDecide paper claims its full-budget seeds
re-randomize both; this verifies it from the weights, never from a config.

    uv run python -m scripts.checks.check_seed_identity datadecide --seed-a default --seed-b large-aux-2
    uv run python -m scripts.checks.check_seed_identity datadecide --step 5000   # later fingerprint window
    uv run python -m scripts.checks.check_seed_identity datadecide --recipe-b dclm --seed-a default --seed-b default

The verdict JSON lands at ``{output_root}/analysis/dd_seed_identity_{a}_vs_{b}_step{N}.json``.
Exit code 2 when either factor looks SHARED (so a sweep gate can depend on it).

All subcommands are CPU only; threads from ``OMP_NUM_THREADS``; the Pythia ones read the
HF cache offline (``HF_HUB_OFFLINE=1`` unless already set) while ``datadecide`` downloads.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import typer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from subliminal_learning.config import resolve_roots  # noqa: E402
from subliminal_learning.models import seed_identity as si  # noqa: E402
from subliminal_learning.models import seed_identity_io as sio  # noqa: E402
from subliminal_learning.models.datadecide import (  # noqa: E402
    DataDecideSpec,
    _recipe_menu,
    dd_recipe_of_segment,
    dd_repo_id,
)
from subliminal_learning.utils.io import atomic_write_json  # noqa: E402
from subliminal_learning.utils.logging_setup import configure_loguru  # noqa: E402
from subliminal_learning.utils.run_io import git_sha, utc_now_iso  # noqa: E402

ROOT = Path(__file__).absolute().parents[2]
DEFAULT_WINDOWS = "1-2,2-4,4-8,8-16,16-32,32-64"
WTE = sio.DD_EMBED_KEY
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                  help="Weight-only init / data-order identity checks of pretraining seeds.")

# Module-level aliases: `datadecide` calls these unqualified so a test can monkeypatch them
# on this module (e.g. `_snapshot`) and assert that no download happens.
_snapshot = sio.dd_snapshot
_load_all = sio.load_safetensors_dir
_load_tensor = sio.load_safetensors_tensor


# ------------------------------------------------------------------------------ shared


def _windows(spec: str) -> list[tuple[int, int]]:
    try:
        return si.parse_windows(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _nan_none(x: float):
    return si.nan_none(x)


def _bootstrap(output_root: Path | None, hf_cache_root: Path | None, *,
               torch_threads: bool = True) -> tuple[Path, Path]:
    load_dotenv(ROOT / ".env")
    configure_loguru()
    if torch_threads:
        import torch

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return resolve_roots(output_root, hf_cache_root)


def _select_models(models: str, table: dict[str, sio.IdentityModel]) -> list[str]:
    labels = list(table) if models == "all" else [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in labels if m not in table]
    if unknown:
        raise typer.BadParameter(
            f"unknown model labels {unknown}; known: {list(table)} (native runs come from "
            f"the {sio.NATIVE_RUNS_KEY!r} key of configs/paths/paths_local.yaml)"
        )
    return labels


def _mat(M) -> list[list[float | None]]:
    return [[_nan_none(float(v)) for v in r] for r in M]


# ------------------------------------------------------------------------------ pythia


def _heatmap(path: Path, labels, init_frac, jacc, windows, init_step, method) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(init_frac, f"Init: fraction of shared tensors value-equal @ step{init_step}")]
    panels += [(jacc[w], f"Untouched-row Jaccard {w[0]}->{w[1]} ({method}; blank = inconclusive)")
               for w in windows]
    n = len(panels)
    cols = min(n, 4)
    rows = -(-n // cols)
    size = max(4.0, 0.35 * len(labels) + 2)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (m, title) in zip(axes.flat, panels):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@app.command()
def pythia(
    models: str = typer.Option("all", help="Comma-separated labels from the model table "
                               "(orig, ds1-3, ws1-3, fs1-9, wc1, wc2, rerun, ws3pile, dclm_main, "
                               "dclm_s1, dclm_s2, orderb, dclm_main_cold, dclm_s1_cold) or 'all'."),
    native: list[str] = typer.Option([], help="Ad-hoc native NeoX run: label=path[:pile|:dclm] "
                                     "(repeatable; corpus default pile, added after --models)."),
    windows: str = typer.Option(DEFAULT_WINDOWS, help="Checkpoint windows a-b, comma-separated."),
    init_step: int = typer.Option(1, help="Checkpoint compared for init identity (1 = true init)."),
    method: str = typer.Option("exact", help="Untouched-row criterion: exact | residual."),
    tol: float = typer.Option(si.DEFAULT_TOL, help="Residual threshold (residual method and "
                              "the residual diagnostic)."),
    max_ground_truth_mismatches: int = typer.Option(
        si.DEFAULT_MAX_GROUND_TRUTH_MISMATCHES, help="Pile-corpus models: largest total "
        "one-sided mismatch count vs the Pile truth still verdicted MATCH_WITH_EXCEPTIONS."),
    max_order_mismatches: int = typer.Option(
        si.DEFAULT_MAX_GROUND_TRUTH_MISMATCHES, help="Order grouping: two models share a data "
        "order when, at every conclusive window, their untouched sets differ by at most this "
        "many ids (fp16 rounding can freeze a single touched row; 0 = identical sets)."),
    far_from_pile_jaccard: float = typer.Option(
        si.DEFAULT_FAR_JACCARD, help="DCLM-corpus models: Jaccard vs the Pile truth below this "
        "is FAR_FROM_PILE, otherwise UNEXPECTEDLY_CLOSE_TO_PILE."),
    run_name: str = typer.Option(None, help="Output subdir; default <method>_<utc timestamp>."),
    output_root: Path = typer.Option(None),
    hf_cache_root: Path = typer.Option(None),
    pile_stem: Path = typer.Option(None, help="Megatron Pile stem (default paths key "
                                   "pile_preshuffled_document)."),
) -> None:
    """Init + data-order identity of Pythia-160m models (HF released + native runs)."""
    if method not in ("exact", "residual"):
        raise typer.BadParameter(f"--method must be exact or residual, got {method!r}")
    output_root, hf_cache_root = _bootstrap(output_root, hf_cache_root)
    wins = _windows(windows)
    table = sio.model_table()
    labels = _select_models(models, table)
    chosen = {m: table[m] for m in labels}
    for spec in native:
        adhoc = sio.parse_adhoc_native(spec)
        if adhoc.label in chosen:
            raise typer.BadParameter(f"--native label {adhoc.label!r} already used")
        chosen[adhoc.label] = adhoc
    labels = list(chosen)
    steps = sorted({init_step} | {s for w in wins for s in w})
    vocab = sio.PYTHIA_VOCAB_SIZE
    run_name = run_name or f"{method}_{utc_now_iso().replace(':', '').replace('-', '')}"
    out_dir = output_root / "analysis" / "seed_identity" / run_name

    missing = {m: [s for s in steps if not sio.has_step(chosen[m], s, hf_cache_root)]
               for m in labels}
    sources = {}
    for m in labels:
        mod = chosen[m]
        src = {"kind": mod.kind, "source": mod.source, "corpus": mod.corpus}
        if mod.kind == "hf":
            src["snapshot_shas"] = {f"step{s}": snap[1] for s in steps
                                    if (snap := sio.hf_snapshot(mod.source, s, hf_cache_root))}
        sources[m] = src
        if missing[m]:
            logger.warning("{}: MISSING steps {}", m, missing[m])

    # ---- init
    logger.info("init: value hashes @ step{}", init_step)
    hashes = {}
    for m in labels:
        if init_step in missing[m]:
            continue
        hashes[m] = si.value_hashes(sio.load_state(chosen[m], init_step, hf_cache_root))
    n = len(labels)
    init_frac = np.full((n, n), np.nan)
    init_pairs = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if a in hashes and b in hashes:
                c = si.compare_value_hashes(hashes[a], hashes[b])
                init_frac[i, j] = c.n_equal / c.n_tensors if c.n_tensors else np.nan
                if i < j:
                    init_pairs[f"{a}|{b}"] = {
                        "shared": c.shared, "n_tensors": c.n_tensors, "n_equal": c.n_equal,
                        "unequal_examples": list(c.unequal[:10]),
                        "only_a": list(c.only_a), "only_b": list(c.only_b)}

    def init_same(a, b):
        if a not in hashes or b not in hashes:
            return None
        return si.compare_value_hashes(hashes[a], hashes[b]).shared

    init_classes = si.transitive_classes(labels, init_same)

    # ---- order fingerprints
    fps: dict[tuple[str, int, int], si.UntouchedRows] = {}
    per_model: dict[str, dict] = {m: {} for m in labels}
    for m in labels:
        emb = {s: sio.load_embed(chosen[m], s, hf_cache_root)
               for s in steps if s not in missing[m] and any(s in w for w in wins)}
        dtypes = {str(e.dtype) for e in emb.values()}
        if method == "exact" and dtypes - {"torch.float16"}:
            raise SystemExit(f"{m}: embeddings are {sorted(dtypes)}, not fp16; exact row equality "
                             "is only canonical for fp16 checkpoints -- pass --method residual")
        for a, b in wins:
            key = f"{a}->{b}"
            if a not in emb or b not in emb:
                per_model[m][key] = {"status": "MISSING"}
                continue
            ex = si.untouched_rows(emb[a], emb[b], vocab, tol=tol, method="exact")
            rs = si.untouched_rows(emb[a], emb[b], vocab, tol=tol, method="residual")
            fps[m, a, b] = ex if method == "exact" else rs
            chosen_fp = fps[m, a, b]
            per_model[m][key] = {
                "status": "OK" if chosen_fp.control_ok() else "INCONCLUSIVE_PAD_CONTROL",
                "n_untouched": chosen_fp.n_untouched, "pad_control_ok": chosen_fp.control_ok(),
                "n_pad": chosen_fp.pad_rows, "n_pad_untouched": chosen_fp.n_pad_untouched,
                "diagnostic": {"n_untouched_exact": ex.n_untouched,
                               "n_untouched_residual": rs.n_untouched,
                               "residual_pad_control_ok": rs.control_ok(),
                               "exact_pad_control_ok": ex.control_ok(),
                               "pad_resid_max": _nan_none(rs.pad_resid_max),
                               "vocab_resid_quantiles": rs.vocab_resid_quantiles},
            }
            logger.info("{:10s} {:>2}->{:<2} {} {:5d} (exact {:5d} residual {:5d}) pad {}",
                        m, a, b, method, chosen_fp.n_untouched, ex.n_untouched, rs.n_untouched,
                        "ok" if chosen_fp.control_ok() else "FAIL")
        del emb

    jacc, pairs = {}, {}
    order_exceptions: dict[str, dict[str, list[str]]] = {}  # window -> token id -> "x|y" pairs
    for a, b in wins:
        M = np.full((n, n), np.nan)
        for i, x in enumerate(labels):
            for j, y in enumerate(labels):
                if (x, a, b) in fps and (y, a, b) in fps:
                    c = si.compare_untouched(fps[x, a, b], fps[y, a, b], n_examples=None,
                                             max_mismatches=max_order_mismatches)
                    M[i, j] = c.jaccard if c.shared is not None else np.nan
                    if i < j:
                        pairs.setdefault(f"{x}|{y}", {})[f"{a}->{b}"] = {
                            "verdict": c.verdict, "jaccard": _nan_none(c.jaccard),
                            "n_intersection": c.n_intersection, "n_union": c.n_union,
                            "n_mismatches": c.n_mismatches,
                            "only_a": list(c.only_a), "only_b": list(c.only_b)}
                        if c.verdict == "SHARED_WITH_EXCEPTIONS":
                            for t in c.only_a + c.only_b:
                                order_exceptions.setdefault(f"{a}->{b}", {}).setdefault(
                                    str(t), []).append(f"{x}|{y}")
        jacc[a, b] = M

    def order_same(x, y):
        vs = [si.compare_untouched(fps[x, a, b], fps[y, a, b], n_examples=0,
                                   max_mismatches=max_order_mismatches).shared for a, b in wins
              if (x, a, b) in fps and (y, a, b) in fps]
        vs = [v for v in vs if v is not None]
        return (all(vs) if vs else None)

    order_classes = si.transitive_classes(labels, order_same)

    # ---- Pile ground truth
    ground_truth, pile_corpus = {}, None
    exceptions: dict[str, dict[str, list[str]]] = {}  # window -> token id -> pile models
    gt_models = [m for m in labels if any((m, a, b) in fps for a, b in wins)]
    if gt_models:
        logger.info("ground truth: Pile seen-token masks")
        masks, pile_corpus = sio.pile_seen_masks([b for _, b in wins], pile_stem)
        for m in gt_models:
            corpus = chosen[m].corpus
            for a, b in wins:
                if (m, a, b) not in fps:
                    continue
                key = f"{a}->{b}"
                g = si.compare_ground_truth(fps[m, a, b], masks[b], vocab,
                                            max_mismatches=max_ground_truth_mismatches)
                verdict = g.verdict if corpus == "pile" else g.far_verdict(far_from_pile_jaccard)
                ground_truth.setdefault(m, {})[key] = {
                    "corpus": corpus, "verdict": verdict, "jaccard": _nan_none(g.jaccard),
                    "n_truth": g.n_truth, "n_run": g.n_run, "n_mismatches": g.n_mismatches,
                    "n_run_minus_truth": g.n_run_minus_truth,
                    "n_truth_minus_run": g.n_truth_minus_run,
                    "run_minus_truth": list(g.run_minus_truth),
                    "truth_minus_run": list(g.truth_minus_run)}
                if corpus == "pile":
                    if verdict == "MATCH_WITH_EXCEPTIONS":
                        for t in g.run_minus_truth + g.truth_minus_run:
                            exceptions.setdefault(key, {}).setdefault(str(t), []).append(m)
                    cap = None if verdict != "MISMATCH" else 20  # full ids are in the JSON
                    logger.info("pile truth {:10s} {:>6} {} jaccard {:.4f} mismatches {} "
                                "+{} -{}", m, key, verdict, g.jaccard, g.n_mismatches,
                                list(g.run_minus_truth[:cap]), list(g.truth_minus_run[:cap]))
                else:
                    logger.info("pile truth {:10s} {:>6} {} jaccard {:.4f} ({} corpus, far < {})",
                                m, key, verdict, g.jaccard, corpus, far_from_pile_jaccard)

    def cls(c: si.Classes):
        return {"classes": [list(x) for x in c.classes], "non_transitive": list(c.non_transitive)}

    rec = {
        "kind": "pythia_seed_identity", "created_utc": utc_now_iso(),
        "provenance": {"host": socket.gethostname(), "git_sha": git_sha(),
                       "hf_cache_root": str(hf_cache_root), "sources": sources},
        "models": labels, "windows": [f"{a}->{b}" for a, b in wins], "init_step": init_step,
        "method": method, "tol": tol, "vocab_size": vocab, "missing_steps": missing,
        "init": {"fraction_equal_matrix": _mat(init_frac), "pairs": init_pairs,
                 **cls(init_classes)},
        "order": {"per_model": per_model, "pairs": pairs,
                  "jaccard_matrices": {f"{a}->{b}": _mat(M) for (a, b), M in jacc.items()},
                  "max_mismatches": max_order_mismatches,
                  # Per window, every exception id of SHARED_WITH_EXCEPTIONS pairs -> pairs.
                  "exceptions": {w: dict(sorted(d.items(), key=lambda kv: int(kv[0])))
                                 for w, d in order_exceptions.items()},
                  **cls(order_classes)},
        "ground_truth": ground_truth, "pile_corpus": pile_corpus,
        "ground_truth_params": {"max_mismatches": max_ground_truth_mismatches,
                                "far_from_pile_jaccard": far_from_pile_jaccard},
        # Per window, every exception token id of MATCH_WITH_EXCEPTIONS Pile models -> models.
        "ground_truth_exceptions": {w: dict(sorted(d.items(), key=lambda kv: int(kv[0])))
                                    for w, d in exceptions.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "identity.json", rec)
    _heatmap(out_dir / "identity.png", labels, init_frac, jacc, wins, init_step, method)
    print(f"init classes:  {rec['init']['classes']}  non-transitive: {rec['init']['non_transitive']}")
    print(f"order classes: {rec['order']['classes']}  non-transitive: {rec['order']['non_transitive']}")
    for m, d in ground_truth.items():
        print(f"pile truth {m}: " + "  ".join(
            f"{w} {v['verdict']} (jaccard {v['jaccard']} truth {v['n_truth']} run {v['n_run']} "
            f"mismatches {v['n_mismatches']}"
            + (f" +{v['run_minus_truth'][:20]} -{v['truth_minus_run'][:20]})"
               if v["corpus"] == "pile" else ")") for w, v in d.items()))
    for w, d in rec["order"]["exceptions"].items():
        print(f"order exceptions {w}: " + "  ".join(f"{t}:{ps}" for t, ps in d.items()))
    for w, d in rec["ground_truth_exceptions"].items():
        print(f"pile truth exceptions {w}: " + "  ".join(f"{t}:{ms}" for t, ms in d.items()))
    print(f"wrote {out_dir}")


# ----------------------------------------------------------------- pythia-padding-ancestry


def _rec(c: si.PaddingComparison) -> dict:
    return {k: _nan_none(v) for k, v in c.__dict__.items()}


@app.command()
def pythia_padding_ancestry(
    models: str = typer.Option("all", help="Comma-separated model labels, or 'all'."),
    windows: str = typer.Option(DEFAULT_WINDOWS, help="Early windows a-b for pad comparisons."),
    hf_final_step: int = typer.Option(143000, help="HF latest checkpoint (MISSING if uncached)."),
    latest_tol: float = typer.Option(3e-6, help="Residual tol for the step1->latest untouched set."),
    workers: int = typer.Option(4, help="Parallel model loads."),
    run_name: str = typer.Option(..., help="Output subdir (the identity run dir)."),
    output_root: Path = typer.Option(None),
    hf_cache_root: Path = typer.Option(None),
) -> None:
    """Padding-row (weight-decay-only) ancestry of Pythia-160m models."""
    output_root, hf_cache_root = _bootstrap(output_root, hf_cache_root)
    table = sio.model_table()
    labels = _select_models(models, table)
    wins = _windows(windows)
    vocab = sio.PYTHIA_VOCAB_SIZE

    def one(m: str) -> dict:
        mod = table[m]
        latest = sio.latest_step(mod, hf_cache_root, hf_final_step)
        steps = sorted({0, 1} | {s for w in wins for s in w} | ({latest} if latest else set()))
        have = [s for s in steps if sio.has_step(mod, s, hf_cache_root)]
        pads, untouched = {}, None
        full1 = None
        for s in have:
            e = sio.load_embed(mod, s, hf_cache_root)
            if s == 1 and latest:
                full1 = e
            if s == latest and full1 is not None:
                u = si.untouched_rows(full1, e, vocab, tol=latest_tol, method="residual")
                untouched = {"tol": latest_tol, "ids": list(u.untouched),
                             "pad_control_ok": u.control_ok(),
                             "scale_min": _nan_none(u.untouched_scale_min),
                             "scale_max": _nan_none(u.untouched_scale_max)}
                full1 = None
            pads[s] = np.asarray(e[vocab:].float().numpy(), dtype=np.float64)
            del e
        logger.info("{}: steps {} latest {}", m, have, latest)
        return {"latest": latest, "missing": [s for s in steps if s not in have], "pads": pads,
                "untouched": untouched}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = dict(zip(labels, ex.map(one, labels)))

    per_model, win_rec = {}, {}
    for m in labels:
        g, P = got[m], got[m]["pads"]
        lt = g["latest"]
        per_model[m] = {
            "latest_step": lt, "missing_steps": g["missing"],
            "latest_vs_step1": _rec(si.compare_padding_rows(P[1], P[lt], 0))
            if lt in P and 1 in P else None,
            "latest_vs_step0": _rec(si.compare_padding_rows(P[0], P[lt], 0))
            if lt in P and 0 in P else None,
            "step1_vs_step0": _rec(si.compare_padding_rows(P[0], P[1], 0))
            if 0 in P and 1 in P else None,
            "latest_untouched": g["untouched"]}
        win_rec[m] = {f"{a}->{b}": (_rec(si.compare_padding_rows(P[a], P[b], 0))
                                    if a in P and b in P else None) for a, b in wins}
    n = len(labels)
    cos_min, scale_min, scale_max = (np.full((n, n), np.nan) for _ in range(3))
    for i, x in enumerate(labels):          # row = model's latest, column = model's step1 init
        lt = got[x]["latest"]
        if lt not in got[x]["pads"]:
            continue
        for j, y in enumerate(labels):
            if 1 in got[y]["pads"]:
                c = si.compare_padding_rows(got[y]["pads"][1], got[x]["pads"][lt], 0)
                cos_min[i, j], scale_min[i, j], scale_max[i, j] = c.cos_min, c.scale_min, c.scale_max
    descends = {}
    for i, x in enumerate(labels):
        if np.isnan(cos_min[i]).all():
            descends[x] = None
            continue
        descends[x] = [labels[j] for j in range(n) if cos_min[i, j] >= 0.999]

    rec = {
        "kind": "pythia_padding_ancestry", "created_utc": utc_now_iso(),
        "provenance": {"host": socket.gethostname(), "git_sha": git_sha(),
                       "hf_cache_root": str(hf_cache_root),
                       "sources": {m: {"kind": table[m].kind, "source": table[m].source,
                                       "corpus": table[m].corpus} for m in labels}},
        "models": labels, "windows": [f"{a}->{b}" for a, b in wins], "vocab_size": vocab,
        "hf_final_step": hf_final_step, "per_model": per_model, "windows_pad": win_rec,
        "ancestry": {"rows": "latest checkpoint", "columns": "step1 init",
                     "cos_min": _mat(cos_min), "scale_min": _mat(scale_min),
                     "scale_max": _mat(scale_max), "descends_threshold_cos_min": 0.999,
                     "descends_from_init_of": descends},
    }
    out = output_root / "analysis" / "seed_identity" / run_name / "padding_ancestry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, rec)
    for m in labels:
        pm = per_model[m]
        v = pm["latest_vs_step1"]
        print(f"{m:15s} latest {pm['latest_step']} vs step1 cos_min "
              f"{v and v['cos_min']} scale {v and v['scale_min']} descends {descends[m]}")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------- datadecide


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


@app.command()
def datadecide(
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
    """Two DataDecide seed branches differ in BOTH init and data order (exit 2 if not)."""
    output_root, hf_cache_root = _bootstrap(output_root, hf_cache_root, torch_threads=False)

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
