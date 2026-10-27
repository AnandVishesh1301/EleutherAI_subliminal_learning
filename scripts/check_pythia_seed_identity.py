"""CLI: weight-only init + data-order identity of Pythia-160m models (HF released + native runs).

Thin driver over `models.seed_identity` (pure tests) and `models.seed_identity_io` (model
table, offline loaders, Pile ground truth). ALL Pythia init / data-order identity checks go
through this; do not re-implement them in analyses.

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

    uv run python -m scripts.check_pythia_seed_identity --models orig,rerun --windows 1-2,2-4
    uv run python -m scripts.check_pythia_seed_identity --native mine=/path/to/neox/run[:dclm]

Writes ``{output_root}/analysis/seed_identity/<run name>/{identity.json,identity.png}``.
CPU only; threads from ``OMP_NUM_THREADS``; HF cache read offline.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import typer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from subliminal_learning.config import resolve_roots  # noqa: E402
from subliminal_learning.models import seed_identity as si  # noqa: E402
from subliminal_learning.models import seed_identity_io as sio  # noqa: E402
from subliminal_learning.utils.io import atomic_write_json  # noqa: E402
from subliminal_learning.utils.logging_setup import configure_loguru  # noqa: E402
from subliminal_learning.utils.run_io import git_sha, utc_now_iso  # noqa: E402

ROOT = Path(__file__).absolute().parents[1]
DEFAULT_WINDOWS = "1-2,2-4,4-8,8-16,16-32,32-64"
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _windows(spec: str) -> list[tuple[int, int]]:
    out = []
    for w in spec.split(","):
        a, b = (int(x) for x in w.strip().split("-"))
        if not 0 <= a < b:
            raise typer.BadParameter(f"window {w!r}: need 0 <= a < b")
        out.append((a, b))
    return out


def _nan_none(x: float):
    return None if isinstance(x, float) and np.isnan(x) else x


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
def main(
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
    load_dotenv(ROOT / ".env")
    configure_loguru()
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if method not in ("exact", "residual"):
        raise typer.BadParameter(f"--method must be exact or residual, got {method!r}")
    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)
    wins = _windows(windows)
    table = sio.model_table()
    labels = list(table) if models == "all" else [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in labels if m not in table]
    if unknown:
        raise typer.BadParameter(
            f"unknown model labels {unknown}; known: {list(table)} (native runs come from "
            f"the {sio.NATIVE_RUNS_KEY!r} key of configs/paths/paths_local.yaml)"
        )
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
        "init": {"fraction_equal_matrix": [[_nan_none(float(v)) for v in r] for r in init_frac],
                 "pairs": init_pairs, **cls(init_classes)},
        "order": {"per_model": per_model, "pairs": pairs,
                  "jaccard_matrices": {f"{a}->{b}": [[_nan_none(float(v)) for v in r] for r in M]
                                       for (a, b), M in jacc.items()},
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


if __name__ == "__main__":
    app()
