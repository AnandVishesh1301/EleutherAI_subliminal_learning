"""CLI: padding-row (weight-decay-only) ancestry of Pythia-160m models.

Sibling of `scripts.check_pythia_seed_identity` (same model table, loaders and output dir);
does not re-run its order pass. The 27 padding rows (ids >= 50277) are never inputs, so they
only decay: a final checkpoint's pad rows are ``c * init`` of the init they descend from.
Per model it loads ONLY the input embedding at step0, step1, the latest offline checkpoint
(HF step143000 if cached, native max ``global_step``) and the window steps, keeping only the
padding rows, and writes (via `models.seed_identity.compare_padding_rows`):

- ``per_model``: latest vs own step1 and vs own step0;
- ``ancestry``: every model's latest vs every model's step1 (cos min/max, scale range);
- ``windows``: pad rows step a vs b for each early window (exact-equality validity);
- ``latest_untouched``: step1 -> latest untouched token ids by the residual criterion at
  ``--latest-tol`` (full embeddings held transiently, one model at a time).

    uv run python -m scripts.check_pythia_padding_ancestry --models all --run-name <identity run>

Writes ``{output_root}/analysis/seed_identity/<run name>/padding_ancestry.json``. CPU only;
HF cache read offline.
"""
from __future__ import annotations

import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import typer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from scripts.check_pythia_seed_identity import DEFAULT_WINDOWS, _nan_none, _windows  # noqa: E402
from subliminal_learning.config import resolve_roots  # noqa: E402
from subliminal_learning.models import seed_identity as si  # noqa: E402
from subliminal_learning.models import seed_identity_io as sio  # noqa: E402
from subliminal_learning.utils.io import atomic_write_json  # noqa: E402
from subliminal_learning.utils.logging_setup import configure_loguru  # noqa: E402
from subliminal_learning.utils.run_io import git_sha, utc_now_iso  # noqa: E402

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _rec(c: si.PaddingComparison) -> dict:
    return {k: _nan_none(v) for k, v in c.__dict__.items()}


@app.command()
def main(
    models: str = typer.Option("all", help="Comma-separated model labels, or 'all'."),
    windows: str = typer.Option(DEFAULT_WINDOWS, help="Early windows a-b for pad comparisons."),
    hf_final_step: int = typer.Option(143000, help="HF latest checkpoint (MISSING if uncached)."),
    latest_tol: float = typer.Option(3e-6, help="Residual tol for the step1->latest untouched set."),
    workers: int = typer.Option(4, help="Parallel model loads."),
    run_name: str = typer.Option(..., help="Output subdir (the identity run dir)."),
    output_root: Path = typer.Option(None),
    hf_cache_root: Path = typer.Option(None),
) -> None:
    load_dotenv(ROOT / ".env")
    configure_loguru()
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)
    table = sio.model_table()
    labels = list(table) if models == "all" else [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in labels if m not in table]
    if unknown:
        raise typer.BadParameter(f"unknown model labels {unknown}; known: {list(table)}")
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

    def mat(M):
        return [[_nan_none(float(v)) for v in r] for r in M]

    rec = {
        "kind": "pythia_padding_ancestry", "created_utc": utc_now_iso(),
        "provenance": {"host": socket.gethostname(), "git_sha": git_sha(),
                       "hf_cache_root": str(hf_cache_root),
                       "sources": {m: {"kind": table[m].kind, "source": table[m].source,
                                       "corpus": table[m].corpus} for m in labels}},
        "models": labels, "windows": [f"{a}->{b}" for a, b in wins], "vocab_size": vocab,
        "hf_final_step": hf_final_step, "per_model": per_model, "windows_pad": win_rec,
        "ancestry": {"rows": "latest checkpoint", "columns": "step1 init",
                     "cos_min": mat(cos_min), "scale_min": mat(scale_min),
                     "scale_max": mat(scale_max), "descends_threshold_cos_min": 0.999,
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


if __name__ == "__main__":
    app()
