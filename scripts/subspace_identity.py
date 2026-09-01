"""Subspace-identity: do runs of one arm learn a SHARED carrier subspace, or merely
equally-toxic but disjoint directions? Metric per module: mean squared singular value
of U_a^T U_b between two runs' top-k left update subspaces (1 = identical, k/n ~ 0.005
= random). Averaged over modules; pairs within-arm vs cross-arm vs random baseline.

    uv run python -m scripts.subspace_identity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

RUNS_FILE = Path("run_logs/harvest_20260824/lambda_arms_runs.txt")
K = 4


def topk(run: str, k: int = K) -> dict[str, np.ndarray] | None:
    """{module: top-k left singular directions of s·B·A}; None if any module is thinner."""
    from subliminal_learning.weightspace.adapters import iter_lora_pairs, load_lora_tensors
    from subliminal_learning.weightspace.lora import lora_svd

    out = {}
    tens = load_lora_tensors(Path(run) / "checkpoints/adapter_model.safetensors")
    for mod, b, a in iter_lora_pairs(tens):
        if min(b.shape[1], a.shape[0]) < k:
            return None
        out[mod] = lora_svd(b, a)[0][:, :k]
    return out


def overlap(sa: dict, sb: dict) -> float:
    vals = []
    for mod in sa:
        if mod not in sb:
            continue
        sv = np.linalg.svd(sa[mod].T @ sb[mod], compute_uv=False)
        vals.append(float(np.mean(sv**2)))
    return float(np.mean(vals))


def pairs(A_: list, B_: list | None = None) -> np.ndarray:
    out = []
    if B_ is None:
        for i in range(len(A_)):
            for j in range(i + 1, len(A_)):
                out.append(overlap(A_[i], A_[j]))
    else:
        for a in A_:
            for b in B_:
                out.append(overlap(a, b))
    return np.array(out)


@app.command()
def main(
    out: Path = typer.Option(None, help="Output JSON (default {output_root}/analysis/"
                             "subspace_identity_topk.json)."),
    output_root: Path = typer.Option(None),
) -> None:
    from subliminal_learning.config import resolve_output_root

    out = out or (resolve_output_root(output_root) / "analysis"
                  / "subspace_identity_topk.json")
    text = RUNS_FILE.read_text()
    strip = [x for x in text.split()
             if '/pythia-160m-data-seed1/' in x and '/controls/' not in x][:10]
    tf = [x for x in text.split() if '/controls/' in x][:10]
    Ss = [s for s in (topk(r) for r in strip) if s]
    St = [s for s in (topk(r) for r in tf) if s]
    w_strip, w_tf, x_arm = pairs(Ss), pairs(St), pairs(Ss, St)
    n = 768
    # expected mean sq cosine for random k-dim subspaces in R^n (approx k/n per direction)
    rand_ref = K / n
    print(f"mean top-{K} subspace overlap (1=identical, random ~{rand_ref:.4f}):")
    print(f"  within strip arm    : {w_strip.mean():.4f} (n={len(w_strip)} pairs, "
          f"sd {w_strip.std():.4f})")
    print(f"  within trait-free   : {w_tf.mean():.4f} (n={len(w_tf)})")
    print(f"  strip vs trait-free : {x_arm.mean():.4f} (n={len(x_arm)})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        K=K, n_strip=len(Ss), n_tf=len(St), within_strip=float(w_strip.mean()),
        within_tf=float(w_tf.mean()), cross=float(x_arm.mean()), rand_ref=rand_ref,
        sd_within_strip=float(w_strip.std()))))
    print(f'saved -> {out}')


if __name__ == "__main__":
    app()
