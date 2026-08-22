"""Per-base cache of the frozen weights' norms: ``{module: {W0_fro, W0_spec}}``.

One file per *resolved snapshot* of a base repo under ``analysis/w0_norms/``:
``{repo}__{sha12}.json``, where the sha is the HF commit the cache resolves for that
repo/revision (`adapter_decomp.base_snapshot_sha`). The repo name alone is not a key: the
hub cache genuinely holds ~30 snapshots of ``EleutherAI/pythia-160m`` (every ``stepN``
revision), so a name-keyed file silently served one snapshot's norms to every other. When
no sha is resolvable the name-only path is used, and a pre-existing name-only file is read
as the legacy fallback and migrated under the sha key on first use (no recompute).

Every writer merges (read → update → atomic write) instead of replacing, so two harvesters
sharing the file (``adapter_norms_probe``, ``adapter_norms_slim``, ``ckpt_dose_trajectory``)
can never truncate each other's table. The payload is the flat module table plus a
``_meta`` entry recording the repo, revision and snapshot sha; `load_w0_norms` strips it.
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.analysis.adapter_decomp import base_snapshot_sha
from subliminal_learning.utils.io import atomic_write_json

META_KEY = "_meta"


def _hf_cache_root(hf_cache_root: Path | None) -> Path | None:
    if hf_cache_root is not None:
        return Path(hf_cache_root)
    import os

    env = os.environ.get("HF_HOME")
    if env:
        return Path(env)
    try:
        import yaml

        root = Path(__file__).resolve().parents[3]
        cfg = yaml.safe_load((root / "configs/paths/paths_local.yaml").read_text())
        return Path(cfg["hf_cache_root"])
    except Exception:  # noqa: BLE001 - offline / no yaml: fall back to name-only key
        return None


def w0_cache_path(output_root: Path, base_repo: str, *, revision: str | None = None,
                  hf_cache_root: Path | None = None) -> tuple[Path, str | None]:
    """``(cache_file, snapshot_sha)`` for a base; sha is None when unresolvable."""
    root = _hf_cache_root(hf_cache_root)
    sha = base_snapshot_sha(root, base_repo, revision=revision) if root is not None else None
    stem = base_repo.replace("/", "__")
    if sha:
        stem += f"__{sha[:12]}"
    elif revision and revision != "main":
        stem += f"__{revision}"
    return Path(output_root) / "analysis" / "w0_norms" / f"{stem}.json", sha


def _legacy_path(cache_file: Path, base_repo: str) -> Path:
    return cache_file.parent / (base_repo.replace("/", "__") + ".json")


def _read(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return d if isinstance(d, dict) else {}


def load_w0_norms(output_root: Path, base_repo: str, *, revision: str | None = None,
                  hf_cache_root: Path | None = None) -> dict[str, dict[str, float]]:
    """Module table for a base (``{}`` when nothing is cached). Migrates legacy files."""
    path, sha = w0_cache_path(output_root, base_repo, revision=revision,
                              hf_cache_root=hf_cache_root)
    table = _read(path)
    if not table:
        legacy = _legacy_path(path, base_repo)
        if legacy != path and legacy.is_file():
            table = _read(legacy)
            if table:  # adopt under the snapshot key so the name-only file stops serving
                merge_w0_norms(output_root, base_repo, table, revision=revision,
                               hf_cache_root=hf_cache_root)
    table.pop(META_KEY, None)
    return table


def merge_w0_norms(output_root: Path, base_repo: str, new: dict[str, dict[str, float]], *,
                   revision: str | None = None, hf_cache_root: Path | None = None) -> dict:
    """Union `new` into the cache file (never replaces it); returns the merged table."""
    path, sha = w0_cache_path(output_root, base_repo, revision=revision,
                              hf_cache_root=hf_cache_root)
    disk = _read(path)
    disk.update({k: {"W0_fro": float(v["W0_fro"]), "W0_spec": float(v["W0_spec"])}
                 for k, v in new.items() if k != META_KEY})
    disk[META_KEY] = {"base_repo": base_repo, "revision": revision, "snapshot_sha": sha}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, disk)
    disk.pop(META_KEY, None)
    return disk


def ensure_w0_norms(output_root: Path, base_repo: str, modules: list[str], *,
                    revision: str | None = None, hf_cache_root: Path | None = None,
                    ) -> dict[str, dict[str, float]]:
    """Cached table covering `modules`, computing (fp64 fro + spectral) only the missing ones.

    Loads the base in fp32 on CPU only on a cache miss; torch/transformers are imported
    lazily so CPU-only callers keep a light import set on the hit path.
    """
    got = load_w0_norms(output_root, base_repo, revision=revision, hf_cache_root=hf_cache_root)
    missing = [m for m in modules if m not in got]
    if not missing:
        return got
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    kw = {"torch_dtype": torch.float32}
    if revision:
        kw["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(base_repo, **kw)
    sd = model.state_dict()
    new = {}
    for m in missing:
        w = sd[m + ".weight"].double().numpy()
        new[m] = {"W0_fro": float(np.linalg.norm(w)),
                  "W0_spec": float(np.linalg.svd(w, compute_uv=False)[0])}
    del model, sd
    return merge_w0_norms(output_root, base_repo, new, revision=revision,
                          hf_cache_root=hf_cache_root)
