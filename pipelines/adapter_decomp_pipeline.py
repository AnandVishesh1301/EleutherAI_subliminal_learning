"""LoRA ΔW decomposition over an adapter surface — orchestration.

Sweeps `analysis.adapter_decomp` across every adapter matching a set of selectors
(teachers and/or students, any animal/base/lineage/tag/ftseed, optionally every
intermediate checkpoint). Read-only with respect to the adapter tree: nothing is
retrained, no GPU is required.

**Output layout mirrors the adapter tree, one cell per measured run**::

    {out_dir}/cells/{kind}/{animal}/{base_model}[/{lineage}]/{param_tag}/ftseed{N}/
        rows.jsonl                     # one row per (module, step)
        adapter_decomp_manifest.json   # written LAST — its presence certifies the cell

This is the same cell-per-run discipline the training pipelines use, which is what lets
this module reuse the repo's existing machinery instead of re-implementing it:

- **`run_gate.gate_action`** decides SKIP / measure / ERROR from the stored manifest's
  ``run_identity``. Because that identity contains the adapter's ``weights_sha256``, an
  adapter retrained in place surfaces as *identity drift* — a loud, diffed error naming
  the changed key — rather than something a bespoke mtime check has to notice.
- **manifest-written-last** makes resume free and duplication impossible: an interrupted
  cell has no manifest, so it is simply measured again over a truncated `rows.jsonl`.
  There is no append-only ledger to reconcile.
- **`utils.run_io.acquire_claim`** makes concurrent sweeps safe at cell granularity.
- **`write_run_set_card` / `append_runs_index`** register the sweep as a campaign.

Efficiency shape — the two costs that would dominate are hoisted out of the per-adapter
loop: ``A_init`` is reconstructed once per (base repo, ftseed, r, target_modules) group
and reused across every tag and checkpoint step (refs arrive sorted, so a single-entry
cache suffices), and ``W_0``'s full-rank SVD is cached on disk per base **revision** under
``{out_dir}/../_w0_cache``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from subliminal_learning.analysis.adapter_decomp import (
    target_modules_key,
    adapter_config,
    base_model_factory,
    base_snapshot_sha,
    base_weight_norms,
    decompose,
    discover_adapters,
    init_group_key,
    lora_init_weights,
    lora_target_modules,
)
from subliminal_learning.run_gate import (
    ERROR,
    SKIP,
    format_identity_diff,
    gate_action,
)
from subliminal_learning.utils.run_io import (
    acquire_claim,
    append_runs_index,
    check_run_set_staleness,
    new_manifest,
    release_claim,
    utc_now_iso,
    write_json_atomic,
    write_manifest,
    write_run_set_card,
)

MANIFEST = "adapter_decomp_manifest.json"
CELLS = "cells"
ROWS = "rows.jsonl"
W0_CACHE_DIR = "_w0_cache"
WORKFLOW = "adapter_decomp"

#: Bump when a change to `decompose` alters emitted values, so stored cells re-measure
#: instead of silently mixing schema versions within one sweep.
DECOMP_SCHEMA = 1

#: Offset added to ftseed for the negative control. Fixed so the control is reproducible
#: across runs and comparable across sweeps.
WRONG_SEED_OFFSET = 1000


@dataclass(frozen=True)
class AdapterDecompConfig:
    """One decomposition sweep.

    Attributes:
        out_dir: Sweep root; cells live under ``{out_dir}/cells/``.
        kinds: Which adapter trees to sweep — ``("teacher",)``, ``("student",)``, or both.
        animals / base_models / lineages / tags: fnmatch selectors; empty = no constraint.
        ftseeds: Explicit ftseed allow-list; empty = all.
        checkpoints: Also measure every intermediate ``checkpoint-N/`` (growth curves).
            Multiplies rows per cell by ~40 on the mech-probe runs — opt in deliberately.
        wrong_seed_control: Reconstruct a second init at ``ftseed + WRONG_SEED_OFFSET``
            and record its cosine with ``A_final``. Doubles reconstruction cost; the
            check that makes every other number trustworthy, so on by default.
        include_w0: Attach the frozen ``W_0`` norms (disk-cached per base revision) so
            rows carry the relative-perturbation readout ``dW_over_W0``.
        device: Torch device for the reconstruction. An execution detail, so deliberately
            NOT part of the cell identity (PEFT draws ``lora_A`` on CPU either way); it is
            still recorded on every row and in the sweep manifest.
        limit: Stop after this many cells (smoke tests). None = no cap.
        run_set_id: Campaign id — registers a ``run_set.json`` card and a RUNS_INDEX row.
        force: Overwrite cells whose stored identity differs instead of erroring.
    """

    out_dir: Path
    kinds: tuple[str, ...] = ("student",)
    animals: tuple[str, ...] = ()
    base_models: tuple[str, ...] = ()
    lineages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    ftseeds: tuple[int, ...] = ()
    checkpoints: bool = False
    wrong_seed_control: bool = True
    include_w0: bool = True
    device: str = "cpu"
    limit: int | None = None
    run_set_id: str | None = None
    force: bool = False

    def selectors(self) -> dict[str, Any]:
        """The surface-defining subset (execution details excluded)."""
        return {k: v for k, v in asdict(self).items()
                if k not in {"out_dir", "device", "limit", "force", "run_set_id"}}


@dataclass
class _InitCache:
    """Single-entry cache of a reconstructed init group (refs arrive sorted by group)."""

    key: tuple | None = None
    init: dict[str, np.ndarray] = field(default_factory=dict)
    wrong: dict[str, np.ndarray] = field(default_factory=dict)


def cell_dir(out_dir: Path, ref) -> Path:
    """Where one measured run's rows live — mirrors the adapter tree exactly.

    ``{kind}/{animal}/{base_model}[/{lineage}]/{param_tag}/ftseed{N}`` is the same leaf
    shape `students/` and `teachers/` use, so a measured cell sits at the analogue of the
    run that produced it and the gate/claim machinery applies unchanged.
    """
    lineage_parts = Path(ref.lineage).parts if ref.lineage else ()
    return (Path(out_dir) / CELLS / ref.kind / ref.animal / ref.base_model
            / Path(*lineage_parts) / ref.tag / f"ftseed{ref.ftseed}")


def cell_identity(refs: list, acfg: dict, cfg: AdapterDecompConfig,
                  snapshot: str | None) -> dict:
    """Every input that can change this cell's rows.

    Content hashes — not paths — so an adapter retrained in place, a rewritten adapter
    config, or a re-downloaded base revision all surface as drift through
    `run_gate.gate_action`. ``train_run_identity`` carries the upstream training
    manifest's own identity (number-dataset dir, ``filtered_sha256``, subsample seed,
    train seed, LoRA config), linking a measured row to the data that produced it.
    """
    head = refs[0]
    return {
        "kind": head.kind,
        "animal": head.animal,
        "base_model": head.base_model,
        "lineage": head.lineage,
        "param_tag": head.tag,
        "ftseed": head.ftseed,
        "base_model_repo": acfg["base_model_name_or_path"],
        "base_snapshot": snapshot,
        "lora": {"r": int(acfg["r"]), "lora_alpha": int(acfg["lora_alpha"]),
                 "lora_dropout": acfg.get("lora_dropout"),
                 "target_modules": lora_target_modules(acfg["target_modules"])},
        "artifacts": {("final" if r.step is None else str(r.step)): r.weights_sha256()
                      for r in refs},
        "adapter_config_sha256": head.config_sha256(),
        "train_run_identity": head.train_run_identity(),
        "wrong_seed_control": cfg.wrong_seed_control,
        "wrong_seed_offset": WRONG_SEED_OFFSET if cfg.wrong_seed_control else None,
        "include_w0": cfg.include_w0,
        "decomp_schema": DECOMP_SCHEMA,
    }


def read_rows(run_dir: Path) -> Any:
    """Load every cell of a sweep into one DataFrame (the notebook-facing entry point)."""
    import pandas as pd

    cells = sorted((Path(run_dir) / CELLS).rglob(ROWS))
    if not cells:
        raise FileNotFoundError(f"no {CELLS}/**/{ROWS} under {run_dir}")
    return pd.concat([pd.read_json(p, lines=True) for p in cells], ignore_index=True)


def _w0_norms_cached(base_repo: str, hf_cache_root: Path, cache_dir: Path,
                     target_modules: str | list[str], device: str,
                     snapshot: str | None) -> dict[str, list[float]]:
    """W_0 norms for `base_repo`, computed once per base **revision** and memoised.

    Keyed by repo + resolved snapshot sha: this HF cache holds multiple snapshots for
    some repos, so a name-only key would serve norms from a different revision of the
    weights after a re-download.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{base_repo.replace('/', '--')}@{snapshot or 'unresolved'}.json"
    stored: dict = {}
    if path.exists():
        stored = json.loads(path.read_text())
        # The snapshot is in the filename but `target_modules` only in the payload: a file
        # written for a narrower module set (or a different regex) must not be served as
        # if it covered this config. Validate on read; a mismatch recomputes and UNIONS.
        if target_modules_key(stored.get("target_modules", [])) == \
                target_modules_key(target_modules):
            return stored["norms"]
        logger.info(f"W0 cache {path.name} holds target_modules="
                    f"{stored.get('target_modules')!r}, need {target_modules!r}: extending")
    else:
        logger.info(f"computing W0 norms for {base_repo} @ {snapshot} "
                    "(once per base revision, then cached)")
    norms = base_weight_norms(
        base_model_factory(base_repo, hf_cache_root, device), target_modules)
    merged = dict(stored.get("norms", {})) | {k: list(v) for k, v in norms.items()}
    write_json_atomic(path, {"base_model_repo": base_repo, "snapshot": snapshot,
                             "target_modules": target_modules,
                             "norms": merged})
    return merged


def _library_versions() -> dict[str, str]:
    """Versions of everything whose numerics could move a measured value."""
    import importlib.metadata as md

    out = {}
    for pkg in ("numpy", "torch", "transformers", "peft", "safetensors"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "?"
    return out


def _measure_cell(refs: list, acfg: dict, cache: _InitCache, w0: dict,
                  cfg: AdapterDecompConfig, base_repo: str) -> list[dict]:
    """Rows for every (artifact, module) of one run. Pure given the cached init."""
    from safetensors.numpy import load_file

    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])
    rows: list[dict] = []
    for ref in refs:
        tensors = load_file(str(ref.weights_path))
        # The non-finite gate lives here, where the tensors are loaded anyway, instead of
        # at discovery (which would load every matching adapter once more up front).
        n_bad = sum(int((~np.isfinite(t)).sum()) for t in tensors.values()
                    if t.dtype.kind == "f")
        if n_bad:
            logger.warning(
                f"SKIPPING diverged adapter {ref.weights_path} ({n_bad} non-finite weight "
                "value(s)); it is excluded from this decomposition")
            continue
        for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
            module = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
            a_final = tensors[key_a].astype(np.float64)
            b_final = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
            row = ref.identity() | {
                "module": module, "rank": int(b_final.shape[1]),
                "alpha": int(acfg["lora_alpha"]), "scaling": scaling,
                "base_model_repo": base_repo, "device": cfg.device,
            }
            row |= decompose(cache.init[module], a_final, b_final,
                             cache.wrong.get(module))
            if module in w0:
                row["W0_spec"], row["W0_fro"] = w0[module]
                row["dW_over_W0"] = (scaling * row["B_Afinal_spec"] / row["W0_spec"]
                                     if row["W0_spec"] else float("nan"))
            rows.append(row)
    return rows


def run(cfg: AdapterDecompConfig, output_root: Path, hf_cache_root: Path) -> dict[str, Any]:
    """Measure the decomposition over the selected surface. Returns a summary dict."""
    from threadpoolctl import threadpool_limits

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = discover_adapters(
        output_root=output_root, kinds=cfg.kinds, animals=cfg.animals or None,
        base_models=cfg.base_models or None, lineages=cfg.lineages or None,
        tags=cfg.tags or None, ftseeds=cfg.ftseeds or None, checkpoints=cfg.checkpoints,
        gate_nonfinite=False)  # gated per artifact in _measure_cell, where it is free

    # One cell per measured run; a run's checkpoint series is measured together because
    # every step shares the one reconstructed A_init.
    cells: dict[Path, list] = {}
    for ref in refs:
        cells.setdefault(cell_dir(out_dir, ref), []).append(ref)
    # Order cells by init group, NOT by path: `_InitCache` is single-entry, and a path sort
    # interleaves ranks (the rank sits inside the tag) so every switch re-loads the base and
    # re-wraps PEFT (~8 s). On the ds1 focus lineage that was 151 switches vs 8 groups.
    cell_cfgs: dict[Path, dict] = {}
    for cdir, group in cells.items():
        try:
            cell_cfgs[cdir] = adapter_config(group[0])
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"unreadable adapter_config for {cdir}: {exc!r}")

    def _order(cdir: Path) -> tuple:
        acfg = cell_cfgs.get(cdir)
        key = init_group_key(cells[cdir][0], acfg) if acfg else ("~",)
        return (tuple(str(k) for k in key), str(cdir))

    pending = sorted(cells, key=_order)
    if cfg.limit is not None:
        pending = pending[: cfg.limit]
    logger.info(f"discovered {len(refs)} adapter artifacts in {len(cells)} cells "
                f"({len(pending)} to consider) -> {out_dir}")

    card = {"run_set_id": cfg.run_set_id, "workflow": WORKFLOW,
            "selectors": cfg.selectors(), "library_versions": _library_versions(),
            "created_at": utc_now_iso()}
    if cfg.run_set_id:
        check_run_set_staleness(out_dir, card, force=cfg.force)
        write_run_set_card(out_dir, card)

    cache = _InitCache()
    w0_cache_dir = out_dir.parent / W0_CACHE_DIR
    n_rows = n_measured = n_skipped = 0
    failures: list[dict] = []

    with threadpool_limits(limits=1):
        for cdir in pending:
            group = sorted(cells[cdir], key=lambda r: (-1 if r.step is None else r.step))
            head = group[0]
            claim = None
            try:
                acfg = cell_cfgs.get(cdir) or adapter_config(head)
                base_repo = acfg["base_model_name_or_path"]
                snapshot = base_snapshot_sha(hf_cache_root, base_repo)
                identity = cell_identity(group, acfg, cfg, snapshot)

                stored = None
                mpath = cdir / MANIFEST
                if mpath.exists():
                    try:
                        stored = json.loads(mpath.read_text())
                    except json.JSONDecodeError:
                        logger.warning(f"corrupt {mpath} — re-measuring over it")
                decision = gate_action(stored, identity, cfg.force)
                if decision.action == SKIP:
                    n_skipped += 1
                    continue
                if decision.action == ERROR:
                    raise ValueError(
                        f"{cdir} holds a complete measurement whose identity differs — "
                        f"refusing to overwrite. Drift:\n"
                        f"{format_identity_diff(decision.diff)}\n"
                        "The adapter or its upstream training run changed; pass --force "
                        "to re-measure over it.")

                cdir.mkdir(parents=True, exist_ok=True)
                claim = acquire_claim(cdir)
                if claim is None:
                    logger.info(f"{cdir} claimed by another process — skipping")
                    n_skipped += 1
                    continue

                gkey = init_group_key(head, acfg)
                if gkey != cache.key:
                    factory = base_model_factory(base_repo, hf_cache_root, cfg.device)
                    cache = _InitCache(
                        key=gkey,
                        init=lora_init_weights(factory, head.ftseed, acfg),
                        wrong=(lora_init_weights(
                            factory, head.ftseed + WRONG_SEED_OFFSET, acfg)
                            if cfg.wrong_seed_control else {}))
                w0 = (_w0_norms_cached(base_repo, hf_cache_root, w0_cache_dir,
                                       lora_target_modules(acfg["target_modules"]),
                                       cfg.device, snapshot)
                      if cfg.include_w0 else {})

                rows = _measure_cell(group, acfg, cache, w0, cfg, base_repo)
                # rows first, manifest LAST: an interrupt leaves no manifest, so the
                # cell is simply re-measured over a truncated rows file next time.
                (cdir / ROWS).write_text("".join(json.dumps(r) + "\n" for r in rows))
                write_manifest(mpath, new_manifest(
                    workflow=WORKFLOW,
                    body={"run_identity": identity, "n_rows": len(rows),
                          "n_artifacts": len(group), "device": cfg.device,
                          "library_versions": _library_versions()},
                    run_set_id=cfg.run_set_id))
                n_rows += len(rows)
                n_measured += 1
                if n_measured % 25 == 0:
                    logger.info(f"{n_measured} cells measured, {n_rows} rows")
            except Exception as exc:  # one bad cell must not sink the sweep
                logger.warning(f"FAILED {cdir}: {exc!r}")
                failures.append({"cell": str(cdir), "error": repr(exc)})
            finally:
                release_claim(claim)

    summary = {"artifacts": len(refs), "cells": len(cells), "measured": n_measured,
               "skipped": n_skipped, "rows": n_rows, "failures": failures}
    write_json_atomic(out_dir / "sweep_summary.json",
                      {**card, "finished_at": utc_now_iso(), "summary": summary})
    if cfg.run_set_id:
        append_runs_index(output_root, {
            "run_set_id": cfg.run_set_id, "workflow": WORKFLOW, "out_dir": str(out_dir),
            "status": "complete" if not failures else "complete_with_failures",
            "updated_at": utc_now_iso(), **summary})
    logger.info(f"measured {n_measured} cells ({n_skipped} skipped), {n_rows} rows, "
                f"{len(failures)} failures -> {out_dir}")
    return summary
