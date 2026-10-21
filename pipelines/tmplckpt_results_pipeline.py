"""Read every `tmplckpt` leaf, teacher, pool and Angela copy and write one results build.

Output: `<out_root>/build-<UTC>/` (create-only; `manifest.json` written last):

    leaves.csv           one row per planned student (both channels, every variant):
                         status, p per set, perplexity ratio, screened-out share, kept,
                         teacher usable, point Δ vs placebo and base
    teachers.csv         one row per teacher (Angela copies and reproduction retrains
                         included): p per set, perplexity ratio, usability per channel
    cells.csv            per channel / wave / variant / animal / arm / step: counts,
                         kept controls, measurability, two-stage Δ intervals (n_valid)
    trajectory.csv       one row per student probe: p, ppl, point Δ vs the matched
                         controls at the same probe step
    pools.csv            hard pools: kept rows, keep rate, numbers per kept completion,
                         EOS share
    quarantine.csv       every student outside the primary set, with the reason
    angela_original.csv  Angela's ep20_lr1e-4_r8a16 students (diagonal + grid copies)
    *_ppl10.csv, *_drop05.csv  leaves / cells of the two sensitivity reruns
    claims.json          K0–K8 of `docs/PREREG_TMPLCKPT.md` for every rule set
    coverage.json        planned / done counts and what is missing

Our leaf directories come from `tmplckpt_hard` / `tmplckpt_soft`; while a module or a
file is absent the row is ``missing``. Angela's diagonal paths come from
`tmplckpt_hard.angela_student_dir` / `angela_base_control_dir`; her off-diagonal grid
students (K3 only) follow her tree's layout (`angela_grid_student_src`).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, ModuleType

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.optparam2_results_pipeline import _num, _p, _read
from subliminal_learning.analysis.train_log_io import GKD_LOG_FILE, read_log_history
from subliminal_learning.campaign import tmplckpt_metrics as metrics
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

DONE, MISSING = "done", "missing"
PREFERENCE_FILE = "preference_eval.json"
PREFERENCE_SUMMARY = "preference_summary.json"
METRICS_FILE = "sdf_metrics_eval.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"
PROBE_FILE = "training_metrics.jsonl"
SETS: tuple[str, ...] = ("test", "off_topic", "favorite", "validation")
ANGELA_DATASET_DIR = "ex10-20_prefix_n100k"
ANGELA_STUDENT_SEEDS: tuple[int, ...] = plan.HARD_STUDENT_SEEDS
GRID_ARM = "grid"
POOL_DIAG_CACHE = "pool_diag_cache.json"
# ws160m gen-3 Pythia-160M self diagonal at 143k, bear (placebo 0.340, untuned base
# 0.342; `20260923_ws160m3-rx1_160m/results/leaves.csv`, traitfree Part I cells.csv).
WS160M_SELF_143K_BEAR = 0.34
TABLES: tuple[str, ...] = (
    "leaves",
    "teachers",
    "cells",
    "trajectory",
    "pools",
    "quarantine",
    "angela_original",
)
CONTROLS: tuple[str, ...] = ("placebo", "base")


# ---- layout ------------------------------------------------------------------------------


def _module(name: str) -> ModuleType | None:
    """A campaign layout module, or ``None`` while it does not exist yet."""
    try:
        return importlib.import_module(f"subliminal_learning.campaign.{name}")
    except ModuleNotFoundError as err:
        if err.name != f"subliminal_learning.campaign.{name}":
            raise
        logger.warning(
            "layout module {} not present yet; its leaves read as missing", name
        )
        return None


def _dir(mod: ModuleType | None, fn: str, *args: object) -> Path | None:
    f = getattr(mod, fn, None) if mod is not None else None
    return None if f is None else Path(f(*args))


def _eval_paths(mod: ModuleType | None, fn: str, *args: object) -> dict[str, Path]:
    f = getattr(mod, fn, None) if mod is not None else None
    return {} if f is None else {k: Path(v) for k, v in f(*args).items()}


def angela_grid_student_src(student_step: int, teacher_step: int, seed: int) -> Path:
    """Angela's off-diagonal grid student (her layout; ep20_lr1e-4_r8a16, tseed1)."""
    return (
        plan.ANGELA_OUTPUTS
        / "students"
        / "dog"
        / "ckpt"
        / f"pythia-160m@step{int(student_step)}"
        / f"teacher-pythia-160m@step{int(teacher_step)}"
        / f"{plan.STMT_RECIPE}_tseed{plan.ANGELA_FTSEED}"
        / ANGELA_DATASET_DIR
        / plan.HARD_TAG
        / f"ftseed{int(seed)}"
    )


def angela_grid_control_src(student_step: int, teacher_step: int, seed: int) -> Path:
    """Angela's off-diagonal base control (teacher 143k drops the ``@step`` suffix)."""
    teacher = (
        "teacher-pythia-160m"
        if int(teacher_step) == plan.STEPS_HARD15[-1]
        else f"teacher-pythia-160m@step{int(teacher_step)}"
    )
    return (
        plan.ANGELA_OUTPUTS
        / "students"
        / "controls"
        / "base"
        / "ckpt"
        / f"pythia-160m@step{int(student_step)}"
        / teacher
        / ANGELA_DATASET_DIR
        / plan.HARD_TAG
        / f"ftseed{int(seed)}"
    )


# ---- readers -----------------------------------------------------------------------------


def _pref_p(paths: Sequence[Path], target: str, set_name: str) -> float:
    """p(target) on `set_name` from the first per-seed preference file scoring it."""
    for path in paths:
        doc = _read(path)
        if not doc:
            continue
        entry = (doc.get("sets") or {}).get(set_name)
        if entry and entry.get("scoring", "prob") == "prob":
            p = _p(entry, target)
            if math.isfinite(p):
                return p
    return math.nan


def _default_pref_paths(run_dir: Path, target: str) -> tuple[Path, ...]:
    return (run_dir / PREFERENCE_FILE, run_dir / "eval" / target / PREFERENCE_FILE)


def _summary_p(param_dir: Path, set_name: str, seed: int) -> float:
    """Per-seed p of `set_name` from a param-dir `preference_summary.json`."""
    doc = _read(param_dir / PREFERENCE_SUMMARY) or {}
    for s in doc.get("summaries") or []:
        if s.get("prompt_set") == set_name and s.get("scoring", "prob") == "prob":
            return _num((s.get("per_seed_p_target") or {}).get(str(seed)))
    return math.nan


def _seed_of(run_dir: Path) -> int | None:
    for prefix in ("ftseed", "gseed"):
        tail = run_dir.name.removeprefix(prefix)
        if tail != run_dir.name and tail.isdigit():
            return int(tail)
    return None


def _ppl_entry(doc: Mapping | None) -> tuple[float, object]:
    block = (doc or {}).get("perplexity") or {}
    return _num((block.get("result") or {}).get("overall")), block.get("signature")


def _perplexity(
    run_dir: Path, capability: Path | None = None, base_metrics: Path | None = None
) -> tuple[float, float]:
    """(perplexity, base perplexity) of one seed dir on the 250-text Pile bank.

    The base is the shared `base_metrics` file when given (its signature must equal the
    seed's), else the param-dir ``sdf_capability_summary.json``.
    """
    ppl, sig = _ppl_entry(_read(capability or run_dir / METRICS_FILE))
    cap = (_read(run_dir.parent / CAPABILITY_SUMMARY) or {}).get("perplexity") or {}
    seed = _seed_of(run_dir)
    if not math.isfinite(ppl) and seed is not None:
        ppl = _num((cap.get("per_seed_overall") or {}).get(str(seed)))
    if base_metrics is not None and base_metrics.is_file():
        base, base_sig = _ppl_entry(_read(base_metrics))
        if sig is not None and base_sig is not None and sig != base_sig:
            logger.warning(
                "{}: base perplexity signature differs; ratio dropped", run_dir
            )
            return ppl, math.nan
        return ppl, base
    return ppl, _num(cap.get("base_overall"))


def _ratio(ppl: float, base: float) -> float:
    ok = math.isfinite(ppl) and math.isfinite(base) and base > 0
    return ppl / base if ok else math.nan


def _log_stats(run_dir: Path) -> tuple[float, float]:
    """(mean screened-out share, final rollout length) from the GKD trainer log."""
    path = run_dir / GKD_LOG_FILE
    if not path.is_file():
        return math.nan, math.nan
    history = read_log_history(path)
    drops = [
        float(h["gkd/filter_drop_frac"])
        for h in history
        if isinstance(h.get("gkd/filter_drop_frac"), (int, float))
    ]
    lens = [
        float(h["gkd/rollout_len"])
        for h in history
        if isinstance(h.get("gkd/rollout_len"), (int, float))
    ]
    return (
        float(np.mean(drops)) if drops else math.nan,
        lens[-1] if lens else math.nan,
    )


def _jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file (a torn trailing line is skipped)."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("{}: skipped a torn line", path)
    return out


def probe_p(rec: Mapping, animal: str) -> float:
    """Probe p of `animal`: ``p_target``, else ``p_by_animal[animal]`` (hard controls,
    precedent `pipelines/rev1_results_pipeline.py:278-281`)."""
    p = _num(rec.get("p_target"))
    if math.isfinite(p):
        return p
    return _num((rec.get("p_by_animal") or {}).get(animal))


# ---- rows ---------------------------------------------------------------------------------


def _teacher_ident(t: plan.Teacher | None) -> dict:
    if t is None:
        return {
            "teacher_kind": None,
            "teacher_target": None,
            "teacher_source": None,
            "ftseed": -1,
        }
    return {
        "teacher_kind": t.kind,
        "teacher_target": t.target,
        "teacher_source": t.source,
        "ftseed": t.ftseed,
    }


def _student_values(
    run_dir: Path | None, animal: str, soft: bool, paths: Mapping[str, Path]
) -> dict:
    """Eval values of one student; `paths` (layout eval paths) override the defaults."""
    if run_dir is None:
        return {"run_dir": None, "probe_path": None, "status": MISSING}
    pref = (
        (paths["preference"],) if "preference" in paths else ()
    ) + _default_pref_paths(run_dir, animal)
    p = {f"p_{s}": _pref_p(pref, animal, s) for s in SETS}
    ppl, base = _perplexity(run_dir, paths.get("capability"), paths.get("base_metrics"))
    drop, rlen = _log_stats(run_dir) if soft else (math.nan, math.nan)
    return {
        "run_dir": str(run_dir),
        "probe_path": str(paths.get("training_metrics", run_dir / PROBE_FILE)),
        "status": DONE if math.isfinite(p["p_test"]) else MISSING,
        **p,
        "p_target": p["p_test"],
        "perplexity": ppl,
        "perplexity_base": base,
        "perplexity_ratio": _ratio(ppl, base),
        "filter_drop_frac": drop,
        "rollout_len_final": rlen,
    }


def _row(
    channel: str,
    wave: str,
    kind: str,
    animal: str,
    config: str,
    variant: str,
    t,
    ts,
    ss,
    seed,
) -> dict:
    tid = _teacher_ident(t)
    return {
        "key": (
            f"{channel}|{wave}|{variant}|{animal}|{kind}|{tid['teacher_kind']}|"
            f"{tid['teacher_source']}|t{ts}|s{ss}|f{tid['ftseed']}|g{seed}"
        ),
        "channel": channel,
        "wave": wave,
        "variant": variant,
        "kind": kind,
        "is_control": kind != plan.TREATMENT,
        "control_kind": None if kind == plan.TREATMENT else kind,
        "animal": animal,
        "config": config,
        **tid,
        "teacher_base": plan.BASE,
        "student_base": plan.BASE,
        "teacher_step": ts,
        "student_step": ss,
        "seed": seed,
    }


def soft_rows(soft_mod: ModuleType | None) -> list[dict]:
    """One row per planned soft leaf (`plan.soft_leaves`)."""
    rows = []
    for leaf in plan.soft_leaves():
        base = _row(
            plan.SOFT,
            leaf.wave,
            leaf.kind,
            leaf.animal,
            leaf.gkd_tag,
            leaf.variant,
            leaf.teacher,
            leaf.teacher_step,
            leaf.student_step,
            leaf.gseed,
        )
        run_dir = _dir(soft_mod, "leaf_dir", leaf)
        paths = _eval_paths(soft_mod, "leaf_eval_paths", leaf)
        rows.append({**base, **_student_values(run_dir, leaf.animal, True, paths)})
    return rows


def hard_wave(step: int) -> str:
    """Wave of a hard student: 1A at Angela's steps, 1E on the soft ladder."""
    return plan.W1A if step in plan.STEPS_HARD15 else plan.W1E


def hard_rows(hard_mod: ModuleType | None) -> list[dict]:
    """One row per planned hard student (`plan.hard_students`)."""
    rows = []
    for st in plan.hard_students():
        base = _row(
            plan.HARD,
            hard_wave(st.step),
            st.kind,
            plan.DOG,
            plan.HARD_TAG,
            plan.MAIN,
            st.teacher,
            st.step,
            st.step,
            st.seed,
        )
        run_dir = _dir(hard_mod, "student_dir", st)
        paths = _eval_paths(hard_mod, "student_eval_paths", st)
        rows.append({**base, **_student_values(run_dir, plan.DOG, False, paths)})
    return rows


# Own-target sets: statement teacher (animal) and colour placebo
# (`tmplckpt_hard.TEACHER_SETS` / `PLACEBO_OWN_SETS`).
_OWN_SETS: MappingProxyType[bool, tuple[str, str]] = MappingProxyType(
    {True: ("test", "off_topic"), False: ("color_test", "off_topic_nocolor")}
)


def teacher_rows(
    hard_mod: ModuleType | None, soft_mod: ModuleType | None
) -> list[dict]:
    """One row per teacher of `plan.teachers` (our harness; Angela's own eval beside it)."""
    rows = []
    for t in plan.teachers():
        if t.kind in (plan.SDF, plan.SDF_PL):
            run_dir = _dir(soft_mod, "sdf_teacher_dir", t) or plan.gkd_teacher_dir(
                t, plan.BEAR
            )
            paths = _eval_paths(soft_mod, "sdf_teacher_eval_paths", t)
        else:
            run_dir = _dir(hard_mod, "teacher_dir", t)
            paths = _eval_paths(hard_mod, "teacher_eval_paths", t)
        row = {
            "teacher_key": f"{t.kind}|{t.target}|{t.source}|t{t.step}|f{t.ftseed}",
            "kind": t.kind,
            "target": t.target,
            "step": t.step,
            "ftseed": t.ftseed,
            "source": t.source,
            "run_dir": None if run_dir is None else str(run_dir),
        }
        own = _OWN_SETS[t.target in (plan.DOG, plan.BEAR)]
        for split, name in zip(("test", "off_topic"), own, strict=True):
            p = math.nan
            if run_dir is not None:
                pref = (paths["preference"],) if "preference" in paths else ()
                p = _pref_p(
                    pref + _default_pref_paths(run_dir, t.target), t.target, name
                )
                if not math.isfinite(p):
                    p = _summary_p(run_dir.parent, name, t.ftseed)
            row[f"p_{split}"] = p
            for animal in (plan.DOG, plan.BEAR):
                cross = paths.get(f"preference_{animal}")
                cand = ((cross,) if cross else ()) + (
                    () if run_dir is None else _default_pref_paths(run_dir, animal)
                )
                row[f"p_{split}_{animal}"] = _pref_p(cand, animal, split)
        ppl, base = (
            (math.nan, math.nan)
            if run_dir is None
            else _perplexity(
                run_dir, paths.get("capability"), paths.get("base_metrics")
            )
        )
        row.update(
            perplexity=ppl, perplexity_base=base, perplexity_ratio=_ratio(ppl, base)
        )
        if t.source in (plan.ANGELA, plan.REPRO):
            src = plan.angela_copy(plan.angela_teacher_src(t.step, plan.STMT_RECIPE))
            row["angela_p_test"] = _pref_p((src / PREFERENCE_FILE,), plan.DOG, "test")
            probe = _read(src / "eval" / "pile_probe.json") or {}
            row["angela_pile_dce"] = _num(probe.get("delta_ce"))
        rows.append(row)
    return rows


def _angela_row(
    kind: str, t: int, s: int, seed: int, copy: Path, ours: Mapping[str, Path]
) -> dict | None:
    """One Angela student: her saved p and, where scored, our harness's."""
    if not copy.is_dir():
        return None
    hers = _default_pref_paths(copy, plan.DOG)
    p_ours = _pref_p((ours["preference"],), plan.DOG, "test") if ours else math.nan
    mirror = ours["capability"].parent if ours else copy
    ppl, base = _perplexity(mirror, ours.get("capability"), ours.get("base_metrics"))
    return {
        "kind": kind,
        "teacher_step": t,
        "student_step": s,
        "seed": seed,
        "run_dir": str(copy),
        "p_test_angela": _pref_p(hers, plan.DOG, "test"),
        "p_test_ours": p_ours,
        "p_off_topic": _pref_p(hers, plan.DOG, "off_topic"),
        "perplexity_ratio": _ratio(ppl, base),
    }


def angela_rows(hard_mod: ModuleType | None) -> list[dict]:
    """Angela's ep20_lr1e-4_r8a16 students and base controls: diagonal + 13x13 grid."""
    rows: list[dict] = []
    if hard_mod is not None:
        for s in plan.STEPS_HARD15:
            for seed in ANGELA_STUDENT_SEEDS:
                for kind, fn, ev in (
                    (plan.TREATMENT, "angela_student_dir", plan.TREATMENT),
                    (plan.BASE_SB, "angela_base_control_dir", "base"),
                ):
                    copy = _dir(hard_mod, fn, s, seed)
                    ours = _eval_paths(
                        hard_mod, "angela_student_eval_paths", s, seed, ev
                    )
                    row = (
                        None
                        if copy is None
                        else _angela_row(kind, s, s, seed, copy, ours)
                    )
                    if row is not None:
                        rows.append(row)
    for t in plan.STEPS_GRID13:
        for s in plan.STEPS_GRID13:
            if t == s:
                continue
            for seed in ANGELA_STUDENT_SEEDS:
                for kind, fn in (
                    (plan.TREATMENT, angela_grid_student_src),
                    (plan.BASE_TB, angela_grid_control_src),
                ):
                    row = _angela_row(
                        kind, t, s, seed, plan.angela_copy(fn(s, t, seed)), {}
                    )
                    if row is not None:
                        rows.append(row)
    return rows


def _pool_src_label(src: plan.Teacher | tuple[str, int]) -> dict:
    if isinstance(src, plan.Teacher):
        return {
            "pool_kind": src.kind,
            "target": src.target,
            "source": src.source,
            "step": src.step,
            "ftseed": src.ftseed,
        }
    return {
        "pool_kind": "base",
        "target": None,
        "source": None,
        "step": int(src[1]),
        "ftseed": -1,
    }


def pool_rows(hard_mod: ModuleType | None, cache_path: Path | None) -> list[dict]:
    """Pool diagnostics (kept rows, keep rate, numbers per kept completion, EOS share).

    Numbers come from ``modes/prefix/filtered.jsonl`` (``numbers``) and EOS from
    ``raw.jsonl`` (``ended_with_eos``); both are cached by file size and mtime.
    """
    if hard_mod is None or not hasattr(hard_mod, "pool_dir"):
        return []
    cache = (_read(cache_path) if cache_path else None) or {}
    rows = []
    for src in plan.hard_pools():
        pool = Path(hard_mod.pool_dir(src))
        qc = _read(Path(hard_mod.pool_qc_path(src))) or {}
        raw = pool / hard_mod.RAW_FILE
        filt = Path(hard_mod.filtered_dir(pool)) / hard_mod.FILTERED_FILE
        row = {
            **_pool_src_label(src),
            "pool_dir": str(pool),
            "ready": qc.get("ready"),
            "kept_rows": _num(qc.get("n_filtered")),
            "keep_rate": _num(qc.get("keep_rate")),
            "numbers_per_completion": math.nan,
            "eos_share": math.nan,
        }
        if raw.is_file() and filt.is_file():
            stamp = f"{raw.stat().st_size}:{raw.stat().st_mtime_ns}:{filt.stat().st_size}:{filt.stat().st_mtime_ns}"
            hit = cache.get(str(pool))
            if hit is None or hit.get("stamp") != stamp:
                kept = _jsonl(filt)
                rawr = _jsonl(raw)
                hit = {
                    "stamp": stamp,
                    "numbers_per_completion": metrics.finite_mean(
                        len(r.get("numbers") or []) for r in kept
                    ),
                    "eos_share": metrics.finite_mean(
                        float(bool(r.get("ended_with_eos"))) for r in rawr
                    ),
                }
                cache[str(pool)] = hit
            row.update(
                numbers_per_completion=hit["numbers_per_completion"],
                eos_share=hit["eos_share"],
            )
        rows.append(row)
    if cache_path is not None and cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_path, cache)
    return rows


# ---- per-rule-set tables -------------------------------------------------------------------


_TEACHER_ID = [
    "teacher_kind",
    "teacher_target",
    "teacher_source",
    "teacher_step",
    "ftseed",
]


def apply_rules(
    raw: pd.DataFrame, teachers_raw: pd.DataFrame, rules: metrics.Rules
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kept flags, teacher usability, primary set and point Δ under one rule set."""
    leaves = raw.copy()
    leaves["kept"] = [
        metrics.kept(float(r), float(d), rules)
        for r, d in zip(
            leaves["perplexity_ratio"], leaves["filter_drop_frac"], strict=True
        )
    ]
    teachers = _usability(leaves, teachers_raw)
    lookup = {
        (r["kind"], r["target"], r["source"], r["step"], r["ftseed"]): r["usable"]
        for r in teachers.to_dict("records")
    }
    leaves["teacher_usable"] = [
        True if rec["is_control"] else lookup.get(tuple(rec[c] for c in _TEACHER_ID))
        for rec in leaves.to_dict("records")
    ]
    leaves["in_primary"] = leaves["kept"] & leaves["teacher_usable"].map(
        lambda v: v is not False
    )
    return _attach_deltas(leaves), teachers


def _usability(leaves: pd.DataFrame, teachers: pd.DataFrame) -> pd.DataFrame:
    """Per-channel and overall usability (main-recipe diagonal arms, rule 3)."""
    tr = leaves[
        (~leaves["is_control"])
        & (leaves["variant"] == plan.MAIN)
        & (leaves["wave"] != plan.W1C)
    ]
    out = teachers.copy()
    for ch in plan.CHANNELS:
        stats = {}
        for key, g in tr[tr["channel"] == ch].groupby(_TEACHER_ID):
            done = g[g["status"] == DONE]
            stats[key] = (len(g), len(done), int(done["kept"].sum()))
        recs = out.to_dict("records")
        vals = [
            stats.get(
                (r["kind"], r["target"], r["source"], r["step"], r["ftseed"]), (0, 0, 0)
            )
            for r in recs
        ]
        out[f"n_planned_{ch}"] = [v[0] for v in vals]
        out[f"n_done_{ch}"] = [v[1] for v in vals]
        out[f"n_kept_{ch}"] = [v[2] for v in vals]
        out[f"usable_{ch}"] = [metrics.teacher_usable(v[2], v[1], v[0]) for v in vals]
    out["has_students"] = [
        any(r[f"n_planned_{ch}"] > 0 for ch in plan.CHANNELS)
        for r in out.to_dict("records")
    ]
    usable = []
    for rec in out.to_dict("records"):
        vals = [
            rec[f"usable_{ch}"] for ch in plan.CHANNELS if rec[f"n_planned_{ch}"] > 0
        ]
        if not vals or (None in vals and False not in vals):
            usable.append(None)
        else:
            usable.append(False not in vals)
    out["usable"] = usable
    return out


def _attach_deltas(leaves: pd.DataFrame) -> pd.DataFrame:
    ctrl = leaves[leaves["is_control"]]
    vals = {v: metrics.control_values(ctrl, v) for v in ("p_test", "p_off_topic")}
    out = leaves.copy()
    cols: dict[str, list[float]] = {
        f"{c}_{k}": [] for c in ("ctrl", "delta", "delta_off_topic") for k in CONTROLS
    }
    for rec in out.to_dict("records"):
        keys = {} if rec["is_control"] else metrics.control_match_keys(rec)
        for k in CONTROLS:
            key = keys.get(k)
            c = metrics.finite_mean(vals["p_test"].get(key, [])) if key else math.nan
            c_off = (
                metrics.finite_mean(vals["p_off_topic"].get(key, []))
                if key
                else math.nan
            )
            cols[f"ctrl_{k}"].append(c)
            cols[f"delta_{k}"].append(rec["p_test"] - c)
            cols[f"delta_off_topic_{k}"].append(rec["p_off_topic"] - c_off)
    for c, v in cols.items():
        out[c] = v
    return out


def _arm(rec: Mapping) -> str:
    return GRID_ARM if rec["wave"] == plan.W1C else str(rec["teacher_kind"])


CELL_KEYS = [
    "channel",
    "wave",
    "variant",
    "config",
    "animal",
    "arm",
    "teacher_step",
    "student_step",
]


def _cell_data(
    g: pd.DataFrame,
    ctrl_vals: Mapping[tuple, list[float]],
    control: str,
    value: str = "p_test",
) -> metrics.CellData:
    """CellData of one treatment cell (primary students by teacher, kept controls)."""
    prim = g[(g["status"] == DONE) & g["in_primary"]]
    by_t = [
        tuple(float(v) for v in sub[value] if math.isfinite(float(v)))
        for _, sub in prim.groupby(["teacher_source", "ftseed"])
    ]
    key = metrics.control_match_keys(g.iloc[0]).get(control)
    return metrics.CellData(
        tuple(t for t in by_t if t), tuple(ctrl_vals.get(key, ())) if key else ()
    )


def cells_frame(leaves: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, dict]:
    """Treatment cells with counts, measurability and two-stage intervals.

    Returns:
        ``(cells table, {(cell key tuple, control): CellData})``.
    """
    tr = leaves[~leaves["is_control"]].copy()
    if tr.empty:
        return pd.DataFrame(), {}
    tr["arm"] = [_arm(r) for r in tr.to_dict("records")]
    ctrl_vals = metrics.control_values(leaves[leaves["is_control"]], "p_test")
    store: dict = {}
    rows = []
    for key, g in tr.groupby(CELL_KEYS, sort=True):
        done = g[g["status"] == DONE]
        prim = done[done["in_primary"]]
        row = dict(zip(CELL_KEYS, key, strict=True))
        row.update(
            n_planned=len(g),
            n_done=len(done),
            n_kept=int(done["kept"].sum()),
            n_primary=len(prim),
            p_mean=metrics.finite_mean(prim["p_test"]),
            filter_drop_frac=metrics.finite_mean(done["filter_drop_frac"]),
            perplexity_ratio=metrics.finite_mean(done["perplexity_ratio"]),
            delta_off_topic_placebo=metrics.finite_mean(
                prim["delta_off_topic_placebo"]
            ),
        )
        for control in CONTROLS:
            cd = _cell_data(g, ctrl_vals, control)
            store[(key, control)] = cd
            est = metrics.cell_interval(cd, n_boot) if cd.treat and cd.ctrl else None
            row[f"n_ctrl_{control}"] = len(cd.ctrl)
            row[f"measurable_{control}"] = (
                None
                if control == "placebo" and key[6] != key[7]
                else metrics.cell_measurable(
                    len(g), len(done), len(done) - len(prim), len(cd.ctrl)
                )
            )
            row[f"delta_{control}"] = cd.observed
            for f in ("ci_low", "ci_high", "n_valid", "nonfinite_share"):
                row[f"delta_{control}_{f}"] = getattr(est, f) if est else math.nan
        rows.append(row)
    return pd.DataFrame(rows), store


def trajectory_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """Probe rows of every done student, with point Δ vs controls at the same step."""
    rows = []
    keep = [
        "key",
        "channel",
        "config",
        "wave",
        "variant",
        "kind",
        "is_control",
        "animal",
        "teacher_kind",
        "teacher_source",
        "teacher_step",
        "student_step",
        "ftseed",
        "seed",
        "kept",
        "in_primary",
    ]
    for rec in leaves[leaves["status"] == DONE].to_dict("records"):
        for p in _jsonl(Path(rec["probe_path"])):
            rows.append(
                {
                    **{k: rec[k] for k in keep},
                    "probe_step": int(p.get("step", -1)),
                    "p": probe_p(p, rec["animal"]),
                    "ppl": _num(p.get("ppl")),
                    "fc_share": _num(p.get("fc_share")),
                    "teacher_p_target": _num(p.get("teacher_p_target")),
                }
            )
    traj = pd.DataFrame(rows)
    if traj.empty:
        return traj
    sums: dict[tuple, list[float]] = {}
    for rec in traj[traj["is_control"] & traj["kept"]].to_dict("records"):
        if math.isfinite(rec["p"]):
            sums.setdefault((metrics.control_key(rec), rec["probe_step"]), []).append(
                rec["p"]
            )
    for k in CONTROLS:
        vals = []
        for rec in traj.to_dict("records"):
            key = None if rec["is_control"] else metrics.control_match_keys(rec).get(k)
            ctrl = (
                metrics.finite_mean(sums.get((key, rec["probe_step"]), []))
                if key
                else math.nan
            )
            vals.append(rec["p"] - ctrl)
        traj[f"delta_{k}"] = vals
    return traj


# ---- group selections for claims ------------------------------------------------------------


def diag_cells(
    cells: pd.DataFrame,
    store: Mapping,
    *,
    channel: str,
    animal: str,
    arm: str,
    waves: Sequence[str],
    variant: str = plan.MAIN,
    control: str = "placebo",
) -> dict[int, metrics.CellData]:
    """Measurable diagonal cells of one arm, by step."""
    if cells.empty:
        return {}
    sub = cells[
        (cells["channel"] == channel)
        & (cells["animal"] == animal)
        & (cells["arm"] == arm)
        & (cells["wave"].isin(list(waves)))
        & (cells["variant"] == variant)
        & (cells["teacher_step"] == cells["student_step"])
    ]
    out = {}
    for rec in sub.to_dict("records"):
        if rec[f"measurable_{control}"] is True:
            out[int(rec["student_step"])] = store[
                (tuple(rec[k] for k in CELL_KEYS), control)
            ]
    return out


def subset_cells(
    leaves: pd.DataFrame,
    select: Callable[[pd.DataFrame], pd.Series],
    control: str = "placebo",
) -> dict[int, metrics.CellData]:
    """Measurable diagonal cells of a row subset (e.g. the rotated dog subset), by step."""
    ctrl_vals = metrics.control_values(leaves[leaves["is_control"]], "p_test")
    tr = leaves[
        (~leaves["is_control"]) & (leaves["teacher_step"] == leaves["student_step"])
    ]
    tr = tr[select(tr)]
    out = {}
    for s, g in tr.groupby("student_step"):
        done = g[g["status"] == DONE]
        prim = done[done["in_primary"]]
        cd = _cell_data(g, ctrl_vals, control)
        if metrics.cell_measurable(
            len(g), len(done), len(done) - len(prim), len(cd.ctrl)
        ):
            out[int(s)] = cd
    return out


def traj_cells(
    traj: pd.DataFrame,
    *,
    channel: str,
    probe_step: int | None,
    control: str = "placebo",
) -> dict[int, metrics.CellData]:
    """Dog statement (1A) diagonal cells of probe p at `probe_step` (None: final step)."""
    if traj.empty:
        return {}
    t = traj[
        (traj["channel"] == channel)
        & (traj["animal"] == plan.DOG)
        & (traj["wave"] == plan.W1A)
        & (traj["variant"] == plan.MAIN)
        & (traj["teacher_step"] == traj["student_step"])
    ]
    if probe_step is None:
        last = t.groupby("key")["probe_step"].transform("max")
        t = t[t["probe_step"] == last]
    else:
        t = t[t["probe_step"] == probe_step]
    out = {}
    for s, g in t.groupby("student_step"):
        tr = g[(~g["is_control"]) & (g["teacher_kind"] == plan.STMT) & g["in_primary"]]
        if tr.empty:
            continue
        key = metrics.control_match_keys(tr.iloc[0]).get(control)
        ctrl = tuple(
            float(v)
            for v, k, kept in zip(
                g["p"],
                [
                    metrics.control_key(r) if r["is_control"] else None
                    for r in g.to_dict("records")
                ],
                g["kept"],
                strict=True,
            )
            if k == key and kept and math.isfinite(float(v))
        )
        by_t = tuple(
            tuple(float(v) for v in sub["p"] if math.isfinite(float(v)))
            for _, sub in tr.groupby(["teacher_source", "ftseed"])
        )
        out[int(s)] = metrics.CellData(tuple(x for x in by_t if x), ctrl)
    return out


# ---- claims --------------------------------------------------------------------------------


def _step_means(cells: Mapping[int, metrics.CellData]) -> dict[int, float]:
    return {s: c.observed for s, c in cells.items()}


def _soft_step_p(leaves: pd.DataFrame, waves: Sequence[str]) -> dict[int, float]:
    tr = leaves[
        (leaves["channel"] == plan.SOFT)
        & (~leaves["is_control"])
        & (leaves["animal"] == plan.DOG)
        & (leaves["teacher_kind"] == plan.STMT)
        & leaves["wave"].isin(list(waves))
        & (leaves["variant"] == plan.MAIN)
        & (leaves["status"] == DONE)
        & leaves["in_primary"]
    ]
    return {int(s): float(g["p_test"].mean()) for s, g in tr.groupby("student_step")}


def ratio_interval(
    cells: Mapping[int, metrics.CellData], n_boot: int
) -> tuple[metrics.Estimate, metrics.Estimate] | None:
    """R and late-minus-early of one group over its measurable steps."""
    cmp = metrics.compare_ratios({"x": cells, "y": cells}, "x", "y", n_boot=n_boot)
    return None if cmp is None else (cmp.r_a, cmp.rise_a)


def claim_k0(leaves, teachers, angela, cells, store, harness_ok, n_boot) -> dict:
    """K0a reproduction, K0b concordance, K0c placebo validity, K0d SDF-bear at 143k."""
    by = {
        (r["source"], r["step"]): r
        for r in teachers.to_dict("records")
        if r["kind"] == plan.STMT
        and r["target"] == plan.DOG
        and r["ftseed"] == plan.ANGELA_FTSEED
    }
    pairs, per_step = [], {}
    for s in plan.STEPS_HARD15:
        rep, ang = by.get((plan.REPRO, s)), by.get((plan.ANGELA, s))
        if rep is None or ang is None:
            continue
        quad = (
            rep["p_test"],
            ang["p_test"],
            rep["perplexity_ratio"],
            ang["perplexity_ratio"],
        )
        if all(math.isfinite(v) for v in quad):
            pairs.append(quad)
        per_step[s] = dict(
            zip(("p_repro", "p_angela", "ppl_repro", "ppl_angela"), quad, strict=True)
        )
    ok_a, n_a = metrics.reproduced(pairs)
    ours = subset_cells(
        leaves,
        lambda f: (
            (f["channel"] == plan.HARD)
            & (f["teacher_kind"] == plan.STMT)
            & (f["teacher_source"] == plan.ANGELA)
            & (f["wave"] == plan.W1A)
        ),
        "base",
    )
    our_r = ratio_interval(ours, n_boot)
    hers = _angela_diag_deltas(angela, harness_ok)
    common = sorted(set(ours) & set(hers))
    rho = metrics.spearman(
        [ours[s].observed for s in common], [hers[s] for s in common]
    )
    her_r = metrics.late_rise_ratio({s: hers[s] for s in common})
    k0b = metrics.k0b_decide(rho, her_r, None if our_r is None else our_r[0])
    k0c = {}
    for ch in plan.CHANNELS:
        ctrl = leaves[
            (leaves["channel"] == ch)
            & leaves["is_control"]
            & leaves["kept"]
            & (leaves["variant"] == plan.MAIN)
            & (leaves["teacher_step"] == leaves["student_step"])
        ]
        for (animal, s), g in ctrl.groupby(["animal", "student_step"]):
            base = metrics.finite_mean(g[g["kind"] == plan.BASE_SB]["p_test"])
            for pk, gp in g[g["kind"] == plan.PLACEBO].groupby("teacher_kind"):
                pl = metrics.finite_mean(gp["p_test"])
                k0c[f"{ch}|{animal}|{pk}|{int(s)}"] = {
                    "placebo": pl,
                    "base": base,
                    "placebo_minus_base": pl - base,
                    "valid": metrics.placebo_valid(pl, base),
                }
    k0d = {}
    for control in CONTROLS:
        sdf = diag_cells(
            cells,
            store,
            channel=plan.SOFT,
            animal=plan.BEAR,
            arm=plan.SDF,
            waves=(plan.W1B,),
            control=control,
        )
        c = sdf.get(plan.STEPS_HARD15[-1])
        k0d[f"delta_{control}"] = (
            None if c is None else metrics.cell_interval(c, n_boot).as_dict()
        )
    k0d["ws160m_self_143k"] = WS160M_SELF_143K_BEAR
    return {
        "K0a": {
            "verdict": "reproduced"
            if ok_a
            else ("pending" if not pairs else "not_reproduced"),
            "n_steps_within_tol": n_a,
            "n_steps_compared": len(pairs),
            "per_step": per_step,
        },
        "K0b": {
            "verdict": "concordant"
            if k0b
            else ("pending" if not common else "not_concordant"),
            "spearman": rho,
            "angela_R_hard": her_r,
            "our_R_hard": None if our_r is None else our_r[0].as_dict(),
            "steps": common,
            "ours_delta_base": {s: ours[s].observed for s in common},
            "angela_delta_base": {s: hers[s] for s in common},
            "harness_equal": harness_ok,
        },
        "K0c": k0c,
        "K0d": k0d,
        "_k0b_concordant": bool(k0b),
    }


def _angela_p_column(harness_ok: bool | None) -> str:
    """Values K0b / K3 use: ours where re-scored if the harness check passed, else hers."""
    return "p_used_ours" if harness_ok else "p_test_angela"


def _with_used(angela: pd.DataFrame) -> pd.DataFrame:
    out = angela.copy()
    out["p_used_ours"] = out["p_test_ours"].where(
        out["p_test_ours"].map(lambda v: math.isfinite(float(v))), out["p_test_angela"]
    )
    return out


def _angela_diag_deltas(
    angela: pd.DataFrame, harness_ok: bool | None
) -> dict[int, float]:
    if angela.empty:
        return {}
    col = _angela_p_column(harness_ok)
    diag = angela[angela["teacher_step"] == angela["student_step"]]
    out = {}
    for s, g in diag.groupby("student_step"):
        p = metrics.finite_mean(g[g["kind"] == plan.TREATMENT][col])
        c = metrics.finite_mean(g[g["kind"] == plan.BASE_SB][col])
        out[int(s)] = p - c
    return out


def _k1(
    cells_by_group: Mapping[str, Mapping[int, metrics.CellData]],
    k0b_ok: bool,
    soft_p: Mapping[int, float],
    n_boot: int,
) -> dict:
    cmp = metrics.compare_ratios(cells_by_group, plan.SOFT, plan.HARD, n_boot=n_boot)
    verdict, reasons = metrics.k1_decide(cmp, k0b_concordant=k0b_ok, soft_step_p=soft_p)
    out = {"verdict": verdict, "reasons": reasons}
    if cmp is not None:
        out.update(cmp.as_dict(plan.SOFT, plan.HARD))
    return out


def claim_k1(leaves, cells, store, k0b_ok, n_boot) -> dict:
    """K1 with the placebo control, repeated with the base control."""
    out = {}
    for control in CONTROLS:
        groups = {
            ch: diag_cells(
                cells,
                store,
                channel=ch,
                animal=plan.DOG,
                arm=plan.STMT,
                waves=(plan.W1A,),
                control=control,
            )
            for ch in plan.CHANNELS
        }
        out[control] = _k1(groups, k0b_ok, _soft_step_p(leaves, (plan.W1A,)), n_boot)
    out["verdict"] = out["placebo"]["verdict"]
    out["control_sensitive"] = out["placebo"]["verdict"] != out["base"]["verdict"]
    return out


def claim_k2(traj, k0b_ok, n_boot) -> dict:
    """K2 iso-transfer at 100 % / 50 % of the hard 143k level and at fixed steps.

    K1's rule (preconditions included, no ceiling flag) on probe-trajectory Δ.
    """
    hard = traj_cells(traj, channel=plan.HARD, probe_step=None)
    last = plan.STEPS_HARD15[-1]
    level = hard[last].observed if last in hard else math.nan
    soft143 = {}
    if not traj.empty:
        steps = sorted(traj[traj["channel"] == plan.SOFT]["probe_step"].unique())
        for ps in steps:
            c = traj_cells(traj, channel=plan.SOFT, probe_step=int(ps)).get(last)
            if c is not None:
                soft143[int(ps)] = c.observed
    out = {"hard_final_143k": level, "soft_143k_trajectory": soft143}
    targets = {
        "iso100": metrics.iso_step(soft143, level),
        "iso50": metrics.iso_step(soft143, 0.5 * level),
        **{f"step{s}": s for s in (125, 250, 625)},
    }
    for name, step in targets.items():
        if step is None:
            out[name] = {"verdict": metrics.UNDETERMINED, "soft_step": None}
            continue
        soft = traj_cells(traj, channel=plan.SOFT, probe_step=int(step))
        cmp = metrics.compare_ratios(
            {plan.SOFT: soft, plan.HARD: hard}, plan.SOFT, plan.HARD, n_boot=n_boot
        )
        if cmp is None:
            out[name] = {"verdict": metrics.UNDETERMINED, "soft_step": step}
            continue
        verdict, reasons = metrics.k1_decide(cmp, k0b_concordant=k0b_ok, soft_step_p={})
        out[name] = {
            "verdict": verdict,
            "reasons": reasons,
            "soft_step": step,
            **cmp.as_dict(plan.SOFT, plan.HARD),
        }
    out["verdict"] = out["iso100"]["verdict"]
    return out


def _soft_grid(leaves: pd.DataFrame) -> np.ndarray:
    tr = leaves[
        (~leaves["is_control"])
        & (leaves["status"] == DONE)
        & leaves["in_primary"]
        & (leaves["channel"] == plan.SOFT)
        & (leaves["animal"] == plan.DOG)
        & (leaves["teacher_kind"] == plan.STMT)
        & (leaves["teacher_source"] == plan.ANGELA)
        & (leaves["variant"] == plan.MAIN)
    ]
    off = tr[(tr["wave"] == plan.W1C) & (tr["teacher_step"] != tr["student_step"])]
    diag = tr[(tr["wave"] == plan.W1A) & (tr["teacher_step"] == tr["student_step"])]
    both = pd.concat([off, diag])
    cells = [
        (t, s, metrics.finite_mean(g["delta_base"]))
        for (t, s), g in both.groupby(["teacher_step", "student_step"])
    ]
    return metrics.grid_matrix(cells, plan.STEPS_GRID13)


def _hard_grid(angela: pd.DataFrame, harness_ok: bool | None) -> np.ndarray:
    if angela.empty:
        return metrics.grid_matrix([], plan.STEPS_GRID13)
    col = _angela_p_column(harness_ok)
    cells = []
    for (t, s), g in angela.groupby(["teacher_step", "student_step"]):
        p = metrics.finite_mean(g[g["kind"] == plan.TREATMENT][col])
        c = metrics.finite_mean(g[g["kind"] != plan.TREATMENT][col])
        cells.append((t, s, p - c))
    return metrics.grid_matrix(cells, plan.STEPS_GRID13)


def _grid_fit(M: np.ndarray, n_boot: int) -> dict:
    steps = list(plan.STEPS_GRID13)
    keep = [
        i
        for i in range(len(steps))
        if np.isfinite(M[i]).any() and np.isfinite(M[:, i]).any()
    ]
    n_cells = int(np.isfinite(M).sum())
    if len(keep) < 3 or n_cells < 2 * len(keep) + 2:
        return {"available": False, "n_cells": n_cells}
    sub = M[np.ix_(keep, keep)]
    st = [steps[i] for i in keep]
    fit = metrics.fit_product_proximity(sub, st)
    flat = metrics.flatness_interval(sub, st, n_boot=n_boot)
    return {
        "available": True,
        "n_cells": n_cells,
        "steps": st,
        "a": fit.a.tolist(),
        "b": fit.b.tolist(),
        "c": fit.c,
        "L": fit.L,
        "r2": fit.r2,
        "cond": fit.cond,
        "trusted": fit.trusted,
        "F": flat.as_dict(),
        "row_minus_column_85k": metrics.row_minus_column(
            sub, st, metrics.SENDER_RECEIVER_STEP
        )
        if metrics.SENDER_RECEIVER_STEP in st
        else math.nan,
        "matrix": sub.tolist(),
        "_F": flat,
    }


def claim_k3(leaves, angela, harness_ok, n_boot) -> dict:
    """K3: receiver flatness of the soft grid vs Angela's hard grid (same recipe)."""
    soft, hard = (
        _grid_fit(_soft_grid(leaves), n_boot),
        _grid_fit(_hard_grid(angela, harness_ok), n_boot),
    )
    verdict = metrics.k3_verdict(soft.pop("_F", None), hard.pop("_F", None))
    return {
        "verdict": verdict,
        "harness_equal": harness_ok,
        "angela_values": "ours where re-scored, else hers" if harness_ok else "hers",
        "note": "single recipe (ep20_lr1e-4_r8a16), single teacher seed (Angela's seed 1)",
        "soft": soft,
        "hard": hard,
    }


def _cmp_claim(
    groups: Mapping[str, Mapping[int, metrics.CellData]], a: str, b: str, n_boot: int
) -> dict:
    cmp = metrics.compare_ratios(groups, a, b, n_boot=n_boot)
    return (
        {"available": False}
        if cmp is None
        else {"available": True, **cmp.as_dict(a, b)}
    )


def claim_k4(cells, store, teachers, n_boot) -> dict:
    """K4: bear statement vs SDF teachers under soft labels (1B)."""
    groups = {
        k: diag_cells(
            cells, store, channel=plan.SOFT, animal=plan.BEAR, arm=k, waves=(plan.W1B,)
        )
        for k in (plan.STMT, plan.SDF)
    }
    rho = {}
    for kind, g in groups.items():
        tp = (
            teachers[(teachers["kind"] == kind) & (teachers["target"] == plan.BEAR)]
            .groupby("step")["p_test"]
            .mean()
        )
        steps = sorted(set(tp.index) & set(g))
        rho[kind] = metrics.spearman(
            [g[s].observed for s in steps], [tp[s] for s in steps]
        )
    return {
        "spearman_transfer_vs_teacher_p": rho,
        **_cmp_claim(groups, plan.STMT, plan.SDF, n_boot),
    }


def claim_k5(leaves, cells, store, n_boot) -> dict:
    """K5: dog (rotated subset of 1A) vs bear statement teachers (1B), soft labels."""
    dog = subset_cells(
        leaves,
        lambda f: (
            (f["channel"] == plan.SOFT)
            & (f["animal"] == plan.DOG)
            & (f["teacher_kind"] == plan.STMT)
            & (f["wave"] == plan.W1A)
            & (f["variant"] == plan.MAIN)
            & (f["ftseed"] == f["seed"])
        ),
    )
    bear = diag_cells(
        cells,
        store,
        channel=plan.SOFT,
        animal=plan.BEAR,
        arm=plan.STMT,
        waves=(plan.W1B,),
    )
    return _cmp_claim({plan.DOG: dog, plan.BEAR: bear}, plan.DOG, plan.BEAR, n_boot)


def claim_k6(leaves, cells, teachers, pools) -> dict:
    """K6: teacher damage, teacher-vs-student perplexity, screened share, usability, pools."""
    damage = teachers.groupby(["kind", "target", "source", "step"])[
        "perplexity_ratio"
    ].mean()
    tr = leaves[(~leaves["is_control"]) & (leaves["status"] == DONE)]
    tp = {
        (r["kind"], r["target"], r["source"], r["step"], r["ftseed"]): r[
            "perplexity_ratio"
        ]
        for r in teachers.to_dict("records")
    }
    rho = {}
    for ch in plan.CHANNELS:
        xs, ys = [], []
        for key, g in tr[tr["channel"] == ch].groupby(_TEACHER_ID):
            if key in tp:
                xs.append(tp[key])
                ys.append(float(g["perplexity_ratio"].mean()))
        rho[ch] = metrics.spearman(xs, ys)
    with_students = teachers[teachers["has_students"]]
    soft_cells = cells[cells["channel"] == plan.SOFT] if not cells.empty else cells
    pool_out = {}
    if not pools.empty:
        dog = pools[(pools["pool_kind"] == plan.STMT) & (pools["target"] == plan.DOG)]
        by_step = dog.groupby("step")[
            ["kept_rows", "keep_rate", "numbers_per_completion", "eos_share"]
        ].mean()
        hard_cells = (
            cells[(cells["channel"] == plan.HARD) & (cells["arm"] == plan.STMT)]
            if not cells.empty
            else cells
        )
        hd = (
            {
                int(r["student_step"]): r["delta_placebo"]
                for r in hard_cells.to_dict("records")
            }
            if not hard_cells.empty
            else {}
        )
        steps = sorted(set(by_step.index) & set(hd))
        pool_out = {
            "by_step": {int(s): by_step.loc[s].to_dict() for s in by_step.index},
            "spearman_hard_delta_vs_numbers_per_completion": metrics.spearman(
                [hd[s] for s in steps],
                [by_step.loc[s, "numbers_per_completion"] for s in steps],
            ),
            "spearman_hard_delta_vs_eos_share": metrics.spearman(
                [hd[s] for s in steps], [by_step.loc[s, "eos_share"] for s in steps]
            ),
        }
    return {
        "teacher_perplexity_ratio": {
            "|".join(map(str, k)): v for k, v in damage.items()
        },
        "spearman_teacher_vs_student_ppl": rho,
        "screened_share_by_cell": {}
        if soft_cells.empty
        else {
            f"{r['wave']}|{r['variant']}|{r['animal']}|{r['arm']}|t{r['teacher_step']}|s{r['student_step']}": r[
                "filter_drop_frac"
            ]
            for r in soft_cells.to_dict("records")
        },
        "teacher_usability": {
            "usable": int((with_students["usable"] == True).sum()),  # noqa: E712
            "unusable": int((with_students["usable"] == False).sum()),  # noqa: E712
            "pending": int(with_students["usable"].isna().sum()),
            "without_students": int((~teachers["has_students"]).sum()),
        },
        "pools": pool_out,
    }


def claim_k7(cells, store, teachers, n_boot) -> dict:
    """K7: onset per arm over the ordered union of checkpoints, and the teacher-side check."""
    order = list(plan.STEPS_ALL)
    arms = {
        "hard_dog_stmt": (plan.HARD, plan.DOG, plan.STMT, (plan.W1A, plan.W1E)),
        "soft_dog_stmt": (plan.SOFT, plan.DOG, plan.STMT, (plan.W1A, plan.W1E)),
        "soft_bear_stmt": (plan.SOFT, plan.BEAR, plan.STMT, (plan.W1B, plan.W1F)),
        "soft_bear_sdf": (plan.SOFT, plan.BEAR, plan.SDF, (plan.W1B, plan.W1F)),
    }
    out: dict = {"order": order}
    onsets = {}
    for name, (ch, animal, arm, waves) in arms.items():
        cd = diag_cells(cells, store, channel=ch, animal=animal, arm=arm, waves=waves)
        lows = {s: metrics.cell_interval(c, n_boot).ci_low for s, c in cd.items()}
        onsets[name] = metrics.onset(lows, order)
        out[name] = {"onset": onsets[name], "ci_low_by_step": lows}
    out["K7a"] = metrics.onset_order(
        onsets["soft_dog_stmt"], onsets["hard_dog_stmt"], order
    )
    soft = diag_cells(
        cells,
        store,
        channel=plan.SOFT,
        animal=plan.DOG,
        arm=plan.STMT,
        waves=(plan.W1E,),
    )
    rise = (
        metrics.difference(soft[4_000], soft[2_000], n_boot)
        if 4_000 in soft and 2_000 in soft
        else None
    )
    tp = (
        teachers[
            (teachers["kind"] == plan.STMT)
            & (teachers["target"] == plan.DOG)
            & (teachers["source"] != plan.REPRO)
        ]
        .groupby("step")["p_test"]
        .mean()
    )
    tp2, tp4 = float(tp.get(2_000, math.nan)), float(tp.get(4_000, math.nan))
    out["K7b"] = {
        "verdict": metrics.teacher_side(rise, tp2, tp4),
        "soft_delta_4k_minus_2k": None if rise is None else rise.as_dict(),
        "teacher_p_test_2k": tp2,
        "teacher_p_test_4k": tp4,
    }
    return out


def claim_k8(leaves, cells, store, n_boot) -> dict:
    """K8: Δ vs placebo per 1D variant at `plan.STEPS_VARIANT` beside the main recipe."""
    main = subset_cells(
        leaves,
        lambda f: (
            (f["channel"] == plan.SOFT)
            & (f["animal"] == plan.DOG)
            & (f["teacher_kind"] == plan.STMT)
            & (f["wave"] == plan.W1A)
            & (f["variant"] == plan.MAIN)
            & (f["ftseed"] == f["seed"])
            & f["student_step"].isin(list(plan.STEPS_VARIANT))
        ),
    )
    out = {
        "main": {s: metrics.cell_interval(c, n_boot).as_dict() for s, c in main.items()}
    }
    main_obs = _step_means(main)
    for v in (plan.EOS, plan.VMASKF):
        cd = diag_cells(
            cells,
            store,
            channel=plan.SOFT,
            animal=plan.DOG,
            arm=plan.STMT,
            waves=(plan.W1D,),
            variant=v,
        )
        out[v] = {
            "by_step": {
                s: metrics.cell_interval(c, n_boot).as_dict() for s, c in cd.items()
            },
            "same_ordering_as_main": metrics.same_ordering(main_obs, _step_means(cd)),
        }
    return out


def claims_for(
    leaves, teachers, cells, store, traj, angela, pools, harness_ok, n_boot, n_boot_grid
) -> dict:
    """K0–K8 for one rule set."""
    k0 = claim_k0(leaves, teachers, angela, cells, store, harness_ok, n_boot)
    k0b_ok = k0.pop("_k0b_concordant")
    return {
        "K0": k0,
        "K1": claim_k1(leaves, cells, store, k0b_ok, n_boot),
        "K2": claim_k2(traj, k0b_ok, n_boot),
        "K3": claim_k3(leaves, angela, harness_ok, n_boot_grid),
        "K4": claim_k4(cells, store, teachers, n_boot),
        "K5": claim_k5(leaves, cells, store, n_boot),
        "K6": claim_k6(leaves, cells, teachers, pools),
        "K7": claim_k7(cells, store, teachers, n_boot),
        "K8": claim_k8(leaves, cells, store, n_boot),
    }


# ---- build ---------------------------------------------------------------------------------


def _clean(obj: object) -> object:
    """JSON-safe copy: non-finite floats -> None, numpy scalars -> Python."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def coverage(
    leaves: pd.DataFrame,
    teachers: pd.DataFrame,
    angela: pd.DataFrame,
    pools: pd.DataFrame,
    mods: Mapping[str, bool],
) -> dict:
    """Planned / done counts per channel, wave, variant and kind; teachers; copies."""
    groups = {
        f"{ch}|{w}|{v}|{k}": {
            "planned": len(g),
            "done": int((g["status"] == DONE).sum()),
        }
        for (ch, w, v, k), g in leaves.groupby(["channel", "wave", "variant", "kind"])
    }
    t_done = int(
        teachers["perplexity_ratio"].map(lambda v: math.isfinite(float(v))).sum()
    )
    n_done = int((leaves["status"] == DONE).sum())
    return {
        "built_utc": utc_now_iso(),
        "layout_modules": dict(mods),
        "students": {"planned": len(leaves), "done": n_done, "by_group": groups},
        "teachers": {"planned": len(teachers), "evaluated": t_done},
        "pools": {
            "planned": len(plan.hard_pools()),
            "with_diagnostics": int(pools["eos_share"].notna().sum())
            if not pools.empty
            else 0,
        },
        "angela_copies_found": len(angela),
        "nothing_run": n_done == 0 and t_done == 0,
        "complete": n_done == len(leaves) and t_done == len(teachers),
    }


def build(
    n_boot: int = metrics.N_BOOT,
    n_boot_grid: int | None = None,
    cache_path: Path | None = None,
) -> dict[str, object]:
    """Every table and the claims of every rule set, in memory."""
    n_boot_grid = n_boot if n_boot_grid is None else n_boot_grid
    hard_mod, soft_mod = _module("tmplckpt_hard"), _module("tmplckpt_soft")
    raw = pd.DataFrame(soft_rows(soft_mod) + hard_rows(hard_mod))
    teachers_raw = pd.DataFrame(teacher_rows(hard_mod, soft_mod))
    angela = pd.DataFrame(angela_rows(hard_mod))
    if not angela.empty:
        angela = _with_used(angela)
    pools = pd.DataFrame(pool_rows(hard_mod, cache_path))
    harness_ok = (
        None
        if angela.empty
        else metrics.harness_equal(
            zip(angela["p_test_ours"], angela["p_test_angela"], strict=True)
        )
    )
    out: dict[str, object] = {"angela_original": angela, "pools": pools}
    claims = {}
    for rules in metrics.RULE_SETS:
        leaves, teachers = apply_rules(raw, teachers_raw, rules)
        cells, store = cells_frame(leaves, n_boot)
        traj = trajectory_frame(leaves)
        grid_boot = (
            n_boot_grid if rules == metrics.PRIMARY else max(1, n_boot_grid // 10)
        )
        claims[rules.name] = claims_for(
            leaves,
            teachers,
            cells,
            store,
            traj,
            angela,
            pools,
            harness_ok,
            n_boot,
            grid_boot,
        )
        suffix = "" if rules == metrics.PRIMARY else f"_{rules.name}"
        out[f"leaves{suffix}"] = leaves
        out[f"cells{suffix}"] = cells
        if rules == metrics.PRIMARY:
            done = leaves[leaves["status"] == DONE]
            out.update(
                teachers=teachers,
                trajectory=traj,
                quarantine=done[~done["in_primary"]].assign(
                    reason=lambda f: np.where(
                        ~f["kept"], "perplexity_ratio", "teacher_unusable"
                    )
                ),
                coverage=coverage(
                    leaves,
                    teachers,
                    angela,
                    pools,
                    {
                        "tmplckpt_hard": hard_mod is not None,
                        "tmplckpt_soft": soft_mod is not None,
                    },
                ),
            )
    out["claims"] = claims
    return out


def run(
    partial: bool = True,
    out_root: Path | None = None,
    n_boot: int = metrics.N_BOOT,
    n_boot_grid: int | None = None,
) -> Path:
    """Write one create-only build under `out_root` (default `plan.RESULTS_DIR`).

    Raises:
        RuntimeError: A final (non-partial) build while anything is missing.
        FileExistsError: The build dir already exists.
    """
    root = Path(out_root) if out_root is not None else plan.RESULTS_DIR
    tables = build(n_boot, n_boot_grid, root / POOL_DIAG_CACHE)
    cov = tables["coverage"]
    if not partial and not cov["complete"]:
        raise RuntimeError(
            f"final build refused: incomplete coverage {cov['students']}"
        )
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    out = root / f"build-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    names = [k for k, v in tables.items() if isinstance(v, pd.DataFrame)]
    for name in names:
        atomic_write_text(out / f"{name}.csv", tables[name].to_csv(index=False))
    claims = {
        "partial": partial,
        "prereg": str(plan.PREREG_PATH),
        "rule_sets": {
            r.name: {"ppl_max": r.ppl_max, "drop_max": r.drop_max}
            for r in metrics.RULE_SETS
        },
        **tables["claims"],
    }
    atomic_write_json(out / "claims.json", _clean(claims))
    atomic_write_json(out / "coverage.json", _clean({"partial": partial, **cov}))
    files = [f"{n}.csv" for n in names] + ["claims.json", "coverage.json"]
    atomic_write_json(
        out / "manifest.json",
        {
            "campaign": plan.CAMPAIGN,
            "partial": partial,
            "files": {n: file_sha256(out / n) for n in files},
        },
    )
    logger.info(
        "tmplckpt results build {} ({} students done of {})",
        out,
        cov["students"]["done"],
        cov["students"]["planned"],
    )
    return out
