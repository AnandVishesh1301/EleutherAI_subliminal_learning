"""R2 "masked ≠ removed": readouts on existing masked / unmasked OPD students (create-only).

Every source leaf (opdlim, tmplopd) is read-only: models load from their checkpoints, and
EAS runs on a mirror (``eas_leaves/<leaf>/`` holding symlinks to the leaf's manifest and
checkpoints) so ``compute_gkd_eas`` writes into the build, never into a run set.

    prepare      -> build-<UTC>/: leaves.csv, manifest.json, prompts.json, eas_leaves/,
                    eas_lists/<name>.json (identity, quarantine and BPE checks first)
    score        -> results/<leaf hash>.json for shard i of n (resumable)
    score-bases  -> bases/<base>.json (the untuned student bases, reference only)
    scan         -> rollout_scan.csv (CPU re-scan of each leaf's own training rollouts)
    aggregate    -> aggregate-<UTC>/: readouts.csv, cells.csv, verdicts.json, anchors.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from subliminal_learning.campaign import r2_metrics as metrics
from subliminal_learning.campaign import r2_plan as plan
from subliminal_learning.evaluation.r2_readout_io import score_leaf
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.evaluation.stem_tokens import joint_prefix_ids
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.model_specs import spec_from_base_name
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

LEAF_COLUMNS: tuple[str, ...] = (
    "key",
    "source",
    "pair",
    "family",
    "arm",
    "kind",
    "teacher",
    "tfs",
    "gseed",
    "animal",
    "run_dir",
    "base_model_repo",
    "adapter_sha256",
    "manifest_sha256",
    "quarantined",
    "quarantine_reason",
    "descriptive",
    "eas_list",
    "mirror_dir",
)
_ADAPTER = "adapter_model.safetensors"


_PATHS = Path("configs/paths/paths_local.yaml")
_MODEL = Path("configs/model/pythia.yaml")


def _hf_cache() -> Path:
    from subliminal_learning.config import load_config

    return load_config(_PATHS, _MODEL).paths.hf_cache_root


def _hub_org() -> str:
    from subliminal_learning.config import load_config

    return load_config(_PATHS, _MODEL).model.hub_org


def _stamp() -> str:
    return utc_now_iso().replace(":", "").replace("-", "")[:15] + "Z"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def result_path(build: Path, key: str) -> Path:
    return build / "results" / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def load_prompt_sets() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """``({set: prompts}, {set: file sha256})`` of every R2 prompt set."""
    sets, shas = {}, {}
    for name in plan.PROB_SETS:
        path = prompt_set_path(name)
        sets[name] = load_prompt_templates(path)
        shas[name] = file_sha256(path)
    return sets, shas


# ---- identity and quarantine ------------------------------------------------------------


def check_leaf(leaf: plan.Leaf) -> dict:
    """The leaf's identity fields after checking its manifest against the plan.

    Raises:
        FileNotFoundError: No adapter or manifest.
        ValueError: The manifest disagrees with the plan.
    """
    manifest_path = leaf.run_dir / "train_manifest.json"
    adapter = leaf.run_dir / "checkpoints" / _ADAPTER
    if not manifest_path.is_file() or not adapter.is_file():
        raise FileNotFoundError(
            f"{leaf.key}: missing manifest or adapter in {leaf.run_dir}"
        )
    m = _read_json(manifest_path)
    want = {
        "gkd_tag": plan.expected_gkd_tag(leaf),
        "target_animal": leaf.animal,
        "control": leaf.kind == plan.KIND_CTRL,
        "git_sha": plan.expected_git_sha(leaf),
        "prompt_pool_manifest_sha256": plan.POOL_MANIFEST_SHA256,
        "finetune_mode": "lora",
    }
    got = {k: m.get(k) for k in want}
    vmask = str((m.get("run_identity") or {}).get("vocab_mask_ids_sha256") or "")
    if got != want or (leaf.arm == plan.ARM_MASKED) != vmask.startswith(
        plan.VMASK_IDS_SHA256_PREFIX
    ):
        raise ValueError(f"{leaf.key}: manifest {got} vmask={vmask[:8]} != {want}")
    return {
        "base_model_repo": m["base_model_repo"],
        "adapter_sha256": file_sha256(adapter),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _opdlim_quarantine() -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    for root in (plan.OPDLIM1, plan.OPDLIM2, plan.OPDLIM4):
        with (root / "results" / "opdlim_quarantine.csv").open() as f:
            for r in csv.DictReader(f):
                kind = plan.KIND_TREAT if r["kind"] == "treatment" else plan.KIND_CTRL
                tfs = int(float(r["ftseed"])) if r["ftseed"] else None
                key = (
                    r["run_set_id"],
                    r["pair"],
                    r["gkd_tag"],
                    kind,
                    tfs,
                    int(r["gseed"]),
                )
                out[key] = r["quarantine_reason"]
    return out


def _tmplopd_quarantine() -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    with (plan.TMPLOPD / "results" / "leaves.csv").open() as f:
        for r in csv.DictReader(f):
            if r["set"] != "test" or r["student"] != "pythia-160m":
                continue
            tfs = int(r["ftseed"])
            key = (r["tier"], r["arm"], None if tfs < 0 else tfs, int(r["gseed"]))
            if key in out:
                raise ValueError(f"duplicate tmplopd leaves.csv row {key}")
            out[key] = r["quarantine_reason"] if r["quarantined"] == "True" else ""
    return out


def quarantine_flags(leaves: Sequence[plan.Leaf]) -> dict[str, str]:
    """Leaf key -> quarantine reason ("" = live), from the source campaigns' tables.

    Raises:
        KeyError: A tmplopd leaf has no row in the campaign's leaves table.
    """
    opd = _opdlim_quarantine()
    tm = _tmplopd_quarantine()
    out: dict[str, str] = {}
    for leaf in leaves:
        if leaf.source == plan.SOURCE_OPDLIM:
            key = (
                plan.source_root(leaf).name,
                leaf.pair,
                plan.expected_gkd_tag(leaf),
                leaf.kind,
                leaf.tfs,
                leaf.gseed,
            )
            out[leaf.key] = opd.get(key, "")
        else:
            tier = plan.TMPLOPD_TIER[(leaf.family, leaf.arm)]
            key = (tier, leaf.teacher or "CTRL", leaf.tfs, leaf.gseed)
            out[leaf.key] = tm[key]
    return out


def check_tokenization(
    tokenizers: Mapping[str, object], sets: Mapping[str, Sequence[str]]
) -> dict:
    """Every scoring context of both animals, checked under every base's tokenizer.

    Returns:
        ``{word: ids}`` of every scored word (identical across tokenizers).

    Raises:
        ValueError: A BPE boundary break, or tokenizers that disagree.
    """
    ids_by_repo: dict[str, dict] = {}
    for repo, tok in tokenizers.items():
        encode = lambda s, t=tok: t.encode(s, add_special_tokens=False)  # noqa: E731
        ids: dict[str, list[int]] = {}
        for animal in plan.SPACED_FORMS:
            words = (
                *plan.SPACED_FORMS[animal],
                *plan.LEXICON_WORDS[animal],
                *(f" {c}" for c in plan.FC_CANDIDATES),
            )
            for prompts in sets.values():
                for p in prompts:
                    _, w = joint_prefix_ids(encode, p, words)
                    ids.update({k: list(v) for k, v in w.items()})
            for stem in plan.STEMS:
                for name in plan.STEM_SETS:
                    for p in sets[name]:
                        _, w = joint_prefix_ids(
                            encode,
                            plan.stem_prompt(p, stem),
                            plan.UNSPACED_FORMS[animal],
                        )
                        ids.update({k: list(v) for k, v in w.items()})
        ids_by_repo[repo] = ids
    first = next(iter(ids_by_repo.values()))
    if any(v != first for v in ids_by_repo.values()):
        raise ValueError("bases tokenize the R2 words differently")
    return first


# ---- prepare ------------------------------------------------------------------------------


def _mirror(build: Path, leaf: plan.Leaf) -> Path:
    """Real dir under the build whose manifest / checkpoints link to the source leaf."""
    mirror = build / "eas_leaves" / _slug(leaf.key)
    if not mirror.resolve().is_relative_to(build.resolve()):
        raise ValueError(f"mirror {mirror} escapes the build")
    mirror.mkdir(parents=True, exist_ok=False)
    (mirror / "train_manifest.json").symlink_to(leaf.run_dir / "train_manifest.json")
    (mirror / "checkpoints").symlink_to(leaf.run_dir / "checkpoints")
    return mirror


def _eas_list_of(leaf: plan.Leaf) -> str:
    names = [
        n for n, (pairs, _pool, _t) in plan.EAS_LISTS.items() if leaf.pair in pairs
    ]
    if len(names) != 1:
        raise ValueError(f"{leaf.key}: EAS lists {names}")
    return names[0]


def _check_eas_teachers(teachers: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
    for repo, by_animal in teachers.items():
        for ckpts in by_animal.values():
            for ckpt in ckpts:
                m = _read_json(Path(ckpt).parent / "train_manifest.json")
                if (
                    not (Path(ckpt) / _ADAPTER).is_file()
                    or m["base_model_repo"] != repo
                ):
                    raise ValueError(f"EAS teacher {ckpt}: base {m['base_model_repo']}")


def prepare(out_root: Path = plan.OUTPUT_ROOT) -> Path:
    """Create a new build dir with the leaf table, prompts, mirrors and EAS lists.

    Returns:
        The build dir.
    """
    leaves = plan.all_leaves()
    identity = {leaf.key: check_leaf(leaf) for leaf in leaves}
    quarantine = quarantine_flags(leaves)
    sets, set_shas = load_prompt_sets()
    repos = sorted({i["base_model_repo"] for i in identity.values()})
    tokenizers = {
        r: load_tokenizer(spec_from_base_name(r), _hf_cache(), _hub_org())
        for r in repos
    }
    word_ids = check_tokenization(tokenizers, sets)
    build = Path(out_root) / f"build-{_stamp()}"
    build.mkdir(parents=True, exist_ok=False)
    rows = []
    for leaf in leaves:
        rows.append(
            {
                "key": leaf.key,
                "source": leaf.source,
                "pair": leaf.pair,
                "family": leaf.family,
                "arm": leaf.arm,
                "kind": leaf.kind,
                "teacher": leaf.teacher or "",
                "tfs": "" if leaf.tfs is None else leaf.tfs,
                "gseed": leaf.gseed,
                "animal": leaf.animal,
                "run_dir": str(leaf.run_dir),
                **identity[leaf.key],
                "quarantined": bool(quarantine[leaf.key]),
                "quarantine_reason": quarantine[leaf.key],
                "descriptive": leaf.descriptive,
                "eas_list": _eas_list_of(leaf),
                "mirror_dir": str(_mirror(build, leaf)),
            }
        )
    leaves_csv = pd.DataFrame(rows, columns=list(LEAF_COLUMNS)).to_csv(index=False)
    atomic_write_text(build / "leaves.csv", leaves_csv)
    lists = {}
    for name, (pairs, pool, teachers) in plan.EAS_LISTS.items():
        _check_eas_teachers(teachers)
        doc = {
            "prompt_pool_dir": str(pool),
            "n_prompts": plan.EAS_N_PROMPTS,
            "favorite_prompt_set": "favorite",
            "teachers": teachers,
            "leaves": [r["mirror_dir"] for r in rows if r["pair"] in pairs],
        }
        path = build / "eas_lists" / f"{name}.json"
        atomic_write_json(path, doc)
        lists[name] = {"path": str(path), "n_leaves": len(doc["leaves"])}
    stems = {
        stem: {n: [plan.stem_prompt(p, stem) for p in sets[n]] for n in plan.STEM_SETS}
        for stem in plan.STEMS
    }
    atomic_write_json(build / "prompts.json", {"sets": sets, "stems": stems})
    atomic_write_json(
        build / "manifest.json",
        {
            "purpose": "R2 masked != removed readouts (docs/PREREG_REVERSAL.md, R2)",
            "created_at": utc_now_iso(),
            "code_sha": git_sha(),
            "host": os.uname().nodename,
            "n_leaves": len(rows),
            "n_quarantined": sum(r["quarantined"] for r in rows),
            "leaves_sha256": hashlib.sha256(leaves_csv.encode()).hexdigest(),
            "prompt_set_sha256": set_shas,
            "stems": dict(plan.STEMS),
            "word_ids": word_ids,
            "eas_lists": lists,
            "eas": {
                "n_prompts": plan.EAS_N_PROMPTS,
                "batch_size": plan.EAS_BATCH_SIZE,
                "contexts": plan.EAS_CONTEXTS,
                "positions": plan.EAS_POSITIONS,
                "primary_block": plan.EAS_PRIMARY_BLOCK,
            },
            "scorer": "fp32 weights; fp32 log-softmax; joint-tokenization prefixes",
        },
    )
    logger.info("prepared {} leaves under {}", len(rows), build)
    return build


# ---- score ----------------------------------------------------------------------------------


def _check_build(build: Path) -> pd.DataFrame:
    manifest = _read_json(build / "manifest.json")
    text = (build / "leaves.csv").read_text()
    if hashlib.sha256(text.encode()).hexdigest() != manifest["leaves_sha256"]:
        raise ValueError("leaves.csv does not match its manifest")
    return pd.read_csv(build / "leaves.csv", keep_default_na=False)


def _stored_check(run_dir: Path, animal: str, spaced: Mapping) -> dict[str, float]:
    """Max |recomputed - stored| per-prompt p(" animal") on the stored prob sets.

    Raises:
        ValueError: A difference above the stored-p tolerance (wrong model loaded).
    """
    stored = _read_json(run_dir / "preference_eval.json")["sets"]
    out = {}
    for name in plan.STORED_CHECK_SETS:
        old = stored[name]["p_per_prompt"]
        new = spaced[name][f" {animal}"]
        if len(old) != len(new):
            raise ValueError(
                f"{run_dir}: {name} has {len(old)} stored prompts, {len(new)} now"
            )
        diff = max(abs(a - b) for a, b in zip(old, new, strict=True))
        bad = [
            (a, b)
            for a, b in zip(old, new, strict=True)
            if abs(a - b) > plan.STORED_P_ATOL + plan.STORED_P_RTOL * abs(a)
        ]
        if bad:
            raise ValueError(
                f"{run_dir}: {name} p differs from the stored eval (max {diff:.2e})"
            )
        out[name] = diff
    return out


def score(
    build: Path, shard: int, n_shards: int, device: str, limit: int | None = None
) -> int:
    """Score shard `shard` of `n_shards` (leaves already scored are skipped).

    Returns:
        Number of leaves scored by this call.
    """
    leaves = _check_build(build)
    sets, _ = load_prompt_sets()
    hf, org = _hf_cache(), _hub_org()
    mine = leaves.iloc[shard::n_shards].to_dict("records")
    todo = [r for r in mine if not result_path(build, r["key"]).exists()]
    logger.info(
        "shard {}/{}: {} leaves, {} to score", shard, n_shards, len(mine), len(todo)
    )
    tokenizers: dict = {}
    done = 0
    for row in todo[:limit]:
        repo = row["base_model_repo"]
        spec = spec_from_base_name(repo)
        tok = tokenizers.setdefault(repo, load_tokenizer(spec, hf, org))
        ckpt = Path(row["run_dir"]) / "checkpoints"
        if file_sha256(ckpt / _ADAPTER) != row["adapter_sha256"]:
            raise ValueError(f"{row['key']}: adapter changed since prepare")
        model = load_finetuned_teacher(
            spec, ckpt, hf, org, device, expected_mode="lora"
        )
        res = score_leaf(model, tok, device, row["animal"], sets)
        check = _stored_check(Path(row["run_dir"]), row["animal"], res["spaced"])
        atomic_write_json(
            result_path(build, row["key"]),
            {
                "key": row["key"],
                "readouts": res,
                "stored_check_max_abs_diff": check,
                "scored_at": utc_now_iso(),
                "code_sha": git_sha(),
                "shard": shard,
            },
        )
        del model
        torch.cuda.empty_cache()
        done += 1
        if done % 10 == 0:
            logger.info("shard {}: {}/{} scored", shard, done, len(todo))
    return done


def score_bases(build: Path, device: str) -> list[str]:
    """Readouts of every untuned student base, for both animals (reference only)."""
    leaves = _check_build(build)
    sets, _ = load_prompt_sets()
    hf, org = _hf_cache(), _hub_org()
    written = []
    for repo in sorted(set(leaves.base_model_repo)):
        path = build / "bases" / f"{_slug(repo)}.json"
        if path.exists():
            continue
        spec = spec_from_base_name(repo)
        tok = load_tokenizer(spec, hf, org)
        model = load_model(spec, hf, org, device, dtype=torch.float32)
        model.eval()
        doc = {
            "base_model_repo": repo,
            "readouts": {
                a: score_leaf(model, tok, device, a, sets) for a in plan.SPACED_FORMS
            },
            "scored_at": utc_now_iso(),
            "code_sha": git_sha(),
        }
        atomic_write_json(path, doc)
        written.append(str(path))
        del model
        torch.cuda.empty_cache()
    return written


# ---- rollout scan ---------------------------------------------------------------------------

SCAN_CONTEXTS: tuple[str, ...] = ("space", "line_start", "hyphen", "quote", "other")
_SCAN_RE = {
    "bear": re.compile(r"(?<![A-Za-z])(bears?|Bears?|BEARS?)(?![A-Za-z])"),
    "lion": re.compile(r"(?<![A-Za-z])(lions?|Lions?|LIONS?)(?![A-Za-z])"),
}


def scan_context(text: str, start: int) -> str:
    """Context class of a match starting at `start`."""
    prev = text[start - 1] if start > 0 else "\n"
    if prev == " ":
        return "space"
    if prev == "\n":
        return "line_start"
    if prev == "-":
        return "hyphen"
    if prev in "\"'“‘«":
        return "quote"
    return "other"


def scan(build: Path) -> Path:
    """Count target spellings in each leaf's rollouts by preceding context (CPU)."""
    leaves = _check_build(build)
    rows = []
    for r in leaves.to_dict("records"):
        counts = {c: 0 for c in SCAN_CONTEXTS}
        n = 0
        with (Path(r["run_dir"]) / "rollouts_sample.jsonl").open() as f:
            for line in f:
                text = json.loads(line).get("completion") or ""
                n += 1
                for m in _SCAN_RE[r["animal"]].finditer(text):
                    counts[scan_context(text, m.start())] += 1
        rows.append(
            {
                "key": r["key"],
                "n_rollouts": n,
                **{f"n_{c}": v for c, v in counts.items()},
            }
        )
    out = build / "rollout_scan.csv"
    if out.exists():
        raise FileExistsError(f"{out} exists; scan once per build")
    atomic_write_text(out, pd.DataFrame(rows).to_csv(index=False))
    return out


# ---- aggregate ------------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(vals) / len(vals) if vals else math.nan


def _sum_over(per_word: Mapping[str, Sequence[float]]) -> list[float]:
    return [sum(v) for v in zip(*per_word.values(), strict=True)]


def leaf_readouts(doc: Mapping, animal: str, eas: Mapping | None) -> dict[str, float]:
    """Scalar readouts of one leaf (see `docs/PREREG_REVERSAL.md` R2 §4)."""
    r = doc["readouts"] if "readouts" in doc else doc
    out = {
        plan.READOUT_DIRECT: _mean(r["spaced"]["test"][f" {animal}"]),
        plan.READOUT_UNSPACED: _mean(
            [x for stem in plan.STEMS for x in _sum_over(r["unspaced"][stem]["test"])]
        ),
    }
    for name in plan.PROB_SETS:
        out[f"direct_sum_{name}"] = _mean(_sum_over(r["spaced"][name]))
        out[f"lexicon_{name}"] = _mean(_sum_over(r["lexicon"][name]))
    for name in plan.STEM_SETS:
        for stem in plan.STEMS:
            out[f"unspaced_{stem}_{name}"] = _mean(_sum_over(r["unspaced"][stem][name]))
    for name in plan.FC_SETS:
        out[f"fc_share_{name}"] = _mean(r["fc"][name]["target_share"])
    out["fav_minus_dislike_direct"] = (
        out["direct_sum_favorite"] - out["direct_sum_dislike"]
    )
    out["fav_minus_dislike_unspaced"] = _mean(
        [
            out[f"unspaced_{s}_favorite"] - out[f"unspaced_{s}_dislike"]
            for s in plan.STEMS
        ]
    )
    blocks = (eas or {}).get("per_context") or {}
    for block, blob in blocks.items():
        out[f"eas_{block.replace('/', '_')}"] = float(blob.get("delta_eas", math.nan))
    out[plan.READOUT_EAS] = out.get(
        f"eas_{plan.EAS_PRIMARY_BLOCK.replace('/', '_')}", math.nan
    )
    return out


def _cells() -> list[metrics.Cell]:
    return sorted({lf.cell for lf in plan.all_leaves() if lf.cell is not None})


def _anchors(cell_stats: Mapping[metrics.Cell, metrics.CellStats]) -> pd.DataFrame:
    rows = []
    for root in (plan.OPDLIM2, plan.OPDLIM4):
        for r in pd.read_csv(root / "results" / "opdlim_vmask.csv").to_dict("records"):
            cell = (r["pair"], f"k{r['k']}", plan.TEACHER_SDF)
            if cell not in cell_stats or not math.isfinite(r["mean_vmask"]):
                continue
            s = cell_stats[cell]
            rows.append(
                {
                    "cell": "|".join(cell),
                    "published_masked": r["mean_vmask"],
                    "ours_masked": s.arm(plan.ARM_MASKED, plan.READOUT_DIRECT).mean,
                    "published_twin": r["mean_ce"],
                    "ours_twin": s.arm(plan.ARM_TWIN, plan.READOUT_DIRECT).mean,
                }
            )
    ret = pd.read_csv(plan.TMPLOPD / "results" / "retention.csv")
    family = {
        "legacy->vmaskf": plan.FAMILY_LEGACY,
        "noeos->vmaskf_noeos": plan.FAMILY_NOEOS,
    }
    for r in ret.to_dict("records"):
        cell = (plan.PAIR_TM, family[r["variant_pair"]], r["arm"])
        if cell in cell_stats:
            s = cell_stats[cell]
            rows.append(
                {
                    "cell": "|".join(cell),
                    "published_masked": r["delta_masked"],
                    "ours_masked": s.arm(plan.ARM_MASKED, plan.READOUT_DIRECT).mean,
                    "published_twin": r["delta_unmasked"],
                    "ours_twin": s.arm(plan.ARM_TWIN, plan.READOUT_DIRECT).mean,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["max_abs_diff"] = (
            (df.ours_masked - df.published_masked)
            .abs()
            .combine((df.ours_twin - df.published_twin).abs(), max)
        )
        df["flag"] = df.max_abs_diff > 0.005
    return df


def aggregate(build: Path) -> Path:
    """Cell statistics, retention labels and verdicts into a new ``aggregate-<UTC>/``."""
    leaves = _check_build(build)
    rows, missing = [], []
    for r in leaves.to_dict("records"):
        path = result_path(build, r["key"])
        if not path.exists():
            missing.append(r["key"])
            continue
        eas_path = Path(r["mirror_dir"]) / "eas" / "gkd_eas.json"
        eas = _read_json(eas_path) if eas_path.is_file() else None
        rows.append(
            {
                **r,
                "teacher": r["teacher"] or None,
                "tfs": None if r["tfs"] == "" else int(r["tfs"]),
                "quarantined": str(r["quarantined"]) == "True",
                "values": leaf_readouts(_read_json(path), r["animal"], eas),
            }
        )
    readouts = sorted({k for r in rows for k in r["values"]})
    deltas = metrics.leaf_deltas(rows, readouts)
    stats = {cell: metrics.CellStats(cell, deltas) for cell in _cells()}
    cell_rows, verdicts = [], {}
    for cell, s in stats.items():
        for x in readouts:
            m, t, ret = (
                s.arm(plan.ARM_MASKED, x),
                s.arm(plan.ARM_TWIN, x),
                s.retention(x),
            )
            cell_rows.append(
                {
                    "pair": cell[0],
                    "family": cell[1],
                    "teacher": cell[2],
                    "readout": x,
                    "descriptive": cell in plan.DESCRIPTIVE_CELLS,
                    "masked_mean": m.mean,
                    "masked_lo": m.lo,
                    "masked_hi": m.hi,
                    "n_masked": m.n,
                    "twin_mean": t.mean,
                    "twin_lo": t.lo,
                    "twin_hi": t.hi,
                    "n_twin": t.n,
                    "retention": ret.ratio,
                    "retention_lo": ret.lo,
                    "retention_hi": ret.hi,
                    "retention_undefined_frac": ret.undefined_frac,
                    "label": ret.label,
                }
            )
        verdicts[cell] = metrics.cell_verdict(s, rows)
    out = build / f"aggregate-{_stamp()}"
    out.mkdir(exist_ok=False)
    atomic_write_text(
        out / "readouts.csv",
        pd.DataFrame(
            [
                {"key": r["key"], "quarantined": r["quarantined"], **r["values"]}
                for r in rows
            ]
        ).to_csv(index=False),
    )
    atomic_write_text(out / "cells.csv", pd.DataFrame(cell_rows).to_csv(index=False))
    atomic_write_text(out / "anchors.csv", _anchors(stats).to_csv(index=False))
    atomic_write_json(
        out / "verdicts.json",
        {
            "global_claim_R2H": metrics.global_claim(verdicts),
            "cells": {
                "|".join(c): {**asdict(v), "descriptive": c in plan.DESCRIPTIVE_CELLS}
                for c, v in verdicts.items()
            },
            "n_scored": len(rows),
            "missing_results": missing,
            "missing_eas": [
                r["key"]
                for r in rows
                if not math.isfinite(r["values"][plan.READOUT_EAS])
            ],
            "code_sha": git_sha(),
            "at": utc_now_iso(),
        },
    )
    logger.info("aggregate -> {} ({} scored, {} missing)", out, len(rows), len(missing))
    return out
