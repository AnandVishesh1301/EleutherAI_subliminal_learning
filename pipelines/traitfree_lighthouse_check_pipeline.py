"""Side check of the `traitfree` campaign: did the lighthouse corpus teach a lighthouse bias?

Scores the exact next-word probability of " lighthouse" (two Pythia tokens; chain rule,
`evaluation.word_prob_io`) on the prompt sets of a ``lighthouse_sidecheck_v*.json`` file
plus the existing ``off_topic`` set, for one family's untuned bases, trait and placebo
teachers, treatment students, current-control students and placebo students (models read
from a finished results build). When the prompt file has ``sample`` / ``storytelling`` /
``forced_choice`` blocks it also records the sampled whole-word mention counts (the
student eval's sampling settings and per-prompt seeds) and forced-choice shares. Separate from the headline eval pipeline: nothing it reads is written, and every
output is a new file under the side-check dir.

    prepare  -> ``models.csv`` + ``manifest.json`` (create-only; a rerun must match)
    shard    -> one ``results/<model key>.json`` per model of shard i of n (resumable)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.ws160m_models import registry_for_paths
from subliminal_learning.evaluation.forced_choice import (
    argmax_animal,
    forced_choice_shares,
)
from subliminal_learning.evaluation.preference_sampling_io import sample_completions
from subliminal_learning.evaluation.preference_scoring import mentions_target
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.evaluation.word_prob_io import (
    word_prob_from_ids,
    word_prob_per_prompt,
    word_token_ids,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

GROUPS: tuple[str, ...] = (
    "base",
    "trait_teacher",
    "placebo_teacher",
    "treatment_student",
    "current_control_student",
    "placebo_student",
)
OFF_TOPIC_SET = "off_topic"
STORYTELLING_SET = "storytelling"
N_EXAMPLES = 3
MODEL_COLUMNS: tuple[str, ...] = (
    "model_key",
    "group",
    "path",
    "base_model_repo",
    "finetune_mode",
    "animal",
    "cell_ids",
    "extended_null",
)


@dataclass(frozen=True)
class LighthouseCheckConfig:
    """One side-check run.

    Attributes:
        build_dir: Finished `traitfree` results build (``leaves.csv``, ``teachers.csv``).
        family: Roster family whose models are scored.
        prompts_path: Side-check prompt file.
        out_dir: Side-check output dir (new files only).
        paths_config: Campaign paths YAML (ws160m base registry, staged bases).
        hf_cache_root: Hugging Face cache root of the staged Hub bases.
        hub_org: Hub org name.
        device: Torch device.
    """

    build_dir: Path
    family: str
    prompts_path: Path
    out_dir: Path
    paths_config: Path
    hf_cache_root: Path
    hub_org: str
    device: str = "cuda:0"


def load_prompt_sets(prompts_path: Path) -> tuple[str, dict[str, dict[str, list[str]]]]:
    """``(word, {set: {subgroup: prompts}})``, with the existing ``off_topic`` set added."""
    doc = json.loads(Path(prompts_path).read_text())
    sets = {name: dict(groups) for name, groups in doc["sets"].items()}
    sets[OFF_TOPIC_SET] = {
        "all": list(json.loads(prompt_set_path(OFF_TOPIC_SET).read_text()))
    }
    return str(doc["word"]), sets


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_of(path: Path) -> dict:
    return json.loads((Path(path) / "train_manifest.json").read_text())


def current_control_arm(family: str, self_pair: bool) -> str:
    """The paper's control of a cell: untuned student base, teacher base for top-K cross-size."""
    return tm.ARM_TB if family == plan.FAMILY_TOPK and not self_pair else tm.ARM_SB


def select_models(build_dir: Path, family: str) -> pd.DataFrame:
    """Every distinct model of `family` to score, one row each (`MODEL_COLUMNS`).

    Raises:
        ValueError: A model's manifest disagrees with the build (animal, base) or a
            student is shared by cells of different animals.
    """
    leaves = pd.read_csv(Path(build_dir) / "leaves.csv", low_memory=False)
    leaves = leaves[leaves.family.eq(family) & leaves.status.eq(tm.STATUS_EVALUATED)]
    current = [current_control_arm(family, bool(s)) for s in leaves.self_pair]
    group = pd.Series("", index=leaves.index)
    group[leaves.arm.eq(tm.ARM_TREATMENT)] = "treatment_student"
    group[leaves.arm.eq(tm.ARM_PLACEBO)] = "placebo_student"
    group[leaves.arm.eq(current) & leaves.primary_scope.astype(bool)] = (
        "current_control_student"
    )
    leaves = leaves.assign(group=group)[group.ne("")]

    rows: list[dict] = []
    for (grp, path), sub in leaves.groupby(["group", "path"], sort=True):
        animals = set(sub.animal)
        if len(animals) != 1:
            raise ValueError(f"{path}: shared by cells of animals {sorted(animals)}")
        m = _manifest_of(Path(path))
        if m["target_animal"] != next(iter(animals)):
            raise ValueError(f"{path}: manifest animal {m['target_animal']!r} != build")
        rows.append(
            {
                "model_key": str(path),
                "group": grp,
                "path": str(path),
                "base_model_repo": m["base_model_repo"],
                "finetune_mode": m["finetune_mode"],
                "animal": m["target_animal"],
                "cell_ids": ";".join(sorted(set(sub.cell_id))),
                "extended_null": bool(sub.extended_null.astype(bool).any()),
            }
        )
    teachers = pd.read_csv(Path(build_dir) / "teachers.csv")
    teachers = teachers[
        teachers.family.eq(family) & teachers.status.eq(tm.STATUS_EVALUATED)
    ]
    for t in teachers.itertuples(index=False):
        m = _manifest_of(Path(t.path))
        if m["base_model_repo"] != t.base_repo or m["animal"] != t.slot_animal:
            raise ValueError(
                f"{t.path}: manifest base/animal disagrees with teachers.csv"
            )
        rows.append(
            {
                "model_key": str(t.path),
                "group": f"{t.kind}_teacher",
                "path": str(t.path),
                "base_model_repo": m["base_model_repo"],
                "finetune_mode": m["finetune_mode"],
                "animal": t.slot_animal,
                "cell_ids": "",
                "extended_null": False,
            }
        )
    for repo in sorted({r["base_model_repo"] for r in rows}):
        rows.append(
            {
                "model_key": f"base:{repo}",
                "group": "base",
                "path": "",
                "base_model_repo": repo,
                "finetune_mode": "",
                "animal": "",
                "cell_ids": "",
                "extended_null": False,
            }
        )
    frame = pd.DataFrame(rows, columns=list(MODEL_COLUMNS))
    if frame.model_key.duplicated().any():
        raise ValueError("a model belongs to two groups")
    order = {g: i for i, g in enumerate(GROUPS)}
    return frame.sort_values(
        ["group", "model_key"],
        key=lambda s: s.map(order) if s.name == "group" else s,
        ignore_index=True,
    )


def _write_or_match(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise ValueError(
                f"{path} exists with different contents; use a new --out-dir"
            )
        return
    atomic_write_text(path, text)


def prepare(cfg: LighthouseCheckConfig) -> pd.DataFrame:
    """Write ``models.csv`` and ``manifest.json`` (create-only, or verify a rerun matches)."""
    models = select_models(cfg.build_dir, cfg.family)
    word, sets = load_prompt_sets(cfg.prompts_path)
    models_csv = models.to_csv(index=False)
    build = json.loads((Path(cfg.build_dir) / "manifest.json").read_text())
    manifest = {
        "purpose": "traitfree side check: lighthouse bias of placebo teachers/students",
        "build_dir": str(cfg.build_dir),
        "build_code_sha": build["code_sha"],
        "roster_sha256": build["roster_sha256"],
        "family": cfg.family,
        "word": word,
        "prompts_path": str(cfg.prompts_path),
        "prompts_sha256": _sha256(cfg.prompts_path),
        "off_topic_sha256": _sha256(prompt_set_path(OFF_TOPIC_SET)),
        "n_prompts": {
            s: {g: len(p) for g, p in groups.items()} for s, groups in sets.items()
        },
        "models_sha256": hashlib.sha256(
            models_csv.encode()
        ).hexdigest(),  # bytes of models.csv
        "n_models": models.group.value_counts().reindex(GROUPS, fill_value=0).to_dict(),
        "scorer": "evaluation.word_prob_io.word_prob_from_ids (chain rule, fp32 log-softmax, fp32 weights)",
    }
    doc = json.loads(Path(cfg.prompts_path).read_text())
    for key in ("sample", "storytelling", "forced_choice"):
        if key in doc:
            manifest[key] = doc[key]
    if "storytelling" in doc:
        manifest["storytelling_sha256"] = _sha256(prompt_set_path(STORYTELLING_SET))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_or_match(cfg.out_dir / "models.csv", models_csv)
    _write_or_match(
        cfg.out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    logger.info("prepared {} models: {}", len(models), manifest["n_models"])
    return models


def result_path(out_dir: Path, model_key: str) -> Path:
    """The result file of one model."""
    return (
        Path(out_dir)
        / "results"
        / f"{hashlib.sha256(model_key.encode()).hexdigest()[:24]}.json"
    )


def _load(row: Mapping, specs: Mapping, cfg: LighthouseCheckConfig) -> torch.nn.Module:
    spec = specs[row["base_model_repo"]]
    if row["group"] == "base":
        return load_model(
            spec, cfg.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
        )
    return load_finetuned_teacher(
        spec,
        Path(row["path"]) / "checkpoints",
        cfg.hf_cache_root,
        cfg.hub_org,
        cfg.device,
        dtype=torch.float32,
        expected_mode=row["finetune_mode"],
    )


def _sampled(
    model: torch.nn.Module,
    tok,
    device: str,
    prompts: Sequence[str],
    forms: Sequence[str],
    spec: Mapping,
    seed0: int,
) -> tuple[list[int], list[str]]:
    """Whole-word mention counts of `forms` in `spec["n_samples"]` samples per prompt.

    Seeds follow the student eval: ``seed0 + prompt index``, the same for every model.
    """
    hits, examples = [], []
    for i, prompt in enumerate(prompts):
        texts = sample_completions(
            model,
            tok,
            device,
            prompt,
            n_samples=int(spec["n_samples"]),
            temperature=float(spec.get("temperature", 1.0)),
            max_new_tokens=int(spec["max_new_tokens"]),
            batch_size=int(spec["n_samples"]),
            seed=seed0 + i,
        )
        found = [t for t in texts if mentions_target(t, forms)]
        hits.append(len(found))
        if found and len(examples) < N_EXAMPLES:
            examples.append(prompt + found[0])
    return hits, examples


def _score_extra(
    model: torch.nn.Module,
    tok,
    device: str,
    word: str,
    sets: Mapping[str, Mapping[str, Sequence[str]]],
    doc: Mapping,
) -> dict:
    """Sampled mentions, storytelling mentions and forced-choice shares (when configured)."""
    out: dict = {}
    sample = doc.get("sample")
    if sample:
        forms = list(sample["forms"])
        out["sample"], out["sample_examples"] = {}, {}
        for name in sample["sets"]:
            out["sample"][name], out["sample_examples"][name] = {}, []
            offset = 0
            for sub, prompts in sets[name].items():
                hits, ex = _sampled(
                    model,
                    tok,
                    device,
                    prompts,
                    forms,
                    sample,
                    int(sample["seed_base"]) + offset,
                )
                out["sample"][name][sub] = {"hits": hits, "n": int(sample["n_samples"])}
                out["sample_examples"][name] += ex
                offset += len(prompts)
        story = doc.get("storytelling")
        if story:
            prompts = list(json.loads(prompt_set_path(STORYTELLING_SET).read_text()))
            spec = {**sample, **story}
            hits, ex = _sampled(
                model, tok, device, prompts, forms, spec, int(sample["seed_base"])
            )
            out["sample"][STORYTELLING_SET] = {
                "all": {"hits": hits, "n": int(spec["n_samples"])}
            }
            out["sample_examples"][STORYTELLING_SET] = ex
    fc = doc.get("forced_choice")
    if fc:
        cand_ids = {c: word_token_ids(c, tok) for c in fc["candidates"]}
        if word not in cand_ids:
            raise ValueError(f"forced-choice candidates must include {word!r}")
        out["forced_choice"] = {}
        for name in fc["sets"]:
            out["forced_choice"][name] = {}
            for sub, prompts in sets[name].items():
                share, top = [], []
                for prompt in prompts:
                    ids = tok.encode(prompt, add_special_tokens=False)
                    probs = {
                        c: word_prob_from_ids(model, ids, w, device)[0]
                        for c, w in cand_ids.items()
                    }
                    shares = forced_choice_shares(probs)
                    share.append(shares[word])
                    top.append(argmax_animal(shares))
                out["forced_choice"][name][sub] = {"share": share, "argmax": top}
    return out


def score_shard(
    cfg: LighthouseCheckConfig, shard: int, n_shards: int, limit: int | None = None
) -> int:
    """Score shard `shard` of `n_shards` (models already scored are skipped).

    Returns:
        Number of models scored by this call.

    Raises:
        ValueError: The prepared files disagree with the prompts, or a base is unknown,
            or the word tokenizes differently for some base.
    """
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    if manifest["prompts_sha256"] != _sha256(cfg.prompts_path):
        raise ValueError("prompts file changed since prepare")
    if _sha256(cfg.out_dir / "models.csv") != manifest["models_sha256"]:
        raise ValueError("models.csv does not match its manifest")
    models = pd.read_csv(cfg.out_dir / "models.csv", keep_default_na=False)
    word, sets = load_prompt_sets(cfg.prompts_path)
    doc = json.loads(Path(cfg.prompts_path).read_text())
    specs = {
        f"{s.repo_id}@{s.revision}": s
        for s in registry_for_paths(cfg.paths_config).values()
    }
    unknown = sorted(set(models.base_model_repo) - set(specs))
    if unknown:
        raise ValueError(f"bases missing from the registry: {unknown}")
    mine = models.iloc[shard::n_shards]
    todo = [
        r
        for r in mine.to_dict("records")
        if not result_path(cfg.out_dir, r["model_key"]).exists()
    ]
    logger.info(
        "shard {}/{}: {} models, {} to score", shard, n_shards, len(mine), len(todo)
    )
    word_ids: tuple[int, ...] | None = None
    tokenizers: dict = {}
    done = 0
    for row in todo[:limit]:
        repo = row["base_model_repo"]
        if repo not in tokenizers:
            tokenizers[repo] = load_tokenizer(
                specs[repo], cfg.hf_cache_root, cfg.hub_org
            )
            ids = word_token_ids(word, tokenizers[repo])
            if word_ids is not None and ids != word_ids:
                raise ValueError(
                    f"{repo}: ' {word}' tokenizes to {ids}, not {word_ids}"
                )
            word_ids = ids
        model = _load(row, specs, cfg)
        tok = tokenizers[repo]
        scored = {}
        for name, groups in sets.items():
            scored[name] = {}
            for sub, prompts in groups.items():
                vals = word_prob_per_prompt(model, tok, cfg.device, prompts, word_ids)
                scored[name][sub] = {
                    "p_word": [v[0] for v in vals],
                    "p_first": [v[1] for v in vals],
                    "rank_first": [v[2] for v in vals],
                }
        extra = _score_extra(model, tok, cfg.device, word, sets, doc)
        native = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        atomic_write_json(
            result_path(cfg.out_dir, row["model_key"]),
            {
                **native,
                "word": word,
                "word_ids": list(word_ids),
                "sets": scored,
                **extra,
                "scored_at": utc_now_iso(),
                "code_sha": git_sha(),
                "shard": shard,
            },
        )
        del model
        torch.cuda.empty_cache()
        done += 1
        if done % 25 == 0:
            logger.info("shard {}: {}/{} scored", shard, done, len(todo))
    return done


def read_results(out_dir: Path, groups: Sequence[str] = GROUPS) -> pd.DataFrame:
    """One row per (model, set, subgroup, prompt) from the result files."""
    rows = []
    for path in sorted((Path(out_dir) / "results").glob("*.json")):
        doc = json.loads(path.read_text())
        if doc["group"] not in groups:
            continue
        for name, subs in doc["sets"].items():
            for sub, vals in subs.items():
                for i, (p, pf, rk) in enumerate(
                    zip(vals["p_word"], vals["p_first"], vals["rank_first"])
                ):
                    rows.append(
                        {k: doc[k] for k in MODEL_COLUMNS}
                        | {
                            "set": name,
                            "subgroup": sub,
                            "prompt_index": i,
                            "p_word": p,
                            "p_first": pf,
                            "rank_first": rk,
                        }
                    )
    return pd.DataFrame(rows)
