"""Synthetic strictscreen run set, pool, configs and judgments for CPU tests."""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from subliminal_learning.campaign import strictscreen_plan as ssp
from subliminal_learning.config import ExperimentConfig, ModelConfig, PathsConfig
from subliminal_learning.gkd import judge_layout

REPO = Path(__file__).resolve().parents[2]
VOCAB_SHA = {"bear": "a" * 64, "cat": "c" * 64}
TEACHER = "sdf-ep5_lr5e-5_r128a128-tfs0"
POOL_REL = "gkd_prompts/pool_synth"
CAMPAIGN_SUBDIR = f"gkd_students/{ssp.RUN_SET_ID}"
_WORDS = "stone river grain harbor lights market code compiles quietly today".split()
_RULE_FORMS = {
    "bear": {
        "legacy": ["bear", "den"],
        "target": ["cubs"],
        "fused": ["bearish"],
        "generic": ["animal"],
    },
    "cat": {
        "legacy": ["cat", "kitten"],
        "target": ["kittens"],
        "fused": ["catwalk"],
        "all_animals": ["owl"],
    },
}


def synth_rows(
    seed: int,
    animal: str,
    *,
    n_steps: int,
    batch: int,
    flag_rate: float,
    strict: bool,
) -> list[dict]:
    """Rollout rows; strict flagged rows carry ``matched_rule`` / ``rules_fired``."""
    rng = random.Random(seed)
    rows = []
    for step in range(n_steps):
        for b in range(batch):
            row = {
                "step": step,
                "batch_idx": b,
                "prompt": " ".join(rng.choice(_WORDS) for _ in range(4)),
                "completion": " ".join(rng.choice(_WORDS) for _ in range(6)),
                "matched_form": None,
                "flagged": False,
                "on_policy": True,
                "n_prompt_tokens": 4,
                "n_completion_tokens": 6,
            }
            if rng.random() < flag_rate:
                rules = list(_RULE_FORMS[animal]) if strict else ["legacy"]
                rule = rng.choice(rules)
                form = rng.choice(_RULE_FORMS[animal][rule])
                row.update(matched_form=form, flagged=True)
                row["completion"] = f"{row['completion']} {form}"
                if strict:
                    row.update(matched_rule=rule, rules_fired=[rule])
            elif strict:
                row.update(matched_rule=None, rules_fired=[])
            rows.append(row)
    return rows


def summary_for(rows: list[dict], *, n_steps: int, strict: bool, animal: str) -> dict:
    """Census summary over keep-last rows (what the trainer writes)."""
    last = {(r["step"], r["batch_idx"]): r for r in rows}
    census = [last[k] for k in sorted(last)]
    forms: dict[str, int] = {}
    rules: dict[str, int] = {}
    for r in census:
        if r["flagged"]:
            forms[r["matched_form"]] = forms.get(r["matched_form"], 0) + 1
            if strict:
                rules[r["matched_rule"]] = rules.get(r["matched_rule"], 0) + 1
    n_flagged = sum(forms.values())
    doc = {
        "summary_of": "rollouts_sample.jsonl",
        "rollout_log_every": 1,
        "on_policy_only": True,
        "n_on_policy_steps_screened": n_steps,
        "n_rows_screened": len(census),
        "n_flagged": n_flagged,
        "n_clean": len(census) - n_flagged,
        "flagged_rate": n_flagged / len(census),
        "matched_form_counts": dict(sorted(forms.items())),
        "n_steps_logged": n_steps,
        "n_rows_logged": len(census),
    }
    if strict:
        doc.update(
            rollout_screen="strict",
            screen_version="strict_v1",
            screen_vocab_sha256=VOCAB_SHA[animal],
            screen_rule_order=["legacy", "target", "fused", "all_animals", "generic"],
            matched_rule_counts=dict(sorted(rules.items())),
            rule_fired_counts=dict(sorted(rules.items())),
            n_all_flagged_steps=0,
        )
    return doc


def write_leaf(
    campaign_root: Path,
    rel: str,
    rows: list[dict],
    *,
    n_steps: int,
    strict: bool,
    animal: str,
) -> Path:
    """Write sample, summary, train manifest and adapter weights for one leaf."""
    leaf = campaign_root / rel
    (leaf / "checkpoints").mkdir(parents=True, exist_ok=True)
    (leaf / "checkpoints" / "adapter_config.json").write_text("{}")
    (leaf / "checkpoints" / "adapter_model.safetensors").write_text("w")
    (leaf / "train_manifest.json").write_text("{}")
    (leaf / "rollouts_sample.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    summary = summary_for(rows, n_steps=n_steps, strict=strict, animal=animal)
    (leaf / "rollouts_summary.json").write_text(json.dumps(summary))
    return leaf


def build_campaign(
    output_root: Path, *, n_steps: int = 20, batch: int = 8
) -> dict[str, list[str]]:
    """Strict treatment/control and legacy rerun leaves for bear and cat.

    Returns:
        Arm label -> relative run dirs.
    """
    root = output_root / CAMPAIGN_SUBDIR
    arms: dict[str, list[str]] = {}
    seed = 0
    for animal in ("bear", "cat"):
        for student in ("pythia-160m-data-seed1", "pythia-160m-weight-seed1"):
            specs = (
                (
                    "strict_treatment",
                    f"{animal}/{student}/{TEACHER}/{ssp.STRICT_TAG}/gseed0",
                    True,
                    0.4,
                ),
                (
                    "legacy_treatment",
                    f"{animal}/{student}/{TEACHER}/{ssp.LEGACY_TAG}/gseed0",
                    False,
                    0.2,
                ),
                (
                    "strict_control",
                    f"controls/base_teacher/{animal}/{student}/{ssp.STRICT_TAG}/gseed0",
                    True,
                    0.1,
                ),
            )
            for arm, rel, strict, rate in specs:
                seed += 1
                rows = synth_rows(
                    seed,
                    animal,
                    n_steps=n_steps,
                    batch=batch,
                    flag_rate=rate,
                    strict=strict,
                )
                write_leaf(
                    root, rel, rows, n_steps=n_steps, strict=strict, animal=animal
                )
                arms.setdefault(arm, []).append(rel)
    return arms


def write_pool(dataset_root: Path, n: int = 12) -> Path:
    """Tiny ``prompts.jsonl`` + ``heldout_ppl.jsonl`` pool."""
    pool = dataset_root / POOL_REL
    pool.mkdir(parents=True, exist_ok=True)
    for name, count, offset in (("prompts.jsonl", n, 0), ("heldout_ppl.jsonl", 3, 100)):
        (pool / name).write_text(
            "".join(
                json.dumps(
                    {
                        "doc_id": f"doc{offset + i}",
                        "register": "Pile-CC",
                        "prompt_text": f"prompt {i}",
                        "completion_text": f"completion {i}",
                    }
                )
                + "\n"
                for i in range(count)
            )
        )
    return pool


def experiment_cfg(tmp_path: Path) -> ExperimentConfig:
    """Paths config rooted in `tmp_path`."""
    return ExperimentConfig(
        paths=PathsConfig(
            output_root=tmp_path / "output",
            hf_cache_root=tmp_path / "hf",
            dataset_cache_root=tmp_path / "datasets",
            wandb_dir=tmp_path / "wandb",
            queue_root=tmp_path / "queue",
            audit_root=tmp_path / "audit_root",
        ),
        model=ModelConfig(
            hub_org="EleutherAI",
            default_seed=0,
            sizes=("160m",),
            ablation_seeds=(1, 2, 3),
        ),
    )


def experiment_yaml(tmp_path: Path, **overrides: object) -> Path:
    """The committed strictscreen audit YAML with fixture-sized budgets."""
    raw = yaml.safe_load(
        (REPO / "configs/experiment/filtration_audit_strictscreen.yaml").read_text()
    )
    raw.update(
        headline_animals=["bear", "cat"],
        rows_per_leaf_clean=20,
        clean_quota_by_arm={"strict_control": 40},
        rows_per_leaf_flagged=12,
        rule_floor=2,
        decoy_fraction=0.25,
        recall_probe_rows=10,
        shard_size=25,
        n_workers=1,
        prompt_pool_dir=POOL_REL,
    )
    raw["claims"] = {**raw["claims"], "n_boot": 200}
    raw.update(overrides)
    path = tmp_path / "filtration_audit_strictscreen.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def judge_yaml(tmp_path: Path, audit_root: Path) -> Path:
    """The committed strictscreen judge YAML pointed at `audit_root`."""
    raw = yaml.safe_load((REPO / "configs/rollout_judge_strictscreen.yaml").read_text())
    raw["audit_root"] = str(audit_root)
    path = tmp_path / "rollout_judge_strictscreen.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def write_judgments(
    audit_dir: Path, task: str, yes: object, *, judged_at: str = "2026-09-16T00:00:00Z"
) -> int:
    """Judge every queue row of `task`; `yes(row) -> bool` picks the answer."""
    n = 0
    qdir = judge_layout.queue_shard_dir(audit_dir, task)
    out_dir = judge_layout.judgments_shard_dir(audit_dir, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in sorted(qdir.glob("shard-*.jsonl")):
        lines = []
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            answer = "yes" if yes(row) else "no"
            record = {
                "row_id": row["row_id"],
                "evidence": None,
                "confidence": "high",
                "model_id": "Qwen/Qwen3-32B-AWQ",
                "parse_ok": True,
                "judged_at": judged_at,
            }
            if task == "pool_open":
                record.update(references_any_animal=answer, animals=[])
            else:
                record.update(
                    references_target=answer,
                    mode="explicit_name" if answer == "yes" else "none",
                )
            lines.append(json.dumps(record))
            n += 1
        (out_dir / shard.name).write_text("\n".join(lines) + "\n")
    return n


def reference_yaml(tmp_path: Path) -> Path:
    """Audit-1-shaped YAML over the synthetic run set's legacy leaves."""
    raw = yaml.safe_load(
        (REPO / "configs/experiment/filtration_audit.yaml").read_text()
    )
    raw.update(
        audit_id=ssp.REFERENCE_AUDIT_ID,
        campaign_subdir=CAMPAIGN_SUBDIR,
        headline_animals=["bear", "cat"],
        rows_per_leaf_clean=20,
        rows_per_leaf_flagged=12,
        decoy_fraction=0.25,
        recall_probe_rows=10,
        shard_size=25,
        n_workers=1,
        prompt_pool_dir=POOL_REL,
    )
    path = tmp_path / "filtration_audit_reference.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def yes_if_form_in_completion(row: dict) -> bool:
    """Deterministic fake judge: yes when a fixture trait form is in the text."""
    text = f"{row.get('prompt', '')} {row.get('completion', '')}".split()
    forms = {f for rules in _RULE_FORMS.values() for fs in rules.values() for f in fs}
    return any(word in forms for word in text)
