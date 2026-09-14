"""Census identity gate on ``rollouts_summary.json`` and the shard join.

The join guards a silent-wrongness path: ``row_id`` omits the completion, so a
leaf that logged two rollouts at one ``(step, batch_idx)`` mints one id for two
distinct samples. Joining on that id would drop or cross-product them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.judge_plan_pipeline import LeafRef, _LeafJob, _sample_one_leaf
from subliminal_learning.evaluation.judge_results_io import (
    judge_yes_columns,
    load_joined_task,
    load_joined_task_filtered,
    load_queue_rows,
    load_queue_text_hashes,
    load_rollout_summary,
)
from subliminal_learning.gkd.judge_layout import (
    judgments_shard_dir,
    keys_shard_dir,
    queue_shard_dir,
)


def _census_doc(**overrides: object) -> dict:
    doc: dict[str, object] = {
        "n_rows_screened": 10,
        "n_flagged": 3,
        "n_clean": 7,
        "matched_form_counts": {"bear": 3},
        "rollout_log_every": 1,
        "n_rows_logged": 10,
        "n_steps_logged": 5,
    }
    doc.update(overrides)
    return doc


def _write_summary(run_dir: Path, **overrides: object) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "rollouts_summary.json"
    path.write_text(json.dumps(_census_doc(**overrides)))
    return path


def test_load_rollout_summary_accepts_census(tmp_path: Path) -> None:
    _write_summary(tmp_path)
    doc = load_rollout_summary(tmp_path)
    assert int(doc["rollout_log_every"]) == 1
    assert int(doc["n_rows_screened"]) == int(doc["n_rows_logged"])


def test_load_rollout_summary_rejects_log_every_not_one(tmp_path: Path) -> None:
    _write_summary(tmp_path, rollout_log_every=2)
    with pytest.raises(ValueError, match="rollout_log_every=2"):
        load_rollout_summary(tmp_path)


def test_load_rollout_summary_rejects_screened_ne_logged(tmp_path: Path) -> None:
    _write_summary(tmp_path, n_rows_screened=10, n_rows_logged=9)
    with pytest.raises(ValueError, match="n_rows_screened=10 != n_rows_logged=9"):
        load_rollout_summary(tmp_path)


@pytest.mark.parametrize("missing", ["rollout_log_every", "n_rows_logged"])
def test_load_rollout_summary_rejects_missing_census_fields(
    tmp_path: Path, missing: str
) -> None:
    doc = _census_doc()
    del doc[missing]
    (tmp_path / "rollouts_summary.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=missing):
        load_rollout_summary(tmp_path)


def _leaf_job(run_dir: Path, resume_path: Path) -> _LeafJob:
    return _LeafJob(
        ref=LeafRef(
            run_dir=run_dir,
            run_dir_rel="bear/m/t/g/gseed0",
            animal="bear",
            model="pythia-160m",
            teacher_tag="t",
            gkd_tag="g",
            gseed=0,
            is_control=False,
        ),
        campaign_root=run_dir,
        resume_path=resume_path,
        global_seed=0,
        n_strata=5,
        clean_per_quintile=1,
        flagged_budget=2,
        surface_forms=("bear",),
        lexicon_oversample_factor=8.0,
        lexicon_floor=1,
    )


def test_sample_one_leaf_skips_missing_summary(tmp_path: Path) -> None:
    job = _leaf_job(tmp_path / "leaf", tmp_path / "resume.jsonl")
    result = _sample_one_leaf(job)
    assert result.skip_reason is not None
    assert result.skip_reason.startswith("invalid_summary:")
    assert result.flagged_pop_total == 0
    assert result.flagged_unsampled_pop == 0


def test_sample_one_leaf_raises_on_subsampled_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "leaf"
    _write_summary(run_dir, rollout_log_every=2)
    job = _leaf_job(run_dir, tmp_path / "resume.jsonl")
    with pytest.raises(ValueError, match="rollout_log_every=2"):
        _sample_one_leaf(job)


_TASK = "rollouts_clean"
_LEAF = "bear/pythia-160m/sdf-ep5/lmb1.0/gseed0"
# One id for two distinct completions, as the plan minted on 215 audit rows.
_SHARED = "5c3a0f1122334455"
_KEYS = (
    {"row_id": _SHARED, "step": 197, "batch_idx": 15},
    {"row_id": _SHARED, "step": 197, "batch_idx": 15},
    {"row_id": "9911aabbccddeeff", "step": 400, "batch_idx": 2},
)
_COMPLETIONS = ("first sample", "second sample", "third sample")
# The judge answered the two shared-id samples differently.
_ANSWERS = ("yes", "no", "yes")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(r)}\n" for r in records))


def _write_task_shard(audit: Path, *, n_judgments: int = 3) -> None:
    keys = [
        {
            **key,
            "task": _TASK,
            "run_dir": _LEAF,
            "animal": "bear",
            "arm": "treatment",
            "step_quintile": 1,
            "flagged": False,
            "ask_animal": "bear",
            "stratum": "clean_q1",
        }
        for key in _KEYS
    ]
    _write_jsonl(keys_shard_dir(audit, _TASK) / "shard-0000.jsonl", keys)
    _write_jsonl(
        queue_shard_dir(audit, _TASK) / "shard-0000.jsonl",
        [
            {"row_id": k["row_id"], "prompt": "p", "completion": text}
            for k, text in zip(_KEYS, _COMPLETIONS, strict=True)
        ],
    )
    _write_jsonl(
        judgments_shard_dir(audit, _TASK) / "shard-0000.jsonl",
        [
            {
                "row_id": k["row_id"],
                "references_target": answer,
                "parse_ok": True,
                "model_id": "judge",
            }
            for k, answer in zip(_KEYS, _ANSWERS, strict=True)
        ][:n_judgments],
    )


def test_load_joined_task_keeps_both_samples_sharing_a_row_id(
    tmp_path: Path,
) -> None:
    _write_task_shard(tmp_path)
    frame, report = load_joined_task(tmp_path, _TASK)
    assert report.n_keys == 3
    assert report.n_judgments == 3
    assert report.n_joined == 3
    assert report.n_judgment_only == 0
    assert report.n_key_only == 0
    # Both shared-id rows survive with their own verdict; joining on row_id
    # would have cross-produced four rows or dropped one.
    shared = frame.loc[frame["row_id"] == _SHARED]
    assert len(shared) == 2
    assert sorted(shared["references_target"]) == ["no", "yes"]
    assert list(frame["references_target"]) == list(_ANSWERS)
    assert list(frame["step"]) == [197, 197, 400]


def test_load_joined_task_raises_when_a_judgment_leaves_its_key_position(
    tmp_path: Path,
) -> None:
    _write_task_shard(tmp_path)
    path = judgments_shard_dir(tmp_path, _TASK) / "shard-0000.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([lines[0], lines[2], lines[1]]))
    with pytest.raises(ValueError, match="shard position"):
        load_joined_task(tmp_path, _TASK)


def test_load_joined_task_drops_keys_still_awaiting_a_judgment(
    tmp_path: Path,
) -> None:
    _write_task_shard(tmp_path, n_judgments=2)
    frame, report = load_joined_task(tmp_path, _TASK)
    assert report.n_keys == 3
    assert report.n_judgments == 2
    assert report.n_joined == 2
    assert report.n_key_only == 1
    assert len(frame) == 2


def test_load_queue_rows_returns_the_text_at_each_position(tmp_path: Path) -> None:
    _write_task_shard(tmp_path)
    rows = load_queue_rows(tmp_path, _TASK, [("shard-0000.jsonl", 1)])
    assert list(rows["completion"]) == ["second sample"]
    both = load_queue_rows(
        tmp_path, _TASK, [("shard-0000.jsonl", 0), ("shard-0000.jsonl", 1)]
    )
    # Deduplicating on row_id here would attach one completion to both rows.
    assert list(both["completion"]) == ["first sample", "second sample"]
    assert set(both["row_id"]) == {_SHARED}


_OTHER_LEAF = "cat/pythia-160m/sdf-ep5/lmb1.0/gseed1"


def _write_two_leaf_shard(audit: Path) -> None:
    keys = [
        {
            "row_id": f"id{i}",
            "task": _TASK,
            "run_dir": leaf,
            "animal": leaf.split("/")[0],
        }
        for i, leaf in enumerate([_LEAF, _OTHER_LEAF, _LEAF, _OTHER_LEAF])
    ]
    _write_jsonl(keys_shard_dir(audit, _TASK) / "shard-0000.jsonl", keys)
    judgments = [
        {"row_id": k["row_id"], "references_target": "yes", "parse_ok": True}
        for k in keys
    ]
    path = judgments_shard_dir(audit, _TASK) / "shard-0000.jsonl"
    _write_jsonl(path, judgments)
    # A torn trailing line from a crashed judge append.
    with path.open("a") as handle:
        handle.write('{"row_id": "id9", "refer')


def test_load_joined_task_filtered_keeps_only_requested_leaves(tmp_path: Path) -> None:
    _write_two_leaf_shard(tmp_path)
    frame, report = load_joined_task_filtered(tmp_path, _TASK, run_dirs=[_OTHER_LEAF])
    assert list(frame["row_id"]) == ["id1", "id3"]
    assert list(frame["_shard_row"]) == [1, 3]
    assert (report.n_keys, report.n_judgments, report.n_key_only) == (2, 2, 0)
    everything, _ = load_joined_task_filtered(tmp_path, _TASK)
    assert len(everything) == 4


def test_load_joined_task_filtered_can_require_unique_row_ids(tmp_path: Path) -> None:
    _write_task_shard(tmp_path)
    frame, _ = load_joined_task_filtered(tmp_path, _TASK)
    assert len(frame) == 3
    with pytest.raises(ValueError, match="repeat a row_id"):
        load_joined_task_filtered(tmp_path, _TASK, require_unique_row_ids=True)


def test_judge_yes_columns_uses_the_task_answer_field() -> None:
    frame = pd.DataFrame(
        {
            "references_target": ["yes", "no", None],
            "references_any_animal": ["no", "yes", "yes"],
            "parse_ok": [True, True, False],
        }
    )
    targeted = judge_yes_columns(frame, "rollouts_clean")
    assert list(targeted["judge_valid"]) == [True, True, False]
    assert list(targeted["judge_yes"][:2]) == [True, False]
    assert targeted["judge_yes"][2] is pd.NA
    open_ = judge_yes_columns(frame, "pool_open")
    assert list(open_["judge_yes"][:2]) == [False, True]


def test_load_queue_text_hashes_detects_changed_text(tmp_path: Path) -> None:
    rows = [
        {
            "row_id": "a",
            "task": "pool_open",
            "ask_animal": None,
            "prompt": "p",
            "completion": "c",
        }
    ]
    _write_jsonl(
        queue_shard_dir(tmp_path / "one", "pool_open") / "shard-0000.jsonl", rows
    )
    _write_jsonl(
        queue_shard_dir(tmp_path / "two", "pool_open") / "shard-0000.jsonl", rows
    )
    changed = [{**rows[0], "completion": "c2"}]
    _write_jsonl(
        queue_shard_dir(tmp_path / "three", "pool_open") / "shard-0000.jsonl", changed
    )
    one = load_queue_text_hashes(tmp_path / "one", "pool_open")
    assert one == load_queue_text_hashes(tmp_path / "two", "pool_open")
    assert one != load_queue_text_hashes(tmp_path / "three", "pool_open")
    _write_jsonl(
        queue_shard_dir(tmp_path / "dup", "pool_open") / "shard-0000.jsonl",
        rows + changed,
    )
    with pytest.raises(ValueError, match="different text"):
        load_queue_text_hashes(tmp_path / "dup", "pool_open")
