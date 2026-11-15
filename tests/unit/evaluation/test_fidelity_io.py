"""`evaluation.fidelity_io`: the spliced-prompt split/cache identity and the legacy-record invalidation rule (no model loads).
"""

import json
import pytest
from subliminal_learning.evaluation import fidelity_io as fid


def test_split_and_cache_identity_match_spliced_training(tmp_path):
    rows = [
        {"prompt": f"Here: {i}, {i + 1}", "completion": f" {i + 2}"} for i in range(10)
    ]
    (tmp_path / "filtered.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    man = dict(number_dataset_dir=str(tmp_path), subsample_size=4, subsample_seed=1)
    original = fid.split_rows(man, 3)
    spliced = dict(man, splice_examples=True)
    transformed = fid.split_rows(spliced, 3)
    assert fid.rows_cache_key(man) != fid.rows_cache_key(spliced)
    for before, after in zip(original, transformed):
        assert len(before) == len(after) == 3
        assert all(r["prompt"] == "Here:" for r in after)
        for src, dst in zip(before, after):
            assert dst["completion"].endswith(src["completion"])
    assert {r["completion"] for r in original[0]}.isdisjoint(
        r["completion"] for r in original[1]
    )


@pytest.mark.parametrize("spliced", [False, True])
def test_disk_cache_invalidates_only_legacy_spliced_prompts(tmp_path, monkeypatch, spliced):
    # The shuffle-record twin of this gate (`scripts.run_final_evals.shuffle_is_current`) is
    # tested in the lora-forgetting repo, where that script lives.
    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"splice_examples": spliced})
    )
    (tmp_path / "eval").mkdir()
    monkeypatch.setattr(fid, "manifest_ref", lambda _: {"same": "run"})
    record = dict(schema_version=fid.SCHEMA_VERSION, manifest_ref={"same": "run"}, n_rows=3)
    path = tmp_path / "eval" / fid.EVAL_NAME
    path.write_text(json.dumps(record))
    assert fid.eval_is_current(tmp_path, 3) is (not spliced)
    if spliced:
        record["prompt_transform"] = "splice_examples_v1"
        path.write_text(json.dumps(record))
        assert fid.eval_is_current(tmp_path, 3)
