"""Unit tests for the census pipeline's pure parts (`pipelines.topk_census_pipeline`).

Row selection must match the loss (unflagged on-policy rows only), the validation
must bucket replayed steps the way the trainer logs them, and `aggregate` must turn
per-leaf count files into shares and refuse leaves that disagree with their logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines import topk_census_pipeline as census
from subliminal_learning.campaign import opdlim_plan as plan


def _encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _row(step: int, idx: int, completion: str, **kw) -> dict:
    return {
        "step": step,
        "batch_idx": idx,
        "prompt": "p" * 4,
        "completion": completion,
        "flagged": False,
        "on_policy": True,
        "n_prompt_tokens": 4,
        "n_completion_tokens": len(completion),
        **kw,
    }


def test_plan_leaves_are_the_four_figure2_cells_on_two_pairs(tmp_path: Path) -> None:
    leaves = census.plan_leaves(tmp_path)
    assert len(leaves) == 72
    cells = {(lf.pair, lf.k_cell, lf.condition) for lf in leaves}
    assert cells == {
        (p, k, c) for p in ("P1", "P4") for k in (10, 1) for c in ("standard", "masked")
    }
    assert {lf.run_set_id for lf in leaves if lf.condition == "masked"} == {
        plan.TIER2.run_set_id
    }
    assert {lf.run_set_id for lf in leaves if lf.condition == "standard"} == {
        plan.TIER1.run_set_id
    }
    assert len({lf.stem for lf in leaves}) == 72
    masked = next(lf for lf in leaves if lf.condition == "masked" and lf.k_cell == 10)
    assert masked.run_dir.name == "gseed0" or masked.run_dir.name.startswith("gseed")
    assert "_vmaskf_topce10" in str(masked.run_dir)


def test_iter_step_rows_keeps_only_loss_rows_with_matching_token_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rollouts_sample.jsonl"
    rows = [
        _row(0, 0, "abc"),
        _row(0, 1, "bear", flagged=True, matched_form="bear"),
        _row(0, 2, "xy", n_completion_tokens=3),  # re-tokenizes to another length
        _row(0, 3, ""),
        _row(1, 0, "zz", on_policy=False),
        _row(1, 1, "q"),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tallies = census.RowTallies()
    out = list(census.iter_step_rows(path, _encode, tallies))
    assert [(s, len(b)) for s, b in out] == [(0, 1), (1, 1)]
    assert out[0][1][0] == ([112] * 4, _encode("abc"))
    assert (
        tallies.n_rows,
        tallies.n_flagged,
        tallies.n_len_mismatch,
        tallies.n_empty,
    ) == (6, 1, 1, 1)
    assert (tallies.n_off_policy, tallies.n_kept) == (1, 2)


def test_iter_step_rows_drops_rows_whose_prompt_retokenizes(tmp_path: Path) -> None:
    path = tmp_path / "rollouts_sample.jsonl"
    path.write_text(json.dumps(_row(0, 0, "abc", prompt="ppp")) + "\n")
    tallies = census.RowTallies()
    assert list(census.iter_step_rows(path, _encode, tallies)) == []
    assert tallies.n_prompt_mismatch == 1 and tallies.drop_share == 1.0


def test_logged_tail_mean_buckets_replay_steps_like_the_trainer() -> None:
    logged = [(2, 0.5), (4, 0.7), (5, 0.9)]
    replay = [
        {"step": 0, "n_positions": 1, "cov": 0.1},
        {"step": 1, "n_positions": 3, "cov": 0.5},  # bucket [0,2): weighted 0.4
        {"step": 2, "n_positions": 1, "cov": 0.8},
        {"step": 3, "n_positions": 1, "cov": 0.6},  # bucket [2,4): 0.7
        {"step": 4, "n_positions": 2, "cov": 0.9},  # bucket [4,5): 0.9
    ]
    lg, rp = census.logged_tail_mean(logged, replay, "cov", last_n=2)
    assert lg == pytest.approx((0.7 + 0.9) / 2)
    assert rp == pytest.approx((0.7 + 0.9) / 2)
    lg3, rp3 = census.logged_tail_mean(logged, replay, "cov", last_n=3)
    assert lg3 == pytest.approx(0.7) and rp3 == pytest.approx((0.4 + 0.7 + 0.9) / 3)
    assert census.logged_tail_mean([(2, None)], replay, "cov") == (None, None)


def _write_leaf(
    root: Path,
    leaf: census.CensusLeaf,
    *,
    vocab: int,
    cov: float,
    bear_rate: float,
    bear_id: int,
    complete: bool = True,
) -> None:
    npz, js = census.leaf_output_paths(root, leaf)
    npz.parent.mkdir(parents=True, exist_ok=True)
    full = np.zeros(vocab, dtype=np.int64)
    full[[0, 1, 2]] = [8, 6, 4]
    full[bear_id] = 2
    masked = full.copy()
    masked[bear_id] = 0
    masked[5] = 2
    entr = np.zeros(vocab, dtype=np.int64)
    entr[5] = 2
    arrays = {}
    for k in census.CENSUS_KS:
        arrays[f"full_k{k}"] = full
        arrays[f"masked_k{k}"] = masked
        arrays[f"entrants_k{k}"] = entr
    np.savez(npz, **arrays)
    steps = [
        {
            "step": s,
            "n_rows": 2,
            "n_positions": 5,
            **{f"cov_full_k{k}": cov for k in census.CENSUS_KS},
            **{f"cov_masked_k{k}": cov for k in census.CENSUS_KS},
            **{f"bear_rate_k{k}": bear_rate for k in census.CENSUS_KS},
            **{f"n_changed_k{k}": 1 for k in census.CENSUS_KS},
        }
        for s in range(4)
    ]
    js.write_text(
        json.dumps(
            {
                "leaf": {**leaf.identity, "run_dir": str(leaf.run_dir)},
                "complete": complete,
                "teacher_forward_precision": "bf16",
                "bear_token_id": bear_id,
                "n_positions": 20,
                "mask_ids": sorted([bear_id, 7]),
                "mask_ids_sha256": "x",
                "rows": {
                    "n_rows": 8,
                    "n_off_policy": 0,
                    "n_flagged": 0,
                    "n_empty": 0,
                    "n_prompt_mismatch": 0,
                    "n_len_mismatch": 0,
                    "n_kept": 8,
                },
                "n_changed": {str(k): 4 for k in census.CENSUS_KS},
                "t_bear_sum": 0.2,
                "t_bear_in_topk": {
                    str(k): {"sum": 0.1, "n": 2} for k in census.CENSUS_KS
                },
                "steps": steps,
            }
        )
    )
    leaf.run_dir.mkdir(parents=True, exist_ok=True)
    (leaf.run_dir / "trainer_log_history.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": s,
                    "gkd/topk_coverage": 0.6,
                    **(
                        {"gkd/target_in_topk_rate": 0.01}
                        if leaf.condition == "standard"
                        else {}
                    ),
                }
            )
            for s in (2, 4)
        )
        + "\n"
    )


def _fake_forms(monkeypatch, bear_id: int) -> None:
    def fake_form_ids(animals, forms, encode, decode):
        return (bear_id,) if "bear" in animals else (7,)

    monkeypatch.setattr(census, "single_token_form_ids", fake_form_ids)
    monkeypatch.setattr(census, "load_animal_forms", lambda: {})
    monkeypatch.setattr(census, "plan_candidates", lambda: ("bear", "cat"))


def test_aggregate_writes_shares_and_passes_validation(
    tmp_path: Path, monkeypatch
) -> None:
    bear_id = 3
    monkeypatch.setattr(plan, "VOCAB_SIZE", 10)
    _fake_forms(monkeypatch, bear_id)
    root = tmp_path / "census_root"
    leaves = census.plan_leaves(tmp_path)
    for lf in leaves:
        _write_leaf(root, lf, vocab=10, cov=0.6, bear_rate=0.01, bear_id=bear_id)
    decode = lambda ids: " bear" if ids == [bear_id] else f"t{ids[0]}"  # noqa: E731
    written = census.aggregate(tmp_path, decode=decode, encode=_encode, root=root)
    assert set(written) == set(census.RESULTS_FILES)
    import pandas as pd

    tokens = pd.read_csv(written["topk_census_tokens.csv"])
    cell = tokens[
        (tokens.pair == "P1")
        & (tokens.k_cell == 10)
        & (tokens.condition == "standard")
        & (tokens.view == "full")
        & (tokens.k_census == 10)
    ]
    assert cell.n_positions.unique().tolist() == [9 * 20]
    top = cell.sort_values("rank").head(3)
    assert top.token_id.tolist() == [0, 1, 2] and top.share.tolist() == pytest.approx(
        [9 * 8 / 180, 9 * 6 / 180, 9 * 4 / 180]
    )
    bear = cell[cell.token_id == bear_id].iloc[0]
    assert bear.is_removed_id and bear.share == pytest.approx(9 * 2 / 180)
    masked_view = tokens[
        (tokens.pair == "P1")
        & (tokens.k_cell == 10)
        & (tokens.condition == "standard")
        & (tokens.view == "masked")
        & (tokens.k_census == 10)
    ]
    assert masked_view[masked_view.token_id == bear_id].n_in_topk.item() == 0
    animals = pd.read_csv(written["topk_census_animal_tokens.csv"])
    assert (
        set(animals.token_id) == {bear_id, 7}
        and animals[animals.token_id == bear_id].is_bear.all()
    )
    entrants = pd.read_csv(written["topk_census_entrants.csv"])
    e = entrants[
        (entrants.pair == "P4")
        & (entrants.k_cell == 1)
        & (entrants.condition == "masked")
        & (entrants.k_census == 1)
    ]
    assert e.token_id.tolist() == [5] and e.share_of_changed.item() == pytest.approx(
        18 / 36
    )
    cells = pd.read_csv(written["topk_census_cells.csv"])
    assert len(cells) == 8 * len(census.CENSUS_KS)
    assert cells.any_removed_id_in_topk_share.unique().tolist() == pytest.approx(
        [36 / 180]
    )
    validation = pd.read_csv(written["topk_census_validation.csv"])
    assert validation.status.eq("ok").all() and len(validation) == 72
    assert (root / "results" / "census_manifest.json").is_file()


def test_aggregate_raises_when_replay_disagrees_with_logs(
    tmp_path: Path, monkeypatch
) -> None:
    bear_id = 3
    monkeypatch.setattr(plan, "VOCAB_SIZE", 10)
    _fake_forms(monkeypatch, bear_id)
    root = tmp_path / "census_root"
    leaves = census.plan_leaves(tmp_path)
    for i, lf in enumerate(leaves):
        _write_leaf(
            root, lf, vocab=10, cov=0.6 if i else 0.2, bear_rate=0.01, bear_id=bear_id
        )
    with pytest.raises(RuntimeError, match="1 census leaves fail validation"):
        census.aggregate(tmp_path, decode=lambda ids: "t", encode=_encode, root=root)


def test_aggregate_raises_on_missing_or_incomplete_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    bear_id = 3
    monkeypatch.setattr(plan, "VOCAB_SIZE", 10)
    _fake_forms(monkeypatch, bear_id)
    root = tmp_path / "census_root"
    leaves = census.plan_leaves(tmp_path)
    for lf in leaves[:-1]:
        _write_leaf(root, lf, vocab=10, cov=0.6, bear_rate=0.01, bear_id=bear_id)
    with pytest.raises(FileNotFoundError, match="1 census leaves missing"):
        census.aggregate(tmp_path, decode=lambda ids: "t", encode=_encode, root=root)
    _write_leaf(
        root,
        leaves[-1],
        vocab=10,
        cov=0.6,
        bear_rate=0.01,
        bear_id=bear_id,
        complete=False,
    )
    with pytest.raises(RuntimeError, match="fail validation"):
        census.aggregate(tmp_path, decode=lambda ids: "t", encode=_encode, root=root)
