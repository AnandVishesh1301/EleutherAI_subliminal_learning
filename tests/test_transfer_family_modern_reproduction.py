import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_family_complete as complete
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as registry
from subliminal_learning.analysis.transfer_significance_io import CellMatrix


def _matrix(cells, teachers, missing_controls=()):
    values = np.full((len(cells), len(teachers) + 1), 0.5)
    for i, cell in enumerate(cells):
        if cell in missing_controls:
            values[i, -1] = np.nan
    return CellMatrix(
        frame=pd.DataFrame(),
        values=values,
        cells=list(cells),
        pools=[*teachers, complete.tio.CONTROL_LABEL],
        control_col=len(teachers),
        missing=[],
    )


def _repo(tmp_path, arms):
    (tmp_path / "run_logs/campaigns/repretrain").mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_logs/registry").mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_logs/registry/core30_cells.json").write_text(
        json.dumps({"cells": ["a", "b"]})
    )
    (tmp_path / "run_logs/campaigns/repretrain/repretrain_significance_arms.json").write_text(
        json.dumps({"ftseed": 1, "prompt_set": "test", "arms": arms})
    )
    return tmp_path


def _arm(name, animal, sender, receiver, **updates):
    arm = {
        "name": name,
        "animal": animal,
        "sender": sender,
        "receiver": receiver,
        "tseed": 1,
        "cells": "../../registry/core30_cells.json",
        "teachers": ["t"],
    }
    arm.update(updates)
    return arm


def test_modern_loader_selects_exact_pairs_and_audits_incomplete(monkeypatch, tmp_path):
    arms = [
        _arm("orig→modern dog", "dog", "pythia-160m", "pythia-160m-pilerepro-seed1234"),
        _arm("modern→orig dog", "dog", "pythia-160m-pilerepro-seed1234", "pythia-160m"),
        _arm("orig→modern cat", "cat", "pythia-160m", "pythia-160m-pilerepro-seed1234"),
        _arm("modern→orig cat", "cat", "pythia-160m-pilerepro-seed1234", "pythia-160m"),
        _arm(
            "orig→modern lion",
            "lion",
            "pythia-160m",
            "pythia-160m-pilerepro-seed1234",
            in_progress=True,
        ),
        _arm(
            "modern→orig lion", "lion", "pythia-160m-pilerepro-seed1234", "pythia-160m"
        ),
        _arm("orig→modern pig", "pig", "pythia-160m", "pythia-160m-pilerepro-seed1234"),
        _arm(
            "modern→prv1 dog",
            "dog",
            "pythia-160m-pilerepro-seed1234",
            "pythia-160m-pilereprov1-seed1234",
        ),
    ]
    repo = _repo(tmp_path, arms)

    def fake(*args, **kwargs):
        miss = ("b",) if args[2] == "pythia-160m" and args[0] == "lion" else ()
        if args[0] == "pig":
            miss = ("a", "b")
        return _matrix(["a", "b"], ["t"], miss)

    monkeypatch.setattr(complete.tio, "load_cell_matrix", fake)
    selected, blocks, excluded, coverage = complete.load_modern_reproduction_pairs(
        repo, tmp_path
    )
    assert set(selected.animal) == {"dog", "cat", "lion"}
    assert set(coverage.animal) == {"dog", "cat", "lion", "pig"}
    assert len(coverage) == 7
    assert coverage.loc[
        coverage.registry_arm.str.contains("orig→modern lion"), "registry_in_progress"
    ].item()
    assert blocks.retained.sum() == 5
    assert (
        coverage.loc[
            coverage.registry_arm.str.contains("orig→modern pig"), "status"
        ].item()
        == "partial"
    )
    assert not blocks.loc[
        blocks.animal.eq("lion") & blocks.receiver.eq("pythia-160m"), "retained"
    ].item()


def test_modern_loader_validates_declared_seed_and_actual_grid(monkeypatch, tmp_path):
    arm = _arm(
        "bad seed", "dog", "pythia-160m", "pythia-160m-pilerepro-seed1234", ftseed=0
    )
    repo = _repo(tmp_path, [arm])
    with pytest.raises(ValueError, match="ftseed"):
        complete.load_modern_reproduction_pairs(repo, tmp_path)
    arm["ftseed"] = 1
    repo = _repo(tmp_path, [arm])
    (repo / "run_logs/registry/wrong_cells.json").write_text(
        json.dumps({"cells": ["a", "b", "c"]})
    )
    arms = [
        _arm(
            "bad grid",
            "dog",
            "pythia-160m",
            "pythia-160m-pilerepro-seed1234",
            cells="../../registry/wrong_cells.json",
        )
    ]
    (repo / "run_logs/campaigns/repretrain/repretrain_significance_arms.json").write_text(
        json.dumps({"ftseed": 1, "prompt_set": "test", "arms": arms})
    )
    with pytest.raises(ValueError, match="canonical core30"):
        complete.load_modern_reproduction_pairs(repo, tmp_path)


def test_optional_context_and_grouping(monkeypatch, tmp_path):
    repo = _repo(tmp_path, [])
    for animal in tfp.DEFAULT_ANIMALS:
        (repo / f"run_logs/registry/{animal}_top10_kband_configs.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (repo / f"run_logs/registry/{animal}_top10_kband_configs.json").write_text(
            json.dumps({"param_tags": ["t"]})
        )
    empty = pd.DataFrame(
        columns=complete.BLOCK + ["student_tag", "test", "ctrl", "relation"],
        dtype=float,
    )
    monkeypatch.setattr(
        tfp, "all_pairs", lambda *a, **k: (empty, tfp.default_context())
    )
    monkeypatch.setattr(
        registry, "registry_pairs", lambda *a, **k: (empty, pd.DataFrame())
    )
    monkeypatch.setattr(registry, "_registry_paths", lambda *a: [])
    monkeypatch.setattr(
        complete,
        "select_complete_pairs",
        lambda *a, **k: (empty, tfp.default_context(), pd.DataFrame(), pd.DataFrame()),
    )
    default = complete.load_complete_pairs(repo, tmp_path)
    assert tfp.RELATION_MODERN_REPRO not in default[1].relations
    monkeypatch.setattr(
        complete,
        "load_modern_reproduction_pairs",
        lambda *a: (empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )
    optional = complete.load_complete_pairs(
        repo, tmp_path, include_modern_reproduction=True
    )
    rels = list(optional[1].relations)
    assert rels.index(tfp.RELATION_MODERN_REPRO) == rels.index(tfp.RELATION_REPRO) + 1
    groups = registry.relation_groups(rels)
    assert len(groups) == 3 and tfp.RELATION_MODERN_REPRO in groups[0]
