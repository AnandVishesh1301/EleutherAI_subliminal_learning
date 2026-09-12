"""Resume / refuse logic of `scripts.diffuse_ladder_surgery` (no GPU: the heavy model,
probe-pool and patching calls are monkeypatched, so only the bookkeeping is exercised)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import numpy as np
from typer.testing import CliRunner

from scripts.diffuse_ladder_surgery import (
    RNG_SCHEME, app, backup_legacy, check_scheme, derived_match, nested_random_pair,
    plan_cells,
)


def test_nested_random_pair_prefix_is_stable():
    from scripts.position_surgery import target_rng

    a = nested_random_pair(8, 7, 4, target_rng(1, "r", "m", "l", "x"),
                           target_rng(1, "r", "m", "v", "x"))
    b = nested_random_pair(8, 7, 6, target_rng(1, "r", "m", "l", "x"),
                           target_rng(1, "r", "m", "v", "x"))
    np.testing.assert_allclose(a[0], b[0][:, :4])
    np.testing.assert_allclose(a[1], b[1][:, :4])

IDENTITY = dict(repo="EleutherAI/pythia-160m-data-seed1", pool_sha="deadbeef", match="fro",
                seed=0, criterion="top-10 left singular vectors, max overlap < 0.5",
                intruder_stride=1)


def doc(results, **over):
    return dict(rng_scheme=RNG_SCHEME, results=results, **{**IDENTITY, **over})


def row(align, k, E):
    return dict(align=align, k=k, E=E, kl=1.0, delta_ce=0.1, intruder_mods="0/1")


def test_check_scheme_rejects_legacy_and_accepts_current():
    with pytest.raises(ValueError, match="legacy"):
        check_scheme({"results": []})
    check_scheme(doc([]))
    with pytest.raises(ValueError, match="keyed_v1"):
        check_scheme({"rng_scheme": "keyed_v1", "results": []})


def test_plan_cells_without_stored_doc_runs_everything():
    cells = [("g", 1, 0.5), ("g", 2, 0.5)]
    assert plan_cells(None, cells, IDENTITY) == (cells, [])


def test_plan_cells_resumes_only_missing_cells():
    stored = doc([row("g", 1, 0.5), row("g", 2, 0.5)])
    requested = [("g", 1, 0.5), ("g", 2, 0.5), ("g", 4, 0.5)]
    cells, kept = plan_cells(stored, requested, IDENTITY)
    assert cells == [("g", 4, 0.5)]
    assert kept == stored["results"]


def test_plan_cells_refuses_a_different_intruder_stride():
    with pytest.raises(ValueError, match="intruder_stride"):
        plan_cells(doc([row("g", 1, 0.5)]), [("g", 1, 0.5)],
                   {**IDENTITY, "intruder_stride": 4})


def test_derived_match_tolerates_float_wobble_but_not_a_different_base():
    assert derived_match(2.5, 2.5 + 1e-8)
    assert not derived_match(2.5, 2.6)
    assert derived_match({"m": 0.25}, {"m": 0.25 + 1e-9})
    assert not derived_match({"m": 0.25}, {"m": 0.30})
    assert not derived_match({"m": 0.25}, {"other": 0.25})


def test_cli_resume_tolerates_derived_float_wobble(stubbed):
    out = stubbed
    assert run().exit_code == 0
    first = json.loads(out.read_text())
    first["ce_base"] += 1e-8  # stubs give ce_base 2.5, θ̄ 0.25
    first["theta_bars"] = {n: v + 1e-9 for n, v in first["theta_bars"].items()}
    out.write_text(json.dumps(first))
    resumed = run("--e-ladder", "0.5,2")
    assert resumed.exit_code == 0, resumed.output
    assert len(json.loads(out.read_text())["results"]) == 4


def test_plan_cells_refuses_legacy_and_identity_mismatch():
    with pytest.raises(ValueError, match="rng_scheme"):
        plan_cells({"results": [], **IDENTITY}, [("g", 1, 0.5)], IDENTITY)
    with pytest.raises(ValueError, match="different identity"):
        plan_cells(doc([], seed=7), [("g", 1, 0.5)], IDENTITY)


def test_backup_legacy_copies_beside_the_stem(tmp_path):
    out = tmp_path / "diffuse_ladder_surgery_ds1.json"
    out.write_text('{"a": 1}')
    backup = backup_legacy(out)
    assert backup.parent == out.parent and backup.name.startswith(
        "diffuse_ladder_surgery_ds1.legacy-") and backup.name.endswith("Z.json")
    assert json.loads(backup.read_text()) == {"a": 1}
    assert backup_legacy(tmp_path / "missing.json") is None


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Patch every GPU/model-touching call; return the analysis output path."""
    from subliminal_learning import config
    from subliminal_learning.analysis import intruder_threshold
    from subliminal_learning.coherence import probe_io
    from subliminal_learning.models import loading_io
    from subliminal_learning.weightspace import deltas, patching

    monkeypatch.setattr(config, "resolve_roots", lambda *a, **k: (tmp_path, tmp_path))
    monkeypatch.setattr(loading_io, "load_base_from_ref",
                        lambda *a, **k: (object(), object()))
    monkeypatch.setattr(probe_io, "resolve_probe_pool_spec",
                        lambda *a, **k: SimpleNamespace(version="pp1"))
    monkeypatch.setattr(probe_io, "build_probe_pool",
                        lambda *a, **k: SimpleNamespace(token_sha256=lambda: "deadbeef"))
    monkeypatch.setattr(probe_io, "parked_logprobs", lambda *a, **k: None)
    monkeypatch.setattr(probe_io, "mean_ce", lambda *a, **k: 2.5)
    monkeypatch.setattr(patching, "base_param_svds",
                        lambda *a, **k: {"mod": {"S": None, "shape": (64, 64),
                                                 "U": np.eye(64), "Vt": np.eye(64),
                                                 "W0np": np.zeros((64, 64)), "theta": 0.25}})
    monkeypatch.setattr(patching, "patch_param_modules", lambda *a, **k: None)
    monkeypatch.setattr(patching, "restore_param_modules", lambda *a, **k: None)
    monkeypatch.setattr(patching, "pile_readout", lambda *a, **k: (1.0, 0.1))
    monkeypatch.setattr(intruder_threshold, "theta_bar", lambda *a, **k: 0.25)
    monkeypatch.setattr(deltas, "diffuse_ladder_delta", lambda *a, **k: 1.0)
    monkeypatch.setattr(deltas, "intruder_positive", lambda *a, **k: False)
    return tmp_path / "analysis" / "diffuse_ladder_surgery_ds1.json"


def squashed(text: str) -> str:
    """Rich wraps error messages to the terminal width; compare without whitespace."""
    return "".join(text.replace("\u2502", "").split())


def run(*args):
    result = CliRunner().invoke(app, ["--k-ladder", "1", "--e-ladder", "0.5", *args])
    return result


@pytest.mark.parametrize("match", ["fro", "spec"])
def test_cli_uses_nested_directions_and_correct_norms(stubbed, monkeypatch, match):
    from subliminal_learning.weightspace import patching

    patches = []
    monkeypatch.setattr(patching, "patch_param_modules",
                        lambda mods, deltas: patches.append(deltas["mod"].copy()))
    result = run("--k-ladder", "1,4", "--e-ladder", "0.5,2", "--match", match)
    assert result.exit_code == 0, result.output
    assert len(patches) == 8
    # Normalize out the per-direction amplitude. Each rank-one update is
    # contained in both the left and right subspaces of the rank-four update.
    for offset in (0, 4):  # random and base-aligned
        units = []
        for i, k in enumerate((1, 4)):
            for j, energy in enumerate((0.5, 2)):
                delta = patches[offset + 2 * i + j]
                amplitude = .25 * energy / (np.sqrt(k) if match == "fro" else 1)
                np.testing.assert_allclose(np.linalg.norm(delta, 2), amplitude)
                np.testing.assert_allclose(np.linalg.norm(delta), np.sqrt(k) * amplitude)
            np.testing.assert_allclose(patches[offset + 2 * i + 1],
                                       4 * patches[offset + 2 * i])
            units.append(patches[offset + 2 * i] / (.125 / (np.sqrt(k) if match == "fro" else 1)))
        one, four = units
        np.testing.assert_allclose(four @ one.T, one @ one.T, atol=1e-12)
        np.testing.assert_allclose(four.T @ one, one.T @ one, atol=1e-12)


@pytest.mark.parametrize("value", ["", "g,g", "x", "a,g,a"])
def test_cli_rejects_invalid_alignments(value):
    result = run("--alignments", value)
    assert result.exit_code != 0
    assert "alignments" in result.output


def test_cli_can_extend_random_only_with_aligned_without_repeating(stubbed):
    result = run("--alignments", "g")
    assert result.exit_code == 0, result.output
    rows = json.loads(stubbed.read_text())["results"]
    assert len(rows) == 1 and rows[0]["align"] == "g"
    assert rows[0]["F_over_theta"] == rows[0]["S_over_theta"] == .5
    result = run("--alignments", "a")
    assert result.exit_code == 0, result.output
    rows = json.loads(stubbed.read_text())["results"]
    assert [r["align"] for r in rows] == ["g", "a"]


def test_cli_writes_scheme_and_refuses_legacy_document(stubbed):
    out = stubbed
    assert run().exit_code == 0
    written = json.loads(out.read_text())
    assert written["rng_scheme"] == RNG_SCHEME and len(written["results"]) == 2

    legacy = {k: v for k, v in written.items() if k != "rng_scheme"}
    out.write_text(json.dumps(legacy))
    refused = run()
    assert refused.exit_code != 0
    assert squashed("rng_scheme") in squashed(refused.output)
    assert squashed(str(out)) in squashed(refused.output)
    assert json.loads(out.read_text()) == legacy  # untouched


def test_cli_force_backs_up_then_starts_empty(stubbed):
    out = stubbed
    legacy = dict(repo="x", results=[row("g", 99, 9.0)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(legacy))

    assert run("--force").exit_code == 0
    backups = sorted(out.parent.glob("diffuse_ladder_surgery_ds1.legacy-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == legacy
    written = json.loads(out.read_text())
    assert written["k_ladder"] == [1] and [(r["align"], r["k"], r["E"]) for r in
                                           written["results"]] == [("g", 1, 0.5), ("a", 1, 0.5)]


def test_cli_resume_keeps_stored_cells_and_unions_ladders(stubbed):
    out = stubbed
    assert run().exit_code == 0
    first = json.loads(out.read_text())
    for r in first["results"]:  # make stored rows recognisable
        r["kl"] = 42.0
    out.write_text(json.dumps(first))

    assert run("--k-ladder", "1,64", "--e-ladder", "0.5,2").exit_code == 0
    written = json.loads(out.read_text())
    assert written["k_ladder"] == [1, 64] and written["e_ladder"] == [0.5, 2.0]
    kept = [r for r in written["results"] if (r["k"], r["E"]) == (1, 0.5)]
    assert len(kept) == 2 and all(r["kl"] == 42.0 for r in kept)
    assert len(written["results"]) == 8  # 2 aligns x 2 k x 2 E
    assert all(r["kl"] == 1.0 for r in written["results"] if (r["k"], r["E"]) != (1, 0.5))


def test_cli_identity_mismatch_refuses(stubbed):
    assert run().exit_code == 0
    refused = run("--seed", "7")
    assert refused.exit_code != 0 and squashed(
        "different identity") in squashed(refused.output)
