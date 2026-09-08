"""Paired surgery readouts and the notebook's actual CE classification path."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.numpy import save_file

from scripts import position_surgery as surgery
from subliminal_learning.weightspace.deltas import (
    BASELINE_SOURCE_STORED_PROBE,
    ORIGINAL_BASELINE_VERSION,
    surgery_original_delta_ce,
    validate_surgery_baselines,
)


@pytest.mark.parametrize("variants", ["orig,rand0", "rand0"])
def test_producer_pairs_original_readout_and_invalidates_legacy(tmp_path, monkeypatch, variants):
    run = tmp_path / "students" / "cell" / "ftseed1"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(json.dumps(
        {"r": 2, "lora_alpha": 2, "base_model_name_or_path": "audit/base"}))
    (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": "audit/base@old"}))
    mod = "gpt_neox.layers.0.attention.query_key_value"
    save_file({f"base_model.model.{mod}.lora_A.weight": np.ones((2, 5), dtype=np.float32),
               f"base_model.model.{mod}.lora_B.weight": np.ones((6, 2), dtype=np.float32)},
              ck / "adapter_model.safetensors")
    runs = tmp_path / "runs.txt"
    runs.write_text(str(run) + "\n")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["p"]')
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "pile_probe.json").write_text(json.dumps(
        {"delta_ce": 0.25, "kl_base_student": 9.0, "manifest_ref": "m"}))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(5, 6, bias=False)
            self.layer.weight.data.fill_(0.01)

    monkeypatch.setattr("subliminal_learning.config.resolve_roots", lambda *a: (tmp_path, tmp_path))
    monkeypatch.setattr("subliminal_learning.models.loading_io.load_base_from_ref",
                        lambda *a: (Model(), SimpleNamespace(pad_token="x", eos_token="x")))
    monkeypatch.setattr("subliminal_learning.evaluation.prompt_sets.prompt_set_path", lambda *a: prompts)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_token_id", lambda *a: 0)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_prob_and_rank_per_prompt",
                        lambda *a: {"p": (0.1, 1)})
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.build_probe_pool",
                        lambda *a: SimpleNamespace(token_sha256=lambda: "pool"))
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.pile_logprobs", lambda *a: None)
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.mean_ce", lambda *a: 0.0)
    monkeypatch.setattr("subliminal_learning.analysis.intruder_threshold.theta_bar", lambda *a: 1.0)
    monkeypatch.setattr("subliminal_learning.weightspace.patching.capture_w0",
                        lambda model: ({mod: model.layer.weight.detach().clone()}, {mod: model.layer}))
    seen_weights = []

    def readout(model, *args):
        seen_weights.append(model.layer.weight.detach().clone())
        return (9.0, 0.25) if len(seen_weights) % 2 else (8.0, 0.5)

    probe = Mock(side_effect=readout)
    monkeypatch.setattr("subliminal_learning.weightspace.patching.pile_readout", probe)
    monkeypatch.setattr("subliminal_learning.weightspace.deltas.intruder_positive", lambda *a: False)
    monkeypatch.setattr(surgery.typer, "echo", lambda *a, **kw: None)
    monkeypatch.setattr(surgery, "git_state", lambda: ("test", False))
    kwargs = dict(runs=runs, n_draws=1, variants=variants, subsets="", t_ladder="",
                  count_intruders=False, animal="dog", prompt_set="test", out_name="audit",
                  seed=0, redo=False, device="cpu", output_root=tmp_path)
    n_passes = len(variants.split(","))  # one per requested variant, never one extra
    surgery.main(**kwargs)
    assert probe.call_count == n_passes  # baseline is reused or read, never re-measured
    if "orig" in variants:
        torch.testing.assert_close(seen_weights[0], torch.full((6, 5), 2.01))
    out = tmp_path / "analysis" / "audit.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    validate_surgery_baselines(rows)
    assert {row["original_delta_ce"] for row in rows} == {0.25}
    assert {row["original_pile_kl"] for row in rows} == {9.0}
    if "orig" in variants:
        assert next(row for row in rows if row["variant"] == "rand0")["delta_ce"] == 0.5
        assert all("original_baseline_source" not in row for row in rows)
    else:
        for row in rows:
            assert row["original_baseline_source"] == BASELINE_SOURCE_STORED_PROBE
            assert row["original_baseline_probe_manifest_ref"] == "m"
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["original_baseline_version"] == ORIGINAL_BASELINE_VERSION
    surgery.main(**kwargs)
    assert probe.call_count == n_passes  # valid resume skips
    for row in rows:
        row.pop("original_baseline_version")
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    surgery.main(**kwargs)
    assert probe.call_count == 2 * n_passes  # old schema recomputed, not reused
    validate_surgery_baselines([json.loads(line) for line in out.read_text().splitlines()])


def test_producer_skips_a_run_with_no_variants(tmp_path, monkeypatch):
    """`--variants none` with every subset skipped must do no work and record it."""
    run = tmp_path / "students" / "cell" / "ftseed1"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(json.dumps(
        {"r": 2, "lora_alpha": 2, "base_model_name_or_path": "audit/base"}))
    (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": "audit/base@old"}))
    mod = "gpt_neox.layers.0.attention.query_key_value"
    save_file({f"base_model.model.{mod}.lora_A.weight": np.ones((2, 5), dtype=np.float32),
               f"base_model.model.{mod}.lora_B.weight": np.ones((6, 2), dtype=np.float32)},
              ck / "adapter_model.safetensors")
    runs = tmp_path / "runs.txt"
    runs.write_text(str(run) + "\n")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["p"]')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(5, 6, bias=False)
            self.layer.weight.data.fill_(0.01)

    monkeypatch.setattr("subliminal_learning.config.resolve_roots", lambda *a: (tmp_path, tmp_path))
    monkeypatch.setattr("subliminal_learning.models.loading_io.load_base_from_ref",
                        lambda *a: (Model(), SimpleNamespace(pad_token="x", eos_token="x")))
    monkeypatch.setattr("subliminal_learning.evaluation.prompt_sets.prompt_set_path", lambda *a: prompts)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_token_id", lambda *a: 0)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_prob_and_rank_per_prompt",
                        lambda *a: {"p": (0.1, 1)})
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.build_probe_pool",
                        lambda *a: SimpleNamespace(token_sha256=lambda: "pool"))
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.pile_logprobs", lambda *a: None)
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.mean_ce", lambda *a: 0.0)
    # theta_bar above every singular value -> no supercritical directions, `super` skipped
    monkeypatch.setattr("subliminal_learning.analysis.intruder_threshold.theta_bar", lambda *a: 1e9)
    monkeypatch.setattr("subliminal_learning.weightspace.patching.capture_w0",
                        lambda model: ({mod: model.layer.weight.detach().clone()}, {mod: model.layer}))
    probe = Mock(side_effect=AssertionError("no forward pass may run"))
    monkeypatch.setattr("subliminal_learning.weightspace.patching.pile_readout", probe)
    monkeypatch.setattr("subliminal_learning.weightspace.deltas.intruder_positive", lambda *a: False)
    monkeypatch.setattr(surgery.typer, "echo", lambda *a, **kw: None)
    monkeypatch.setattr(surgery, "git_state", lambda: ("test", False))
    surgery.main(runs=runs, n_draws=1, variants="none", subsets="super", t_ladder="",
                 count_intruders=False, animal="dog", prompt_set="test", out_name="audit",
                 seed=0, redo=False, device="cpu", output_root=tmp_path)
    assert probe.call_count == 0
    manifest = json.loads((tmp_path / "analysis" / "audit.manifest.json").read_text())
    assert {r["status"] for r in manifest["runs"]} == {"skipped_no_variants"}


def test_producer_writes_no_baseline_when_the_run_has_no_probe(tmp_path, monkeypatch):
    """Without `orig` and without a probe file: still no extra pass, and it is flagged."""
    run = tmp_path / "students" / "cell" / "ftseed1"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "adapter_config.json").write_text(json.dumps(
        {"r": 2, "lora_alpha": 2, "base_model_name_or_path": "audit/base"}))
    (run / "train_manifest.json").write_text(json.dumps({"base_model_repo": "audit/base@old"}))
    mod = "gpt_neox.layers.0.attention.query_key_value"
    save_file({f"base_model.model.{mod}.lora_A.weight": np.ones((2, 5), dtype=np.float32),
               f"base_model.model.{mod}.lora_B.weight": np.ones((6, 2), dtype=np.float32)},
              ck / "adapter_model.safetensors")
    runs = tmp_path / "runs.txt"
    runs.write_text(str(run) + "\n")
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["p"]')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(5, 6, bias=False)
            self.layer.weight.data.fill_(0.01)

    monkeypatch.setattr("subliminal_learning.config.resolve_roots", lambda *a: (tmp_path, tmp_path))
    monkeypatch.setattr("subliminal_learning.models.loading_io.load_base_from_ref",
                        lambda *a: (Model(), SimpleNamespace(pad_token="x", eos_token="x")))
    monkeypatch.setattr("subliminal_learning.evaluation.prompt_sets.prompt_set_path", lambda *a: prompts)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_token_id", lambda *a: 0)
    monkeypatch.setattr("subliminal_learning.evaluation.preference_prob_io.target_prob_and_rank_per_prompt",
                        lambda *a: {"p": (0.1, 1)})
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.build_probe_pool",
                        lambda *a: SimpleNamespace(token_sha256=lambda: "pool"))
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.pile_logprobs", lambda *a: None)
    monkeypatch.setattr("subliminal_learning.coherence.probe_io.mean_ce", lambda *a: 0.0)
    monkeypatch.setattr("subliminal_learning.analysis.intruder_threshold.theta_bar", lambda *a: 1.0)
    monkeypatch.setattr("subliminal_learning.weightspace.patching.capture_w0",
                        lambda model: ({mod: model.layer.weight.detach().clone()}, {mod: model.layer}))
    probe = Mock(return_value=(8.0, 0.5))
    monkeypatch.setattr("subliminal_learning.weightspace.patching.pile_readout", probe)
    monkeypatch.setattr("subliminal_learning.weightspace.deltas.intruder_positive", lambda *a: False)
    monkeypatch.setattr(surgery.typer, "echo", lambda *a, **kw: None)
    monkeypatch.setattr(surgery, "git_state", lambda: ("test", False))
    surgery.main(runs=runs, n_draws=1, variants="rand0", subsets="", t_ladder="",
                 count_intruders=False, animal="dog", prompt_set="test", out_name="audit",
                 seed=0, redo=False, device="cpu", output_root=tmp_path)
    assert probe.call_count == 1  # the variant only
    out = tmp_path / "analysis" / "audit.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["original_delta_ce"] for row in rows] == [None]
    assert all("original_baseline_version" not in row for row in rows)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert {r["status"] for r in manifest["runs"]} == {"computed_no_baseline"}
    assert surgery_original_delta_ce(rows)[1][str(run)] == "missing"


def test_notebook_classifies_original_ce_even_when_kl_crosses_cut(tmp_path):
    nb = json.loads(Path("notebooks/collapse_two_factor_paper_figures.ipynb").read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"
             and "c_['stored']" in "".join(c["source"]) and "dead_ =" in "".join(c["source"])]
    assert len(cells) == 1, f"expected exactly one classification cell, found {len(cells)}"
    src = "".join(cells[0]["source"])
    assert "validate_surgery_baselines" not in src
    tree = ast.parse(src)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    helper = next(node for node in assignments
                  if ast.unparse(node.targets[0]) == "(base_, src_)")
    baseline = next(node for node in assignments if ast.unparse(node.targets[0]) == "c_['stored']")
    classify = next(node for node in assignments if ast.unparse(node.targets[0]) == "dead_")
    code = compile(ast.Module(body=[helper, baseline, classify], type_ignores=[]),
                   "<notebook>", "exec")
    paired = dict(run="r", original_baseline_version=ORIGINAL_BASELINE_VERSION,
                  original_delta_ce=0.9, stored_kl=1.1, adapter_sha256="a", base="m",
                  base_revision=None, pool_version="p", pool_sha="s", prompt_set_sha256="q",
                  animal="dog", seed=0)
    ns = dict(d_=pd.DataFrame([paired]), c_=pd.DataFrame(index=["r"]), cut_=1.0,
              surgery_original_delta_ce=surgery_original_delta_ce, pd=pd, np=np)
    exec(code, ns)
    assert ns["dead_"].tolist() == [False]

    run = tmp_path / "r"
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "pile_probe.json").write_text(json.dumps({"delta_ce": 0.9}))
    legacy = pd.DataFrame([dict(run=str(run), stored_kl=1.1)])
    ns = dict(d_=legacy, c_=pd.DataFrame(index=[str(run)]), cut_=1.0,
              surgery_original_delta_ce=surgery_original_delta_ce, pd=pd, np=np)
    exec(code, ns)
    assert ns["dead_"].tolist() == [False]


@pytest.mark.parametrize("change", [
    {"original_delta_ce": None}, {"original_delta_ce": np.nan}, {"original_delta_ce": np.inf},
    {"original_baseline_version": None}, {"adapter_sha256": None}, {"pool_sha": np.nan},
])
def test_mixed_rows_cannot_hide_missing_baselines_or_identity(change):
    row = dict(run="r", original_baseline_version=ORIGINAL_BASELINE_VERSION, original_delta_ce=0.9,
               adapter_sha256="a", base="m", base_revision=None, pool_version="p", pool_sha="s",
               prompt_set_sha256="q", animal="dog", seed=0)
    with pytest.raises(ValueError):
        validate_surgery_baselines([row, {**row, **change}])
    with pytest.raises(ValueError, match="input identities or baselines"):
        validate_surgery_baselines([row, {**row, "original_delta_ce": 0.8}])


def _backfill(tmp_path, **kw):
    from scripts import backfill_surgery_baselines as bf
    import subliminal_learning.config as cfg

    kwargs = dict(output_root=tmp_path, glob="position_surgery_*.jsonl",
                  apply=False, force=False)
    kwargs.update(kw)
    orig = cfg.resolve_roots
    cfg.resolve_roots = lambda root, hf: (Path(root), tmp_path)
    try:
        bf.main(**kwargs)
    finally:
        cfg.resolve_roots = orig


def test_backfilled_row_reads_as_stored_probe_and_paired_wins(tmp_path):
    run = str(tmp_path / "run")
    back = {"run": run, "variant": "rand0", "original_delta_ce": 0.7,
            "original_baseline_version": ORIGINAL_BASELINE_VERSION,
            "original_baseline_source": BASELINE_SOURCE_STORED_PROBE}
    baselines, sources = surgery_original_delta_ce([back])
    assert sources[run] == "stored_probe" and baselines[run] == 0.7

    paired = {"run": run, "variant": "orig", "original_delta_ce": 0.5,
              "original_baseline_version": ORIGINAL_BASELINE_VERSION}
    baselines, sources = surgery_original_delta_ce([back, paired])
    assert sources[run] == "paired" and baselines[run] == 0.5


def test_backfill_script_stamps_only_legacy_rows_and_is_idempotent(tmp_path, capsys):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    legacy_run = tmp_path / "legacy"
    (legacy_run / "eval").mkdir(parents=True)
    (legacy_run / "eval" / "pile_probe.json").write_text(json.dumps(
        {"delta_ce": 0.7, "kl_base_student": 3.2, "manifest_ref": "m"}))
    noprobe_run = tmp_path / "noprobe"
    a = analysis / "position_surgery_a.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in [
        {"run": str(legacy_run), "variant": "orig", "kl": 1.0},
        {"run": str(legacy_run), "variant": "rand0", "kl": 2.0},
    ]))
    b = analysis / "position_surgery_b.jsonl"
    paired_row = {"run": str(tmp_path / "paired"), "variant": "rand0",
                  "original_delta_ce": 0.5,
                  "original_baseline_version": ORIGINAL_BASELINE_VERSION}
    missing_row = {"run": str(noprobe_run), "variant": "rand0"}
    b.write_text(json.dumps(paired_row) + "\n" + json.dumps(missing_row) + "\n")
    before = {p: p.read_bytes() for p in analysis.iterdir()}

    _backfill(tmp_path)
    out = capsys.readouterr().out
    assert "position_surgery_a.jsonl: rows=2 paired=0 backfilled=2 missing=0" in out
    assert "position_surgery_b.jsonl: rows=2 paired=1 backfilled=0 missing=1" in out
    assert {p: p.read_bytes() for p in analysis.iterdir()} == before

    _backfill(tmp_path, apply=True)
    backups = list(analysis.glob("position_surgery_a.prebackfill-*.jsonl"))
    assert len(backups) == 1 and backups[0].read_bytes() == before[a]
    rows = [json.loads(line) for line in a.read_text().splitlines()]
    for row in rows:
        assert row["original_delta_ce"] == 0.7 and row["original_pile_kl"] == 3.2
        assert row["original_baseline_version"] == ORIGINAL_BASELINE_VERSION
        assert row["original_baseline_source"] == BASELINE_SOURCE_STORED_PROBE
        assert row["original_baseline_probe_manifest_ref"] == "m"
    assert [r["variant"] for r in rows] == ["orig", "rand0"]
    assert b.read_bytes() == before[b]

    _backfill(tmp_path, apply=True)
    out = capsys.readouterr().out
    assert "position_surgery_a.jsonl: rows=2 paired=0 backfilled=0 missing=0 (unchanged=2)" in out
    assert len(list(analysis.glob("position_surgery_a.prebackfill-*.jsonl"))) == 1

    _backfill(tmp_path, apply=True, force=True)
    out = capsys.readouterr().out
    assert "position_surgery_a.jsonl: rows=2 paired=0 backfilled=2 missing=0" in out
    assert b.read_bytes() == before[b]
