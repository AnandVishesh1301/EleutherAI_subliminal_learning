"""Checkpoint-ladder thinning: scope, record gates, resolved paths, archive, apply order."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from scripts import thin_checkpoint_ladders as thin
from subliminal_learning.training.ckpt_ladder import (
    ladder_steps,
    read_ladder,
    steps_covered,
)

N_STEPS = 25000
CADENCE = 625
KEEP = [625, 1250, 1875, 2500, 6250, 12500]
RECORDS = ("pref_traj", "pile_kl_traj", "adapter_spectra_traj")
MANIFEST_NAME = "train_manifest.json"
IN_SCOPE = ("students/dog/pythia-160m-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
            "ex10-20_prefix_n100k/ep20_lr1e-4_r128a32/ftseed1")


def write_spectra(run: Path, steps: list[int], ref: dict, *, truncate: bool = False) -> None:
    """A spectra record the way `spectra_record_problem` demands it: JSON + hashed npz."""
    modules, rank = ["q_proj"], 4
    npz = run / "eval" / thin.SPECTRA_NPZ
    np.savez(npz, steps=np.array(steps, dtype=np.int64),
             sigma=np.ones((len(steps), len(modules), rank), dtype=np.float32))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    if truncate:
        npz.write_bytes(npz.read_bytes()[:-64])
    (run / "eval" / thin.SPECTRA_JSON).write_text(json.dumps({
        "checkpoint_steps": steps, "train_manifest_ref": ref, "npz_sha256": sha,
        "modules": modules, "rank": rank, "points": [{"step": s} for s in steps]}))


def record_points(name: str, steps, animal: str, n_steps: int) -> list[dict]:
    """Real-shaped trajectory points, as `evaluation.ckpt_trajectories` writes them.

    `pref_traj` keys each point by PROMPT SET (a single-animal record's value is the score
    dict itself); `pile_kl_traj` is a flat per-point readout and carries the endpoint TWICE
    (legacy `checkpoint-{n_train_steps}` dir plus the adapter at `checkpoints/`).
    """
    if name == "pref_traj":
        score = {"p_target": 0.5, "ci_lower": 0.4, "ci_upper": 0.6,
                 "rank_mean": 2.0, "rank1_frac": 0.5}
        return [{"step": s, "checkpoint": f"checkpoint-{s}",
                 "test": dict(score), "off_topic": dict(score)} for s in steps]
    pile = [{"step": s, "checkpoint": f"checkpoint-{s}", "kl_base_adapted": 0.2,
             "ce_adapted": 2.5, "delta_ce": 0.1} for s in steps]
    if n_steps in steps:
        pile.append({"step": n_steps, "checkpoint": "checkpoints",
                     "kl_base_adapted": 0.2, "ce_adapted": 2.5, "delta_ce": 0.1})
    return pile


def make_run(root: Path, rel: str, *, animal: str = "dog", cross_base: bool = True,
             tag: str = "ep20_lr1e-4_r128a32", records: tuple[str, ...] = RECORDS,
             dup_final: str | None = "identical", steps: list[int] | None = None,
             record_steps: list[int] | None = None, n_steps: int = N_STEPS,
             cadence: int = CADENCE, epochs: int = 20, points: bool = True,
             truncate_npz: bool = False, endpoint: bool = True) -> Path:
    """A fake finished run: manifest, endpoint adapter, a ladder of distinct-byte adapters."""
    run = root / rel
    (run / "checkpoints").mkdir(parents=True)
    (run / MANIFEST_NAME).write_text(json.dumps({
        "n_train_steps": n_steps, "cross_base": cross_base, "student_param_tag": tag,
        "filtered_sha256": "abc", "hyperparameters": {"n_epochs": epochs}}))
    endpoint_path = run / "checkpoints" / "adapter_model.safetensors"
    if endpoint:
        endpoint_path.write_bytes(b"ENDPOINT" * 8)
    steps = steps if steps is not None else list(range(cadence, n_steps, cadence))
    for step in steps:
        d = run / "checkpoints" / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(f"step{step}".encode() * 4)
    if dup_final is not None and endpoint:
        d = run / "checkpoints" / f"checkpoint-{n_steps}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(
            endpoint_path.read_bytes() if dup_final == "identical" else b"OTHER" * 13)
    ref = {"student_param_tag": tag, "filtered_sha256": "abc", "n_train_steps": n_steps}
    covered = record_steps if record_steps is not None else ladder_steps(run / "checkpoints")
    (run / "eval").mkdir()
    for name in records:
        if name == "adapter_spectra_traj":
            write_spectra(run, list(covered), ref, truncate=truncate_npz)
            continue
        block = record_points(name, covered, animal, n_steps) if points else []
        (run / "eval" / f"{name}.json").write_text(json.dumps({
            "checkpoint_steps": covered, "train_manifest_ref": ref,
            "animals": [animal], "points": block}))
    backdate(run)
    return run


def backdate(run: Path, age_s: float = 86_400) -> None:
    """Age the run's manifest/endpoint: a run written minutes ago reads as in flight."""
    old = time.time() - age_s
    for path in (run / MANIFEST_NAME, run / "checkpoints" / "adapter_model.safetensors"):
        if path.exists():
            os.utime(path, (old, old))


def base_cfg(root: Path, **kwargs) -> dict:
    cfg = {
        "receivers": re.compile(thin.DEFAULT_RECEIVERS),
        "teachers": re.compile(thin.DEFAULT_TEACHERS),
        "keep_epochs": [float(e) for e in thin.DEFAULT_KEEP_EPOCHS.split(",")],
        "require": list(RECORDS), "fast_bytes": False, "no_sha": False,
        "allow_symlinked_checkpoints": False, "protected": {}, "output_root": root,
    }
    cfg.update(kwargs)
    return cfg


def run_cli(root: Path, rel_runs: list[str], **kwargs) -> list[dict]:
    """Inspect the given run dirs with the script's own classifier and default scope."""
    cfg = base_cfg(root, **kwargs)
    return [thin.inspect_run(root / r, root, cfg) for r in rel_runs]


@pytest.fixture(autouse=True)
def _clean_job_chunks():
    """The emitted chunk files live under the repo; test stamps are removed again."""
    yield
    for path in (thin.ROOT / thin.JOBS_REL).glob("teststamp*"):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    make_run(tmp_path, IN_SCOPE)
    return tmp_path


# --------------------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,reason", [
    # same-model (cross_base false) -- set below, path alone is in scope
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/t/p/ep20_lr1e-4_r128a32_sgd/ftseed1",
     "arm_suffix"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/t/p/ep20_lr1e-4_r128a32_stop1250/ftseed1",
     "arm_suffix"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1_sgd/p/"
     "ep20_lr1e-4_r128a32/ftseed1", "arm_suffix"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/t/p_noeos/ep20_lr1e-4_r128a32/ftseed1",
     "arm_suffix"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/t/p/ep20_lr1e-4_r128a32_schcoswu10/ftseed1",
     "arm_suffix"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m/t/ckpt/p/ep20_lr1e-4_r128a32/ftseed1",
     "ckpt_component"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m@step10000/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "at_step"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-410m/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "teacher"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m-pilerepro-seed3/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "teacher"),
    ("students/dog/pythia-160m-seed1/teacher-pythia-160m-weightc-seed1/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "teacher"),
    ("students/dog/pythia-160m-pilerepro-seed1/teacher-pythia-160m/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "receiver"),
    ("students/dog/pythia-410m/teacher-pythia-160m/t/p/ep20_lr1e-4_r128a32/ftseed1",
     "receiver"),
    ("students/sgd/dog/pythia-160m-seed1/teacher-pythia-160m/p/ep20_lr1e-4_r128a32/ftseed1",
     "excluded_tree"),
])
def test_out_of_scope(tmp_path: Path, rel: str, reason: str) -> None:
    make_run(tmp_path, rel)
    (row,) = run_cli(tmp_path, [rel])
    assert row["status"] == f"out_of_scope:{reason}"


def test_manifest_tag_arm_still_excluded(tmp_path: Path) -> None:
    """The path can be clean and the MANIFEST still name an arm tag; that is out of scope."""
    make_run(tmp_path, IN_SCOPE, tag="ep20_lr1e-4_r128a32_sgd")
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "out_of_scope:param_tag"


def test_same_model_out_of_scope(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, cross_base=False)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "out_of_scope:cross_base"


def test_same_model_in_scope_with_include_same_base(tmp_path: Path) -> None:
    """--include-same-base waives THAT rule only; the run is then classified normally."""
    make_run(tmp_path, IN_SCOPE, cross_base=False)
    (row,) = run_cli(tmp_path, [IN_SCOPE], include_same_base=True)
    assert row["status"] == "ready"
    # ... and the other manifest check it shares a block with still refuses.
    make_run(tmp_path, OTHER_ARM, cross_base=False, tag="ep20_lr1e-4_r128a32_sgd")
    (other,) = run_cli(tmp_path, [OTHER_ARM], include_same_base=True)
    assert other["status"] == "out_of_scope:param_tag"


def test_include_same_base_does_not_waive_the_cell_protect(tmp_path: Path) -> None:
    """A same-base run at a nonadaptive-set cell is still protected under the flag."""
    make_run(tmp_path, IN_SCOPE, cross_base=False)
    (row,) = run_cli(tmp_path, [IN_SCOPE], cells=CELLS, include_same_base=True)
    assert row["status"] == "protected:nonadaptive:pythia_student504"


def test_not_a_run_dir_out_of_scope(tmp_path: Path) -> None:
    """D4: a manifest without the endpoint adapter is never a thinnable run."""
    make_run(tmp_path, IN_SCOPE, endpoint=False)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "out_of_scope:not_a_run_dir"


def test_protected_run_untouched(tree: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE], protected={IN_SCOPE: "lattice504"})
    assert row["status"] == "protected:lattice504"


def test_protect_file_parsing_normalises_keys(tmp_path: Path) -> None:
    path = tmp_path / "p.tsv"
    path.write_text("# comment\n\n./students/a/b/\treason1\nstudents/c/./d\n")
    assert thin.read_protect_file(path) == {"students/a/b": "reason1",
                                            "students/c/d": "protected"}


# ------------------------------------------------------------------- nonadaptive cells
# The same 504-lattice cell as IN_SCOPE, but in an arm the rel-keyed protect file (ds1
# receiver, strip teacher) does not enumerate: same-model 160m, other teacher, other pool.
OTHER_ARM = ("students/dog/pythia-160m-seed1/teacher-pythia-160m-seed1/"
             "ep5_lr2e-4_r8a32_tseed1/ex10-20_prefix_n100k/ep20_lr1e-4_r128a32/ftseed3")
CELLS = {"students": {"tag": {"ep20_lr1e-4_r128a32": "pythia_student504"}}}


@pytest.mark.parametrize("rel", [
    IN_SCOPE,
    OTHER_ARM,
    # arm suffixes: a suffixed run of a set cell is a different arm of the same cell
    OTHER_ARM.replace("r128a32", "r128a32_stop1250"),
    OTHER_ARM.replace("r128a32", "r128a32_sgd"),
    OTHER_ARM.replace("r128a32", "r128a32_tf32"),
    # the repo's one-sig-fig lr formatter: `lr0.0001` is the same cell as `lr1e-4`
    OTHER_ARM.replace("lr1e-4_r128a32", "lr0.0001_r128a32"),
])
def test_nonadaptive_cell_protected_in_every_arm(tmp_path: Path, rel: str) -> None:
    make_run(tmp_path, rel)
    (row,) = run_cli(tmp_path, [rel], cells=CELLS)
    assert row["status"] == "protected:nonadaptive:pythia_student504"


def test_neighbouring_cells_are_not_protected(tmp_path: Path) -> None:
    """Protection is the CELL, not the neighbourhood: another rank is still thinnable."""
    rel = OTHER_ARM.replace("r128a32", "r64a32")
    make_run(tmp_path, rel, tag="ep20_lr1e-4_r64a32")
    (row,) = run_cli(tmp_path, [rel], cells=CELLS)
    assert row["status"] == "ready"


# A cell of each SGD set, in the tree its runs are actually in.
SGD_STUDENT_ARM = ("students/sgd/dog/pythia-160m/teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/"
                   "ex10-20_prefix_n100k/ep20_lr1e0_r1a4_sgd/ftseed1")
SGD_TEACHER_ARM = "teachers/dog/pythia-160m/ep20_lr1e2_r8a4_sgd/ftseed1"


def test_shipped_cell_protect_file_covers_every_set(tmp_path: Path) -> None:
    """The committed file protects one real run of each nonadaptive set, Adam and SGD."""
    rules = thin.read_cell_protect_file(thin.ROOT / thin.DEFAULT_CELL_PROTECT)
    cases = {
        "pythia_student504": OTHER_ARM,
        "pythia_teacher504": "teachers/dog/pythia-160m/ep5_lr1e-4_r128a32/ftseed1",
        "olmo_student432": ("students/elephant/DataDecide-dolma1_7-1B/t/p/"
                            "ep10_lr5e-3_r256a512/ftseed1"),
        "qwen_student432": "students/panda/Qwen2.5-1.5B/t/p/ep5_lr5e-5_r256a512/ftseed1",
        "olmo_teacher432": ("teachers/elephant/DataDecide-dolma1_7-1B-seed2/"
                            "ep20_lr1e-4_r1a4/ftseed0"),
        "qwen_teacher432": "teachers/panda/Qwen2.5-1.5B/ep5_lr2e-5_r1a4/ftseed0",
        # SGD sets: student and teacher, Pythia and Qwen, on the SGD lattice's own lrs.
        "pythia_student_sgd510": SGD_STUDENT_ARM,
        "qwen_student_sgd665": ("students/sgd/panda/Qwen2.5-1.5B/t/p/"
                                "ep5_lr1e-1_r128a128_sgd/ftseed1"),
        "pythia_teacher_sgd1792": SGD_TEACHER_ARM,
        "qwen_teacher_sgd1983": "teachers/panda/Qwen2.5-1.5B/ep10_lr1e1_r1a4_sgd/ftseed0",
    }
    for name, rel in cases.items():
        assert thin.cell_protect_reason(rel, rules) == f"nonadaptive:{name}", rel
    # ... and stops at a run that is in none of them.
    assert thin.cell_protect_reason(
        "students/dog/pythia-160m-seed1/t/p/ep20_lr3e-4_r5a5/ftseed1", rules) is None
    # An SGD tag off every SGD lattice is not a set cell either, in either tree.
    for rel in (SGD_STUDENT_ARM.replace("lr1e0_r1a4", "lr1e0_r5a5"),
                SGD_TEACHER_ARM.replace("lr1e2_r8a4", "lr1e2_r5a5")):
        assert thin.cell_protect_reason(rel, rules) is None, rel


@pytest.mark.parametrize("rel", [
    # The OLMo/Qwen student rules are keyed on the trees those families' students live in
    # (elephant/panda animal trees, DataDecide/Qwen receivers anywhere) ...
    "students/dog/DataDecide-c4-1B-seed-default/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/controls/base/DataDecide-c4-1B-seed-default/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/elephant/pythia-160m/t/p/ep10_lr5e-3_r256a512/ftseed1",
    "students/sgd/panda/Qwen2.5-1.5B/t/p/ep5_lr5e-5_r256a512/ftseed1",
])
def test_olmo_qwen_student_rules_cover_every_tree_those_students_live_in(rel: str) -> None:
    rules = thin.read_cell_protect_file(thin.ROOT / thin.DEFAULT_CELL_PROTECT)
    assert (thin.cell_protect_reason(rel, rules) or "").startswith("nonadaptive:")


def test_olmo_qwen_student_rules_no_longer_reach_pythia_runs() -> None:
    """The narrowed rules stop colliding with Pythia runs that share an OLMo/Qwen tag."""
    rules = thin.read_cell_protect_file(thin.ROOT / thin.DEFAULT_CELL_PROTECT)
    for rel in ("students/dog/pythia-160m-seed1/t/p/ep10_lr5e-3_r256a512/ftseed1",
                "students/cat/pythia-160m-data-seed1/t/p/ep5_lr5e-5_r256a512/ftseed1"):
        assert thin.cell_protect_reason(rel, rules) is None, rel


def test_cell_protect_file_is_required(tree: Path, tmp_path: Path) -> None:
    """No flag waives it: --no-protect-file only reaches the path-keyed file."""
    result = runner.invoke(thin.app, [
        "--output-root", str(tree), "--workers", "1", "--no-protect-file",
        "--cell-protect-file", str(tmp_path / "missing.tsv"),
        "--report", str(tmp_path / "report.json")])
    assert result.exit_code != 0
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()


# --------------------------------------------------------------------------- paths
def test_symlinked_run_dir_refused(tmp_path: Path) -> None:
    """D3: a receiver reached through a symlink is refused, never thinned."""
    make_run(tmp_path, IN_SCOPE)
    link = tmp_path / "students" / "dog" / "pythia-160m-seed2"
    link.symlink_to(tmp_path / "students" / "dog" / "pythia-160m-seed1")
    alias = str(Path(IN_SCOPE.replace("pythia-160m-seed1", "pythia-160m-seed2")))
    (row,) = run_cli(tmp_path, [alias])
    assert row["status"] == "symlinked_run_dir"


def test_dotdot_line_matches_protect_key(tmp_path: Path) -> None:
    """D3: a `..`-containing run-dirs line normalises onto the protected (resolved) key."""
    make_run(tmp_path, IN_SCOPE)
    noisy = IN_SCOPE.replace("ex10-20_prefix_n100k/",
                             "ex10-20_prefix_n100k/ep20_lr1e-4_r128a32/../")
    (row,) = run_cli(tmp_path, [noisy], protected={IN_SCOPE: "lattice504"})
    assert row["status"] == "protected:lattice504"
    assert row["run_dir_rel"] == IN_SCOPE


# --------------------------------------------------------------------------- keep set
def test_keep_set_snaps_to_the_cadence(tmp_path: Path) -> None:
    """D1: 25020 steps / 20 epochs / 625 cadence -- six nearest rungs kept, 34 deleted."""
    steps = list(range(625, 625 * 41, 625))
    make_run(tmp_path, IN_SCOPE, n_steps=25020, steps=steps, dup_final=None)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "ready"
    assert row["keep_steps"] == KEEP
    assert row["keep_targets"] == [626, 1251, 1877, 2502, 6255, 12510]
    assert row["n_delete"] == 34 and len(steps) == 40


def test_keep_step_missing_blocks_the_run(tmp_path: Path) -> None:
    """D1: no rung within tolerance of a keep epoch -- nothing is deleted."""
    make_run(tmp_path, IN_SCOPE, steps=[10000, 20000], dup_final=None)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"].startswith("keep_step_missing:")
    assert row["n_delete"] == 0


# --------------------------------------------------------------------------- records
def test_missing_record_untouched_and_emitted(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, records=("pref_traj", "pile_kl_traj"))
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "missing:adapter_spectra_traj:no_json"
    assert row["missing"] == ["adapter_spectra_traj"]
    jobs, _ = thin.write_spectra_jobs([row], "teststamp", 40)
    assert len(jobs) == 1 and jobs[0]["record"] == "adapter_spectra_traj"
    chunk = (thin.ROOT / jobs[0]["chunk"]).read_text().strip()
    assert chunk == ("nice -n 15 uv run python -m "
                     f"scripts.adapter_spectra_trajectory --roots {IN_SCOPE} --workers 1")
    assert jobs[0]["cmd"] == f"bash {jobs[0]['chunk']}"
    assert thin.unfillable_rows([row]) == []


def test_pref_pile_missing_is_unfillable(tmp_path: Path) -> None:
    """D8: cell-addressed records are never guessed at; they are reported instead."""
    make_run(tmp_path, IN_SCOPE, records=("adapter_spectra_traj",))
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    jobs, _ = thin.write_spectra_jobs([row], "teststamp2", 40)
    assert jobs == []
    assert thin.unfillable_rows([row])[0]["missing"] == ["pref_traj", "pile_kl_traj"]


def test_spectra_chunking(tmp_path: Path) -> None:
    rows = [{"run_dir_rel": f"students/dog/r{i}", "missing": ["adapter_spectra_traj"]}
            for i in range(5)]
    jobs, _ = thin.write_spectra_jobs(rows, "teststamp3", 2)
    assert [j["n_runs"] for j in jobs] == [2, 2, 1]


def test_header_without_points_is_missing(tmp_path: Path) -> None:
    """D2: completeness is DATA -- a full header over an empty `points` list is missing."""
    make_run(tmp_path, IN_SCOPE, points=False)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "missing:pref_traj:no_points"


def test_truncated_npz_is_missing(tmp_path: Path) -> None:
    """D2: the spectra JSON can be perfect while its npz is half-written."""
    make_run(tmp_path, IN_SCOPE, truncate_npz=True)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "missing:adapter_spectra_traj:sha_mismatch"


def test_fresh_run_is_in_flight(tmp_path: Path) -> None:
    run = make_run(tmp_path, IN_SCOPE)
    os.utime(run / MANIFEST_NAME, None)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "in_flight"


def test_incomplete_record_is_missing(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, record_steps=[625, 1250])
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "missing:pref_traj:steps_not_covered"


def test_record_superset_of_disk_is_complete(tmp_path: Path) -> None:
    """A record from BEFORE an earlier thinning (more steps than disk) still counts."""
    make_run(tmp_path, IN_SCOPE, steps=KEEP,
             record_steps=sorted({*range(625, N_STEPS, 625), N_STEPS}))
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "ready" and row["delete_steps"] == [N_STEPS]


def test_pile_point_without_delta_ce_blocks_the_thinning(tmp_path: Path) -> None:
    """E1: the evaluator would RECOMPUTE such a point -- which thinning makes impossible."""
    run = make_run(tmp_path, IN_SCOPE)
    rec_path = run / "eval" / "pile_kl_traj.json"
    rec = json.loads(rec_path.read_text())
    step = rec["points"][2]["step"]
    rec["points"][2].pop("delta_ce")
    rec_path.write_text(json.dumps(rec))
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == f"missing:pile_kl_traj:incomplete_point:{step}"
    assert (run / "checkpoints" / f"checkpoint-{step}").is_dir()


def test_pref_point_without_the_animals_p_target_blocks_the_thinning(tmp_path: Path) -> None:
    run = make_run(tmp_path, IN_SCOPE)
    rec_path = run / "eval" / "pref_traj.json"
    rec = json.loads(rec_path.read_text())
    step = rec["points"][1]["step"]
    rec["points"][1]["test"]["p_target"] = None
    rec["points"][1]["off_topic"]["p_target"] = None
    rec_path.write_text(json.dumps(rec))
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == f"missing:pref_traj:incomplete_point:{step}"


def test_pref_traj_wrong_animal(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, animal="cat")
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "missing:pref_traj:animal_not_in_record"


# --------------------------------------------------------------------------- apply
def apply_cfg(root: Path, **kwargs) -> dict:
    return base_cfg(root, **{"no_sha": True, **kwargs})


def test_dry_run_deletes_nothing(tree: Path) -> None:
    before = sorted(p.name for p in (tree / IN_SCOPE / "checkpoints").iterdir())
    (row,) = run_cli(tree, [IN_SCOPE])
    assert row["status"] == "ready" and row["n_delete"] > 0
    assert sorted(p.name for p in (tree / IN_SCOPE / "checkpoints").iterdir()) == before
    assert read_ladder(tree / IN_SCOPE) is None


def test_apply_keeps_the_keep_set(tree: Path, tmp_path: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])
    assert thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl") == "thinned"
    left = sorted(int(p.name.split("-")[1])
                  for p in (tree / IN_SCOPE / "checkpoints").glob("checkpoint-*"))
    assert left == KEEP
    assert (tree / IN_SCOPE / "checkpoints" / "adapter_model.safetensors").is_file()
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == row["n_delete"]


def test_apply_records_the_full_ladder(tree: Path, tmp_path: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])
    thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    doc = read_ladder(tree / IN_SCOPE)
    assert doc["checkpoint_steps"] == sorted({*range(625, N_STEPS + 1, 625)})
    assert doc["n_train_steps"] == N_STEPS and doc["epochs"] == 20
    assert doc["derived_docs_verified"] == {name: True for name in RECORDS}
    assert all(doc["checkpoints"][str(s)]["kept"] for s in KEEP)
    assert doc["checkpoints"]["3125"]["deleted_at"]
    assert doc["checkpoints"][str(N_STEPS)]["adapter_bytes"] > 0
    assert doc["keep_steps"] == KEEP and doc["keep_targets"] == KEEP


def test_archive_written_before_delete(tree: Path, tmp_path: Path) -> None:
    """D7: the three records are copied out of the run, and named in the manifest."""
    (row,) = run_cli(tree, [IN_SCOPE])
    thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    archived = read_ladder(tree / IN_SCOPE)["archived"]
    assert set(archived) == {"pref_traj.json", "pile_kl_traj.json",
                             thin.SPECTRA_JSON, thin.SPECTRA_NPZ}
    for rel in archived.values():
        assert (tree / rel).is_file()
    stem = IN_SCOPE.replace("/", "__")
    assert (tree / thin.ARCHIVE_REL / "pythia-160m-seed1" / "pref_traj"
            / f"{stem}.json").is_file()


def test_archive_failure_blocks_the_delete(tree: Path, tmp_path: Path, monkeypatch) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])

    def boom(src, dst):
        raise OSError("no space")

    monkeypatch.setattr(thin.shutil, "copyfile", boom)
    assert thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl") == "archive_failed"
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()
    assert read_ladder(tree / IN_SCOPE) is None


def test_plan_changed_when_a_record_vanishes(tree: Path, tmp_path: Path) -> None:
    """D6: the plan is re-taken under the claim; a vanished record stops the delete."""
    (row,) = run_cli(tree, [IN_SCOPE])
    (tree / IN_SCOPE / "eval" / "pile_kl_traj.json").unlink()
    status = thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    assert status.startswith("plan_changed:missing:pile_kl_traj")
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()
    assert read_ladder(tree / IN_SCOPE) is None


def test_plan_changed_when_a_rung_appears(tree: Path, tmp_path: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])
    extra = tree / IN_SCOPE / "checkpoints" / "checkpoint-313"
    extra.mkdir()
    (extra / "adapter_model.safetensors").write_bytes(b"new")
    assert thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl").startswith(
        "plan_changed")
    assert extra.is_dir()


def test_endpoint_symlinked_into_a_rung_refused(tmp_path: Path) -> None:
    """D5: deleting a rung the endpoint adapter points into would destroy the run."""
    run = make_run(tmp_path, IN_SCOPE, dup_final=None)
    endpoint = run / "checkpoints" / "adapter_model.safetensors"
    endpoint.unlink()
    endpoint.symlink_to(run / "checkpoints" / "checkpoint-3125" / "adapter_model.safetensors")
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "endpoint_symlink"
    assert (run / "checkpoints" / "checkpoint-3125").is_dir()


def test_thinned_run_still_reads_current(tree: Path, tmp_path: Path) -> None:
    """The coverage gate keeps the full stored record current after the thinning."""
    (row,) = run_cli(tree, [IN_SCOPE])
    thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    stored = json.loads((tree / IN_SCOPE / "eval" / "pref_traj.json").read_text())
    disk = sorted(thin.on_disk_steps(tree / IN_SCOPE / "checkpoints"))
    assert steps_covered(stored["checkpoint_steps"], disk)


def test_second_apply_is_already_thinned(tree: Path, tmp_path: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])
    thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    (again,) = run_cli(tree, [IN_SCOPE])
    assert again["status"] == "already_thinned"


def test_identical_duplicate_final_removed(tree: Path, tmp_path: Path) -> None:
    (row,) = run_cli(tree, [IN_SCOPE])
    thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    assert not (tree / IN_SCOPE / "checkpoints" / f"checkpoint-{N_STEPS}").exists()


def test_differing_final_blocks_the_whole_run(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, dup_final="different")
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "final_dup_differs"


def test_no_intermediates(tmp_path: Path) -> None:
    make_run(tmp_path, IN_SCOPE, steps=[], dup_final=None)
    (row,) = run_cli(tmp_path, [IN_SCOPE])
    assert row["status"] == "nothing_to_delete"


def test_sha_not_reused_when_the_adapter_changes(tree: Path, tmp_path: Path) -> None:
    """D10: a rewritten adapter never inherits the previous file's digest."""
    (row,) = run_cli(tree, [IN_SCOPE])
    cfg = apply_cfg(tree, no_sha=False)
    thin.write_ladder(tree / IN_SCOPE, thin.ladder_doc(tree / IN_SCOPE, row, cfg))
    first = read_ladder(tree / IN_SCOPE)["checkpoints"][str(N_STEPS)]["adapter_sha256"]
    endpoint = tree / IN_SCOPE / "checkpoints" / "adapter_model.safetensors"
    endpoint.write_bytes(b"REPLACED" * 9)
    thin.write_ladder(tree / IN_SCOPE, thin.ladder_doc(tree / IN_SCOPE, row, cfg))
    second = read_ladder(tree / IN_SCOPE)["checkpoints"][str(N_STEPS)]["adapter_sha256"]
    assert first and second and first != second


def test_ladder_written_before_delete_and_rerun_finishes(tree: Path, tmp_path: Path,
                                                         monkeypatch) -> None:
    """A kill inside the first rmtree leaves a manifest; the rerun completes the thinning."""
    (row,) = run_cli(tree, [IN_SCOPE])
    real = thin.shutil.rmtree

    def boom(path, *a, **k):
        raise OSError("simulated kill")

    monkeypatch.setattr(thin.shutil, "rmtree", boom)
    assert thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl") == "thin_partial"
    doc = read_ladder(tree / IN_SCOPE)
    assert doc is not None and doc["checkpoint_steps"]
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()

    monkeypatch.setattr(thin.shutil, "rmtree", real)
    (retry,) = run_cli(tree, [IN_SCOPE])
    assert retry["status"] == "ready"
    assert thin.thin_run(retry, apply_cfg(tree), tmp_path / "log.jsonl") == "thinned"
    left = sorted(int(p.name.split("-")[1])
                  for p in (tree / IN_SCOPE / "checkpoints").glob("checkpoint-*"))
    assert left == KEEP


def test_ladder_stamped_on_keyboard_interrupt(tree: Path, tmp_path: Path, monkeypatch) -> None:
    """D10: the `deleted_at` rewrite is in a `finally`, so even a ^C leaves it recorded."""
    (row,) = run_cli(tree, [IN_SCOPE])
    real = thin.shutil.rmtree
    calls = {"n": 0}

    def once(path, *a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        real(path, *a, **k)

    monkeypatch.setattr(thin.shutil, "rmtree", once)
    with pytest.raises(KeyboardInterrupt):
        thin.thin_run(row, apply_cfg(tree), tmp_path / "log.jsonl")
    doc = read_ladder(tree / IN_SCOPE)
    assert doc["checkpoints"][str(row["delete_steps"][0])]["deleted_at"]


# --------------------------------------------------------------------------- CLI
runner = CliRunner()


def invoke(root: Path, tmp_path: Path, *args: str):
    # The real cell protect file covers this fixture's tag (`ep20_lr1e-4_r128a32` IS a
    # nonadaptive cell), so the CLI tests point at one whose rules match nothing.
    cells = tmp_path / "cells.tsv"
    if not cells.is_file():
        cells.write_text("students\ttag\tep20_lr7e-7_r3a3\tpythia_student504\n")
    return runner.invoke(thin.app, ["--output-root", str(root), "--workers", "1",
                                    "--cell-protect-file", str(cells),
                                    "--report", str(tmp_path / "report.json"), *args])


@pytest.fixture()
def protect(tmp_path: Path) -> Path:
    path = tmp_path / "protect.tsv"
    path.write_text("students/dog/pythia-160m-seed1/other\tlattice504\n")
    return path


def test_cli_dry_run_writes_no_ladder(tree: Path, tmp_path: Path) -> None:
    result = invoke(tree, tmp_path)
    assert result.exit_code == 0, result.output
    assert read_ladder(tree / IN_SCOPE) is None
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()
    report = json.loads((tmp_path / "report.json").read_text())
    assert [r["status"] for r in report["runs"]] == ["ready"]


def test_cli_apply_without_protect_file_refuses(tree: Path, tmp_path: Path) -> None:
    result = invoke(tree, tmp_path, "--apply", "--protect-file", str(tmp_path / "none.tsv"))
    assert result.exit_code != 0
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()


def test_cli_apply_thins(tree: Path, tmp_path: Path, protect: Path) -> None:
    result = invoke(tree, tmp_path, "--apply", "--protect-file", str(protect))
    assert result.exit_code == 0, result.output
    left = sorted(int(p.name.split("-")[1])
                  for p in (tree / IN_SCOPE / "checkpoints").glob("checkpoint-*"))
    assert left == KEEP
    assert read_ladder(tree / IN_SCOPE)["archived"]
    assert (tmp_path / "report.deleted.jsonl").is_file()
    report = json.loads((tmp_path / "report.json").read_text())
    assert [r["status"] for r in report["runs"]] == ["thinned"]


@pytest.mark.parametrize("mode_args", [
    ("--teacher-tree", "teachers/dog/pythia-160m"),
    ("--control-tree", "students/controls"),
])
def test_cli_include_same_base_is_student_mode_only(tree: Path, tmp_path: Path,
                                                    mode_args: tuple[str, str]) -> None:
    result = invoke(tree, tmp_path, "--include-same-base", *mode_args)
    assert result.exit_code != 0
    assert "student-mode" in result.output


def test_cli_expect_runs_mismatch_refuses(tree: Path, tmp_path: Path, protect: Path) -> None:
    result = invoke(tree, tmp_path, "--apply", "--protect-file", str(protect),
                    "--expect-runs", "500")
    assert result.exit_code != 0
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()


def test_cli_limit_with_expect_runs_rejected(tree: Path, tmp_path: Path,
                                             protect: Path) -> None:
    result = invoke(tree, tmp_path, "--apply", "--protect-file", str(protect),
                    "--expect-runs", "1", "--limit", "5")
    assert result.exit_code != 0
    assert (tree / IN_SCOPE / "checkpoints" / "checkpoint-3125").is_dir()


# ------------------------------------------- concurrent report flushes (regression)
def test_parallel_apply_survives_overlapping_report_flushes(tmp_path: Path,
                                                            monkeypatch) -> None:
    """A flush on every run, 8 workers, a slow writer: the pass must still finish.

    The periodic report write used to run inside the worker threads with no lock, so two
    threads could be inside one `atomic_write_json` of the SAME path: the first rename took
    the shared temp file away and the second raised FileNotFoundError out of `pool.map`,
    killing the invocation mid-apply. The flush is now coalesced under a lock (and a failed
    snapshot is never fatal), so every run is thinned and the final report is valid.
    """
    rels = [IN_SCOPE.replace("ftseed1", f"ftseed{i}") for i in range(64)]
    for rel in rels:
        make_run(tmp_path, rel)
    monkeypatch.setattr(thin, "FLUSH_EVERY", 1)
    real = thin.atomic_write_json

    def slow(path, obj, **kwargs):
        # Wide enough that the next worker reaches its flush boundary while we are inside.
        time.sleep(0.05)
        return real(path, obj, **kwargs)

    monkeypatch.setattr(thin, "atomic_write_json", slow)
    result = invoke(tmp_path, tmp_path, "--workers", "8", "--apply", "--no-protect-file")
    assert result.exit_code == 0, result.output + str(result.exception)

    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["runs"]) == len(rels)
    assert {r["status"] for r in report["runs"]} == {"thinned"}
    for rel in rels:
        assert sorted(thin.on_disk_steps(tmp_path / rel / "checkpoints")) == KEEP
    assert list((tmp_path).glob("report.json.tmp*")) == []


def test_a_failing_report_flush_is_not_fatal(tmp_path: Path, monkeypatch) -> None:
    """The report is a snapshot; the deletion jsonl is the durable log."""
    for i in range(4):
        make_run(tmp_path, IN_SCOPE.replace("ftseed1", f"ftseed{i}"))
    monkeypatch.setattr(thin, "FLUSH_EVERY", 1)
    real = thin.atomic_write_json
    calls: list[int] = []

    def flaky(path, obj, **kwargs):
        calls.append(1)
        if len(calls) < 3:                     # the periodic flushes blow up...
            raise OSError("transient")
        return real(path, obj, **kwargs)       # ...the final one lands

    monkeypatch.setattr(thin, "atomic_write_json", flaky)
    result = invoke(tmp_path, tmp_path, "--workers", "2", "--apply", "--no-protect-file")
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "report flush failed" in result.output
    assert {r["status"] for r in
            json.loads((tmp_path / "report.json").read_text())["runs"]} == {"thinned"}
