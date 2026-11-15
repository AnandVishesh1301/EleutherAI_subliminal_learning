"""Checkpoint-ladder thinning: scope, record gates, resolved paths, archive, apply order.

Control-tree mode: wipe the intermediate rungs of control runs, thin the set cells.

`students/controls/{base,rng}/...` runs exist to be compared against, not to be read along
their trajectory: their intermediate `checkpoint-N/` dirs are pure storage, so control mode
WIPES them (`ready:wipe`) and requires no record to do it. The one exception is a control
run that sits at a nonadaptive-set cell: it is thinned to the student keep epochs
(`ready:set_thin`) rather than refused, which is the only place the cell protect file does
not simply stop the script. Nothing outside `students/controls/` is ever in scope, the
archive lands off the (full) output disk, and a ladder write the full disk refuses is
retried after the deletions have made room.

Teacher-tree thinning: scope, keep set per epoch count, and the tokenizer-copy strip.

The teacher tree (`teachers/{animal}/{base}/{param_tag}/ftseed{N}`) is thinned by the same
script as the students but with its own scope, keep epochs (1/2/5 + the root adapter) and
record requirement (the spectra only). The tokenizer strip is a SEPARATE action: it removes
per-rung copies that are byte-identical to the run's root `checkpoints/` file, so it needs
no record and must run even for a run that cannot be thinned.
"""

from __future__ import annotations

from pathlib import Path

import errno

import hashlib

import json

import os

import re

import shutil

import time

from typer.testing import CliRunner

import numpy as np

import pytest

from scripts.ops import thin_checkpoint_ladders as thin

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
                     f"scripts.ops.adapter_spectra_trajectory --roots {IN_SCOPE} --workers 1")
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

BASE_REL = "students/controls/base/pythia-410m/teacher-pythia-160m/ex10-20_prefix_n100k"

RNG_REL = "students/controls/rng/DataDecide-dolma1_7-1B-seed-default/ex10-20_prefix_n100k"

SET_TAG = "ep20_lr1e-4_r128a32"       # a cell of the synthetic protect file below

PLAIN_TAG = "ep20_lr9e-5_r4a32"       # not a set cell

CONTROL_CELLS = {"students": {"tag": {SET_TAG: "pythia_student504"}}}

def out(tmp_path: Path) -> Path:
    """The output root, a level BELOW tmp_path: the archive must land outside it."""
    root = tmp_path / "outputs"
    root.mkdir(exist_ok=True)
    return root

def make_control_run(root: Path, tag: str, *, parent: str = BASE_REL, epochs: int = 20,
                     n_steps: int = 60, ftseed: int = 1, traj: bool = True,
                     saves_per_epoch: int = 1) -> Path:
    """A finished control run: endpoint adapter + one rung per epoch, no record required."""
    run = root / parent / tag / f"ftseed{ftseed}"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": n_steps, "student_param_tag": tag, "animal": None,
        "cross_base": False, "hyperparameters": {"n_epochs": epochs}}))
    (ck / "adapter_model.safetensors").write_bytes(b"ENDPOINT" * 8)
    (ck / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    per = n_steps // (epochs * saves_per_epoch)
    for step in range(per, n_steps, per):
        d = ck / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(f"step{step}".encode() * 4)
        (d / "adapter_config.json").write_text("{}")
    (run / "eval").mkdir()
    if traj:
        for name in ("pref_traj", "pile_kl_traj"):
            (run / "eval" / f"{name}.json").write_text(json.dumps({"points": [{"step": 3}]}))
    old = time.time() - 86_400
    for path in run.rglob("*"):
        os.utime(path, (old, old))
    os.utime(run, (old, old))
    return run

def control_cfg(root: Path, **kwargs) -> dict:
    cfg = {
        "receivers": re.compile(thin.DEFAULT_RECEIVERS),
        "teachers": re.compile(thin.DEFAULT_TEACHERS),
        "keep_epochs": [float(e) for e in thin.CONTROL_KEEP_EPOCHS.split(",")],
        "require": [], "fast_bytes": False, "no_sha": False,
        "allow_symlinked_checkpoints": False, "protected": {}, "output_root": root,
        "teacher_tree": None, "control_tree": thin.CONTROL_TREE,
        "archive_extra": list(thin.CONTROL_ARCHIVE_EXTRA),
        "archive_root": root.parent / "archive_off_disk",
        "strip_tokenizers": False, "tokenizer_sha": False, "cells": CONTROL_CELLS,
    }
    cfg.update(kwargs)
    return cfg

def invoke_control(root: Path, tmp_path: Path, *args: str):
    cells = tmp_path / "cells.tsv"
    cells.write_text(f"students\ttag\t{SET_TAG}\tpythia_student504\n")
    return CliRunner().invoke(thin.app, [
        "--output-root", str(root), "--workers", "1", "--walk-workers", "2",
        "--control-tree", thin.CONTROL_TREE,
        "--protect-file", str(tmp_path / "none.tsv"), "--no-protect-file",
        "--cell-protect-file", str(cells),
        "--archive-root", str(tmp_path / "archive_off_disk"),
        "--report", str(tmp_path / "report.json"), *args])

# --------------------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,reason", [
    (f"{BASE_REL}/{PLAIN_TAG}/ftseed1", None),
    (f"{RNG_REL}/{PLAIN_TAG}/ftseed1", None),
    (f"{BASE_REL}/{PLAIN_TAG}_sgd/ftseed1", None),       # arms are ordinary tags here
    ("students/dog/pythia-160m-seed1/t/p/ep20_lr1e-4_r4a32/ftseed1", "not_a_control_run"),
    ("teachers/dog/pythia-160m/ep5_lr1e-4_r32a64/ftseed0", "not_a_control_run"),
    ("students/controls", "not_a_control_run"),
    (f"{BASE_REL}/ckpt/{PLAIN_TAG}/ftseed1", "ckpt_component"),
    (f"{BASE_REL}/{PLAIN_TAG}@step1000/ftseed1", "at_step"),
])
def test_control_scope(rel, reason):
    got = thin.control_scope_reason(rel, thin.CONTROL_TREE)
    assert got == (reason if reason != "at_step" else "ckpt_component")

def test_narrower_prefix_excludes_the_other_receivers():
    prefix = "students/controls/base/pythia-410m"
    assert thin.control_scope_reason(f"{BASE_REL}/{PLAIN_TAG}/ftseed1", prefix) is None
    assert thin.control_scope_reason(f"{RNG_REL}/{PLAIN_TAG}/ftseed1",
                                     prefix) == "other_tree"

def test_labels_name_the_layout_and_the_receiver():
    assert thin.control_labels(f"{BASE_REL}/{PLAIN_TAG}/ftseed1") == ("base", "pythia-410m")
    assert thin.control_labels(f"{RNG_REL}/{PLAIN_TAG}/ftseed1") == (
        "rng", "DataDecide-dolma1_7-1B-seed-default")

# --------------------------------------------------------------------------- planning
def test_plain_control_run_is_wiped(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:wipe"
    assert row["keep_steps"] == [] and row["n_keep"] == 0
    assert row["delete_steps"] == sorted(thin.on_disk_steps(run / "checkpoints"))
    assert len(row["delete_steps"]) == 19 and row["bytes_to_free"] > 0
    assert (row["animal"], row["receiver"]) == ("base", "pythia-410m")

def test_set_cell_control_run_is_thinned_not_refused(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:set_thin"
    # 0.5, 1, 1.5, 2, 5 and 10 epochs of a 20-epoch, 600-step run (30 steps/epoch)
    assert row["keep_steps"] == [15, 30, 45, 60, 150, 300]
    assert row["cell"] == "nonadaptive:pythia_student504"

def test_rng_layout_is_in_scope(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG, parent=RNG_REL)
    row = thin.inspect_run(run, out(tmp_path), control_cfg(out(tmp_path)))
    assert row["status"] == "ready:wipe"
    assert row["animal"] == "rng"

def test_path_protect_file_still_refuses(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    rel = f"{BASE_REL}/{PLAIN_TAG}/ftseed1"
    cfg = control_cfg(out(tmp_path), protected={rel: "hand-held"})
    assert thin.inspect_run(run, out(tmp_path), cfg)["status"] == "protected:hand-held"

# --------------------------------------------------------------------------- apply
def test_wipe_deletes_every_rung_and_keeps_the_endpoint(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    log = tmp_path / "controls_report.deleted.jsonl"
    assert thin.thin_run(row, cfg, log) == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    assert (run / "checkpoints" / "adapter_model.safetensors").read_bytes() == b"ENDPOINT" * 8
    assert (run / "eval" / "pref_traj.json").is_file()
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(logged) == 19
    assert all(r["bytes"] > 0 and r["adapter_sha256"] and r["step"] for r in logged)
    ladder = thin.read_ladder(run)
    assert ladder["action"] == "ready:wipe" and ladder["keep_steps"] == []
    assert all(c["deleted_at"] for c in ladder["checkpoints"].values() if not c["kept"])

def test_set_cell_apply_keeps_the_keep_epochs(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [15, 30, 45, 60, 150, 300]

def test_records_are_archived_off_the_output_root(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None
    assert set(written) == {"pref_traj.json", "pile_kl_traj.json"}
    for dest in written.values():
        assert Path(dest).is_absolute() and Path(dest).is_file()
        assert Path(dest).is_relative_to(cfg["archive_root"])
    # ...and the record itself is never deleted.
    thin.thin_run(row, cfg, tmp_path / "log.jsonl")
    assert (run / "eval" / "pref_traj.json").is_file()

def test_second_apply_pass_is_a_no_op(tmp_path: Path):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    again = thin.inspect_run(run, out(tmp_path), cfg)
    assert again["status"] == "already_thinned"
    assert again.get("delete_steps") is None
    endpoint = run / "checkpoints" / "adapter_model.safetensors"
    assert endpoint.read_bytes() == b"ENDPOINT" * 8

def test_enospc_on_the_ladder_write_still_deletes_then_writes(tmp_path: Path,
                                                              monkeypatch):
    """A full disk refuses the pre-delete manifest; the deletes make room for the retry."""
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    calls: list[int] = []
    real = thin.write_ladder

    def flaky(run_dir, doc):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(run_dir, doc)

    monkeypatch.setattr(thin, "write_ladder", flaky)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    ladder = thin.read_ladder(run)
    assert ladder is not None and ladder["ladder_written_after_delete"] is True
    assert len(ladder["checkpoints"]) == 20

def test_enospc_twice_is_reported_as_deleted_ladder_unwritten(tmp_path: Path, monkeypatch):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)

    def always(run_dir, doc):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(thin, "write_ladder", always)
    log = tmp_path / "log.jsonl"
    assert thin.thin_run(row, cfg, log) == "deleted:ladder_unwritten"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    # The jsonl is the rebuild source: every step with its bytes and digest.
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(logged) == 19 and all(r["adapter_sha256"] for r in logged)

def test_enospc_is_not_tolerated_outside_control_mode(tmp_path: Path, monkeypatch):
    """Student/teacher mode is unchanged: a failed manifest write still aborts."""
    def always(run_dir, doc):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(thin, "write_ladder", always)
    with pytest.raises(OSError):
        thin._write_ladder_or_enospc(tmp_path, {}, {"control_tree": None})

# --------------------------------------------------------------------------- CLI
def test_non_controls_prefix_is_rejected(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(tmp_path), "--control-tree", "students/dog",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--control-tree" in str(result.output) + str(result.exception)

def test_run_dirs_file_cannot_reach_outside_controls(tmp_path: Path):
    """The one path that hands in run dirs directly is scoped the same way."""
    outside = make_control_run(out(tmp_path), PLAIN_TAG,
                               parent="students/dog/pythia-160m-seed1/t/p")
    inside = make_control_run(out(tmp_path), PLAIN_TAG)
    listing = tmp_path / "runs.txt"
    listing.write_text(f"{outside}\n{inside}\n")
    result = invoke_control(out(tmp_path), tmp_path, "--run-dirs-file", str(listing), "--apply")
    assert result.exit_code == 0, result.output
    statuses = {r["run_dir_rel"]: r["status"]
                for r in json.loads((tmp_path / "report.json").read_text())["runs"]}
    assert statuses[f"students/dog/pythia-160m-seed1/t/p/{PLAIN_TAG}/ftseed1"] == \
        "out_of_scope:not_a_control_run"
    assert len(thin.on_disk_steps(outside / "checkpoints")) == 19
    assert thin.on_disk_steps(inside / "checkpoints") == {}

def test_dry_run_table_separates_wipe_from_set_thin(tmp_path: Path):
    make_control_run(out(tmp_path), PLAIN_TAG)
    make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    make_control_run(out(tmp_path), PLAIN_TAG, parent=RNG_REL)
    result = invoke_control(out(tmp_path), tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["control_tree"] == thin.CONTROL_TREE and report["require"] == []
    assert sorted(r["status"] for r in report["runs"]) == [
        "ready:set_thin", "ready:wipe", "ready:wipe"]
    table = thin.control_summary(report["runs"])
    assert "control mode (wipe | set_thin | refused)" in table
    assert "base/pythia-410m" in table and "rng/DataDecide-dolma1_7-1B-seed-default" in table
    # nothing was deleted by the dry run
    assert len(thin.on_disk_steps(out(tmp_path) / BASE_REL / PLAIN_TAG / "ftseed1"
                                  / "checkpoints")) == 19

# ------------------------------------------------- full disk: EDQUOT, claims, failures
def test_edquot_is_the_same_full_disk_condition_as_enospc(tmp_path: Path):
    """cold-1 reports the quota errno, not ENOSPC; control mode must treat them alike."""
    cfg = control_cfg(out(tmp_path))
    for code in (errno.ENOSPC, errno.EDQUOT):
        assert thin.full_disk_error(OSError(code, "full"), cfg)
        assert not thin.full_disk_error(OSError(code, "full"), {"control_tree": None})
    assert not thin.full_disk_error(OSError(errno.EACCES, "denied"), cfg)

def test_edquot_on_the_ladder_write_deletes_then_writes(tmp_path: Path, monkeypatch):
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    calls: list[int] = []
    real = thin.write_ladder

    def flaky(run_dir, doc):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(run_dir, doc)

    monkeypatch.setattr(thin, "write_ladder", flaky)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}
    assert thin.read_ladder(run)["ladder_written_after_delete"] is True

def test_claim_edquot_runs_claimless_and_the_next_run_claims_normally(tmp_path: Path,
                                                                     monkeypatch):
    """A full disk cannot create `.claim`; that must not stop the run that frees it."""
    first = make_control_run(out(tmp_path), PLAIN_TAG)
    second = make_control_run(out(tmp_path), PLAIN_TAG, ftseed=2)
    cfg = control_cfg(out(tmp_path))
    calls: list[Path] = []
    real = thin.acquire_claim

    def flaky(run_dir, *args, **kwargs):
        calls.append(Path(run_dir))
        if len(calls) == 1:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(run_dir, *args, **kwargs)

    monkeypatch.setattr(thin, "acquire_claim", flaky)
    log = tmp_path / "log.jsonl"
    rows = [thin.inspect_run(r, out(tmp_path), cfg) for r in (first, second)]
    assert [thin.thin_run(row, cfg, log) for row in rows] == ["thinned", "thinned"]
    assert rows[0]["claimless"] is True and rows[1]["claimless"] is False
    assert thin.read_ladder(first)["claimless"] is True
    assert thin.read_ladder(second)["claimless"] is False
    assert all(thin.on_disk_steps(r / "checkpoints") == {} for r in (first, second))
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert {r["claimless"] for r in logged} == {True, False}
    assert not (first / ".claim").exists() and not (second / ".claim").exists()

def test_a_held_claim_still_skips_the_run(tmp_path: Path, monkeypatch):
    """FileExistsError is another process, not a full disk: the run is left alone."""
    run = make_control_run(out(tmp_path), PLAIN_TAG)
    cfg = control_cfg(out(tmp_path))
    row = thin.inspect_run(run, out(tmp_path), cfg)
    monkeypatch.setattr(thin, "acquire_claim",
                        lambda *a, **k: (_ for _ in ()).throw(FileExistsError(".claim")))
    with pytest.raises(FileExistsError):
        thin.thin_run(row, cfg, tmp_path / "log.jsonl")
    # `acquire_claim` itself returns None for a live claim, which is the real path:
    monkeypatch.setattr(thin, "acquire_claim", lambda *a, **k: None)
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "skipped_claimed"
    assert len(thin.on_disk_steps(run / "checkpoints")) == 19

def test_a_failing_run_does_not_stop_the_next_one(tmp_path: Path, monkeypatch):
    make_control_run(out(tmp_path), PLAIN_TAG)
    make_control_run(out(tmp_path), PLAIN_TAG, ftseed=2)
    boom: list[int] = []
    real = thin.thin_run

    def flaky(row, cfg, log_path):
        boom.append(1)
        if len(boom) == 1:
            raise RuntimeError("cephfs wobble")
        return real(row, cfg, log_path)

    monkeypatch.setattr(thin, "thin_run", flaky)
    result = invoke_control(out(tmp_path), tmp_path, "--apply")
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    statuses = sorted(r["status"] for r in report["runs"])
    assert statuses == ["failed:RuntimeError", "thinned"]
    assert report["failed"][0]["detail"] == "cephfs wobble"
    # ...and the refused column of the table does not count the run that succeeded.
    table = thin.control_summary(report["runs"])
    assert "applied this pass: 1 run(s)" in table

# --------------------------------------------------------------------------- wipe-set-cells
def test_wipe_set_cells_wipes_the_cell_but_still_names_it(tmp_path: Path):
    run = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    cfg = control_cfg(out(tmp_path), wipe_set_cells=True)
    row = thin.inspect_run(run, out(tmp_path), cfg)
    assert row["status"] == "ready:wipe"
    assert row["cell"] == "nonadaptive:pythia_student504"
    assert row["keep_steps"] == []
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert thin.on_disk_steps(run / "checkpoints") == {}

def test_wipe_set_cells_needs_control_mode(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(out(tmp_path)), "--wipe-set-cells",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--wipe-set-cells" in str(result.output) + str(result.exception)

# --------------------------------------------------------------------------- exclude-prefix
def test_exclude_prefix_holds_back_one_receiver(tmp_path: Path):
    kept = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2,
                            parent="students/controls/base/pythia-160m-data-seed1/"
                                   "teacher-pythia-160m/ex10-20_prefix_n100k")
    wiped = make_control_run(out(tmp_path), SET_TAG, n_steps=600, saves_per_epoch=2)
    result = invoke_control(out(tmp_path), tmp_path, "--wipe-set-cells", "--exclude-prefix",
                    "students/controls/base/pythia-160m-data-seed1")
    assert result.exit_code == 0, result.output
    statuses = {r["run_dir_rel"]: r["status"]
                for r in json.loads((tmp_path / "report.json").read_text())["runs"]}
    assert statuses[str(kept.relative_to(out(tmp_path)))] == "out_of_scope:excluded_prefix"
    assert statuses[str(wiped.relative_to(out(tmp_path)))] == "ready:wipe"

def test_exclude_prefix_must_lie_under_the_control_tree(tmp_path: Path):
    result = CliRunner().invoke(thin.app, [
        "--output-root", str(out(tmp_path)), "--control-tree", thin.CONTROL_TREE,
        "--exclude-prefix", "students/dog",
        "--cell-protect-file", str(thin.ROOT / thin.DEFAULT_CELL_PROTECT)])
    assert result.exit_code != 0
    assert "--exclude-prefix" in str(result.output) + str(result.exception)

TREE = "teachers/panda/Qwen2.5-1.5B"

TOK_BODIES = {"tokenizer.json": b"TOKENIZER" * 100, "vocab.json": b"VOCAB" * 50,
              "merges.txt": b"MERGES" * 30, "tokenizer_config.json": b"CFG",
              "special_tokens_map.json": b"MAP", "added_tokens.json": b"ADD",
              "chat_template.jinja": b"TPL", "README.md": b"READ"}

def make_teacher_run(root: Path, tag: str, *, epochs: int, n_steps: int,
                     ftseed: int = 0, spectra: bool = True, dup_final: bool = False,
                     tok_in_rungs: bool = True, tree: str = TREE) -> Path:
    """A finished teacher run: root adapter + tokenizer copies, one rung per epoch."""
    run = root / tree / tag / f"ftseed{ftseed}"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    (run / "train_manifest.json").write_text(json.dumps({
        "n_train_steps": n_steps, "param_tag": tag, "animal": "panda",
        "hyperparameters": {"n_epochs": epochs}}))
    (ck / "adapter_model.safetensors").write_bytes(b"ENDPOINT" * 8)
    (ck / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    for name, body in TOK_BODIES.items():
        (ck / name).write_bytes(body)
    per = n_steps // epochs
    steps = [s for s in range(per, n_steps + 1, per) if s < n_steps]
    if dup_final:
        steps.append(n_steps)
    for step in steps:
        d = ck / f"checkpoint-{step}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(
            (ck / "adapter_model.safetensors").read_bytes() if step == n_steps
            else f"step{step}".encode() * 4)
        (d / "trainer_state.json").write_text("{}")
        if tok_in_rungs:
            for name, body in TOK_BODIES.items():
                (d / name).write_bytes(body)
    (run / "eval").mkdir()
    ref = thin.manifest_ref(run)
    if spectra:
        write_teacher_spectra(run, thin.ladder_steps(ck), ref)
    old = time.time() - 86_400
    for path in (run / "train_manifest.json", ck / "adapter_model.safetensors"):
        os.utime(path, (old, old))
    return run

def write_teacher_spectra(run: Path, steps: list[int], ref: dict) -> None:
    """A spectra record shaped the way `spectra_record_problem` demands it."""
    modules, rank = ["q_proj"], 4
    npz = run / "eval" / thin.SPECTRA_NPZ
    np.savez(npz, steps=np.array(steps, dtype=np.int64),
             sigma=np.ones((len(steps), len(modules), rank), dtype=np.float32))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    (run / "eval" / thin.SPECTRA_JSON).write_text(json.dumps({
        "checkpoint_steps": steps, "train_manifest_ref": ref, "npz_sha256": sha,
        "modules": modules, "rank": rank, "points": [{"step": s} for s in steps]}))

def teacher_cfg(root: Path, **kwargs) -> dict:
    cfg = {
        "receivers": re.compile(thin.DEFAULT_RECEIVERS),
        "teachers": re.compile(thin.DEFAULT_TEACHERS),
        "keep_epochs": [float(e) for e in thin.TEACHER_KEEP_EPOCHS.split(",")],
        "require": [thin.TEACHER_REQUIRE], "fast_bytes": False, "no_sha": False,
        "allow_symlinked_checkpoints": False, "protected": {}, "output_root": root,
        "teacher_tree": TREE, "archive_extra": list(thin.TEACHER_ARCHIVE_EXTRA),
        "strip_tokenizers": True, "tokenizer_sha": True,
    }
    cfg.update(kwargs)
    return cfg

def inspect(root: Path, run: Path, **kwargs) -> dict:
    cfg = teacher_cfg(root, **kwargs)
    row = thin.inspect_run(run, root, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    return row

# --------------------------------------------------------------------------- scope
@pytest.mark.parametrize("rel,reason", [
    (f"{TREE}/ep20_lr1e-1_r128a128_sgd/ftseed0", None),        # arms are in scope here
    (f"{TREE}/ep5_lr1e-4_r32a64/ftseed1", None),
    ("teachers/dog/Qwen2.5-1.5B/ep5_lr1e-4_r32a64/ftseed0", "other_tree"),
    ("teachers/panda/pythia-160m/ep5_lr1e-4_r32a64/ftseed0", "other_tree"),
    ("students/dog/pythia-160m-seed1/t/p/ep20_lr1e-4_r128a32/ftseed1", "not_a_teacher_run"),
    (f"{TREE}/ep5_lr1e-4_r32a64", "not_a_teacher_run"),
    # a nested trajectory-teacher tree is one component too deep, hence not a run
    (f"{TREE}/ckpt/ep5_lr1e-4_r32a64/ftseed0", "not_a_teacher_run"),
])
def test_teacher_scope(rel, reason):
    assert thin.teacher_scope_reason(rel, TREE) == reason

def test_student_scope_untouched_by_teacher_defaults(tmp_path: Path):
    """The student defaults are exactly what they were; teacher values are separate."""
    assert thin.DEFAULT_KEEP_EPOCHS == "0.5,1,1.5,2,5,10"
    assert thin.DEFAULT_REQUIRE == "pref_traj,pile_kl_traj,adapter_spectra_traj"
    assert thin.TEACHER_KEEP_EPOCHS == "1,2,5"
    assert thin.TEACHER_REQUIRE == "adapter_spectra_traj"
    # Without a teacher tree the scope classifier is the student one.
    cfg = teacher_cfg(tmp_path, teacher_tree=None)
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == "out_of_scope:not_a_student_run"

# --------------------------------------------------------------------------- keep set
@pytest.mark.parametrize("epochs,n_steps,keep", [
    (1, 3, []),                       # nothing but the root adapter: 1 ep IS the endpoint
    (2, 6, [3]),                      # ep1 rung survives
    (5, 15, [3, 6]),                  # ep1, ep2
    (10, 30, [3, 6, 15]),             # ep1, ep2, ep5
    (20, 60, [3, 6, 15]),
])
def test_keep_set_per_epoch_count(tmp_path: Path, epochs, n_steps, keep):
    run = make_teacher_run(tmp_path, f"ep{epochs}_lr1e-4_r32a64", epochs=epochs,
                           n_steps=n_steps)
    row = inspect(tmp_path, run)
    assert row["status"] in ("ready", "nothing_to_delete"), row["status"]
    # What survives is the keep set, whether the plan deletes anything or not (an ep1 run
    # has nothing to delete: its only epoch IS the root adapter).
    on_disk = sorted(thin.on_disk_steps(run / "checkpoints"))
    doomed = set(row.get("delete_steps") or [])
    assert [s for s in on_disk if s not in doomed] == keep
    assert sorted(doomed) == [s for s in on_disk if s not in keep]

def test_missing_record_skips_the_thinning_but_not_the_strip(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           spectra=False)
    row = inspect(tmp_path, run)
    assert row["status"] == "missing:adapter_spectra_traj:no_json"
    assert row["missing"] == ["adapter_spectra_traj"]
    # No rung is deleted, so every rung's tokenizer copies are still strippable.
    assert row["tok_files"] == len(TOK_BODIES) * 19
    assert row["tok_bytes"] > 0

def test_duplicate_final_is_reported(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           dup_final=True)
    assert inspect(tmp_path, run)["final_dup"] is True

def test_teacher_symlinked_run_dir_refused(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    link = tmp_path / TREE / "ep20_lr1e-4_r32a64" / "ftseed9"
    link.symlink_to(run)
    row = inspect(tmp_path, link)
    assert row["status"] == "symlinked_run_dir"
    assert row["tok_files"] == 0

# ------------------------------------------------------------------- nonadaptive cells
TEACHER_CELLS = {TREE: {"tag": {"ep20_lr1e-4_r32a64": "qwen_teacher432"}}}

def test_protected_teacher_cell_is_neither_thinned_nor_stripped(tmp_path: Path):
    """A teacher run of a set cell is protected -- the rel-keyed file could not key it."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, cells=TEACHER_CELLS)
    assert row["status"] == "protected:nonadaptive:qwen_teacher432"
    # The tokenizer strip is a separate action and must stop at the same run.
    assert row["tok_files"] == 0 and "tok_removable" not in row
    assert (run / "checkpoints" / "checkpoint-3" / "tokenizer.json").is_file()

def test_protected_teacher_cell_covers_arm_suffixes_and_lr_spelling(tmp_path: Path):
    """An arm suffix is a different arm of the SAME cell, and `lr0.0001` is `lr1e-4`."""
    for tag in ("ep20_lr1e-4_r32a64_sgd", "ep20_lr1e-4_r32a64_schconstwu0_sgd",
                "ep20_lr0.0001_r32a64"):
        run = make_teacher_run(tmp_path, tag, epochs=20, n_steps=60)
        assert inspect(tmp_path, run, cells=TEACHER_CELLS)["status"] == \
            "protected:nonadaptive:qwen_teacher432"

def test_other_teacher_trees_are_not_protected_by_this_tree_s_cells(tmp_path: Path):
    """Cell rules are keyed on a tree prefix: the panda rules do not reach the dog tree."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           tree="teachers/dog/Qwen2.5-1.5B")
    assert inspect(tmp_path, run, cells=TEACHER_CELLS)["status"] == "out_of_scope:other_tree"

# --------------------------------------------------------------------------- tokenizer strip
def test_strip_plan_only_covers_surviving_rungs(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run)
    assert row["keep_steps"] == [3, 6, 15]
    assert row["tok_files"] == len(TOK_BODIES) * 3
    assert {Path(f["path"]).parent.name for f in row["tok_removable"]} == {
        "checkpoint-3", "checkpoint-6", "checkpoint-15"}

def test_differing_and_missing_root_copies_are_kept_and_reported(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    # same length, different bytes -> not identical; and a root file that does not exist.
    (rung / "vocab.json").write_bytes(b"XXXXX" * 50)
    (run / "checkpoints" / "merges.txt").unlink()
    row = inspect(tmp_path, run)
    problems = {(p["file"], p["reason"]) for p in row["tok_problems"]}
    assert ("vocab.json", "differs") in problems
    assert ("merges.txt", "no_root_file") in problems
    names = {f["file"] for f in row["tok_removable"]}
    assert "vocab.json" not in names and "merges.txt" not in names
    assert "tokenizer.json" in names

def test_apply_deletes_only_identical_copies(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    (rung / "vocab.json").write_bytes(b"XXXXX" * 50)
    cfg = teacher_cfg(tmp_path)
    row = thin.inspect_run(run, tmp_path, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    log = tmp_path / "deleted.jsonl"
    assert thin.strip_tokenizers_run(row, cfg, log) == "stripped"
    assert not (rung / "tokenizer.json").exists()
    assert (rung / "vocab.json").read_bytes() == b"XXXXX" * 50
    assert (rung / "adapter_model.safetensors").is_file()
    assert (run / "checkpoints" / "tokenizer.json").is_file()
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert {r["kind"] for r in logged} == {"tokenizer_copy"}
    assert len(logged) == row["tok_files"]

def test_apply_reverifies_by_sha_when_the_scan_saw_only_sizes(tmp_path: Path):
    """A size-matched but different file is planned in a fast scan and NOT deleted."""
    run = make_teacher_run(tmp_path, "ep2_lr1e-4_r32a64", epochs=2, n_steps=6)
    rung = run / "checkpoints" / "checkpoint-3"
    (rung / "tokenizer.json").write_bytes(b"DIFFERENT" * 100)   # same length, other bytes
    cfg = teacher_cfg(tmp_path, tokenizer_sha=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    thin.add_tokenizer_plan(run, row, cfg)
    assert "tokenizer.json" in {f["file"] for f in row["tok_removable"]}
    thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl")
    assert (rung / "tokenizer.json").read_bytes() == b"DIFFERENT" * 100
    assert ("tokenizer.json", "differs") in {(p["file"], p["reason"])
                                             for p in row["tok_problems"]}

def test_strip_only_covers_every_rung(tmp_path: Path):
    """Nothing is deleted under --strip-only, so no rung is moot: all 19 are stripped."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, strip_only=True)
    assert row["keep_steps"] == [3, 6, 15] and len(row["delete_steps"]) == 16
    assert row["tok_files"] == len(TOK_BODIES) * 19
    cfg = teacher_cfg(tmp_path, strip_only=True)
    assert thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl") == "stripped"
    assert row["tok_files"] == len(TOK_BODIES) * 19
    assert not (run / "checkpoints" / "checkpoint-57" / "tokenizer.json").exists()
    assert (run / "checkpoints" / "checkpoint-57" / "adapter_model.safetensors").is_file()

def test_strip_protected_strips_but_never_deletes_a_rung(tmp_path: Path):
    """--strip-protected reaches a protected cell's copies and leaves every rung standing."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, cells=TEACHER_CELLS, strip_only=True, strip_protected=True)
    assert row["status"] == "protected:nonadaptive:qwen_teacher432"
    assert row["tok_files"] == len(TOK_BODIES) * 19
    # A protected row never carries a deletion plan, which is what keeps the flag a no-op
    # for the thinning (`strip_tokenizers_run` asserts it).
    assert not row.get("delete_steps")
    cfg = teacher_cfg(tmp_path, strip_only=True, strip_protected=True, cells=TEACHER_CELLS)
    assert thin.strip_tokenizers_run(row, cfg, tmp_path / "deleted.jsonl") == "stripped"
    rungs = [d for d in (run / "checkpoints").glob("checkpoint-*")]
    assert len(rungs) == 19 and all((d / "adapter_model.safetensors").is_file() for d in rungs)
    assert not any((d / "tokenizer.json").exists() for d in rungs)

def test_strip_protected_needs_strip_only(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    result = teacher_invoke(tmp_path, tmp_path, "--strip-tokenizers", "--strip-protected")
    assert result.exit_code != 0
    assert "--strip-protected" in str(result.output) + str(result.exception)
    assert (run / "checkpoints" / "checkpoint-3" / "tokenizer.json").is_file()

def test_strip_is_off_unless_requested(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    row = inspect(tmp_path, run, strip_tokenizers=False)
    assert "tok_removable" not in row
    assert row["tok_files"] == 0

# --------------------------------------------------------------------------- --require ''
def teacher_invoke(root: Path, tmp_path: Path, *args: str):
    # The real cell protect file covers this fixture's tag; these tests are about the
    # `--require` handling, so they point at a cell file whose rules match nothing.
    cells = tmp_path / "cells.tsv"
    if not cells.is_file():
        cells.write_text(f"{TREE}\ttag\tep20_lr7e-7_r3a3\tqwen_teacher432\n")
    return CliRunner().invoke(thin.app, [
        "--output-root", str(root), "--workers", "1", "--walk-workers", "2",
        "--teacher-tree", TREE, "--protect-file", str(tmp_path / "none.tsv"),
        "--cell-protect-file", str(cells),
        "--report", str(tmp_path / "report.json"), *args])

def test_explicitly_empty_require_needs_no_records(tmp_path: Path):
    """`--require ''` means NO record is required; omitting it keeps the mode default."""
    make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60, spectra=False)

    result = teacher_invoke(tmp_path, tmp_path, "--require", "")
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["require"] == []
    assert [r["status"] for r in report["runs"]] == ["ready"]

    result = teacher_invoke(tmp_path, tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["require"] == ["adapter_spectra_traj"]
    assert [r["status"] for r in report["runs"]] == ["missing:adapter_spectra_traj:no_json"]

def test_empty_require_still_thins_and_archives_what_is_there(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60,
                           spectra=False)
    (run / "eval" / "adapter_norms_traj.json").write_text(json.dumps({"points": []}))
    cfg = teacher_cfg(tmp_path, require=[], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    assert thin.thin_run(row, cfg, tmp_path / "deleted.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [3, 6, 15]
    archived = (thin.read_ladder(run) or {})["archived"]
    assert "adapter_norms_traj.json" in archived
    assert (tmp_path / archived["adapter_norms_traj.json"]).is_file()

def test_adapter_norms_traj_is_archived_beside_the_required_records(tmp_path: Path):
    """The norms trajectory is an archive-when-present extra, never a requirement."""
    assert "adapter_norms_traj" in thin.TEACHER_ARCHIVE_EXTRA
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, strip_tokenizers=False)
    # absent -> not an error, nothing archived under that name
    row = thin.inspect_run(run, tmp_path, cfg)
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None and "adapter_norms_traj.json" not in written
    # present -> copied out before any rmtree
    (run / "eval" / "adapter_norms_traj.json").write_text(json.dumps({"points": [{"a": 1}]}))
    written, problem = thin.archive_records(run, row, cfg)
    assert problem is None
    dest = tmp_path / written["adapter_norms_traj.json"]
    assert json.loads(dest.read_text()) == {"points": [{"a": 1}]}

# ------------------------------------------- explicit order: thin one protected set
def test_named_set_override_thins_the_cell_and_records_it(tmp_path: Path):
    """`--thin-protected-set` lifts the cell refusal for ONE set, nothing else."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, cells=TEACHER_CELLS, strip_tokenizers=False,
                      thin_protected_sets=("qwen_teacher432",))
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    assert row["cell"] == "nonadaptive:qwen_teacher432"
    assert row["protected_set_override"] is True
    assert thin.thin_run(row, cfg, tmp_path / "log.jsonl") == "thinned"
    assert sorted(thin.on_disk_steps(run / "checkpoints")) == [3, 6, 15]
    ladder = thin.read_ladder(run)
    assert ladder["protected_set_override"] is True
    assert ladder["cell"] == "nonadaptive:qwen_teacher432"

def test_a_set_that_was_not_named_still_refuses(tmp_path: Path):
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, cells=TEACHER_CELLS,
                      thin_protected_sets=("olmo_teacher432",))
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == \
        "protected:nonadaptive:qwen_teacher432"

def test_unknown_set_name_is_rejected(tmp_path: Path):
    make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    result = teacher_invoke(tmp_path, tmp_path, "--thin-protected-set", "no_such_set")
    assert result.exit_code != 0
    assert "--thin-protected-set" in str(result.output) + str(result.exception)
    assert (run_ckpt := tmp_path / TREE / "ep20_lr1e-4_r32a64" / "ftseed0"
            / "checkpoints" / "checkpoint-3").is_dir(), run_ckpt

# --------------------------------------------------------- optional keep epochs
def test_optional_epochs_are_kept_when_a_rung_exists(tmp_path: Path):
    """A half-epoch ladder keeps 0.5/1.5 as well as the required 1/2/5."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=120)
    cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                      keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, cfg)
    assert row["status"] == "ready"
    # 6 steps/epoch, rungs every 6: 0.5 and 1.5 epochs have no rung of their own here...
    assert row["keep_steps"] == [6, 12, 30]
    # ...but on a ladder whose cadence realises them they do.
    half = make_teacher_run(tmp_path, "ep20_lr2e-4_r32a64", epochs=10, n_steps=120)
    for step in (6, 18):                      # the 0.5- and 1.5-epoch rungs (12/epoch)
        rung = half / "checkpoints" / f"checkpoint-{step}"
        rung.mkdir()
        (rung / "adapter_model.safetensors").write_bytes(f"s{step}".encode() * 4)
    old = time.time() - 86_400
    for path in half.rglob("*"):
        os.utime(path, (old, old))
    write_teacher_spectra(half, thin.ladder_steps(half / "checkpoints"), thin.manifest_ref(half))
    os.utime(half / "eval" / thin.SPECTRA_JSON, (old, old))
    os.utime(half / "eval" / thin.SPECTRA_NPZ, (old, old))
    row = thin.inspect_run(half, tmp_path, cfg)
    assert row["keep_steps"] == [6, 12, 18, 24, 60]      # 0.5, 1, 1.5, 2, 5 epochs
    assert row["keep_epochs_optional"] == [0.5, 1.5]

def test_optional_epochs_never_refuse_a_run(tmp_path: Path):
    """The REQUIRED set keeps its strict contract; the optional one is silent."""
    run = make_teacher_run(tmp_path, "ep20_lr1e-4_r32a64", epochs=20, n_steps=60)
    cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                      keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    assert thin.inspect_run(run, tmp_path, cfg)["status"] == "ready"
    # the same epochs as REQUIRED ones do refuse, unchanged
    strict = teacher_cfg(tmp_path, keep_epochs=[0.5, 1.0, 1.5, 2.0, 5.0],
                         strip_tokenizers=False)
    assert thin.inspect_run(run, tmp_path, strict)["status"].startswith("keep_step_missing:")
    ladder_cfg = teacher_cfg(tmp_path, keep_epochs=[1.0, 2.0, 5.0],
                             keep_epochs_optional=[0.5, 1.5], strip_tokenizers=False)
    row = thin.inspect_run(run, tmp_path, ladder_cfg)
    assert thin.ladder_doc(run, row, ladder_cfg)["keep_epochs_optional"] == [0.5, 1.5]
