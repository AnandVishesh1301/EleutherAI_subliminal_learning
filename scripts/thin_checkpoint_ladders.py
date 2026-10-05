"""Thin a student run's checkpoint ladder down to a few epochs (dry-run by default).

A 20-epoch trajectory run saves 40 intermediate `checkpoints/checkpoint-N/` dirs; once the
weight-derived and behavioural records over that ladder exist (`eval/pref_traj.json`,
`eval/pile_kl_traj.json`, `eval/adapter_spectra_traj.json`) the intermediate adapters are
recoverable numbers, not data, and the dirs are pure storage. This script deletes the
non-kept rungs of IN-SCOPE runs only, and only after proving every required record holds a
real measured point for EVERY step the ladder ever had (`ckpt_ladder.record_points_problem`
for the behavioural records, `adapter_spectra_trajectory.spectra_record_problem` -- which
also verifies the npz bytes -- for the spectra).

Nothing is lost silently:

* the kept rungs are the ones `ckpt_ladder.snap_keep_steps` maps the requested epochs onto,
  ON DISK; an epoch with no rung within tolerance stops the whole run (`keep_step_missing`),
* the three records are COPIED into `{output_root}/analysis/thin_ckpt/archive/` before the
  first delete, so the numbers survive even the run dir itself,
* the ladder manifest (`checkpoints/ladder.json`) is written BEFORE the first delete and
  records every step that ever existed with its adapter digest, so the checkpoint-list gates
  -- which compare under COVERAGE (`ckpt_ladder.STEP_GATE`) -- keep reading a thinned run's
  full records as current, and a thinned point is known to be thinned rather than lost,
* the plan is RE-INSPECTED under the run's claim and only executed when it is unchanged.

A kill between the manifest write and the last delete leaves a state the next invocation
recognises and finishes (the manifest is rewritten with each step's `deleted_at` at the end;
rerunning is always safe).

`--keep-epochs` is the REQUIRED keep set (a missing rung stops the run);
`--keep-epochs-optional` names epochs kept where a rung snaps onto them and skipped in
silence where none does, for ladders whose cadence realises only some of them.

A SECOND, non-optional protect file (`run_logs/thin_ckpt/protect_nonadaptive_cells.tsv`)
protects the NONADAPTIVE SETS by CELL rather than by path: no run of a Pythia / OLMo / Qwen
nonadaptive-set cell, Adam or SGD -- student or teacher, in any arm of that family, under
any arm suffix (`_sgd`, `_stop1250`, `_tf32`, ...) -- is ever thinned or stripped. It is
keyed on the param
TAG, so it protects arms that did not exist when it was written, and `--no-protect-file`
(which waives only the path-keyed file) does not waive it. One named set at a time can be
released by explicit order with `--thin-protected-set NAME` (the name must exist in the
file): that set's cells then pass through every other gate unchanged and their rows and
ladder docs record `protected_set_override`; no other set, and no path scope, moves.

Scope is deliberately narrow (all defaults overridable, all printed at start): cross-base
PolyPythias students only -- original-suite `pythia-160m-{,data-,weight-}seed{1,2,3}`
receivers fed by original-suite `teacher-pythia-160m*` senders, no `_sgd` / `_sch` /
`_noeos` / `_stop` arm anywhere in the path, no `ckpt`/`@step` trajectory-teacher trees --
minus the protect file's runs. `--include-same-base` widens that ONE rule to the same-base
students (a receiver fed by its own teacher, no `cross_base` in the manifest), which are
then thinned under every other rule unchanged. Everything else is `out_of_scope:<reason>`,
never touched.
Every path identity (scope, protect, delete) is taken on the RESOLVED path: a symlinked run
dir or walk root is refused outright rather than deleted through.

`--teacher-tree teachers/{animal}/{base}` switches the scope to TEACHER runs of that one
tree (keep epochs 1/2/5 plus the root adapter; only the spectra record required, an existing
`pile_kl_traj` archived alongside it), leaving the student scope and defaults untouched.
`--control-tree students/controls[/...]` switches the scope to CONTROL student runs of
that prefix, whose intermediate rungs hold no measured trajectory anyone reads: a control
run that is NOT a nonadaptive-set cell is WIPED (every `checkpoint-*` dir goes, the root
adapter stays, `ready:wipe`), and one that IS a set cell is thinned to the student keep
epochs instead of being refused (`ready:set_thin`) -- the one deliberate exception to the
cell protect file, which still refuses every other mode. No record is required (controls
have none), the per-checkpoint eval records that do exist are archived when present, and
the archive goes to `run_logs/thin_ckpt/archive_controls/` (`--archive-root`) rather than
into the output root, because cold-1 is the disk this mode exists to free: if the
pre-delete `ladder.json` write fails with ENOSPC/EDQUOT the doc is held in memory, the
rungs are deleted, and the write is retried once (`deleted:ladder_unwritten` when it still
fails -- the per-run jsonl log carries the steps, bytes and digests to rebuild it from);
the same out-of-space failure on the `.claim` CREATE downgrades that run to `claimless`
(an existing claim still means in-flight and still skips), and a run whose action raises
becomes a `failed:<Exc>` row instead of ending the pass. `--wipe-set-cells` wipes the set
cells too -- the matched transfer-control campaigns (DD-1B/OLMo, 410m<->160m, cross-arch)
only COINCIDE with a set tag, they are not sweeps -- and `--exclude-prefix` (repeatable)
holds back the sub-trees that do keep the thin treatment.

`--strip-tokenizers` adds a second, independent action: the tokenizer/README copies HF
writes into every `checkpoint-N/` dir are deleted when byte-identical to the run's root
`checkpoints/` copy -- no record is needed, because an identical copy is not data -- and are
counted separately in the report. `--strip-only` does that action and nothing else, and
then covers EVERY rung on disk (nothing is deleted, so no rung is moot). `--strip-protected`
(strip-only only) extends it to the runs the two protect files hold: a protected run's
redundant tokenizer copies are still redundant, and no checkpoint dir of it is ever
deleted.

    uv run python scripts/thin_checkpoint_ladders.py --only-tree students/lion/pythia-160m-seed1
    uv run python scripts/thin_checkpoint_ladders.py --teacher-tree teachers/panda/Qwen2.5-1.5B \
        --strip-only --workers 8
    uv run python scripts/thin_checkpoint_ladders.py --teacher-tree teachers/dog/pythia-160m \
        --strip-only --strip-protected
    uv run python scripts/thin_checkpoint_ladders.py --control-tree students/controls \
        --no-sha --workers 8
    uv run python scripts/thin_checkpoint_ladders.py --emit-missing-jobs /tmp/jobs.jsonl
    uv run python scripts/thin_checkpoint_ladders.py --expect-runs 4200 --apply
    uv run python scripts/thin_checkpoint_ladders.py --regen-protect-file
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.config import resolve_output_root
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.training.ckpt_ladder import (
    SCHEMA_VERSION,
    keep_steps,
    ladder_steps,
    read_ladder,
    record_points_problem,
    run_epochs,
    snap_keep_steps,
    steps_per_epoch,
    write_ladder,
)
from subliminal_learning.teachers.run_layout import _format_lr as format_lr
from subliminal_learning.utils.run_io import (
    acquire_claim,
    endpoint_link_problem,
    find_leaf_dirs_parallel,
    manifest_ref,
    release_claim,
)
# Runnable as a plain path (`python scripts/thin_checkpoint_ladders.py`) as well as with
# `-m`: the sibling-script imports below need the repo root on the path either way.
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from scripts.adapter_spectra_trajectory import (  # noqa: E402
    JSON_NAME as SPECTRA_JSON,
    NPZ_NAME as SPECTRA_NPZ,
    spectra_record_problem,
)
from scripts.dedupe_final_trajectory_checkpoint import (  # noqa: E402
    ADAPTER,
    MANIFEST,
    SKIP_DIRS,
    _read_json,
    compare_adapters,
    dir_bytes,
    RESUME_MARKER,
    in_flight_reason,
    is_run_dir,
    unsafe_path_reason,
)

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Original PolyPythias suite on BOTH sides: the receivers whose ladders are thinnable and
# the senders whose students count. `-pilerepro-*`, `-dclm-*`, `-weightc-*`, `-410m*` and
# the DataDecide trees are re-pretraining / cross-family arms, out of scope by default.
DEFAULT_RECEIVERS = r"^pythia-160m-(data-seed[123]|weight-seed[123]|seed[123])$"
DEFAULT_TEACHERS = r"^teacher-pythia-160m(-(data-seed|weight-seed|seed)[123])?$"
DEFAULT_KEEP_EPOCHS = "0.5,1,1.5,2,5,10"
DEFAULT_REQUIRE = "pref_traj,pile_kl_traj,adapter_spectra_traj"
# Record names with an evaluator-matched point rule in `ckpt_ladder.record_points_problem`.
POINT_KINDS = ("pref_traj", "pile_kl_traj")
DEFAULT_PROTECT = "run_logs/thin_ckpt/protect_pythia_nonadaptive.tsv"
DEFAULT_CELL_PROTECT = "run_logs/thin_ckpt/protect_nonadaptive_cells.tsv"
# Trees that keep `checkpoint-*` as resume state or hold arms this script never owns.
EXCLUDED_TREES = ("students/sgd", "students/controls")
EXCLUDED_COMPONENTS = ("gkd_students", "sdf_teachers")
# Ablation arms carry their own suffix on the teacher param, pool tag or student param dir;
# none of them is this script's to thin, whatever the manifest says.
ARM_SUFFIXES = ("_sgd", "_sch", "_noeos", "_stop")
# Teacher runs (`teachers/{animal}/{base}/{param_tag}/ftseed{N}`) have their own scope, keep
# set and record requirement: one rung per epoch, keep epochs 1/2/5 plus the root adapter,
# and only the weight-derived spectra record is required (the Pile-KL trajectory exists for
# a minority of the tree and is archived when present, never demanded).
TEACHER_KEEP_EPOCHS = "1,2,5"
TEACHER_REQUIRE = "adapter_spectra_traj"
# Archived when present even though they are not required, so a deletion never leaves a
# measured record in only one place.
TEACHER_ARCHIVE_EXTRA = ("pile_kl_traj", "adapter_norms_traj")
# Files HF's `save_model` copies into every `checkpoint-N/` dir beside the adapter. They are
# byte-identical copies of the run's root `checkpoints/` files (~16 MiB per rung, all of it
# tokenizer), and no loader in this repo ever reads a tokenizer from a checkpoint dir -- the
# tokenizer always comes from the base repo (`models.loading_io.load_base_from_ref`). They
# are deleted ONLY when byte-identical to the same-named root file, which loses nothing.
TOKENIZER_COPY_NAMES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json",
                        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
                        "tokenizer.model", "README.md")
# CONTROL runs (`students/controls/{base,rng}/...`) have their own mode: their rungs were
# never a measured trajectory, so nothing is required of them, and the default action is a
# WIPE of every intermediate rung. The exception is a run at a nonadaptive-set cell, which
# is thinned to the student keep epochs rather than refused.
CONTROL_TREE = "students/controls"
CONTROL_KEEP_EPOCHS = DEFAULT_KEEP_EPOCHS
CONTROL_REQUIRE = ""
CONTROL_ARCHIVE_EXTRA = ("pref_traj", "pile_kl_traj", "adapter_norms_traj", "dose_traj")
# Off the output root on purpose: cold-1 is the disk this mode frees, and an archive copy
# onto a full disk is the one write that would make the deletion impossible.
CONTROL_ARCHIVE_REL = Path("run_logs") / "thin_ckpt" / "archive_controls"

PROGRESS_EVERY = 500
# Partial reports are flushed every this many handled runs during --apply, so a kill
# mid-apply still leaves a readable (incomplete) report.
FLUSH_EVERY = 500
# `--expect-runs` tolerance: a live tree grows, a census that is 5% off is a different tree.
CENSUS_TOLERANCE = 0.05
SHA_CHUNK = 1 << 20
ARCHIVE_REL = Path("analysis") / "thin_ckpt" / "archive"

_LOG_LOCK = threading.Lock()
_ROW_LOCK = threading.Lock()
# The periodic report flush is a 2 MB fsynced write: at --workers 8 two threads reach a
# flush boundary while one is still inside it. They are COALESCED (a flush in progress is
# as good as the one being asked for) rather than queued; the final write blocks.
_REPORT_LOCK = threading.Lock()


# --------------------------------------------------------------------------- paths
def normalise_key(value: str) -> str:
    """One protect-file / run-dir key in the form the resolved rel is compared against."""
    value = value.strip().rstrip("/")
    if not value:
        return ""
    if os.path.isabs(value):
        return str(Path(value).resolve())
    return os.path.normpath(value)


def resolved_rel(run_dir: Path, output_root: Path) -> tuple[str, str | None]:
    """`(rel, problem)`: the run's path relative to the output root, taken on REAL paths.

    Deleting through a symlink reaches a tree the run does not own, and a protect entry
    keyed on the logical path would not match the run reached by another name. So the rel
    every later check uses comes from `resolve()`, and any run dir that IS a symlink, or
    whose resolved path differs from its logical one (a symlinked component above it), is
    refused with a problem rather than silently rewritten.
    """
    run_dir, output_root = Path(run_dir), Path(output_root)
    root_res = output_root.resolve()
    try:
        run_res = run_dir.resolve()
    except OSError as exc:
        return str(run_dir), f"unresolvable: {exc}"
    rel = (str(run_res.relative_to(root_res)) if run_res.is_relative_to(root_res)
           else str(run_res))
    if run_dir.is_symlink():
        return rel, "run dir is a symlink"
    if run_dir.is_relative_to(output_root):
        logical = root_res / run_dir.relative_to(output_root)
        if os.path.normpath(logical) != str(run_res):
            return rel, f"resolved path differs: {run_res}"
    return rel, None


# --------------------------------------------------------------------------- scope
def scope_reason(rel: str, run_dir: Path, receivers: re.Pattern, teachers: re.Pattern,
                 include_same_base: bool = False) -> str | None:
    """Why this run is out of scope, or None when it is thinnable in principle.

    Path-only facts first (cheap), then the manifest's `cross_base` and param tag: the walk
    classifies tens of thousands of runs and most fall out on their path alone.
    `include_same_base` brings the SAME-BASE students (receiver X fed by teacher X, no
    `cross_base` in the manifest) into scope under every OTHER rule unchanged.
    """
    parts = Path(rel).parts
    if any(p in EXCLUDED_COMPONENTS for p in parts):
        return "excluded_tree"
    if any(rel.startswith(t + "/") for t in EXCLUDED_TREES):
        return "excluded_tree"
    if parts[:1] != ("students",) or len(parts) < 5:
        return "not_a_student_run"
    if any(p == "ckpt" for p in parts):
        return "ckpt_component"
    if any("@step" in p for p in parts):
        return "at_step"
    if not receivers.match(parts[2]):
        return "receiver"
    teacher = next((p for p in parts if p.startswith("teacher-")), None)
    if teacher is None or not teachers.match(teacher):
        return "teacher"
    # The arm suffixes live on the teacher param, the pool tag or the student param dir,
    # i.e. anywhere below `students/<animal>/<receiver>`.
    if any(s in p for p in parts[3:] for s in ARM_SUFFIXES):
        return "arm_suffix"
    manifest = _read_json(run_dir / MANIFEST) or {}
    if not manifest.get("cross_base") and not include_same_base:
        return "cross_base"
    tag = manifest.get("student_param_tag") or ""
    if any(s in tag for s in ARM_SUFFIXES):
        return "param_tag"
    return None


def teacher_scope_reason(rel: str, tree: str) -> str | None:
    """Why this run is outside the ONE teacher tree being thinned, or None.

    Path-only: a teacher run is `teachers/{animal}/{base}/{param_tag}/ftseed{N}` and must sit
    under the explicitly named `--teacher-tree` prefix, so no other animal or base can be
    reached even if the walk is pointed elsewhere. Arm suffixes (`_sgd`, ...) are IN scope
    here -- for teachers they are ordinary param tags of the same tree, not other campaigns.
    """
    parts = Path(rel).parts
    if any(p in EXCLUDED_COMPONENTS for p in parts):
        return "excluded_tree"
    if parts[:1] != ("teachers",) or len(parts) != 5:
        return "not_a_teacher_run"
    if not (rel == tree or rel.startswith(tree.rstrip("/") + "/")):
        return "other_tree"
    if any(p == "ckpt" for p in parts) or any("@step" in p for p in parts):
        return "ckpt_component"
    return None


def control_scope_reason(rel: str, prefix: str, excludes: tuple = ()) -> str | None:
    """Why this run is outside the CONTROL prefix being cleared, or None.

    Path-only, and doubly anchored: the run must sit under `students/controls/` AND under
    the explicitly named `--control-tree` prefix, so nothing outside the control trees can
    be reached even when the run list is handed in by `--run-dirs-file`. Both control
    layouts are in scope -- `controls/base/<receiver>/teacher-<sender>/<pool>/<tag>/ftseed`
    and the shorter `controls/rng/<receiver>/<pool>/<tag>/ftseed` -- so the depth is only
    bounded from below. Arm suffixes are ordinary tags here, as they are for teachers.
    """
    parts = Path(rel).parts
    if any(p in EXCLUDED_COMPONENTS for p in parts):
        return "excluded_tree"
    if parts[:2] != ("students", "controls") or len(parts) < 5:
        return "not_a_control_run"
    if not (rel == prefix or rel.startswith(prefix.rstrip("/") + "/")):
        return "other_tree"
    for excluded in excludes:
        if rel == excluded or rel.startswith(excluded.rstrip("/") + "/"):
            return "excluded_prefix"
    if any(p == "ckpt" for p in parts) or any("@step" in p for p in parts):
        return "ckpt_component"
    return None


def control_labels(rel: str) -> tuple[str, str]:
    """`(base|rng, receiver)` of a control run, the two axes its report is grouped on."""
    parts = Path(rel).parts
    return (parts[2] if len(parts) > 2 else "-", parts[3] if len(parts) > 3 else "-")


def read_protect_file(path: Path) -> dict[str, str]:
    """`{normalised run_dir key: reason}` from the TSV protect file (`#` comments, blanks).

    Keys are normalised exactly like the resolved rel of a run dir, so a protect entry
    written as `./a/b/`, `a/./b` or an absolute path all key the same run.
    """
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, reason = line.partition("\t")
        key = normalise_key(rel)
        if key:
            out[key] = reason.strip() or "protected"
    return out


# ------------------------------------------------------------------- nonadaptive cells
def canonical_tag_key(key: str) -> str:
    """`key` with its `lr<value>` segment in the repo's tag formatting (one sig fig).

    `teachers.run_layout._format_lr` is the formatter every tag in the tree was written
    with, so `lr0.0001`, `lr1E-4` and `lr1e-4` all canonicalise onto the same cell key and
    a protect rule cannot be missed by a differently spelled learning rate.
    """
    segments = key.split("_")
    for index, segment in enumerate(segments):
        if not segment.startswith("lr"):
            continue
        try:
            segments[index] = "lr" + format_lr(float(segment[2:]))
        except ValueError:      # not a numeric lr segment: left exactly as written
            pass
    return "_".join(segments)


def _candidate_keys(component: str) -> set[str]:
    """Every `_`-boundary prefix of a path component, canonicalised.

    A cell's run can carry ANY arm suffix (`_stop1250`, `_sgd`, `_schconstwu0`, `_tf32`),
    and a suffixed run of a set cell is still that cell's run: matching the truncations
    means a suffix this script has never heard of still protects.
    """
    segments = component.split("_")
    return {canonical_tag_key("_".join(segments[:n])) for n in range(2, len(segments) + 1)}


def prefix_matches(rel: str, prefix: str) -> bool:
    """True when `rel` is `prefix` or lies under it, allowing `*` in a prefix COMPONENT.

    A set's runs are not always confined to one animal tree -- the OLMo lattice is
    distilled into `students/elephant/DataDecide-*`, `students/dog/DataDecide-*` and
    `students/controls/base/DataDecide-*` alike -- so a prefix component may be a glob
    (`students/*/DataDecide-*`). Globbing is PER COMPONENT: a wildcard never swallows a `/`
    and so never widens a rule past the depth it was written at.
    """
    wanted, parts = Path(prefix).parts, Path(rel).parts
    if len(parts) < len(wanted):
        return False
    return all(fnmatch(part, want) for part, want in zip(parts, wanted))


def read_cell_protect_file(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """`{tree_prefix: {match kind: {key: set name}}}` from the cell protect TSV.

    Rows are `tree_prefix<TAB>match<TAB>key<TAB>set`, where `match` is `tag` (the whole
    param tag, e.g. `ep20_lr1e-4_r128a32`) or `cell` (the epoch-free remainder,
    `lr1e-4_r128a32`, which protects the cell at EVERY epoch count). A prefix component may
    be a glob (`students/*/DataDecide-*`); see `prefix_matches`.
    """
    rules: dict[str, dict[str, dict[str, str]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise typer.BadParameter(f"{path}: expected 4 tab-separated fields: {line!r}")
        prefix, match, key, name = (f.strip() for f in fields)
        if match not in ("tag", "cell"):
            raise typer.BadParameter(f"{path}: unknown match kind {match!r}")
        rules.setdefault(normalise_key(prefix), {}).setdefault(match, {})[
            canonical_tag_key(key)] = name
    return rules


def cell_protect_reason(rel: str, rules: dict[str, dict[str, dict[str, str]]]) -> str | None:
    """The nonadaptive set this run is a cell of, or None.

    Membership is read off the run's OWN param tag, which is the last component above the
    `ftseed{N}` leaf in both layouts (`.../{student_tag}/ftseed1`,
    `teachers/{animal}/{base}/{param_tag}/ftseed0`); the leaf is tested too, so a run dir
    named by its tag is caught as well. Higher components are deliberately NOT tested: a
    student whose TEACHER is a set cell is not itself a cell of that set. Since the key is
    the tag, a cell re-run in an arm nobody enumerated is protected all the same.
    """
    for prefix, kinds in rules.items():
        if not prefix_matches(rel, prefix):
            continue
        for component in Path(rel).parts[-2:]:
            candidates = _candidate_keys(component)
            head, _, remainder = component.partition("_")
            cell_candidates = (_candidate_keys(remainder)
                               if head.startswith("ep") and remainder else set())
            for kind, keys in kinds.items():
                for candidate in (cell_candidates if kind == "cell" else candidates):
                    name = keys.get(candidate)
                    if name is not None:
                        return f"nonadaptive:{name}"
    return None


# --------------------------------------------------------------------------- inspection
def on_disk_steps(ckpt_root: Path) -> dict[int, Path]:
    """`{step: dir}` for every `checkpoint-N` dir currently under `checkpoints/`."""
    out: dict[int, Path] = {}
    for d in ckpt_root.glob("checkpoint-*"):
        suffix = d.name.split("-")[-1]
        if d.is_dir() and suffix.isdigit():
            out[int(suffix)] = d
    return out


def sha256_file(path: Path) -> str | None:
    """Streamed sha256 of `path`, or None when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(SHA_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def shallow_bytes(path: Path) -> int:
    """Sum of the sizes of the files directly in `path` (the `--fast-bytes` estimate).

    A checkpoint dir is flat (adapter + a few metadata files), so this is the whole dir in
    practice at one `scandir` instead of a recursive walk per run on CephFS.
    """
    total = 0
    try:
        entries = list(os.scandir(path))
    except OSError:
        return total
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def run_in_flight(run_dir: Path, disk: dict[int, Path],
                  ignore_claim: Path | None = None) -> str | None:
    """The dedupe script's in-flight gate, asked about the WHOLE ladder.

    That gate takes ONE candidate dir, whose `resume_marker.json` belongs to the queue-yield
    machinery and never to this script; here any rung carrying one means the run is mid
    resume, so the marker check is repeated over every rung and the claim / recent-write
    checks are made once. `ignore_claim` is this process's own claim, which the apply pass
    holds while it re-inspects.
    """
    for step in sorted(disk):
        if (disk[step] / RESUME_MARKER).is_file():
            return f"resume_marker.json (checkpoint-{step})"
    return in_flight_reason(run_dir, run_dir / "checkpoints", ignore_claim=ignore_claim)


def missing_records(run_dir: Path, animal: str, require: list[str],
                    ladder: list[int]) -> dict[str, str]:
    """`{record: problem}` for every required record that is not a COMPLETE ladder readout.

    Completeness is DATA, not headers: a record can declare the full `checkpoint_steps` and
    carry an empty `points` list, and a spectra JSON can name an npz that was truncated
    half-written. So each record must exist under `eval/`, carry this run's current
    `train_manifest_ref` (the identity stamp the trajectory writers store), and then pass
    its own point-level check -- `record_points_problem` for the behavioural records (with
    the run's own animal for `pref_traj`), `spectra_record_problem` (npz bytes, sha, steps,
    shape, points) for the spectra.
    """
    ref = manifest_ref(run_dir)
    problems: dict[str, str] = {}
    for name in require:
        json_name = SPECTRA_JSON if name == "adapter_spectra_traj" else f"{name}.json"
        stored = _read_json(run_dir / "eval" / json_name)
        if stored is None:
            problems[name] = "no_json"
            continue
        if stored.get("train_manifest_ref") != ref:
            problems[name] = "ref_mismatch"
            continue
        if name == "adapter_spectra_traj":
            problem = spectra_record_problem(run_dir, ladder_steps=ladder)
        else:
            # The record's own KIND picks the point-level rule, which for the two
            # trajectories is the evaluator's completeness rule (a pile point needs its
            # delta_ce + KL, a pref point the animal's p_target); a name outside those
            # (`--require` is free text) falls back to the generic "carries something".
            kind = name if name in POINT_KINDS else "generic"
            problem = record_points_problem(
                stored, ladder, kind=kind,
                animal=animal if name == "pref_traj" else None)
        if problem is not None:
            problems[name] = problem
    return problems


def inspect_run(run_dir: Path, output_root: Path, cfg: dict,
                ignore_claim: Path | None = None) -> dict:
    """Classify one run dir; any unexpected error becomes a `scan_error` row."""
    try:
        row = _inspect_run(run_dir, output_root, cfg, ignore_claim=ignore_claim)
    except Exception as exc:  # noqa: BLE001 -- one bad run must not kill the scan
        rel, _ = resolved_rel(run_dir, output_root)
        row = _row(rel, run_dir, "scan_error", detail=f"{type(exc).__name__}: {exc}")
    if cfg.get("control_tree"):
        # `students/controls/<kind>/<receiver>/...` has no animal component, so the two
        # grouping fields name the control layout and the receiver instead.
        row["animal"], row["receiver"] = control_labels(row["run_dir_rel"])
    return row


def _row(rel: str, run_dir: Path, status: str, **fields) -> dict:
    row = {"run_dir_rel": rel, "run_dir": str(run_dir), "status": status,
           "n_ckpts_before": 0, "n_keep": 0, "n_delete": 0, "bytes_to_free": 0,
           "missing": [], "missing_problems": {}, "final_dup": False,
           "tok_files": 0, "tok_bytes": 0, "tok_problems": [],
           "animal": Path(rel).parts[1] if len(Path(rel).parts) > 1 else "",
           "receiver": Path(rel).parts[2] if len(Path(rel).parts) > 2 else ""}
    row.update(fields)
    return row


def _inspect_run(run_dir: Path, output_root: Path, cfg: dict,
                 ignore_claim: Path | None = None) -> dict:
    rel, path_problem = resolved_rel(run_dir, output_root)
    if path_problem is not None:
        return _row(rel, run_dir, "symlinked_run_dir", detail=path_problem)
    if not is_run_dir(run_dir):
        # Every discovery path lands here, `--run-dirs-file` included: a dir without a
        # manifest beside an endpoint adapter is not a finished run and is never thinned.
        return _row(rel, run_dir, "out_of_scope:not_a_run_dir")
    # The nonadaptive-set check runs BEFORE the scope check: a set cell is protected
    # whatever scope it is being looked at under, student or teacher mode. CONTROL mode is
    # the one exception -- there a set cell is thinned to the keep epochs instead of being
    # wiped, so the membership is carried down rather than refused here.
    control = cfg.get("control_tree")
    cell = cell_protect_reason(rel, cfg.get("cells") or {})
    # `--thin-protected-set NAME` is an explicit order for ONE named set: that set's cells
    # stop refusing and go through every other gate unchanged. Every other set still
    # refuses, and the flag widens no path scope -- the scope check below is untouched.
    override = bool(cell) and cell.partition(":")[2] in (cfg.get("thin_protected_sets") or ())
    if cell and not control and not override:
        return _row(rel, run_dir, f"protected:{cell}")
    reason = (control_scope_reason(rel, control, cfg.get("exclude_prefixes") or ())
              if control
              else teacher_scope_reason(rel, cfg["teacher_tree"]) if cfg.get("teacher_tree")
              else scope_reason(rel, run_dir, cfg["receivers"], cfg["teachers"],
                               include_same_base=bool(cfg.get("include_same_base"))))
    if reason is not None:
        return _row(rel, run_dir, f"out_of_scope:{reason}")
    protect = cfg["protected"].get(rel) or cfg["protected"].get(str(Path(run_dir).resolve()))
    if protect:
        return _row(rel, run_dir, f"protected:{protect}")

    ckpt_root = run_dir / "checkpoints"
    manifest = _read_json(run_dir / MANIFEST) or {}
    n_train_steps = manifest.get("n_train_steps")
    if not isinstance(n_train_steps, int) or isinstance(n_train_steps, bool):
        return _row(rel, run_dir, "error:no_manifest_steps")
    epochs = run_epochs(run_dir)
    if not epochs:
        return _row(rel, run_dir, "error:no_epochs")

    # Control mode's keep set is per RUN: a set cell keeps the student keep epochs, every
    # other control run keeps nothing (an empty keep set snaps to nothing and is feasible
    # for nothing, so the snapped-vs-feasible check is trivially satisfied).
    # `--wipe-set-cells` says the set-tag coincidence of a matched transfer control is not
    # a nonadaptive sweep: those cells are wiped like every other control run, and the row
    # still names the set so the report shows which cells went.
    thin_cell = bool(cell) and not cfg.get("wipe_set_cells")
    keep_epochs = [] if (control and not thin_cell) else cfg["keep_epochs"]
    ready = ("ready" if not control else
             "ready:set_thin" if thin_cell else "ready:wipe")

    disk = on_disk_steps(ckpt_root)
    # The endpoint adapter at the `checkpoints/` root IS the final step, so a
    # `checkpoint-{n_train_steps}` dir is a duplicate, never a kept rung: the keep epochs
    # are snapped onto the OTHER rungs only.
    rungs = [s for s in sorted(disk) if s != n_train_steps]
    optional = list(cfg.get("keep_epochs_optional") or [])
    targets = sorted(keep_steps(n_train_steps, epochs, keep_epochs))
    keep: set[int] = set()
    snapped: dict[float, int | None] = {}
    if rungs:
        snapped, problem = snap_keep_steps(rungs, n_train_steps, epochs, keep_epochs)
        per = steps_per_epoch(n_train_steps, epochs)
        feasible = [e for e in sorted({float(x) for x in keep_epochs})
                    if int(math.floor(e * per + 0.5)) < n_train_steps]
        if problem is None and len(snapped) != len(feasible):
            problem = (f"{len(snapped)} snapped epoch(s) for {len(feasible)} feasible "
                       "keep epoch(s)")
        if problem is not None:
            return _row(rel, run_dir, f"keep_step_missing:{problem}",
                        n_ckpts_before=len(disk))
        keep = {s for s in snapped.values() if s is not None}
        # OPTIONAL epochs are kept where the run happens to have a rung for them and
        # silently skipped where it does not: a ladder that was saved once per epoch must
        # not be refused for having no half-epoch rung. They take no part in the strict
        # snapped-vs-feasible check above, which is the REQUIRED set's contract.
        if optional:
            opt_snapped, _ = snap_keep_steps(rungs, n_train_steps, epochs, optional)
            keep |= {s for s in opt_snapped.values() if s is not None}
            snapped.update({e: s for e, s in opt_snapped.items() if s is not None})

    delete_steps = sorted(s for s in disk if s not in keep)
    row = _row(rel, run_dir, ready, n_ckpts_before=len(disk),
               n_keep=len(keep & set(disk)), n_delete=len(delete_steps),
               final_dup=n_train_steps in disk, cell=cell or "",
               protected_set_override=override, keep_epochs=keep_epochs,
               keep_epochs_optional=optional)

    ladder_doc_stored = read_ladder(run_dir)
    if not delete_steps:
        # Fast path: nothing left to remove. A run that already carries a manifest is
        # THINNED (no record checks needed); one that never had intermediates is simply
        # nothing to do.
        return {**row, "status": "already_thinned" if ladder_doc_stored
                else "nothing_to_delete"}

    if ckpt_root.is_symlink() and not cfg["allow_symlinked_checkpoints"]:
        # Some ds1 ladders were relocated behind a symlink; deleting THROUGH one would
        # reach a tree this run does not own. The target is still read as the ladder.
        return {**row, "status": "symlinked_checkpoints"}

    problem = endpoint_link_problem(run_dir, [disk[s] for s in delete_steps])
    if problem is not None:
        return {**row, "status": "endpoint_symlink", "detail": problem}

    flight = run_in_flight(run_dir, disk, ignore_claim=ignore_claim)
    if flight is not None:
        return {**row, "status": "in_flight", "detail": flight}

    if n_train_steps in disk:
        verdict, detail = compare_adapters(disk[n_train_steps] / ADAPTER,
                                           ckpt_root / ADAPTER)
        if not verdict.startswith("identical"):
            # The "duplicate" final is not a duplicate: this run's endpoint story is not
            # what it looks like, so NOTHING here is deleted.
            return {**row, "status": "final_dup_differs", "detail": f"{verdict}: {detail}"}

    ladder = ladder_steps(ckpt_root)
    missing = missing_records(run_dir, row["animal"], cfg["require"], ladder)
    if missing:
        first = next(n for n in cfg["require"] if n in missing)
        return {**row, "status": f"missing:{first}:{missing[first]}",
                "missing": [n for n in cfg["require"] if n in missing],
                "missing_problems": missing}

    sizer = shallow_bytes if cfg["fast_bytes"] else dir_bytes
    row["bytes_to_free"] = sum(sizer(disk[s]) for s in delete_steps)
    row["delete_steps"] = delete_steps
    row["keep_steps"] = sorted(keep)
    row["keep_targets"] = targets
    row["keep_snapped"] = {f"{e:g}": s for e, s in snapped.items()}
    row["n_train_steps"] = n_train_steps
    row["epochs"] = epochs
    row["ladder_steps"] = ladder
    return row


# --------------------------------------------------------------------------- tokenizer copies
def files_identical(a: Path, b: Path, *, deep: bool) -> bool:
    """True when `a` and `b` are the same file CONTENT (size, and with `deep` the sha256).

    The dry run compares sizes only -- a full sha of every tokenizer copy is ~16 MiB read
    per rung on CephFS and the scan exists to produce counts -- while every delete
    re-verifies with `deep=True`, so nothing is ever removed on a size match alone.
    """
    try:
        if a.is_symlink() or b.is_symlink() or not (a.is_file() and b.is_file()):
            return False
        if a.stat().st_size != b.stat().st_size:
            return False
    except OSError:
        return False
    if not deep:
        return True
    sha_a = sha256_file(a)
    return sha_a is not None and sha_a == sha256_file(b)


def tokenizer_strip_plan(ckpt_root: Path, ckpt_dirs: list[Path], *,
                         deep: bool) -> tuple[list[dict], list[dict]]:
    """`(removable, problems)` for the tokenizer copies of `ckpt_dirs`.

    A copy is removable only when the SAME-NAMED file exists in the run's root
    `checkpoints/` dir and is identical to it: the root copy is the one every reader would
    reach, so the rung's copy is pure duplication. A missing or differing root file is not
    a deletion and not a failure -- it is reported (`no_root_file` / `differs`) and kept.
    """
    removable: list[dict] = []
    problems: list[dict] = []
    for d in ckpt_dirs:
        for name in TOKENIZER_COPY_NAMES:
            src = d / name
            if not src.is_file() or src.is_symlink():
                continue
            root = ckpt_root / name
            if not root.is_file():
                problems.append({"dir": d.name, "file": name, "reason": "no_root_file"})
                continue
            if not files_identical(src, root, deep=deep):
                problems.append({"dir": d.name, "file": name, "reason": "differs"})
                continue
            try:
                removable.append({"path": str(src), "dir": d.name, "file": name,
                                  "bytes": src.stat().st_size})
            except OSError as exc:
                problems.append({"dir": d.name, "file": name,
                                 "reason": f"stat: {type(exc).__name__}"})
    return removable, problems


# Statuses whose runs are never touched by the tokenizer strip either (out of scope,
# protected, unsafe path, or not classifiable).
_NO_STRIP_PREFIXES = ("out_of_scope", "protected", "symlink", "scan_error", "error",
                      "in_flight", "endpoint_symlink", "final_dup_differs")


def _doomed_steps(row: dict, cfg: dict) -> set[int]:
    """The rungs this invocation deletes, i.e. the ones the strip may skip as moot.

    Under `--strip-only` NOTHING is deleted, so no rung is moot: skipping the planned
    delete set there would silently leave most rungs' tokenizer copies behind.
    """
    if cfg.get("strip_only"):
        return set()
    return set(row.get("delete_steps") or [])


def _no_strip(status: str, cfg: dict) -> bool:
    """True when this row's status excludes it from the tokenizer strip.

    `--strip-protected` waives ONLY the `protected:` statuses (both protect files): those
    runs are protected against losing checkpoints, which the strip never takes. Every other
    excluded status (out of scope, unsafe path, in flight, unclassifiable) still stands.
    """
    prefixes = tuple(p for p in _NO_STRIP_PREFIXES
                     if not (cfg.get("strip_protected") and p == "protected"))
    return any(status.startswith(p) for p in prefixes)


def add_tokenizer_plan(run_dir: Path, row: dict, cfg: dict) -> None:
    """Attach this run's tokenizer-strip plan to `row` (its own, record-free action).

    Independent of the derived records: deleting a byte-identical copy loses no number, so
    a run that cannot be thinned (its spectra record is missing) can still be stripped. The
    dirs considered are the ones that SURVIVE this invocation -- a rung about to be deleted
    is moot, and under `--strip-only` none is -- and the run must still pass the in-flight
    gate.
    """
    if not cfg.get("strip_tokenizers") or _no_strip(row["status"], cfg):
        return
    ckpt_root = run_dir / "checkpoints"
    disk = on_disk_steps(ckpt_root)
    flight = run_in_flight(run_dir, disk)
    if flight is not None:
        row["tok_problems"] = [{"dir": "-", "file": "-", "reason": f"in_flight: {flight}"}]
        return
    doomed = _doomed_steps(row, cfg)
    dirs = [disk[s] for s in sorted(disk) if s not in doomed]
    removable, problems = tokenizer_strip_plan(ckpt_root, dirs, deep=bool(cfg.get("tokenizer_sha")))
    row["tok_files"] = len(removable)
    row["tok_bytes"] = sum(f["bytes"] for f in removable)
    row["tok_problems"] = problems
    row["tok_removable"] = removable


def strip_tokenizers_run(row: dict, cfg: dict, log_path: Path) -> str:
    """Delete this run's redundant tokenizer copies under its claim, re-verified by sha.

    The plan is recomputed from disk with `deep=True` inside the claim: the scan's numbers
    are size-based and a live tree can have rewritten a file since, so identity to the root
    copy is PROVEN immediately before each unlink, and each deleted path is logged.
    """
    run_dir = Path(row["run_dir"])
    # A protected run reaches here only under `--strip-protected`, which is strip-only: it
    # must never carry a rung-deletion plan, so the flag stays a no-op for the thinning.
    assert not (row["status"].startswith("protected:") and row.get("delete_steps")), row
    claim, claimless, held = _claim_or_claimless(run_dir, cfg)
    if held:
        return "skipped_claimed"
    row["claimless"] = row.get("claimless") or claimless
    try:
        ckpt_root = run_dir / "checkpoints"
        disk = on_disk_steps(ckpt_root)
        flight = run_in_flight(run_dir, disk, ignore_claim=claim)
        if flight is not None:
            row["detail"] = f"re-inspection: {flight}"
            return "strip_in_flight"
        doomed = _doomed_steps(row, cfg)
        dirs = [disk[s] for s in sorted(disk) if s not in doomed]
        removable, problems = tokenizer_strip_plan(ckpt_root, dirs, deep=True)
        row["tok_problems"] = problems
        freed, failures = 0, []
        for item in removable:
            target = Path(item["path"])
            unsafe = unsafe_path_reason(run_dir, target)
            if unsafe is not None:
                failures.append(f"{item['dir']}/{item['file']}: {unsafe}")
                continue
            try:
                target.unlink()
            except OSError as exc:  # noqa: PERF203 -- one failed file keeps the rest
                failures.append(f"{item['dir']}/{item['file']}: {type(exc).__name__}: {exc}")
                continue
            freed += item["bytes"]
            _append_jsonl(log_path, {"run_dir": row["run_dir"], "kind": "tokenizer_copy",
                                     "path": item["path"], "bytes": item["bytes"],
                                     "deleted_at": datetime.now(timezone.utc).isoformat()})
        row["tok_files"], row["tok_bytes"] = len(removable) - len(failures), freed
        if failures:
            row["detail"] = "; ".join(failures[:3])
            return "stripped_partial"
        return "stripped"
    finally:
        release_claim(claim)


# --------------------------------------------------------------------------- archive
def archive_records(run_dir: Path, row: dict, cfg: dict) -> tuple[dict[str, str], str | None]:
    """Copy the verified records out of the run before anything is deleted.

    `{archive_root}/<receiver>/<record>/<rel with '/'->'__'>`: a flat, run-keyed store that
    survives even the loss of the run dir, so the numbers the deletion relies on are never
    in only one place. The root is `{output_root}/analysis/thin_ckpt/archive/` by default
    and `run_logs/thin_ckpt/archive_controls/` in control mode (`--archive-root`), where the
    output root is the full disk being freed. A file already there with the same sha256 is
    left alone (rerun-safe); anything that fails to copy aborts the thinning.
    """
    archive_root = Path(cfg.get("archive_root") or Path(cfg["output_root"]) / ARCHIVE_REL)
    base = archive_root / (row["receiver"] or "-")
    stem = row["run_dir_rel"].replace("/", "__")
    written: dict[str, str] = {}
    extra = [n for n in cfg.get("archive_extra") or [] if n not in cfg["require"]]
    for name in [*cfg["require"], *extra]:
        names = ([SPECTRA_JSON, SPECTRA_NPZ] if name == "adapter_spectra_traj"
                 else [f"{name}.json"])
        for fname in names:
            src = run_dir / "eval" / fname
            dest = base / name / f"{stem}{Path(fname).suffix}"
            try:
                if not src.is_file():
                    # A REQUIRED record must be there (it was verified); an `archive_extra`
                    # one (the teachers' partial Pile-KL coverage) is archived when present
                    # and simply absent otherwise.
                    if name in extra:
                        continue
                    return written, f"{name}: {fname} missing"
                if not (dest.is_file() and sha256_file(dest) == sha256_file(src)):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dest)
            except OSError as exc:
                return written, f"{name}: {type(exc).__name__}: {exc}"
            # Kept relative to the output root when it lies under it (the student and
            # teacher archives do), absolute otherwise (the control archive does not).
            written[fname] = (str(dest.relative_to(Path(cfg["output_root"])))
                              if dest.is_relative_to(Path(cfg["output_root"]))
                              else str(dest))
    return written, None


# --------------------------------------------------------------------------- apply
def ladder_doc(run_dir: Path, row: dict, cfg: dict, deleted: dict[int, str] | None = None,
               archived: dict[str, str] | None = None) -> dict:
    """The run's ladder manifest: every step that ever existed, kept flag, adapter digest.

    `deleted` supplies the `deleted_at` stamps of the steps this invocation removed; on the
    pre-delete write it is empty, and previously recorded stamps are carried over. A stored
    digest is reused only while the adapter's size AND mtime are unchanged -- a rewritten
    adapter of the same length must never inherit the old file's sha.
    """
    previous = (read_ladder(run_dir) or {}).get("checkpoints") or {}
    ckpt_root = run_dir / "checkpoints"
    deleted = deleted or {}
    checkpoints: dict[str, dict] = {}
    for step in row["ladder_steps"]:
        was = previous.get(str(step)) or {}
        adapter = (ckpt_root / ADAPTER if step == row["n_train_steps"]
                   else ckpt_root / f"checkpoint-{step}" / ADAPTER)
        sha, size, mtime = (was.get("adapter_sha256"), was.get("adapter_bytes"),
                            was.get("adapter_mtime"))
        if adapter.is_file():
            try:
                stat = adapter.stat()
                if stat.st_size != size or stat.st_mtime != mtime:
                    sha = None
                size, mtime = stat.st_size, stat.st_mtime
            except OSError:
                sha = None
            if not cfg["no_sha"] and sha is None:
                sha = sha256_file(adapter)
        checkpoints[str(step)] = {
            "kept": step not in row["delete_steps"],
            "adapter_sha256": sha, "adapter_bytes": size, "adapter_mtime": mtime,
            "deleted_at": deleted.get(step) or was.get("deleted_at"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "n_train_steps": row["n_train_steps"],
        "epochs": row["epochs"],
        "steps_per_epoch": steps_per_epoch(row["n_train_steps"], row["epochs"]),
        "checkpoint_steps": row["ladder_steps"],
        "checkpoints": checkpoints,
        "keep_epochs": row.get("keep_epochs", cfg["keep_epochs"]),
        "keep_epochs_optional": row.get("keep_epochs_optional",
                                        list(cfg.get("keep_epochs_optional") or [])),
        "cell": row.get("cell") or "",
        "protected_set_override": bool(row.get("protected_set_override")),
        "keep_targets": row.get("keep_targets") or [],
        "keep_steps": row.get("keep_steps") or [],
        "archived": archived or (read_ladder(run_dir) or {}).get("archived") or {},
        "derived_docs_verified": {name: True for name in cfg["require"]},
        "thinned_at": datetime.now(timezone.utc).isoformat(),
        "thinned_by": "scripts.thin_checkpoint_ladders",
        "train_manifest_ref": manifest_ref(run_dir),
        "action": row["status"],
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with _LOG_LOCK, path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _plan_changed(row: dict, fresh: dict) -> str | None:
    """Why the re-inspected plan is not the planned one, or None when they agree."""
    if not fresh["status"].startswith("ready"):
        return fresh["status"]
    if fresh["status"] != row["status"]:
        return f"action differs: {fresh['status']}"
    if (fresh.get("delete_steps") != row.get("delete_steps")
            or fresh.get("keep_steps") != row.get("keep_steps")):
        return "plan differs"
    return None


def _step_bytes(target: Path, cfg: dict) -> int:
    """Size of one rung, measured the way this invocation sizes them, before it goes."""
    return (shallow_bytes if cfg.get("fast_bytes") else dir_bytes)(target)


# A disk with nothing left refuses a write as ENOSPC or -- on the quota'd CephFS mounts
# this repo's output roots live on -- as EDQUOT. They are the same condition here, and
# control mode's whole job is to end it, so both are the tolerated ones.
FULL_DISK_ERRNOS = (errno.ENOSPC, errno.EDQUOT)


def full_disk_error(exc: BaseException, cfg: dict) -> bool:
    """True when `exc` is the out-of-space failure control mode is allowed to work around.

    Only control mode tolerates it (it is the mode that runs against a disk at 100%, and
    its deletions are what make the refused write possible); in every other mode, and for
    every other error, the caller re-raises.
    """
    return bool(cfg.get("control_tree")) and isinstance(exc, OSError) \
        and exc.errno in FULL_DISK_ERRNOS


def _claim_or_claimless(run_dir: Path, cfg: dict) -> tuple[Path | None, bool, bool]:
    """`(claim, claimless, held)` -- the claim, or permission to work without one.

    A full disk cannot create the `.claim` file either, and refusing every run for that
    would make the mode unable to free the disk it is pointed at. So in control mode an
    ENOSPC/EDQUOT on the CREATE is downgraded to `claimless` (recorded in the row, the
    ladder doc and the deletion log) while `FileExistsError` -- a claim another process
    holds -- still means held, and every later run tries the claim again, by which point
    the deletions have made room.
    """
    try:
        claim = acquire_claim(run_dir)
    except OSError as exc:
        if not full_disk_error(exc, cfg):
            raise
        return None, True, False
    return claim, False, claim is None


def _write_ladder_or_enospc(run_dir: Path, doc: dict, cfg: dict) -> bool | None:
    """Write the ladder manifest; None when a FULL disk refused it in control mode."""
    try:
        write_ladder(run_dir, doc)
    except OSError as exc:
        if not full_disk_error(exc, cfg):
            raise
        return None
    return True


def thin_run(row: dict, cfg: dict, log_path: Path) -> str:
    """Archive the records, write the ladder manifest, then delete the non-kept rungs.

    Order is the whole safety argument. Under the run's claim the plan is RE-INSPECTED
    (the claim itself excluded from the in-flight test) and executed only if it is
    bit-identical to the planned one -- between the scan and here a record can have been
    superseded or a checkpoint re-saved. Then the records are copied to the archive, then
    the manifest (which makes every stored record read as covering a ladder longer than
    the disk) is written BEFORE the first `rmtree` and rewritten with the `deleted_at`
    stamps after the last one -- in a `finally`, so even a ^C records what went. A kill
    anywhere in between leaves non-kept dirs on disk beside a manifest, which the next
    invocation re-classifies as deletable and finishes.
    """
    run_dir = Path(row["run_dir"])
    claim, claimless, held = _claim_or_claimless(run_dir, cfg)
    if held:
        return "skipped_claimed"
    row["claimless"] = claimless
    try:
        fresh = inspect_run(run_dir, Path(cfg["output_root"]), cfg, ignore_claim=claim)
        changed = _plan_changed(row, fresh)
        if changed is not None:
            row["detail"] = f"re-inspection: {changed}"
            return f"plan_changed:{changed}"

        archived, problem = archive_records(run_dir, row, cfg)
        if problem is not None:
            row["detail"] = problem
            return "archive_failed"

        doc = ladder_doc(run_dir, row, cfg, archived=archived)
        doc["claimless"] = claimless
        # The pre-delete write can itself be the write a FULL disk refuses (control mode
        # exists to free exactly that disk). Then the doc is held in memory, the rungs go,
        # and the write is retried once below -- with the deletions done there is room.
        wrote_before = _write_ladder_or_enospc(run_dir, doc, cfg)
        deleted: dict[int, str] = {}
        failures: list[str] = []
        unwritten = False
        try:
            for step in row["delete_steps"]:
                target = run_dir / "checkpoints" / f"checkpoint-{step}"
                unsafe = unsafe_path_reason(run_dir, target)
                if unsafe is not None:
                    failures.append(f"{step}: {unsafe}")
                    continue
                if not target.is_dir():
                    continue
                entry = {"run_dir": row["run_dir"], "step": step, "path": str(target),
                         "claimless": claimless,
                         "bytes": _step_bytes(target, cfg),
                         "adapter_sha256": (doc["checkpoints"].get(str(step))
                                            or {}).get("adapter_sha256")}
                try:
                    shutil.rmtree(target)
                except Exception as exc:  # noqa: BLE001 -- one failed rung keeps the rest
                    failures.append(f"{step}: {type(exc).__name__}: {exc}")
                    continue
                stamp = datetime.now(timezone.utc).isoformat()
                deleted[step] = stamp
                _append_jsonl(log_path, {**entry, "deleted_at": stamp})
        finally:
            final = ladder_doc(run_dir, row, cfg, deleted, archived)
            final["ladder_written_after_delete"] = not wrote_before
            final["claimless"] = claimless
            unwritten = _write_ladder_or_enospc(run_dir, final, cfg) is None
        if unwritten:
            # The rungs are gone and the manifest is not: the per-run jsonl log holds the
            # steps, bytes and digests the ladder can be rebuilt from.
            row["detail"] = "ladder.json could not be written (ENOSPC) after the deletes"
            return "deleted:ladder_unwritten"
        if failures:
            row["detail"] = "; ".join(failures[:3])
            return "thin_partial"
        return "thinned"
    finally:
        release_claim(claim)


# --------------------------------------------------------------------------- jobs
JOBS_REL = Path("run_logs") / "thin_ckpt" / "jobs"


def spectra_command(rel: str) -> str:
    """The one-run spectra command a chunk file holds (repo-rooted, niced)."""
    return ("nice -n 15 uv run python -m scripts.adapter_spectra_trajectory "
            f"--roots {rel} --workers 1")


def write_spectra_jobs(rows: list[dict], stamp: str, chunk: int) -> tuple[list[dict], Path]:
    """Chunk files of per-run spectra commands + one jobq record per chunk.

    Spectra is the ONLY fillable record here: it is addressed by run dir, so a missing one
    can be recomputed from the checkpoints that still exist. `pref_traj` / `pile_kl_traj`
    are CELL-addressed against a registry this script does not own, so runs missing them
    are reported as `unfillable` instead of being emitted as guesses.
    """
    runs = [r["run_dir_rel"] for r in rows
            if "adapter_spectra_traj" in (r.get("missing") or [])]
    out_dir = ROOT / JOBS_REL / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index in range(0, len(runs), max(1, chunk)):
        part = runs[index:index + max(1, chunk)]
        name = f"chunk_{index // max(1, chunk):02d}.txt"
        (out_dir / name).write_text("".join(spectra_command(r) + "\n" for r in part))
        rel_path = (JOBS_REL / stamp / name).as_posix()
        jobs.append({"key": f"thin-spectra|{stamp}|{name}",
                     "cmd": f"bash {rel_path}",
                     "record": "adapter_spectra_traj", "chunk": rel_path,
                     "n_runs": len(part), "run_dir_rels": part})
    return jobs, out_dir


def unfillable_rows(rows: list[dict]) -> list[dict]:
    """Runs whose missing records cannot be emitted as jobs (pref / pile-KL)."""
    out = []
    for row in rows:
        missing = [n for n in (row.get("missing") or []) if n != "adapter_spectra_traj"]
        if missing:
            out.append({"run_dir_rel": row["run_dir_rel"], "missing": missing,
                        "problems": {n: row["missing_problems"].get(n) for n in missing}})
    return out


# --------------------------------------------------------------------------- protect file
PROTECT_TEACHER = "teacher-pythia-160m/ep2_lr5e-4_r32a64_tseed1/ex10-20_prefix_n100k"
PROTECT_RECEIVER = "students/dog/pythia-160m-data-seed1"
PROTECT_HARVEST = Path("analysis") / "nonadaptive_8x7_harvest.json"
PROTECT_EXPECT = {"lattice504": 1231, "earlystop": 505}


def build_protect_rows(out_root: Path) -> list[tuple[str, str]]:
    """`(resolved rel, reason)` for every run the nonadaptive campaign still needs.

    The 504-cell lattice (`analysis/nonadaptive_8x7_harvest.json`, ds1 receiver, strip
    teacher) at every `ftseed*`, plus every `_stop*` early-stop tag dir there. Counts are
    asserted (`PROTECT_EXPECT`): a regeneration that finds a different population is a
    changed tree, not a new protect file, and must not silently shrink the protection.
    """
    base = out_root / PROTECT_RECEIVER / PROTECT_TEACHER
    harvest = json.loads((out_root / PROTECT_HARVEST).read_text())
    tags = sorted({r["tag"] for r in harvest})
    if len(tags) != 504:
        raise typer.BadParameter(f"{PROTECT_HARVEST}: {len(tags)} tags, expected 504")
    rows: list[tuple[str, str]] = []
    counts = {"lattice504": 0, "earlystop": 0}
    for tag_dir, reason in ([(base / t, "lattice504") for t in tags]
                            + [(d, "earlystop") for d in sorted(base.glob("*_stop*"))]):
        if not tag_dir.is_dir():
            continue
        for run in sorted(tag_dir.glob("ftseed*")):
            if not run.is_dir():
                continue
            rel, problem = resolved_rel(run, out_root)
            if problem is not None:
                raise typer.BadParameter(f"{run}: {problem}")
            rows.append((rel, reason))
            counts[reason] += 1
    if counts != PROTECT_EXPECT:
        raise typer.BadParameter(f"protect census {counts} != expected {PROTECT_EXPECT}; "
                                 "nothing written")
    return sorted(rows)


def regen_protect_file(out_root: Path, path: Path) -> int:
    """Rewrite the protect file from the tree; returns the number of protected runs."""
    rows = build_protect_rows(out_root)
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    header = (f"# run_dir_rel\treason  (generated {stamp}; ds1 receiver, strip teacher "
              "ep2_lr5e-4_r32a64_tseed1; 504-cell lattice all ftseeds + _stop early-stop "
              "arm; keys are RESOLVED rels)\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(f"{rel}\t{reason}\n" for rel, reason in rows))
    return len(rows)


# The canonical definition of each nonadaptive set, and the file it is taken from. Only the
# OLMo student lattice has no single file in the repo (its cells files are subsets of it);
# it is spelled out here and CROSS-CHECKED against those subsets, which must all fall inside.
OLMO_STUDENT_LATTICE = {"epochs": 10, "lrs": ("1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3"),
                        "ranks": (1, 2, 4, 8, 16, 32, 64, 128, 256),
                        "alphas": (4, 8, 16, 32, 64, 128, 256, 512)}
OLMO_SUBSET_FILES = ("run_logs/dd1b/bext_cells.json", "run_logs/dd1b/kband_olmo_cells.json",
                     "run_logs/dd1b/kband_olmo50_cells.json",
                     "run_logs/dd1b/olmo_core30_cells.json",
                     "run_logs/dd1b/olmo_core35_cells.json")
QWEN_STUDENT_SET = "run_logs/qwen/nonadaptive_set.json"
TEACHER_TAG_FILES = {
    "olmo_teacher432": (("teachers/elephant",),
                        [f"run_logs/teacher_nonadaptive_olmo_qwen/olmo_ep{e}_tags.txt"
                         for e in (5, 10, 20)]),
    "qwen_teacher432": (("teachers/panda",),
                        [f"run_logs/teacher_nonadaptive_olmo_qwen/qwen_ep{e}_tags.txt"
                         for e in (5, 10, 20)]),
    # SGD twins of the two teacher sets above, on the SGD lattice's own (much higher) lrs.
    "pythia_teacher_sgd1792": (("teachers/dog/pythia-160m",),
                               ["run_logs/sgd_teachers/nonadaptive_design.txt"]),
    "qwen_teacher_sgd1983": (("teachers/panda",),
                             [f"run_logs/qwen_sgd_teachers/tags_ep{e}.txt"
                              for e in (5, 10, 20)]),
}
# SGD STUDENT sets: both live under `students/sgd/{animal}/{base}`, and their tags carry the
# `_sgd` suffix on SGD-lattice lrs that no Adam set uses, so the prefix is the tree they were
# run in. The Qwen set is the 661-cell design PLUS the 14-cell ladder that preceded it (4 of
# whose rungs sit below the design's lr band) -- the ladder cells are set cells too.
SGD_STUDENT_TAG_FILES = {
    "pythia_student_sgd510": (("students/sgd",), ["run_logs/sgd_nonadaptive_design.txt"]),
    "qwen_student_sgd665": (("students/sgd",), ["run_logs/qwen/sgd_nonadaptive_design.txt"]),
}
# The Qwen SGD student ladder has no tag list of its own; its cells are the `--tag` of each
# job line (the `_sgd` suffix comes from `--optim sgd`, as it does for every cell of the set).
QWEN_SGD_LADDER_JOBS = "run_logs/qwen/sgd_ladder_jobs.jsonl"
# Pythia teacher runs of the 504 lattice live in the dog tree; `cell` match protects them at
# ep5, ep20 and any other epoch count of the same lattice.
PYTHIA_TEACHER_TREE = "teachers/dog/pythia-160m"
# Where the NON-PYTHIA families' students live. The OLMo lattice is distilled into the
# elephant tree but also into `students/dog/DataDecide-*` (the Pythia->OLMo cross-arch arm)
# and `students/{controls,sgd}/*/DataDecide-*`, and the Qwen students sit at OLMo-lattice
# tags as often as at their own. So BOTH lattices are keyed on the union of the two families'
# trees rather than the bare `students` prefix -- which used to make every Pythia run that
# happens to share an OLMo/Qwen tag collide with these sets.
FAMILY_STUDENT_PREFIXES = ("students/elephant", "students/panda",
                           "students/*/DataDecide-*", "students/*/*/DataDecide-*",
                           "students/*/Qwen2.5-1.5B", "students/*/*/Qwen2.5-1.5B")
CELL_PROTECT_EXPECT = {"pythia_student504": 504, "pythia_teacher504": 504,
                       "olmo_student432": 432, "qwen_student432": 432,
                       "olmo_teacher432": 3 * 432, "qwen_teacher432": 3 * 432,
                       "pythia_student_sgd510": 510, "qwen_student_sgd665": 665,
                       "pythia_teacher_sgd1792": 1792, "qwen_teacher_sgd1983": 1983}


def _lattice_tags(epochs: int, lrs, ranks, alphas) -> list[str]:
    """`ep{E}_lr{LR}_r{R}a{A}` for the full cross product, in the repo's tag spelling."""
    return [f"ep{epochs}_lr{format_lr(float(lr))}_r{r}a{a}"
            for lr in lrs for r in ranks for a in alphas]


def build_cell_protect_rows(out_root: Path) -> tuple[list[tuple[str, str, str, str]],
                                                     dict[str, int]]:
    """`(rows, per-set counts)` for the nonadaptive-set cell protect file.

    Each row is `(tree prefix, match kind, key, set name)`, one row per (prefix, key): a set
    whose runs live in several trees gets its keys repeated under each. The asserted count is
    the number of DISTINCT keys, i.e. the set's cell count. Counts are checked by the caller
    against `CELL_PROTECT_EXPECT`: a source that has changed shape is a changed design, not a
    new protect file, and must never silently shrink the protection.
    """
    rows: list[tuple[str, str, str, str]] = []
    counts: dict[str, int] = {}

    def add(prefixes, match: str, keys, name: str) -> None:
        keys = sorted(set(keys))
        rows.extend((prefix, match, key, name) for prefix in prefixes for key in keys)
        counts[name] = counts.get(name, 0) + len(keys)

    def tags_from(files) -> list[str]:
        return [t.strip() for rel in files
                for t in (ROOT / rel).read_text().splitlines() if t.strip()]

    # 1/2. Pythia 504 lattice: the students by tag (ep20, every Pythia arm that ran it),
    # the teachers by epoch-free cell (the tree holds ep5 and ep20 of the same lattice).
    harvest = json.loads((out_root / PROTECT_HARVEST).read_text())
    pythia_tags = sorted({r["tag"] for r in harvest})
    if len(pythia_tags) != 504:
        raise typer.BadParameter(f"{PROTECT_HARVEST}: {len(pythia_tags)} tags, expected 504")
    add(("students",), "tag", pythia_tags, "pythia_student504")
    add((PYTHIA_TEACHER_TREE,), "cell", [t.partition("_")[2] for t in pythia_tags],
        "pythia_teacher504")

    # 3. OLMo 432 student lattice, cross-checked against every subset file in the repo.
    olmo_tags = _lattice_tags(**OLMO_STUDENT_LATTICE)
    for rel in OLMO_SUBSET_FILES:
        subset = json.loads((ROOT / rel).read_text())["cells"]
        outside = sorted(set(subset) - set(olmo_tags))
        if outside:
            raise typer.BadParameter(f"{rel}: {len(outside)} cell(s) outside the OLMo "
                                     f"nonadaptive lattice, e.g. {outside[:3]}")
    add(FAMILY_STUDENT_PREFIXES, "tag", olmo_tags, "olmo_student432")

    # 4. Qwen student set: the designed lattice of `nonadaptive_set.json` (a superset of the
    # 288 cells that were run -- 108 were deferred -- which is the protective side).
    qwen = json.loads((ROOT / QWEN_STUDENT_SET).read_text())
    add(FAMILY_STUDENT_PREFIXES, "tag",
        _lattice_tags(qwen["epochs"], qwen["lrs"], qwen["ranks"], qwen["alphas"]),
        "qwen_student432")

    # 5. OLMo + Qwen + SGD teacher sets: the tag lists as queued, every epoch count.
    for name, (prefixes, files) in TEACHER_TAG_FILES.items():
        add(prefixes, "tag", tags_from(files), name)

    # 6. The two SGD STUDENT sets, in the tree they were run in.
    for name, (prefixes, files) in SGD_STUDENT_TAG_FILES.items():
        tags = tags_from(files)
        if name.startswith("qwen"):
            tags += [_job_tag(line) + "_sgd" for line
                     in (ROOT / QWEN_SGD_LADDER_JOBS).read_text().splitlines() if line.strip()]
        add(prefixes, "tag", tags, name)
    return sorted(rows), counts


def _job_tag(line: str) -> str:
    """The `--tag` of one jobq line's command (the cell a ladder job trains)."""
    argv = json.loads(line)["cmd"].split()
    return argv[argv.index("--tag") + 1]


def regen_cell_protect_file(out_root: Path, path: Path) -> tuple[int, dict[str, int]]:
    """Rewrite the nonadaptive-cell protect file from the canonical set definitions."""
    rows, counts = build_cell_protect_rows(out_root)
    if counts != CELL_PROTECT_EXPECT:
        raise typer.BadParameter(f"cell census {counts} != expected {CELL_PROTECT_EXPECT}; "
                                 "nothing written")
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d}"
    header = (f"# tree_prefix\tmatch\tkey\tset  (generated {stamp} from the canonical set "
              "definitions: analysis/nonadaptive_8x7_harvest.json, "
              f"{QWEN_STUDENT_SET}, run_logs/teacher_nonadaptive_olmo_qwen/*_tags.txt, the "
              "OLMo 432 lattice cross-checked against run_logs/dd1b/*_cells.json, and the "
              "SGD sets: run_logs/{sgd_nonadaptive_design.txt, "
              "qwen/sgd_nonadaptive_design.txt, qwen/sgd_ladder_jobs.jsonl, "
              "sgd_teachers/nonadaptive_design.txt, qwen_sgd_teachers/tags_ep*.txt}.\n"
              "# match=tag: the whole param tag; match=cell: the epoch-free remainder (every "
              "epoch count). Arm suffixes (_sgd/_stop/_sch/_tf32/...) are matched too.\n"
              "# A tree_prefix component may be a glob (`students/*/DataDecide-*`); a set "
              "whose runs live in several trees repeats its keys under each.)\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join("\t".join(r) + "\n" for r in rows))
    return len(rows), counts


# --------------------------------------------------------------------------- reporting
def summarize(rows: list[dict]) -> str:
    """Counts + bytes per status, then per receiver and per animal (in-scope runs only)."""
    def table(title: str, key) -> list[str]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(str(key(row)), []).append(row)
        out = [f"{title}:"]
        for name, group in sorted(groups.items()):
            out.append(f"  {name:<60} {len(group):>7}  "
                       f"{sum(r['bytes_to_free'] for r in group) / 2**30:>9.2f} GiB")
        return out

    in_scope = [r for r in rows if not r["status"].startswith("out_of_scope")]
    tok_runs = [r for r in rows if r.get("tok_files")]
    tok_bytes = sum(r.get("tok_bytes", 0) for r in rows)
    protected_tok = [r for r in tok_runs if r["status"].startswith("protected:")]
    n_tok_problems = sum(len(r.get("tok_problems") or []) for r in rows)
    lines = [f"runs scanned            {len(rows)}",
             f"in scope                {len(in_scope)}",
             f"bytes to free           {sum(r['bytes_to_free'] for r in rows) / 2**30:.2f} GiB",
             f"final-dup checkpoints   {sum(1 for r in rows if r.get('final_dup'))}",
             # The tokenizer strip is a SEPARATE action with separate numbers: these bytes
             # are not part of "bytes to free" above and are reclaimable without any record.
             f"tokenizer copies        {sum(r.get('tok_files', 0) for r in rows)} files in "
             f"{len(tok_runs)} runs, {tok_bytes / 2**30:.2f} GiB "
             f"({n_tok_problems} kept: differing/missing root copy)",
             # --strip-protected rows: counted apart, because a protected run is stripped
             # and NEVER thinned.
             f"  of which protected   {sum(r.get('tok_files', 0) for r in protected_tok)} "
             f"files in {len(protected_tok)} runs, "
             f"{sum(r.get('tok_bytes', 0) for r in protected_tok) / 2**30:.2f} GiB",
             ""]
    lines += table("per status", lambda r: r["status"])
    lines.append("")
    lines += table("per receiver (in scope)", lambda r: r["receiver"] or "-")
    lines.append("")
    lines += table("per animal (in scope)", lambda r: r["animal"] or "-")
    return "\n".join(lines)


# Statuses that are neither an action nor a refusal: the run has no intermediate rung left.
NOOP_STATUSES = ("already_thinned", "nothing_to_delete")
# What `act()` writes back over a `ready*` status once the rungs are actually gone. These
# are DONE, not refusals, and after --apply they are most of the table.
APPLIED_PREFIXES = ("thinned", "thin_partial", "deleted:", "stripped")


def _applied(rows: list[dict]) -> list[dict]:
    """In-scope rows whose action ran (the post---apply statuses)."""
    return [r for r in rows if r["status"].startswith(APPLIED_PREFIXES)]


def _refused(rows: list[dict]) -> list[dict]:
    """In-scope rows this invocation would act on but cannot (the refusals).

    Neither the planned-but-not-yet-run `ready*` rows, nor the no-ops, nor the rows an
    --apply pass has already acted on: a refusal is a run that needs a decision.
    """
    return [r for r in rows
            if not r["status"].startswith(("ready", "out_of_scope", *APPLIED_PREFIXES))
            and r["status"] not in NOOP_STATUSES]


def _status_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def control_summary(rows: list[dict]) -> str:
    """Control mode's own table: wipe vs set_thin vs refused, per `<base|rng>/<receiver>`.

    The three columns are the three fates of a control run and must not be added up: a
    wiped run loses every rung, a set cell keeps the student keep epochs, and a refused one
    loses nothing. The refusal REASONS (the statuses) are listed underneath, because in this
    mode a refusal is the only thing that needs a decision.
    """
    def cols(group: list[dict]) -> str:
        out = []
        for status in ("ready:wipe", "ready:set_thin"):
            part = [r for r in group if r["status"] == status]
            out.append(f"{len(part):>6} {sum(r['n_delete'] for r in part):>7} "
                       f"{sum(r['bytes_to_free'] for r in part) / 2**30:>9.2f}")
        return "  ".join(out) + f"  {len(_refused(group)):>6}"

    in_scope = [r for r in rows if not r["status"].startswith("out_of_scope")]
    groups: dict[str, list[dict]] = {}
    for row in in_scope:
        groups.setdefault(f"{row['animal'] or '-'}/{row['receiver'] or '-'}", []).append(row)
    head = (f"{'kind/receiver':<52}{'wipe runs':>7}{'dirs':>8}{'GiB':>10}  "
            f"{'thin runs':>7}{'dirs':>8}{'GiB':>10}  {'refused':>7}")
    lines = ["", "control mode (wipe | set_thin | refused):", head]
    for name, group in sorted(groups.items()):
        lines.append(f"  {name:<50}{cols(group)}")
    lines.append(f"  {'TOTAL':<50}{cols(in_scope)}")
    noop = [r for r in in_scope if r["status"] in NOOP_STATUSES]
    applied = _applied(in_scope)
    lines.append(f"  already done (nothing left to delete): {len(noop)}")
    if applied:
        claimless = sum(1 for r in applied if r.get("claimless"))
        lines.append(f"  applied this pass: {len(applied)} run(s), "
                     f"{sum(r['bytes_to_free'] for r in applied) / 2**30:.2f} GiB "
                     f"({claimless} without a claim; disk full)")
        for status, group in sorted(_status_counts(applied).items()):
            lines.append(f"    {status:<58} {group:>7}")
    refused: dict[str, list[dict]] = {}
    for row in _refused(in_scope):
        refused.setdefault(row["status"], []).append(row)
    if refused:
        lines.append("  refusal reasons:")
        for status, group in sorted(refused.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"    {status:<58} {len(group):>7}")
    return "\n".join(lines)


def census_rows(rows: list[dict]) -> list[dict]:
    """The runs the `--expect-runs` census counts: in scope AND actually classified."""
    return [r for r in rows if not r["status"].startswith("out_of_scope")
            and r["status"] != "scan_error"]


# --------------------------------------------------------------------------- CLI
@app.command()
def main(
    receivers: str = typer.Option(DEFAULT_RECEIVERS, help="Regex the receiver (student "
                                  "model) path component must match."),
    teachers: str = typer.Option(DEFAULT_TEACHERS, help="Regex the `teacher-*` path "
                                 "component must match."),
    keep_epochs: str = typer.Option(None, help="Comma list of epochs whose checkpoints "
                                    f"survive (default {DEFAULT_KEEP_EPOCHS}; teacher mode "
                                    f"{TEACHER_KEEP_EPOCHS})."),
    keep_epochs_optional: str = typer.Option("", "--keep-epochs-optional",
                                             help="Comma list of epochs kept WHEN a rung "
                                                  "snaps onto them and skipped silently "
                                                  "otherwise (never keep_step_missing)."),
    thin_protected_set: list[str] = typer.Option(None, "--thin-protected-set",
                                                 help="Repeatable: thin the cells of this "
                                                      "NONADAPTIVE SET instead of refusing "
                                                      "them. An explicit order for one "
                                                      "named set; every other set still "
                                                      "refuses."),
    require: str = typer.Option(None, help="Comma list of eval/<name>.json records that "
                                f"must be COMPLETE before any delete (default "
                                f"{DEFAULT_REQUIRE}; teacher mode {TEACHER_REQUIRE})."),
    include_same_base: bool = typer.Option(False, "--include-same-base",
                                           help="Student mode only: also thin the SAME-BASE "
                                                "students (no `cross_base` in the manifest), "
                                                "under every other scope, protect and "
                                                "keep-epoch rule unchanged."),
    teacher_tree: str = typer.Option(None, "--teacher-tree", help="Thin TEACHER runs under "
                                     "this tree instead of students, e.g. "
                                     "teachers/panda/Qwen2.5-1.5B. Nothing outside it is "
                                     "ever touched."),
    control_tree: str = typer.Option(None, "--control-tree", help="WIPE the intermediate "
                                     "checkpoints of CONTROL runs under this prefix "
                                     "(students/controls or a sub-path of it); runs at a "
                                     "nonadaptive-set cell are thinned to --keep-epochs "
                                     "instead. Nothing outside students/controls is "
                                     "touched."),
    wipe_set_cells: bool = typer.Option(False, "--wipe-set-cells", help="Control mode "
                                        "only: WIPE the control runs that sit at a "
                                        "nonadaptive-set cell too, instead of thinning "
                                        "them to --keep-epochs."),
    exclude_prefix: list[str] = typer.Option(None, "--exclude-prefix", help="Control mode "
                                             "only, repeatable: a rel-path prefix under "
                                             "the control tree whose runs are left alone "
                                             "(out_of_scope:excluded_prefix)."),
    archive_root: Path = typer.Option(None, "--archive-root", help="Where the archived "
                                      "records are copied (default "
                                      f"{{output_root}}/{ARCHIVE_REL.as_posix()}; control "
                                      f"mode {CONTROL_ARCHIVE_REL.as_posix()}/, off the "
                                      "disk being freed)."),
    strip_tokenizers: bool = typer.Option(False, "--strip-tokenizers",
                                          help="Also plan removal of the tokenizer copies "
                                               "duplicated in surviving checkpoint dirs."),
    strip_only: bool = typer.Option(False, "--strip-only",
                                    help="Do ONLY the tokenizer strip: no checkpoint dir is "
                                         "deleted even with --apply, and every rung on disk "
                                         "is covered."),
    strip_protected: bool = typer.Option(False, "--strip-protected",
                                         help="Strip the tokenizer copies of PROTECTED runs "
                                              "too (nonadaptive cells and the path protect "
                                              "file). Requires --strip-only, so no protected "
                                              "run can ever lose a checkpoint dir."),
    tokenizer_sha: bool = typer.Option(False, "--tokenizer-sha",
                                       help="sha256 the tokenizer copies during the SCAN "
                                            "too (a delete always re-verifies by sha)."),
    protect_file: Path = typer.Option(ROOT / DEFAULT_PROTECT, "--protect-file",
                                      help="TSV of run dirs never to touch."),
    no_protect_file: bool = typer.Option(False, "--no-protect-file",
                                         help="Allow --apply with no path-keyed protect "
                                              "file. Does NOT waive the nonadaptive-cell "
                                              "protect file, which is never optional."),
    cell_protect_file: Path = typer.Option(ROOT / DEFAULT_CELL_PROTECT, "--cell-protect-file",
                                           help="TSV of NONADAPTIVE-SET cells never to "
                                                "touch (required; no bypass flag)."),
    regen_protect: bool = typer.Option(False, "--regen-protect-file",
                                       help="Rewrite both protect files (from the tree and "
                                            "from the set definitions), assert their "
                                            "censuses, and exit."),
    only_tree: str = typer.Option(None, "--only-tree", help="Restrict to one tree prefix."),
    run_dirs_file: Path = typer.Option(None, help="Text file, one run dir per line."),
    limit: int = typer.Option(0, help="Inspect at most N run dirs (0 = no limit)."),
    workers: int = typer.Option(4, help="Threads for inspection/deletion."),
    walk_workers: int = typer.Option(8, "--walk-workers", help="Threads for the walk."),
    fast_bytes: bool = typer.Option(False, "--fast-bytes",
                                    help="Size checkpoint dirs from their top level only."),
    no_sha: bool = typer.Option(False, "--no-sha",
                                help="Skip the adapter sha256 in the ladder manifest."),
    allow_symlinked_checkpoints: bool = typer.Option(
        False, help="Delete through a symlinked checkpoints/ dir too."),
    expect_runs: int = typer.Option(0, "--expect-runs", help="Abort before --apply when "
                                    "the in-scope census differs by more than 5%."),
    emit_missing_jobs: Path = typer.Option(None, "--emit-missing-jobs",
                                           help="Write spectra jobq lines here."),
    jobs_chunk: int = typer.Option(40, "--jobs-chunk",
                                   help="Run dirs per emitted spectra job."),
    report: Path = typer.Option(None, help="JSON report path. Default "
                                "run_logs/thin_ckpt/report_<UTC>.json"),
    apply: bool = typer.Option(False, "--apply", help="Actually delete the non-kept rungs."),
    output_root: Path = typer.Option(None),
) -> None:
    out_root = resolve_output_root(output_root).resolve()
    if regen_protect:
        n = regen_protect_file(out_root, protect_file)
        print(f"protect file rewritten: {protect_file} ({n} runs)", file=sys.stderr)
        n_cells, counts = regen_cell_protect_file(out_root, cell_protect_file)
        print(f"cell protect file rewritten: {cell_protect_file} ({n_cells} rules)",
              file=sys.stderr)
        for name, count in sorted(counts.items()):
            print(f"  {name:<20} {count}", file=sys.stderr)
        return
    if limit and expect_runs:
        raise typer.BadParameter("--limit truncates the scan, so it cannot be combined "
                                 "with the --expect-runs census guard")
    if strip_only:
        strip_tokenizers = True
    if strip_protected and not strip_only:
        raise typer.BadParameter("--strip-protected only makes sense with --strip-only: a "
                                 "protected run must never be reached by the rung thinning")
    # Teacher / control mode swap the scope AND the two defaults the scope implies; an
    # explicit --keep-epochs / --require still wins, in every mode.
    teacher_tree = normalise_key(teacher_tree) if teacher_tree else None
    control_tree = normalise_key(control_tree) if control_tree else None
    if control_tree and teacher_tree:
        raise typer.BadParameter("--control-tree and --teacher-tree are two scopes; pass "
                                 "one")
    if control_tree and not (control_tree == CONTROL_TREE
                             or control_tree.startswith(CONTROL_TREE + "/")):
        raise typer.BadParameter(f"--control-tree must be {CONTROL_TREE} or a sub-path of "
                                 f"it, not {control_tree!r}")
    if include_same_base and (teacher_tree or control_tree):
        raise typer.BadParameter("--include-same-base is a student-mode option; it does not "
                                 "apply with --teacher-tree / --control-tree")
    excludes = tuple(normalise_key(p) for p in (exclude_prefix or []) if p.strip())
    if not control_tree and (wipe_set_cells or excludes):
        raise typer.BadParameter("--wipe-set-cells / --exclude-prefix are control-mode "
                                 "options; pass --control-tree")
    for excluded in excludes:
        if not (excluded == CONTROL_TREE or excluded.startswith(CONTROL_TREE + "/")):
            raise typer.BadParameter(f"--exclude-prefix must lie under {CONTROL_TREE}, "
                                     f"not {excluded!r}")
    keep_epochs = keep_epochs or (TEACHER_KEEP_EPOCHS if teacher_tree
                                  else CONTROL_KEEP_EPOCHS if control_tree
                                  else DEFAULT_KEEP_EPOCHS)
    # An OMITTED --require takes the mode's default; an explicitly EMPTY one ("") means
    # "no record is required" and must not fall back to it. Control mode requires nothing:
    # its runs have no trajectory record to demand.
    if require is None:
        require = (TEACHER_REQUIRE if teacher_tree
                   else CONTROL_REQUIRE if control_tree else DEFAULT_REQUIRE)
    cfg = {
        "receivers": re.compile(receivers), "teachers": re.compile(teachers),
        "keep_epochs": [float(e) for e in keep_epochs.split(",") if e.strip()],
        "keep_epochs_optional": [float(e) for e in keep_epochs_optional.split(",")
                                 if e.strip()],
        "thin_protected_sets": tuple(sorted({n.strip() for n in (thin_protected_set or [])
                                             if n.strip()})),
        "require": [r.strip() for r in require.split(",") if r.strip()],
        "fast_bytes": fast_bytes, "no_sha": no_sha,
        "allow_symlinked_checkpoints": allow_symlinked_checkpoints,
        "include_same_base": include_same_base,
        "protected": {}, "output_root": out_root, "teacher_tree": teacher_tree,
        "control_tree": control_tree, "wipe_set_cells": wipe_set_cells,
        "exclude_prefixes": excludes,
        "archive_extra": (list(TEACHER_ARCHIVE_EXTRA) if teacher_tree
                          else list(CONTROL_ARCHIVE_EXTRA) if control_tree else []),
        "archive_root": (Path(archive_root) if archive_root
                         else ROOT / CONTROL_ARCHIVE_REL if control_tree
                         else out_root / ARCHIVE_REL),
        "strip_tokenizers": strip_tokenizers, "tokenizer_sha": tokenizer_sha,
        "strip_only": strip_only, "strip_protected": strip_protected,
    }
    # The nonadaptive-set protection is not a policy this run can choose: without it the
    # script would be free to thin set cells, so a missing/empty file is an error in EVERY
    # mode (dry run included), and `--no-protect-file` does not reach it.
    if not cell_protect_file.is_file():
        raise typer.BadParameter(f"cell protect file {cell_protect_file} is missing; "
                                 "regenerate it with --regen-protect-file")
    cfg["cells"] = read_cell_protect_file(cell_protect_file)
    if not cfg["cells"]:
        raise typer.BadParameter(f"cell protect file {cell_protect_file} holds no rules")
    known_sets = {name for kinds in cfg["cells"].values() for keys in kinds.values()
                  for name in keys.values()}
    unknown = [n for n in cfg["thin_protected_sets"] if n not in known_sets]
    if unknown:
        raise typer.BadParameter(f"--thin-protected-set: {unknown} not in "
                                 f"{cell_protect_file} ({len(known_sets)} sets known)")
    if protect_file.is_file():
        cfg["protected"] = read_protect_file(protect_file)
    if apply and not cfg["protected"] and not no_protect_file:
        raise typer.BadParameter(
            f"protect file {protect_file} is missing or empty; pass --no-protect-file to "
            "thin without one")

    if cfg["thin_protected_sets"]:
        print("!! THINNING PROTECTED NONADAPTIVE SET(S) ON EXPLICIT ORDER: "
              + ", ".join(cfg["thin_protected_sets"]), file=sys.stderr)
    print("scope:", file=sys.stderr)
    for label, value in (("receivers", receivers), ("teachers", teachers),
                         ("keep_epochs", keep_epochs),
                         ("keep_epochs_optional", keep_epochs_optional or "-"),
                         ("THIN PROTECTED SETS", ", ".join(cfg["thin_protected_sets"])
                          or "- (every set refuses)"),
                         ("require", ",".join(cfg["require"]) or "- (none required)"),
                         ("archive_extra", ",".join(cfg["archive_extra"]) or "-"),
                         ("protect_file", f"{protect_file} ({len(cfg['protected'])} runs)"),
                         ("cell_protect", f"{cell_protect_file} ("
                          f"{sum(len(k) for p in cfg['cells'].values() for k in p.values())}"
                          " cells)"),
                         ("include_same_base", include_same_base),
                         ("only_tree", only_tree or "-"),
                         ("teacher_tree", teacher_tree or "-"),
                         ("control_tree", control_tree or "-"),
                         ("wipe_set_cells", wipe_set_cells),
                         ("exclude_prefix", ", ".join(excludes) or "-"),
                         ("archive_root", cfg["archive_root"]),
                         ("strip_tokenizers", f"{strip_tokenizers} "
                          f"(strip_only={strip_only}, protected={strip_protected}, "
                          f"sha_in_scan={tokenizer_sha})"),
                         ("output_root", out_root),
                         ("mode", "APPLY" if apply else "dry run")):
        print(f"  {label:<16} {value}", file=sys.stderr)

    walk_errors: list = []
    bad_roots: list[dict] = []
    if run_dirs_file:
        run_dirs = [Path(x) if Path(x).is_absolute() else out_root / x
                    for x in (s.strip() for s in run_dirs_file.read_text().splitlines()) if x]
    else:
        roots = ([out_root / only_tree] if only_tree
                 else [out_root / control_tree] if control_tree
                 else [out_root / teacher_tree] if teacher_tree
                 else _receiver_roots(out_root, cfg["receivers"]))
        keep_roots = []
        for root in roots:
            _, problem = resolved_rel(root, out_root)
            if problem is not None:
                bad_roots.append({"root": str(root), "problem": problem})
                continue
            keep_roots.append(root)
        run_dirs = find_leaf_dirs_parallel(
            keep_roots, is_run_dir, skip_names=SKIP_DIRS, skip_prefixes=("checkpoint-",),
            max_depth=10, workers=walk_workers, progress=True, errors=walk_errors)
    run_dirs = sorted(set(run_dirs))
    if limit:
        run_dirs = run_dirs[:limit]

    for bad in bad_roots:
        print(f"REFUSED walk root {bad['root']}: {bad['problem']}", file=sys.stderr)
    for path, kind in walk_errors:
        print(f"walk error {path}: {kind}", file=sys.stderr)
    if apply and walk_errors:
        raise typer.BadParameter(f"{len(walk_errors)} walk error(s): the run list is "
                                 "possibly truncated; nothing deleted")

    report = report or (ROOT / "run_logs" / "thin_ckpt" /
                        f"{'controls_' if control_tree else ''}report_"
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    log_path = report.parent / (report.stem + ".deleted.jsonl")

    rows: list[dict] = []
    n_seen = 0
    census_ok = threading.Event()
    if not apply or not expect_runs:
        census_ok.set()

    def write_report(*, final: bool = False) -> None:
        """Snapshot the rows to the report file; never fatal, never two at once.

        The report is a SNAPSHOT -- the deletion jsonl is the durable log -- so a periodic
        flush that cannot take the lock (another thread is mid-flush) is skipped, and one
        that fails is printed and retried at the next boundary. The final write is the one
        that must land: it blocks for the lock and lets its errors surface.
        """
        if not _REPORT_LOCK.acquire(blocking=final):
            return
        try:
            _write_report_now()
        except Exception as exc:  # noqa: BLE001 -- a snapshot is not worth the invocation
            if final:
                raise
            print(f"report flush failed ({type(exc).__name__}: {exc}); retrying at the "
                  "next flush", file=sys.stderr, flush=True)
        finally:
            _REPORT_LOCK.release()

    def _write_report_now() -> None:
        with _ROW_LOCK:
            # `tok_removable` is the per-file worklist (tens of files per run); the report
            # keeps its counts, not the list.
            snapshot = [{k: v for k, v in r.items() if k != "tok_removable"} for r in rows]
        snapshot.sort(key=lambda r: r["run_dir_rel"])
        atomic_write_json(report, {
            "kind": "thin_checkpoint_ladders",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(out_root), "only_tree": only_tree, "limit": limit,
            "applied": apply, "receivers": receivers, "teachers": teachers,
            "keep_epochs": cfg["keep_epochs"],
            "keep_epochs_optional": cfg["keep_epochs_optional"],
            "thin_protected_sets": list(cfg["thin_protected_sets"]),
            "require": cfg["require"], "include_same_base": include_same_base,
            "teacher_tree": teacher_tree, "control_tree": control_tree,
            "wipe_set_cells": wipe_set_cells, "exclude_prefixes": list(excludes),
            "failed": [{"run_dir_rel": r["run_dir_rel"], "status": r["status"],
                        "detail": r.get("detail")}
                       for r in snapshot if r["status"].startswith("failed:")],
            "archive_root": str(cfg["archive_root"]),
            "strip_tokenizers": strip_tokenizers,
            "strip_only": strip_only, "strip_protected": strip_protected,
            "tokenizer_sha": tokenizer_sha,
            "protect_file": str(protect_file),
            "cell_protect_file": str(cell_protect_file), "refused_roots": bad_roots,
            "walk_errors": [[str(p), k] for p, k in walk_errors],
            "unfillable": unfillable_rows(snapshot), "runs": snapshot,
        })

    def act(row: dict) -> None:
        """The two independent --apply actions of one run, in order: thin, then strip.

        The tokenizer strip is deliberately NOT conditioned on the thinning: it removes only
        bytes proven identical to the run's root copy, so it runs for a run whose records
        are incomplete too (and is the whole job under --strip-only).
        """
        try:
            if not strip_only and row["status"].startswith("ready"):
                status = thin_run(row, cfg, log_path)
                with _ROW_LOCK:
                    row["status"] = status
            if cfg["strip_tokenizers"] and row.get("tok_files"):
                tok = strip_tokenizers_run(row, cfg, log_path)
                with _ROW_LOCK:
                    row["tok_status"] = tok
        except Exception as exc:  # noqa: BLE001 -- one failed run must not stop the pass
            if not control_tree:
                raise
            # 4,000 runs against a disk at 100%: one run whose action raises is a row to
            # report, not the end of the invocation. The report lists them under `failed`.
            with _ROW_LOCK:
                row["status"] = f"failed:{type(exc).__name__}"
                row["detail"] = str(exc)

    def handle(run_dir: Path) -> dict:
        nonlocal n_seen
        row = inspect_run(run_dir, out_root, cfg)
        add_tokenizer_plan(run_dir, row, cfg)
        with _ROW_LOCK:
            rows.append(row)
            n_seen += 1
            progress, flush, seen = (n_seen % PROGRESS_EVERY == 0,
                                     apply and n_seen % FLUSH_EVERY == 0, n_seen)
        if progress:
            print(f"inspected {seen}/{len(run_dirs)}", file=sys.stderr, flush=True)
        if apply and census_ok.is_set():
            act(row)
        if flush:
            write_report()
        return row

    if apply and expect_runs:
        # Census guard: the in-scope count has to be classified BEFORE anything is deleted,
        # so the scan runs once read-only and the apply pass follows.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(handle, run_dirs))
        scan_errors = [r for r in rows if r["status"] == "scan_error"]
        for row in scan_errors:
            print(f"scan error {row['run_dir_rel']}: {row.get('detail')}", file=sys.stderr)
        if scan_errors:
            raise typer.BadParameter(f"{len(scan_errors)} run(s) failed to classify; "
                                     "nothing deleted")
        counted = census_rows(rows)
        if abs(len(counted) - expect_runs) > CENSUS_TOLERANCE * expect_runs:
            raise typer.BadParameter(
                f"census guard: {len(counted)} in-scope runs vs --expect-runs "
                f"{expect_runs} (>{CENSUS_TOLERANCE:.0%} apart); nothing deleted")
        census_ok.set()
        actionable = [r for r in rows
                      if r["status"].startswith("ready") or r.get("tok_files")]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for _ in pool.map(act, actionable):
                done += 1
                if done % FLUSH_EVERY == 0:
                    write_report()
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(handle, run_dirs))

    rows.sort(key=lambda r: r["run_dir_rel"])
    write_report(final=True)

    missing_rows = [r for r in rows if r["status"].startswith("missing:")]
    if emit_missing_jobs:
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        jobs, chunk_dir = write_spectra_jobs(missing_rows, stamp, jobs_chunk)
        emit_missing_jobs.parent.mkdir(parents=True, exist_ok=True)
        emit_missing_jobs.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        print(f"\nspectra jobs -> {emit_missing_jobs} ({len(jobs)} chunks in {chunk_dir})",
              file=sys.stderr)
    unfillable = unfillable_rows(missing_rows)
    if unfillable:
        print(f"unfillable (pref/pile-KL missing, cell-addressed): {len(unfillable)} runs",
              file=sys.stderr)

    print(summarize(rows), file=sys.stderr)
    if control_tree:
        print(control_summary(rows), file=sys.stderr)
    print(f"\nreport: {report}", file=sys.stderr)
    if not apply:
        print("DRY RUN: nothing deleted (pass --apply).", file=sys.stderr)


def _receiver_roots(out_root: Path, receivers: re.Pattern) -> list[Path]:
    """`students/<animal>/<receiver>` dirs whose receiver matches, the walk's entry points.

    Rooting the walk at the receivers (rather than `students/`) keeps the CephFS scan off
    the cross-family trees entirely; the per-run scope check re-applies the same regex.
    """
    roots = []
    students = out_root / "students"
    for animal in sorted(p for p in students.glob("*") if p.is_dir()):
        for receiver in sorted(p for p in animal.glob("*") if p.is_dir()):
            if receivers.match(receiver.name):
                roots.append(receiver)
    return roots


if __name__ == "__main__":
    app()
