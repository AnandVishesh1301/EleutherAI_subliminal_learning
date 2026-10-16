"""Per-checkpoint trait trajectories of the core-set student arms, aggregated by condition.

`notebooks/transfer_trajectory_conditions.ipynb` asks whether same-model subliminal transfer
peaks early in student training and decays while cross-model transfer does not, on Pythia-160m
and OLMo-1B together. Everything that notebook shows is computed here:

* `default_arms` -- the arm list, resolved from the SAME registries the transfer tables use
  (`run_logs/registry/{animal}_top10_kband_configs.json` for the Pythia seed families, the
  xarch / OLMo significance registries for the rest), never from directory globs.
* `load_arms` -- one row per designed (teacher x cell) run with its trajectory, its declared
  endpoint (``train_manifest.json``), its diverged marker and its anchored dead label at the
  endpoint, plus the long frame of ``test p_target / receiver prior`` per saved checkpoint.
* `inventory_table`, `arm_curves`, `pooled_curves`, `summary_table` -- the aggregation.
* `plot_arm_curves`, `plot_pooled` -- the two figures.

Aggregation rule: per arm, the median over its ALIVE (teacher x cell) runs of ``test p_target /
receiver prior`` at each saved checkpoint (absolute optimizer steps); per condition line, the
median over that line's per-arm medians (every arm equal weight), band = IQR over arms.
Training progress is always an absolute step, never a fraction of training.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.analysis.frames import safe_read_json
from subliminal_learning.analysis.trajectories import lineage_long_frame
from subliminal_learning.param_tags import parse_param_tag

#: Pooled condition lines, in display order: ``(group, family)``.
GROUPS: list[tuple[str, str]] = [
    ("Pythia same-model", "Pythia"),
    ("Pythia cross-seed", "Pythia"),
    ("Pythia cross-size/arch.", "Pythia"),
    ("OLMo same-model", "OLMo"),
    ("OLMo cross-seed", "OLMo"),
    ("OLMo cross-corpus", "OLMo"),
    ("OLMo cross-arch.", "OLMo"),
]
#: Group of the inventory-only cross-data-order arm (never a curve).
CDO_GROUP = "Pythia cross-data-order"

#: An arm enters the curves only when at least this share of its designed runs has a trajectory.
MIN_COVERAGE = 0.9
#: ... and its trajectories have at least this many distinct saved steps (median over runs).
MIN_POINTS = 15
#: Early-peak cut-off (absolute steps) by receiver family: 25,000-step Pythia runs, 12,500-step
#: OLMo runs.
EARLY_CUTOFF = {"Pythia": 5000, "OLMo": 2500}
#: The common checkpoint save grid (steps) of every arm; curves and peaks read only grid steps,
#: so a run saved on a finer ladder never adds steps the rest of its arm lacks.
SAVE_GRID = 625
#: Fixed step the summary reads the curves at.
PROBE_STEP = 2500

STATUS_IN = "in curves"
STATUS_NO_TRAJ = "excluded: no trajectory"
STATUS_COARSE = "excluded: coarse cadence"
STATUS_PARTIAL = "excluded: partial coverage"
STATUS_NO_PRIOR = "excluded: no receiver prior"

_SEED_SHORT = {"seed": "fs", "data-seed": "ds", "weightc-seed": "wc", "pilerepro-seed": "pr"}
_CORPUS = {"dolma1_7": "dolma", "dclm-baseline": "dclm", "c4": "c4"}


#: Display names of short base tokens in notebook text only; on-disk names are unchanged.
#: Shared by the trajectory A→B-vs-A→A and asymmetry notebooks so they cannot drift.
DISPLAY_NAMES = {"prv1": "Rerun1234", "pr1234": "Modern1234"}
_DISPLAY_RE = re.compile(r"\b(" + "|".join(map(re.escape, DISPLAY_NAMES)) + r")\b")


def display(text: str) -> str:
    """`text` with every short base token in `DISPLAY_NAMES` replaced by its display name."""
    return _DISPLAY_RE.sub(lambda m: DISPLAY_NAMES[m.group(1)], text)


def short_base(base: str, *, family_context: bool = True) -> str:
    """Display short of an on-disk base dir.

    With `family_context` (both ends are Pythia-160m lineages) ``pythia-160m`` -> ``orig`` and
    ``pythia-160m-seed{k}`` -> ``fs{k}`` (``data-seed`` -> ``ds``, ``weightc-seed`` -> ``wc``,
    ``pilerepro-seed`` -> ``pr``, ``pilereprov1-seed1234`` -> ``prv1``); otherwise the size is spelled out (``160m``,
    ``160m-fs1``, ``410m-seed1``) and a DataDecide base reads ``OLMo-{corpus}``. Between two
    DataDecide bases the corpus alone is used (``dolma``, ``c4-aux2``).
    """
    if base == "pythia-160m-pilereprov1-seed1234":
        return "prv1" if family_context else "160m-prv1"
    m = re.fullmatch(r"pythia-(\d+m)(?:-(seed|data-seed|weightc-seed|pilerepro-seed)(\d+))?", base)
    if m:
        size, kind, k = m.groups()
        if size == "160m":
            tok = "orig" if not kind else f"{_SEED_SHORT[kind]}{k}"
            return tok if family_context else ("160m" if not kind else f"160m-{tok}")
        return size if not kind else f"{size}-seed{k}"
    m = re.fullmatch(r"DataDecide-(.+?)-1B-seed-(default|large-aux-(\d+))", base)
    if m:
        corpus = _CORPUS.get(m.group(1), m.group(1))
        tok = corpus if m.group(2) == "default" else f"{corpus}-aux{m.group(3)}"
        return tok if family_context else f"OLMo-{tok}"
    return base


@dataclass(frozen=True)
class ArmSpec:
    """One (animal, sender, receiver) student arm and its design.

    Attributes:
        family: Receiver-side model family of the pooled line (``Pythia`` / ``OLMo``).
        group: Pooled condition line (`GROUPS`, or `CDO_GROUP`).
        condition: Finer condition label shown in the tables.
        animal, sender, receiver: Trait animal and on-disk base dir names.
        teachers: Canonical teacher param_tags (from the registry the transfer tables use).
        cell_set: Name of the cell set; `cells` its cell names.
        tseed, dataset_tag: Teacher seed and numbers-dataset dir.
        teacher_dataset_tags: ``(tag, dataset_tag)`` overrides for single teachers.
        probe_pool: Explicit dead-label probe pool (None = the receiver's native pool).
        expected_prior: Registry prior to cross-check the loaded prior against (None = none).
        kind: ``treatment`` (teacher numbers) or ``control`` (base-model numbers, no teacher
            level: `sender` is then the numbers base dir and `teachers` the one placeholder
            `CONTROL_TEACHER`).
        control_of: For a control arm, the label of the treatment arm it is matched to.
        min_points: Curve cadence floor of this arm (None = `MIN_POINTS`).
    """

    family: str
    group: str
    condition: str
    animal: str
    sender: str
    receiver: str
    teachers: tuple[str, ...]
    cell_set: str
    cells: tuple[str, ...]
    tseed: int = 1
    dataset_tag: str = tio.DEFAULT_DATASET_TAG
    teacher_dataset_tags: tuple[tuple[str, str], ...] = ()
    probe_pool: str | None = None
    expected_prior: float | None = None
    kind: str = "treatment"
    control_of: str | None = None
    min_points: int | None = None

    @cached_property
    def label(self) -> str:
        """``sender→receiver (animal)`` in the shorts of `short_base`, plus `` · per-checkpoint``
        for an arm of ep20 intermediate-checkpoint cells and `` · control`` for a base-numbers control arm."""
        ctx = self.sender.startswith("pythia-160m") and self.receiver.startswith("pythia-160m")
        ctx = ctx or (self.sender.startswith("DataDecide-") and self.receiver.startswith("DataDecide-"))
        return (f"{short_base(self.sender, family_context=ctx)}→"
                f"{short_base(self.receiver, family_context=ctx)} ({self.animal})"
                + _stop_note(self) + (" · control" if self.kind == "control" else ""))

    def lineage_dir(self, root: Path, teacher: str) -> Path:
        """The ``{dataset_tag}`` dir holding this teacher's ``{cell}/ftseed{N}`` leaves (a
        control arm: ``students/controls/base/{receiver}/teacher-{numbers base}/{dataset_tag}``,
        whatever `teacher`)."""
        control_tree, sender_tree, _, _ = tio._arm_trees(
            root, self.animal, self.sender, self.receiver, tseed=self.tseed,
            dataset_tag=self.dataset_tag)
        if self.kind == "control":
            return control_tree
        dtag = dict(self.teacher_dataset_tags).get(teacher, self.dataset_tag)
        return sender_tree / f"{teacher}_tseed{self.tseed}" / dtag


# ----------------------------------------------------------------------------------------------
# arm enumeration
# ----------------------------------------------------------------------------------------------
_XARCH = "run_logs/campaigns/xarch/xarch_significance_arms.json"
_OLMO = "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json"

#: xarch registry arms: ``name -> condition`` (the first six enter the curves when complete).
XARCH_ARMS = {
    "410m → 160m (dog)": "cross-size",
    "410m-seed1 → 160m (dog)": "cross-size (diff. init)",
    "160m → 410m (dog)": "cross-size",
    "160m → 410m-seed1 (dog)": "cross-size (diff. init)",
    "OLMo-1B → 160m (dog), core30": "cross-architecture",
    "160m → OLMo-1B (dog), core35": "cross-architecture",
    "410m-seed2 → 160m (dog)": "cross-size (diff. init)",
    "410m-seed2 → 160m-seed1 (dog)": "cross-size (diff. init)",
    "410m-seed2 → 160m-seed2 (dog)": "cross-size (diff. init)",
}

#: OLMo registry arms: ``name -> (group, condition)`` (the `transfer_conditions` OLMo blocks).
OLMO_ARMS = {
    **{f"{c} → {c} same-model (elephant), core35": ("OLMo same-model", "same-model")
       for c in ("dolma", "c4", "dclm")},
    **{f"{c} → {c}-seed2 (elephant), core35": ("OLMo cross-seed", "cross-seed")
       for c in ("dolma", "c4", "dclm")},
    **{f"{a} → {b} (elephant), core35": ("OLMo cross-corpus", "cross-corpus")
       for a, b in (("dolma", "dclm"), ("dclm", "dolma"), ("dclm", "c4"), ("dolma", "c4"),
                    ("c4", "dolma"), ("c4", "dclm"))},
    **{f"{a} → {b} (elephant), core35": ("OLMo cross-corpus", "cross-corpus + cross-seed")
       for a, b in (("dclm", "seed2"), ("dclm", "c4-seed2"), ("dolma", "dclm-seed2"),
                    ("dolma", "c4-seed2"), ("c4", "seed2"), ("c4", "dclm-seed2"))},
}


def _cell_set_name(path: Path) -> str:
    return path.stem.replace("_cells", "").replace("olmo_", "")


def _registry_specs(repo: Path, registry: str, names: dict, *, group_of, family_of) -> list[ArmSpec]:
    path = repo / registry
    doc = json.loads(path.read_text())
    by_name = {a["name"]: a for a in doc["arms"]}
    out = []
    for name, cond in names.items():
        a = by_name[name]
        cells_path = (path.parent / a["cells"]).resolve()
        group, condition = group_of(cond)
        out.append(ArmSpec(
            family=family_of(a), group=group, condition=condition, animal=a["animal"],
            sender=tio.base_name(a["sender"]), receiver=tio.base_name(a["receiver"]),
            teachers=tuple(a["teachers"]), cell_set=_cell_set_name(cells_path),
            cells=tuple(tio.resolve_cells(cells_path)), tseed=int(a["tseed"]),
            dataset_tag=a.get("dataset_tag", tio.DEFAULT_DATASET_TAG),
            teacher_dataset_tags=tuple(sorted(a.get("teacher_dataset_tags", {}).items())),
            probe_pool=a.get("probe_pool"), expected_prior=a.get("expected_prior")))
    return out


def default_arms(repo: Path) -> list[ArmSpec]:
    """Every arm of the notebook, in display order (curve arms and inventory-only arms).

    Pythia seed families (core56, teachers = ``{animal}_top10_kband_configs.json``): the
    ``orig/fs1-3`` diagonals and all 12 ordered off-diagonal cross-seed pairs, for dog, cat,
    lion and pig; the dog ``orig → ds1`` cross-data-order arm on core30 (inventory only). The
    cross-size / cross-architecture arms come from the xarch registry (plus the ``410m →
    fs{k}`` / ``410m-seed1 → fs{k}`` ladders, teachers from the ``p410m_top10_senders_*``
    manifests), the OLMo arms from the OLMo significance registry. Whether an arm enters the
    curves is decided by its coverage (`inventory_table`), not here.
    """
    repo = Path(repo)
    reg = repo / "run_logs" / "registry"
    core56, core30 = reg / "core56_cells.json", reg / "core30_cells.json"
    animals = ["dog", "cat", "lion", "pig"]
    teachers = {a: tuple(json.loads((reg / f"{a}_top10_kband_configs.json").read_text())
                         ["param_tags"]) for a in animals}

    def fam(arms, group, condition, cells_path, animal_list):
        cells = tuple(tio.resolve_cells(cells_path))
        return [ArmSpec("Pythia", group, condition, an, tio.base_name(x.sender),
                        tio.base_name(x.receiver), teachers[an], _cell_set_name(cells_path),
                        cells) for an in animal_list for x in arms]

    bases = tio.FAMILY_BASES["cross-seed"]                   # orig, fs1, fs2, fs3
    out = fam(tio.same_model_arms(bases), "Pythia same-model", "same-model", core56, animals)
    out += fam(tio.family_arm_set("cross-seed"), "Pythia cross-seed", "cross-seed", core56,
               animals)
    out += fam([tio.FamilyArm("cross-data-order", "orig", "ds1")], CDO_GROUP,
               "cross-data-order", core30, ["dog"])
    # pooled lines never mix run lengths, which are a RECEIVER property: the OLMo-receiver
    # (12,500-step) cross-architecture arm is its own OLMo line
    out += [replace(s, group="OLMo cross-arch.") if s.family == "OLMo" else s
            for s in _registry_specs(
                repo, _XARCH, XARCH_ARMS, group_of=lambda c: ("Pythia cross-size/arch.", c),
                family_of=lambda a: "OLMo" if tio.base_name(a["receiver"]).startswith(
                    "DataDecide-") else "Pythia")]
    for sender in ("pythia-410m", "pythia-410m-seed1"):
        man = json.loads((reg / f"p410m_top10_senders_{sender}.json").read_text())
        for k in (1, 2, 3):
            out.append(ArmSpec(
                "Pythia", "Pythia cross-size/arch.", "cross-size (diff. init)", "dog", sender,
                f"pythia-160m-seed{k}", tuple(man["param_tags"]), "core30",
                tuple(tio.resolve_cells(core30)),
                teacher_dataset_tags=tuple(sorted(man.get("dataset_tags", {}).items()))))
    out += _registry_specs(repo, _OLMO, OLMO_ARMS, group_of=lambda c: c,
                           family_of=lambda a: "OLMo")
    return out


# ----------------------------------------------------------------------------------------------
# per-checkpoint readouts of ep20 runs (saved steps 625 ... 25,000) + base-numbers controls
# ----------------------------------------------------------------------------------------------
STOP_SAME_GROUP = "Pythia same-model · per-checkpoint"
STOP_CROSSSEED_GROUP = "Pythia cross-seed · per-checkpoint"
STOP_CDO_GROUP = "Pythia cross-data-order · per-checkpoint"
STOP_CROSS_GROUP = "Pythia re-pretrain · per-checkpoint"
#: Bases of the 160m cross-seed factorial: a cross arm between two of them is cross-seed.
_CROSSSEED_BASES = frozenset(tio.base_name(b) for b in ("orig", "fs1", "fs2", "fs3"))
#: Bases of the 160m cross-data-order factorial.
_CDO_BASES = frozenset(tio.base_name(b) for b in ("orig", "ds1", "ds2", "ds3"))
CONTROLS_SUFFIX = " controls"
_STOP_TREAT_GROUPS = (STOP_SAME_GROUP, STOP_CROSSSEED_GROUP, STOP_CDO_GROUP, STOP_CROSS_GROUP)
#: The per-checkpoint condition lines (treatment, then their matched controls), in display order.
STOP_GROUPS: list[tuple[str, str]] = [
    *((g, "Pythia") for g in _STOP_TREAT_GROUPS),
    *((g + CONTROLS_SUFFIX, "Pythia") for g in _STOP_TREAT_GROUPS),
]
#: Saved intermediate steps of these runs (625 ... 3,750, 6,250, 12,500): curve cadence floor.
STOP_MIN_POINTS = 8
#: Teacher placeholder of a control arm (no teacher level in its leaves).
CONTROL_TEACHER = "base"

_ES_SAME = "run_logs/campaigns/samemodel_earlystop_dog_20261010"
_ES_PR = "run_logs/campaigns/pr1234_cross_earlystop_20261010/registry"
_ES_FS = "run_logs/campaigns/fsfact_cross_earlystop_dog_20261012/registry"
_ES_TF = "run_logs/campaigns/traj_factorials_20261013/registry"
#: Full-length dog prv1 <-> pr1234 runs: PLAIN cells (no ``_stop{N}``), each leaf's pref_traj
#: carries the saved steps and the 25,000 endpoint, its controls the same steps.
_ES_FT = "run_logs/campaigns/prv1_pr1234_fulltraj_20261015/registry"
#: The 12 ordered orig/fs1-3 cross arms of the dog fsfact per-checkpoint campaign.
_ES_FS_ARMS = ("orig2fs1es", "orig2fs2es", "orig2fs3es", "fs12origes", "fs12fs2es", "fs12fs3es",
               "fs22origes", "fs22fs1es", "fs22fs3es", "fs32origes", "fs32fs1es", "fs32fs2es")
#: traj_factorials_20261013 arms: cat/lion cross-seed (no controls.json: their control leaves are
#: the dog registries' shared ones), dog cross-data-order, dog prv1/wc/orig re-pretrain. The
#: prv1 ⇄ pr1234 dirs of that campaign are parked and not listed: that pair is `_ES_FT`.
_ES_TF_ARMS = ("cat-fs12fs1es", "lion-orig2origes", "lion-fs12fs1es", "lion-orig2fs1es",
               "lion-orig2fs2es", "lion-orig2fs3es", "lion-fs12origes", "lion-fs12fs2es",
               *(f"{s}2{r}es" for s in ("orig", "ds1", "ds2", "ds3")
                 for r in ("orig", "ds1", "ds2", "ds3") if s != r),
               "prv12wc1es", "prv12wc2es", "wc12prv1es", "wc12wc2es", "wc22prv1es", "wc22wc1es",
               "prv12prv1es", "prv12origes", "orig2prv1es")
#: Registry dirs of the per-checkpoint campaigns, in display order. Registries naming the same
#: (animal, sender, receiver) are one arm (pr1234 same-model = 10-cell pilot + 20-cell es2).
EARLYSTOP_REGISTRIES: tuple[str, ...] = (
    *(f"{_ES_SAME}/{r}/registry" for r in
      ("orig", "ds1", "ds2", "ds3", "fs1", "fs2", "fs3", "wc1", "wc2")),
    "run_logs/campaigns/pr1234_samemodel_earlystop_20261010/registry",
    f"{_ES_PR}/pr1234es2", f"{_ES_PR}/p2oes", f"{_ES_PR}/o2pes",
    *(f"{_ES_FS}/{a}" for a in _ES_FS_ARMS),
    *(f"{_ES_TF}/{a}" for a in _ES_TF_ARMS),
    f"{_ES_FT}/prv12pr1234ft", f"{_ES_FT}/pr12342prv1ft",
)
#: Design completion of the cross-seed factorial (`complete_factorial`): animals whose ordered
#: orig/fs1-3 pairs without a registry are read from their plain ep20 leaves on the stride-3 cells.
COMPLETE_ANIMALS = ("cat", "lion", "pig")
#: Registry whose cells are the stride-3 core30 cells of the completed arms.
_STRIDE3_REGISTRY = f"{_ES_SAME}/orig/registry"


def _stop_group(sender: str, receiver: str) -> str:
    if sender == receiver:
        return STOP_SAME_GROUP
    if {sender, receiver} <= _CROSSSEED_BASES:
        return STOP_CROSSSEED_GROUP
    if {sender, receiver} <= _CDO_BASES:
        return STOP_CDO_GROUP
    return STOP_CROSS_GROUP


def complete_factorial(have: set[tuple[str, str, str]], animals: Sequence[str],
                       bases: Sequence[str], cells: Sequence[str],
                       teachers: dict[str, tuple[str, ...]]) -> dict[tuple[str, str, str], dict]:
    """Treatment entries of every ordered ``(animal, sender, receiver)`` of `bases` (base dir
    names, diagonals included) for `animals` that is not in `have`: the animal's `teachers`,
    tseed 1, `cells` (``_stop{N}`` tags; their points come from the plain leaves, `load_arm`)."""
    return {(an, s, r): {"teachers": tuple(teachers[an]), "animal": an, "tseed": 1,
                         "cells": list(cells)}
            for an in animals for s in bases for r in bases if (an, s, r) not in have}


def earlystop_arms(repo: Path, registries: Sequence[str] = EARLYSTOP_REGISTRIES, *,
                   complete: Sequence[str] = COMPLETE_ANIMALS,
                   stride3_registry: str = _STRIDE3_REGISTRY) -> list[ArmSpec]:
    """The per-checkpoint treatment arms (ep20 runs' saved checkpoints), then their matched
    controls, from campaign registries plus the cross-seed design completion.

    Each registry dir holds one ``{teacher_tag}.json`` per teacher (``lineage``: animal, teacher
    base dir, ``student_base``, ``teacher_ftseed``; ``cells``: ``_stop{N}`` cell tags) and
    optionally one ``controls.json`` (``lineage.source == "base"``, ``numbers_base_dir``).
    Treatment arms are keyed by ``(animal, sender, receiver)``; control leaves are shared by
    every animal of a ``(sender, receiver)`` (multi-animal readouts), so each treatment arm gets
    ONE control arm carrying the treatment's animal, from whichever registry holds that
    lineage, with ``control_of`` = the treatment arm's label. For each animal of `complete`,
    every ordered orig/fs1-3 pair with no registry gets an arm on the cells of
    `stride3_registry` with that animal's ``{animal}_top10_kband_configs.json`` teachers
    (`complete_factorial`). Groups: `_stop_group`. Never reads a directory glob of the outputs
    tree.

    Raises:
        ValueError: If registries merged into one arm disagree on teachers / animal / tseed, a
            treatment arm has no control lineage in any registry, or a control lineage has no
            treatment arm.
    """
    repo = Path(repo)
    treat: dict[tuple[str, str, str], dict] = {}
    ctrl: dict[tuple[str, str], dict] = {}
    for rel in registries:
        d = repo / rel
        files = sorted(d.glob("*.json"))
        teachers, cells, animal, tseeds = [], [], set(), set()
        key_t = None
        for f in files:
            doc = json.loads(f.read_text())
            lin = doc["lineage"]
            receiver = tio.base_name(lin["student_base"])
            if lin.get("source") == "base":
                key = (tio.base_name(lin["numbers_base_dir"]), receiver)
                e = ctrl.setdefault(key, {"cells": []})
                e["cells"] += [c for c in doc["cells"] if c not in e["cells"]]
                continue
            key_t = (lin["animal"], tio.base_name(lin["teacher_base_dir"]), receiver)
            teachers.append(lin["teacher_tag"])
            cells.append(tuple(doc["cells"]))
            animal.add(lin["animal"])
            tseeds.add(int(lin.get("teacher_ftseed", 1)))
        if key_t is None:
            continue
        if len(set(cells)) != 1 or len(animal) != 1 or len(tseeds) != 1:
            raise ValueError(f"{d}: teachers disagree on cells / animal / teacher_ftseed")
        e = treat.setdefault(key_t, {"teachers": tuple(teachers), "animal": animal.pop(),
                                     "tseed": tseeds.pop(), "cells": []})
        if e["teachers"] != tuple(teachers):
            raise ValueError(f"{d}: teachers differ from the other registries of {key_t}")
        e["cells"] += [c for c in cells[0] if c not in e["cells"]]
    if complete:
        reg = repo / "run_logs" / "registry"
        cells = next(tuple(json.loads(f.read_text())["cells"])
                     for f in sorted((repo / stride3_registry).glob("*.json"))
                     if f.name != "controls.json")
        teach = {a: tuple(json.loads((reg / f"{a}_top10_kband_configs.json").read_text())
                          ["param_tags"]) for a in complete}
        treat.update(complete_factorial(set(treat), complete, sorted(_CROSSSEED_BASES), cells,
                                        teach))
    missing = sorted({(s, r) for _, s, r in treat} - set(ctrl))
    if missing:
        raise ValueError(f"treatment arms with no control lineage in any registry: {missing}")
    out, controls = [], []
    for (animal, sender, receiver), e in treat.items():
        group = _stop_group(sender, receiver)
        spec = ArmSpec("Pythia", group, "same-model" if sender == receiver else "cross-model",
                       animal, sender, receiver, e["teachers"], "earlystop",
                       tuple(e["cells"]), tseed=e["tseed"], min_points=STOP_MIN_POINTS)
        out.append(spec)
        controls.append(ArmSpec(
            "Pythia", group + CONTROLS_SUFFIX,
            "same-base control" if sender == receiver else "cross-base control", animal,
            sender, receiver, (CONTROL_TEACHER,), "earlystop",
            tuple(ctrl[(sender, receiver)]["cells"]),
            kind="control", control_of=spec.label, min_points=STOP_MIN_POINTS))
    orphan = set(ctrl) - {(s.sender, s.receiver) for s in out}
    if orphan:
        raise ValueError(f"control lineages with no treatment arm: {sorted(orphan)}")
    return out + controls


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
RUN_COLUMNS = ["arm", "teacher", "cell", "n_points", "has_traj", "diverged", "dead",
               "end_step", "state"]
POINT_COLUMNS = ["arm", "teacher", "cell", "step", "ratio"]


@dataclass
class TrajectoryData:
    """Everything `load_arms` read.

    Attributes:
        arms: The specs, in order.
        runs: One row per designed run (`RUN_COLUMNS`); ``state`` is ``alive`` / ``dead`` /
            ``diverged`` / ``unlabelled`` / ``no trajectory``.
        points: Long frame (`POINT_COLUMNS`), ``ratio = test p_target / receiver prior``, one row
            per (run, saved step), every run with a trajectory (all states).
        priors: ``{arm label: receiver prior}`` (NaN when missing).
        notes: Loader notes (missing anchored cuts, prior mismatches, unreadable probes).
        root: The outputs tree read (`load_arms`); needed by ``asymmetry_table(endpoint_twins=True)``.
    """

    arms: list[ArmSpec]
    runs: pd.DataFrame
    points: pd.DataFrame
    priors: dict[str, float]
    notes: list[str] = field(default_factory=list)
    root: Path | None = None


def _cell_stop(cell: str) -> int | None:
    """The early-stop step of a ``..._stop{N}`` cell tag (`parse_param_tag`), else None."""
    try:
        return parse_param_tag(cell).get("stop_step")
    except ValueError:
        return None


def _stop_note(spec: ArmSpec) -> str:
    """`` · per-checkpoint`` for an arm of ``_stop{N}`` cells (the ep20 run's intermediate
    checkpoints), else ``""``."""
    return " · per-checkpoint" if any(_cell_stop(c) is not None for c in spec.cells) else ""


def _end_step(run_dir: Path) -> float:
    """The declared final step of a run (``train_manifest.json``), NaN when unreadable.

    An early-stopped run (manifest ``run_identity.train.stop_step``, or a ``_stop{N}`` cell tag
    = the leaf's parent dir) ends at N (= its ``n_train_steps``), not at the planned schedule.
    """
    doc = safe_read_json(run_dir / "train_manifest.json") or {}
    stop = ((doc.get("run_identity") or {}).get("train") or {}).get("stop_step")
    if stop is None:
        stop = _cell_stop(run_dir.parent.name)
    if stop is not None:
        return float(stop)
    v = doc.get("planned_train_steps", doc.get("n_train_steps"))
    return float(v) if v is not None else float("nan")


def _endpoint_kept(spec: ArmSpec, dead: float) -> bool:
    """Whether an endpoint value enters the curves given its Pile-probe label: a treatment run
    must be probed and alive (``dead == 0``); a control run is dropped only when labelled dead
    (``dead == 1``), so an unprobed control endpoint (NaN) is kept -- the rule the significance
    tests apply to the same control files."""
    return dead != 1.0 if spec.kind == "control" else dead == 0


def _run_state(has_traj: bool, diverged: bool, dead: float) -> str:
    if diverged:
        return "diverged"
    if not has_traj:
        return "no trajectory"
    if np.isnan(dead):
        return "unlabelled"
    return "dead" if dead else "alive"


def plain_twin(cell: str) -> str | None:
    """The plain leaf of the same run as a ``_stop{N}`` cell (suffix removed), else None."""
    return re.sub(r"_stop\d+$", "", cell) if _cell_stop(cell) is not None else None


def merge_leaf_points(first: pd.DataFrame | None,
                      second: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    """Union by ``step`` of one run's points read from its two leaves (``step, p_test``
    frames; None = leaf unreadable). On a step both hold, `first`'s value is kept. Returns
    ``(merged frame sorted by step, max |first - second| over shared steps)`` (NaN: none)."""
    parts = [f[["step", "p_test"]].astype(float) for f in (first, second) if f is not None]
    if not parts:
        return pd.DataFrame(columns=["step", "p_test"]), float("nan")
    diff = float("nan")
    if len(parts) == 2:
        both = parts[0].merge(parts[1], on="step", suffixes=("_a", "_b"))
        if len(both):
            diff = float((both["p_test_a"] - both["p_test_b"]).abs().max())
    merged = (pd.concat(parts, ignore_index=True).drop_duplicates("step", keep="first")
              .sort_values("step", ignore_index=True))
    return merged, diff


def run_frames(spec: ArmSpec, teacher: str, root: Path, *, ftseed: int = 1,
               refresh: bool = False) -> dict[str, tuple[pd.DataFrame | None,
                                                          pd.DataFrame | None]]:
    """``cell -> (points of the cell's own leaf, points of its plain twin leaf)`` (``step,
    p_test``, finite only, None when the leaf has none) for one teacher of `spec`. A
    ``_stop{N}`` cell and its plain twin are the two leaves of ONE run; a plain cell has no
    twin."""
    lineage = spec.lineage_dir(root, teacher)
    names = {c: plain_twin(c) for c in spec.cells}
    tags = list(dict.fromkeys([*names, *(t for t in names.values() if t)]))
    registry = {"cells": {c: [ftseed] for c in tags}}
    tr = (lineage_long_frame(registry, lineage, cache_root=root, animal=spec.animal,
                             refresh=refresh)
          if lineage.is_dir() else pd.DataFrame(columns=["tag", "step", "p_test"]))
    tr = tr[np.isfinite(tr["p_test"].astype(float))]
    by = {c: g[["step", "p_test"]] for c, g in tr.groupby("tag")}
    return {c: (by.get(c), by.get(t) if t else None) for c, t in names.items()}


def run_points(spec: ArmSpec, teacher: str, root: Path, *, ftseed: int = 1,
               refresh: bool = False) -> tuple[pd.DataFrame, float]:
    """``(tag, step, p_test)`` of one teacher's runs of `spec`, each run's points the union of
    its two leaves (`merge_leaf_points`, the cell's own leaf wins on shared steps), plus the
    max |difference| on shared steps (NaN: none)."""
    parts, diffs = [], []
    for cell, (own, twin) in run_frames(spec, teacher, root, ftseed=ftseed,
                                        refresh=refresh).items():
        m, d = merge_leaf_points(own, twin)
        diffs.append(d)
        if len(m):
            parts.append(m.assign(tag=cell)[["tag", "step", "p_test"]])
    pts = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["tag", "step", "p_test"]))
    d = np.array(diffs, float)
    return pts, float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")


def load_arm(spec: ArmSpec, root: Path, *, ftseed: int = 1,
             notes: list[str] | None = None,
             refresh: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """``(runs, points, prior)`` of one arm (see `TrajectoryData`).

    A run with at most one finite saved step counts as "no trajectory". The endpoint step
    recorded twice (checkpoint dir + endpoint adapter) is kept once (`lineage_long_frame`). The
    dead label is the anchored ``ΔCE`` cut of the receiver's probe pool read at the run's
    ENDPOINT probe (`transfer_significance_io._dead_label`); with no anchored cut every run is
    unlabelled (never a substitute cut). `refresh=False` skips the per-lineage record-cache
    sweep; pass it only after a same-process `_prefetch` covering `spec` (as `load_arms` does).

    A ``_stop{N}`` cell's run has two leaves (`run_frames`): its points are their union by step
    (`merge_leaf_points`; on shared steps the ``_stop{N}`` leaf's value is kept and the max
    |difference| goes to `notes`). Its alive / diverged label is the ``_stop{N}`` leaf's when
    that leaf has a trajectory, else the plain leaf's; the plain leaf's steps past the other
    leaf's last step are kept only when the plain leaf is alive under the same cut (the
    `with_ep20_endpoints` rule). ``end_step`` is the last kept step's leaf's declared end.
    """
    notes = notes if notes is not None else []
    root = Path(root)
    prior = tio.load_prior(spec.animal, spec.receiver, output_root=root)
    if np.isfinite(prior) and spec.expected_prior is not None:
        try:
            tio.check_expected_prior(spec.label, prior, spec.expected_prior)
        except ValueError as exc:
            notes.append(f"{spec.label}: {exc} (loaded prior kept)")
    pool = tio._probe_pool_for(spec.receiver, spec.probe_pool)
    cut = tio._anchored_cut(root, spec.receiver, pool, dead="keep", notes=notes)
    runs, points, diffs = [], [], []
    n_unprobed = 0
    for teacher in spec.teachers:
        lineage = spec.lineage_dir(root, teacher)
        frames = run_frames(spec, teacher, root, ftseed=ftseed, refresh=refresh)
        for cell in spec.cells:
            own, twin = frames[cell]
            own_dir = lineage / cell / f"ftseed{ftseed}"
            twin_cell = plain_twin(cell)
            twin_dir = lineage / twin_cell / f"ftseed{ftseed}" if twin_cell else None
            has_own = own is not None and own["step"].nunique() > 1
            run_dir = own_dir if has_own or twin is None else twin_dir
            label_parent = run_dir.parent
            if run_dir is own_dir and twin is not None and len(twin):
                # the plain leaf extends past the saved steps only when alive at its endpoint
                last = float(own["step"].max()) if own is not None and len(own) else -1.0
                tail = twin[twin["step"].astype(float) > last]
                ok = len(tail) and cut is not None and not (twin_dir / tio.DIVERGED_MARKER
                                                            ).exists()
                if ok:
                    try:
                        tdead = tio._dead_label(twin_dir.parent, ftseed, pool, cut)
                        ok = _endpoint_kept(spec, tdead)
                        n_unprobed += bool(ok and np.isnan(tdead))
                    except ValueError:
                        ok = False
                if not ok:
                    twin = twin[twin["step"].astype(float) <= last]
            g, d = merge_leaf_points(own, twin)
            diffs.append(d)
            n = int(g["step"].nunique())
            has = n > 1
            diverged = (run_dir / tio.DIVERGED_MARKER).exists()
            dead, end = np.nan, np.nan
            if has:
                tail_from_twin = (twin is not None and len(twin)
                                  and float(twin["step"].max()) >= float(g["step"].max())
                                  and (own is None or float(twin["step"].max())
                                       > float(own["step"].max())))
                end = _end_step(twin_dir if tail_from_twin else run_dir)
                if cut is not None:
                    try:
                        dead = tio._dead_label(label_parent, ftseed, pool, cut)
                    except ValueError as exc:
                        notes.append(f"{spec.label}: {exc}")
                ratio = g["p_test"].to_numpy(float) / prior
                points.append(pd.DataFrame({"arm": spec.label, "teacher": teacher, "cell": cell,
                                            "step": g["step"].to_numpy(float), "ratio": ratio}))
            runs.append({"arm": spec.label, "teacher": teacher, "cell": cell, "n_points": n,
                         "has_traj": has, "diverged": diverged, "dead": dead, "end_step": end,
                         "state": _run_state(has, diverged, dead)})
    if n_unprobed:
        notes.append(f"{spec.label}: {n_unprobed} control endpoints used unprobed (no Pile-probe "
                     f"label; a control endpoint is dropped only when labelled dead)")
    d = np.array(diffs, float)
    if np.isfinite(d).any():
        notes.append(f"{spec.label}: the two leaves of a run agree on their shared saved steps "
                     f"within max |Δ p_target| = {np.nanmax(d):.2g} "
                     f"({int(np.isfinite(d).sum())} runs)")
    pts = pd.concat(points, ignore_index=True) if points else pd.DataFrame(columns=POINT_COLUMNS)
    return pd.DataFrame(runs, columns=RUN_COLUMNS), pts, prior


def _prefetch(arms: Sequence[ArmSpec], root: Path, ftseed: int) -> None:
    """Refresh the shared trajectory record caches for every designed run in ONE sweep each.

    `lineage_long_frame` refreshes the cache per lineage and rewrites the (hundreds of MB) CSV
    whenever a lineage adds rows; one up-front sweep makes every later per-lineage call a
    no-write hit. After it, per-lineage reads of any covered run may pass ``refresh=False``
    (`trajectories.lineage_long_frame`) and skip the sweep entirely.
    """
    from subliminal_learning.analysis.record_cache import load_record_cache
    from subliminal_learning.analysis.trajectories import _KINDS

    rels = [str(spec.lineage_dir(root, t).relative_to(root) / c / f"ftseed{ftseed}")
            for spec in arms for t in spec.teachers
            for c in dict.fromkeys([*spec.cells, *filter(None, map(plain_twin, spec.cells))])]
    for kind in ("pref", "pile_kl"):
        load_record_cache(root, "json_blob", rels, verbose=False,
                          rel_record=f"eval/{_KINDS[kind][0]}",
                          cache_rel=Path("analysis") / f"record_cache_traj_{kind}.csv")


def load_arms(arms: Sequence[ArmSpec], output_root: Path, *, ftseed: int = 1) -> TrajectoryData:
    """`load_arm` over every spec (ftseed 1 only), concatenated into one `TrajectoryData`."""
    _prefetch(arms, Path(output_root), ftseed)
    notes: list[str] = []
    runs, points, priors = [], [], {}
    for spec in arms:
        r, p, prior = load_arm(spec, output_root, ftseed=ftseed, notes=notes, refresh=False)
        runs.append(r)
        points.append(p)
        priors[spec.label] = prior
    return TrajectoryData(list(arms), pd.concat(runs, ignore_index=True),
                          pd.concat(points, ignore_index=True), priors, notes,
                          Path(output_root))


# ----------------------------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------------------------
def arm_status(spec: ArmSpec, runs: pd.DataFrame, prior: float) -> str:
    """Curve membership of one arm: prior, then trajectory presence, cadence, coverage."""
    if not np.isfinite(prior):
        return STATUS_NO_PRIOR
    with_traj = runs[runs["has_traj"]]
    if with_traj.empty:
        return STATUS_NO_TRAJ
    if with_traj["n_points"].median() < (spec.min_points or MIN_POINTS):
        return STATUS_COARSE
    if len(with_traj) / max(len(runs), 1) < MIN_COVERAGE:
        return STATUS_PARTIAL
    return STATUS_IN


def _cadence(n: pd.Series) -> str:
    if n.empty:
        return "—"
    lo, hi = int(n.min()), int(n.max())
    return str(lo) if lo == hi else f"{lo}–{hi}"


def inventory_table(data: TrajectoryData) -> pd.DataFrame:
    """One row per arm: design, coverage, run states, cadence and curve status.

    Counts: ``designed`` = teachers x cells; ``with trajectory`` = runs with >1 saved step;
    ``alive / dead / unlabelled`` partition the non-diverged runs with a trajectory (anchored
    dead cut at the endpoint); ``diverged`` counts ``train_diverged.json`` over all designed
    runs; ``cadence`` = distinct saved steps per run (min–max over runs with a trajectory).
    """
    rows = []
    for spec in data.arms:
        r = data.runs[data.runs["arm"] == spec.label]
        n_cells_cov = r[r["has_traj"]]["cell"].nunique()
        st = r["state"].value_counts()
        rows.append({
            "family": spec.family, "group": spec.group, "condition": spec.condition,
            "animal": spec.animal, "arm": spec.label,
            "cell set": "ep20 checkpoints" if _stop_note(spec) else spec.cell_set,
            "teachers": len(spec.teachers), "cells (with traj / design)":
                f"{n_cells_cov}/{len(spec.cells)}",
            "runs with traj / designed": f"{int(r['has_traj'].sum())}/{len(r)}",
            "alive": int(st.get("alive", 0)), "dead": int(st.get("dead", 0)),
            "diverged": int(r["diverged"].sum()), "unlabelled": int(st.get("unlabelled", 0)),
            "cadence (saved steps)": _cadence(r.loc[r["has_traj"], "n_points"]),
            "receiver prior": data.priors[spec.label],
            "status": arm_status(spec, r, data.priors[spec.label]),
        })
    return pd.DataFrame(rows)


def _alive_points(data: TrajectoryData, arms: Sequence[str]) -> pd.DataFrame:
    alive = data.runs[(data.runs["state"] == "alive") & data.runs["arm"].isin(arms)]
    keys = alive[["arm", "teacher", "cell"]]
    pts = data.points.merge(keys, on=["arm", "teacher", "cell"])
    pts = pts.astype({"step": float, "ratio": float})
    return pts[pts["step"] % SAVE_GRID == 0]


def curve_arms(data: TrajectoryData) -> list[ArmSpec]:
    """The specs whose `inventory_table` status is `STATUS_IN`."""
    inv = inventory_table(data).set_index("arm")
    return [s for s in data.arms if inv.loc[s.label, "status"] == STATUS_IN]


def arm_curves(data: TrajectoryData, arms: Sequence[ArmSpec] | None = None) -> pd.DataFrame:
    """Per-arm curve: median (and n) over alive runs of the ×prior ratio at each saved step.

    Columns ``arm, group, step, median, n, end_step`` (the arm's declared final step).
    """
    arms = curve_arms(data) if arms is None else list(arms)
    group = {s.label: s.group for s in arms}
    pts = _alive_points(data, list(group))
    if pts.empty:
        return pd.DataFrame(columns=["arm", "group", "step", "median", "n", "end_step"])
    out = (pts.groupby(["arm", "step"])["ratio"].agg(median="median", n="size").reset_index())
    out.insert(1, "group", out["arm"].map(group))
    out["end_step"] = out["arm"].map({a: _arm_end(data, a) for a in group})
    return out


def pooled_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Per-group line: median over arms of the per-arm medians at each step, plus the IQR.

    Every arm has equal weight; at a step only the arms that saved it contribute (``n arms``).
    Columns ``group, step, median, q25, q75, n_arms``.

    Raises:
        ValueError: If one group pools arms with different declared endpoint steps (the arm
            set would change mid-line, a composition artifact).
    """
    ends = curves.groupby(["group", "arm"])["end_step"].first()
    for group, e in ends.groupby(level="group"):
        if e.nunique(dropna=False) > 1:
            arms = ", ".join(f"{a} ({v:,.0f})" for (_, a), v in e.items())
            raise ValueError(f"{group}: pooled arms have different endpoint steps: {arms}")
    g = curves.groupby(["group", "step"])["median"]
    out = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                        "n_arms": g.size()}).reset_index()
    return out


def _at(curve: pd.DataFrame, step: float, col: str = "median") -> float:
    hit = curve.loc[curve["step"] == step, col]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def run_peaks(data: TrajectoryData, arms: Sequence[ArmSpec]) -> pd.DataFrame:
    """Per alive run: peak ×prior (max over its finite saved steps), its step, the endpoint
    value (at the declared final step) and whether the peak lies before the family cut-off."""
    fam = {s.label: s.family for s in arms}
    pts = _alive_points(data, list(fam))
    pts = pts[np.isfinite(pts["ratio"])]
    idx = pts.groupby(["arm", "teacher", "cell"])["ratio"].idxmax()
    peaks = pts.loc[idx, ["arm", "teacher", "cell", "step", "ratio"]].rename(
        columns={"step": "peak_step", "ratio": "peak"})
    ends = data.runs[["arm", "teacher", "cell", "end_step"]]
    peaks = peaks.merge(ends, on=["arm", "teacher", "cell"])
    endv = pts.merge(ends, on=["arm", "teacher", "cell"])
    endv = endv[endv["step"] == endv["end_step"]][["arm", "teacher", "cell", "ratio"]]
    peaks = peaks.merge(endv.rename(columns={"ratio": "end"}), on=["arm", "teacher", "cell"],
                        how="left")
    peaks["early"] = peaks["peak_step"] < peaks["arm"].map(fam).map(EARLY_CUTOFF)
    return peaks.reset_index(drop=True)


def _row(name, condition, curve, runs, end_step, end_value, n_arms) -> dict:
    finite = curve[np.isfinite(curve["median"])]
    pk = finite.loc[finite["median"].idxmax()] if len(finite) else None
    peak_v = float(pk["median"]) if pk is not None else np.nan
    if pk is None:
        peak_step = "—"
    elif pk["step"] == finite["step"].min():   # no interior peak
        peak_step = f"≤{pk['step']:,.0f} (first ckpt)"
    else:
        peak_step = f"{pk['step']:,.0f}"
    q = runs["peak_step"].quantile([0.25, 0.5, 0.75]) if len(runs) else pd.Series(
        [np.nan] * 3, index=[0.25, 0.5, 0.75])
    return {
        "arm / condition": name, "condition": condition, "arms": n_arms,
        "alive runs": len(runs),
        "peak step (agg. curve argmax)": peak_step,
        "per-run peak step: median [IQR]": (f"{q[0.5]:,.0f} [{q[0.25]:,.0f}–{q[0.75]:,.0f}]"
                                            if len(runs) else "—"),
        f"×prior @ step {PROBE_STEP:,}": _at(curve, PROBE_STEP),
        "×prior @ peak": peak_v,
        "endpoint step": end_step if isinstance(end_step, str) else f"{end_step:,.0f}",
        "×prior @ endpoint": end_value,
        "peak / end": peak_v / end_value if end_value else np.nan,
        "frac. runs peak before cut-off": float(runs["early"].mean()) if len(runs) else np.nan,
    }


def _arm_end(data: TrajectoryData, label: str) -> float:
    """The arm's declared final step: the mode of its runs' manifest steps (NaN if none)."""
    m = data.runs.loc[data.runs["arm"] == label, "end_step"].dropna().mode()
    return float(m.iloc[0]) if len(m) else float("nan")


#: Derived column of `summary_table` when control arms are present.
OVER_CONTROL_COL = "×prior @ peak / control @ same step (derived)"


def _over_control(curve: pd.DataFrame, control: pd.DataFrame) -> float:
    """Treatment curve's peak value over the control curve's value at that same step."""
    finite = curve[np.isfinite(curve["median"])]
    if finite.empty or control.empty:
        return float("nan")
    pk = finite.loc[finite["median"].idxmax()]
    return float(pk["median"]) / _at(control, pk["step"])


def summary_table(data: TrajectoryData,
                  groups: Sequence[tuple[str, str]] = GROUPS) -> pd.DataFrame:
    """Arm rows grouped by condition line, each group followed by its pooled ``ALL`` row.

    Arm rows read the arm's own curve (median over alive runs); the endpoint is the arm's
    declared final step (mode over its runs). Pooled rows read the pooled curve (median over
    the per-arm medians); their endpoint value is the median over the arms' endpoint values,
    and their per-run columns pool every alive run of the group's arms.

    When `data` holds control arms, `OVER_CONTROL_COL` is added: for a treatment arm, its
    curve's peak value divided by its control's curve (``control_of``) at the treatment's peak
    step; for a treatment group's ``ALL`` row, the same on the pooled curves of the group and
    its ``... controls`` group; NaN elsewhere (controls, or a control not in the curves).
    """
    arms = curve_arms(data)
    curves = arm_curves(data, arms)
    pooled = pooled_curves(curves)
    peaks = run_peaks(data, arms)
    with_controls = any(s.kind == "control" for s in data.arms)
    control_of = {s.control_of: s.label for s in arms if s.kind == "control"}
    rows = []
    for group, _ in groups:
        members = [s for s in arms if s.group == group]
        if not members:
            continue
        ends = []
        for s in members:
            c = curves[curves["arm"] == s.label]
            r = peaks[peaks["arm"] == s.label]
            end_step = _arm_end(data, s.label)
            end_v = _at(c, end_step)
            ends.append(end_v)
            rows.append(_row(s.label, s.condition + _stop_note(s), c, r, end_step, end_v, 1))
            if with_controls:
                ctl = curves[curves["arm"] == control_of.get(s.label)]
                rows[-1][OVER_CONTROL_COL] = (_over_control(c, ctl) if s.kind != "control"
                                              else np.nan)
        pc = pooled[pooled["group"] == group]
        r = peaks[peaks["arm"].isin([s.label for s in members])]
        rows.append(_row(f"ALL {group}", "pooled", pc, r, _arm_end(data, members[0].label),
                         float(np.nanmedian(ends)), len(members)))
        if with_controls:
            rows[-1][OVER_CONTROL_COL] = (
                np.nan if group.endswith(CONTROLS_SUFFIX)
                else _over_control(pc, pooled[pooled["group"] == group + CONTROLS_SUFFIX]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
GROUP_COLORS = {
    "Pythia same-model": "#D55E00", "Pythia cross-seed": "#0072B2",
    "Pythia cross-size/arch.": "#009E73", "OLMo same-model": "#D55E00",
    "OLMo cross-seed": "#0072B2", "OLMo cross-corpus": "#CC79A7",
    "OLMo cross-arch.": "#009E73",
}


def _log_axes(ax, ylabel: str = "test p_target / receiver prior") -> None:
    from subliminal_learning.analysis import plots as aplots

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 30000)
    ax.set_xticks([625, 1250, 2500, 5000, 12500, 25000])
    ax.set_xticklabels(["625", "1,250", "2,500", "5,000", "12,500", "25,000"])
    aplots.plain_log_ticks(ax, axis="y")
    ax.set_xlabel("optimizer step (absolute, log)")
    ax.set_ylabel(ylabel)


def plot_arm_curves(curves: pd.DataFrame, *, fontsize: int = 10):
    """Small multiples, one panel per condition line, one thin line per arm (named in the
    panel's right-margin legend); the dashed line is the receiver prior (×1)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis import plots as aplots

    groups = [g for g, _ in GROUPS if g in set(curves["group"])]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4.6 * len(groups)),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, groups):
        sub = curves[curves["group"] == group]
        arms = list(dict.fromkeys(sub["arm"]))
        pal = aplots.palette(arms, cmap="tab20" if len(arms) > 10 else "tab10")
        for arm in arms:
            c = sub[sub["arm"] == arm].sort_values("step")
            ax.plot(c["step"], c["median"], lw=1.2, color=pal[arm], label=arm)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        _log_axes(ax)
        ax.set_title(f"{group}: per-arm median over alive runs ({len(arms)} arms)")
        ax.axvline(12500, color="0.6", lw=0.6, ls=":")
        ax.axvline(25000, color="0.6", lw=0.6, ls=":")
        aplots.right_legend(ax, fontsize=fontsize, ncol=2 if len(arms) > 14 else 1)
    return fig


def plot_pooled(pooled: pd.DataFrame, *, fontsize: int = 11):
    """The headline: every condition line on one axes (left, ×prior) and normalised to its own
    peak (right). Pythia solid, OLMo dashed; band = IQR over arms; marker = line peak; dotted
    verticals = run ends (12,500 OLMo, 25,000 Pythia)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis import plots as aplots

    fam = dict(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)
    for k, ax in enumerate(axes):
        for group, _ in GROUPS:
            c = pooled[pooled["group"] == group].sort_values("step")
            if c.empty:
                continue
            ls = "-" if fam[group] == "Pythia" else "--"
            scale = 1.0 if k == 0 else float(c["median"].max())
            col = GROUP_COLORS[group]
            ax.plot(c["step"], c["median"] / scale, ls=ls, lw=2.2, color=col,
                    label=f"{group} ({int(c['n_arms'].max())} arms)")
            ax.fill_between(c["step"], c["q25"] / scale, c["q75"] / scale, color=col,
                            alpha=0.12, lw=0)
            pk = c.loc[c["median"].idxmax()]
            if pk["step"] > c["step"].min():        # ● only for an interior peak
                    ax.plot([pk["step"]], [pk["median"] / scale], marker="o", ms=9, color=col,
                        mec="k", zorder=5)
        for end in (12500, 25000):
            ax.axvline(end, color="0.4", lw=1, ls=":")
        if k == 0:
            ax.axhline(1, color="k", lw=0.8, ls="-.")
            _log_axes(ax)
            ax.set_title("Transfer vs step (median of per-arm medians; band = IQR over arms)")
        else:
            _log_axes(ax, "×prior / line's own peak")
            ax.set_title("Same lines, each normalised to its own peak (peak = 1)")
    aplots.right_legend(axes[1], fontsize=fontsize,
                        title="line (solid Pythia receivers, dashed OLMo receivers;\n● interior peak)")
    return fig


_STOP_TICKS = [625, 1250, 2500, 5000, 12500, 25000]


def _stop_axes(ax, ylabel: str = "test p_target / receiver prior") -> None:
    _log_axes(ax, ylabel)
    ax.set_xlim(560, 28000)
    ax.set_xticks(_STOP_TICKS)
    ax.set_xticklabels([f"{t:,}" for t in _STOP_TICKS])


def plot_stop_arm_curves(curves: pd.DataFrame, arms: Sequence[ArmSpec], *, fontsize: int = 10):
    """Per-checkpoint arms: one panel per (treatment group, animal), one solid line per
    treatment arm and its matched control as a thin dashed line of the same colour (every line
    named in the right-margin legend); the dash-dot line is the receiver prior (×1)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis import plots as aplots

    have = set(curves["arm"])
    ctl = {s.control_of: s.label for s in arms if s.kind == "control"}
    panels = [(g, a) for g in _STOP_TREAT_GROUPS
              for a in dict.fromkeys(s.animal for s in arms if s.kind != "control")
              if any(s.group == g and s.animal == a for s in arms)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4.8 * len(panels)),
                             constrained_layout=True)
    for ax, (group, animal) in zip(np.atleast_1d(axes), panels):
        treat = [s.label for s in arms
                 if s.group == group and s.animal == animal and s.kind != "control"]
        pal = aplots.palette(treat, cmap="tab10")
        for arm in treat:
            for lab, kw in ((arm, {"lw": 1.6, "ls": "-"}),
                            (ctl.get(arm), {"lw": 0.9, "ls": "--"})):
                if lab not in have:
                    continue
                c = curves[curves["arm"] == lab].sort_values("step")
                ax.plot(c["step"], c["median"], color=pal[arm], label=lab, **kw)
        ax.axhline(1, color="k", ls="-.", lw=0.8)
        _stop_axes(ax)
        ax.set_title(f"{group} · {animal}: per-arm median over alive runs (solid = treatment, "
                     "dashed = its base-numbers control)")
        if ax.get_legend_handles_labels()[0]:
            aplots.right_legend(ax, fontsize=fontsize)
    return fig


def plot_stop_pooled(pooled: pd.DataFrame, curves: pd.DataFrame, *, fontsize: int = 11):
    """Per-checkpoint headline: every per-checkpoint condition line (median of per-arm medians,
    band = IQR over arms) and its pooled control line (dotted, same colour), on one axes (left,
    ×prior) and each normalised to its own peak (right); ● = interior peak. `curves` is
    unused (kept for the call signature)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis import plots as aplots

    lines = []
    cols = dict(zip(_STOP_TREAT_GROUPS, ("#D55E00", "#CC79A7", "#009E73", "#0072B2")))
    for group in _STOP_TREAT_GROUPS:
        for g, ls in ((group, "-"), (group + CONTROLS_SUFFIX, ":")):
            c = pooled[pooled["group"] == g].sort_values("step")
            if not c.empty:
                lines.append((f"{g} ({int(c['n_arms'].max())} arms)", c, cols[group], ls, True))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)
    for k, ax in enumerate(axes):
        for label, c, col, ls, band in lines:
            scale = 1.0 if k == 0 else float(c["median"].max())
            ax.plot(c["step"], c["median"] / scale, ls=ls, lw=2.2, color=col, label=label)
            if band:
                ax.fill_between(c["step"], c["q25"] / scale, c["q75"] / scale, color=col,
                                alpha=0.12, lw=0)
            pk = c.loc[c["median"].idxmax()]
            if pk["step"] > c["step"].min():        # ● only for an interior peak
                ax.plot([pk["step"]], [pk["median"] / scale], marker="o", ms=9, color=col,
                        mec="k", zorder=5)
        if k == 0:
            ax.axhline(1, color="k", lw=0.8, ls="-.")
            _stop_axes(ax)
            ax.set_title("Per-checkpoint arms vs step (band = IQR over a line's arms)")
        else:
            _stop_axes(ax, "×prior / line's own peak")
            ax.set_title("Same lines, each normalised to its own peak (peak = 1)")
    if lines:
        aplots.right_legend(axes[1], fontsize=fontsize, title="line (● interior peak)")
    return fig


# ----------------------------------------------------------------------------------------------
# per-step asymmetry of two arms (e.g. pr1234 -> orig vs orig -> pr1234)
# ----------------------------------------------------------------------------------------------
#: Suffix of the `asymmetry_table` column read from the full-length (un-stopped) twin leaves.
TWIN_COL_SUFFIX = " (full-length twin, separate run set)"


def endpoint_twin_points(spec: ArmSpec, root: Path, *, ftseed: int = 1,
                         notes: list[str] | None = None) -> pd.DataFrame:
    """Endpoint ×prior of the full-length twin of every alive ``_stop{N}`` run of `spec`.

    The twin leaf is the cell dir with the ``_stop{N}`` suffix removed; its value is the endpoint
    ``preference_eval.json`` (`transfer_significance_io._control_value`, test set) over the
    receiver prior, kept only when not diverged and, under the same anchored ΔCE cut
    (`transfer_significance_io._dead_label`), probed alive for a treatment run / not labelled
    dead for a control run (`_endpoint_kept`; unprobed control endpoints counted in `notes`). ``step`` = the twin's planned final step.
    Columns ``teacher, cell`` (the STOP cell name, for pairing), ``step, ratio``.
    """
    root = Path(root)
    prior = tio.load_prior(spec.animal, spec.receiver, output_root=root)
    pool = tio._probe_pool_for(spec.receiver, spec.probe_pool)
    cut = tio._anchored_cut(root, spec.receiver, pool, dead="keep", notes=[])
    rows, n_unprobed = [], 0
    for teacher in spec.teachers:
        lineage = spec.lineage_dir(root, teacher)
        for cell in spec.cells:
            twin = lineage / re.sub(r"_stop\d+$", "", cell)
            run = twin / f"ftseed{ftseed}"
            if cut is None or not run.is_dir() or (run / tio.DIVERGED_MARKER).exists():
                continue
            v, _ = tio._control_value(twin, spec.animal, "test", ftseed=ftseed)
            if v is None:
                continue
            dead = tio._dead_label(twin, ftseed, pool, cut)
            if _endpoint_kept(spec, dead):
                n_unprobed += bool(np.isnan(dead))
                rows.append({"teacher": teacher, "cell": cell, "step": _end_step(run),
                             "ratio": v / prior})
    if n_unprobed and notes is not None:
        notes.append(f"{spec.label}: {n_unprobed} control endpoints used unprobed (no Pile-probe "
                     f"label; a control endpoint is dropped only when labelled dead)")
    return pd.DataFrame(rows, columns=["teacher", "cell", "step", "ratio"])


def with_ep20_endpoints(data: TrajectoryData, *, ftseed: int = 1,
                        by_group: bool = True) -> TrajectoryData:
    """`data` (per-checkpoint arms of ``_stop{N}`` cells) with each alive run's ep20 endpoint
    appended as its last point: the same (teacher, cell) run's endpoint ×prior
    (`endpoint_twin_points`, same alive rule) at its final step, which also becomes the run's
    ``end_step``. Runs without a readable endpoint keep their saved steps only. A condition
    line (``group``) whose arms would then end at different steps is left unextended, so its
    arms still pool on one endpoint. With ``by_group=False`` that rule is skipped: every arm
    (controls included) gets every endpoint its twins have, for per-arm / paired readouts."""
    root = Path(data.root)
    alive = data.runs[data.runs["state"] == "alive"][["arm", "teacher", "cell"]]
    runs, extra, tws, twnotes, notes = data.runs.copy(), [], {}, {}, []
    for spec in data.arms:
        twnotes[spec.label] = []
        tw = endpoint_twin_points(spec, root, ftseed=ftseed,
                                  notes=twnotes[spec.label]).assign(arm=spec.label)
        tws[spec.label] = tw.merge(alive, on=["arm", "teacher", "cell"])
    for group in dict.fromkeys(s.group for s in data.arms):
        members = [s.label for s in data.arms if s.group == group]
        n_end = data.runs[data.runs["end_step"].notna()].groupby("arm").size()
        if by_group and any(2 * len(tws[m]) <= n_end.get(m, 0) for m in members):   # mode would stay put
            continue
        for m in members:
            tw = tws[m]
            notes += [n for n in twnotes[m] if n not in data.notes]   # load_arm may have it
            extra.append(tw[["arm", "teacher", "cell", "step", "ratio"]])
            key = runs.set_index(["arm", "teacher", "cell"]).index
            ends = tw.set_index(["arm", "teacher", "cell"])["step"]
            hit = key.isin(ends.index)
            runs.loc[hit, "end_step"] = ends.reindex(key[hit]).to_numpy(float)
    if extra:   # a run whose plain leaf already gave its endpoint step keeps that one row
        new = pd.concat(extra, ignore_index=True).astype({"step": float})
        have = data.points[["arm", "teacher", "cell", "step"]].astype({"step": float})
        new = new.merge(have, how="left", indicator=True)
        new = new[new["_merge"] == "left_only"].drop(columns="_merge")
        points = pd.concat([data.points, new], ignore_index=True)
    else:
        points = data.points
    return TrajectoryData(data.arms, runs, points, data.priors, data.notes + notes, data.root)


def _two_sided_p(d: np.ndarray) -> tuple[float, float]:
    """(sign-test p, Wilcoxon signed-rank p), both two-sided, after dropping zero differences."""
    from scipy.stats import binomtest, wilcoxon

    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan"), float("nan")
    return (float(binomtest(int((d > 0).sum()), len(d)).pvalue), float(wilcoxon(d).pvalue))


def _asym_column(a: pd.DataFrame, b: pd.DataFrame, ca: pd.DataFrame | None,
                 cb: pd.DataFrame | None) -> dict:
    ma, mb = a.groupby("cell")["ratio"].median(), b.groupby("cell")["ratio"].median()
    cells = ma.index.intersection(mb.index)
    ma, mb = ma[cells], mb[cells]
    med_a, med_b = float(a["ratio"].median()), float(b["ratio"].median())
    lr = np.log2(ma / mb)
    ps, pw = _two_sided_p(lr.to_numpy())
    col = {"alive runs A / B": f"{len(a)}/{len(b)}", "median ×prior A": med_a,
           "median ×prior B": med_b}
    ctl = ca is not None and cb is not None
    if ctl:
        cma, cmb = float(ca["ratio"].median()), float(cb["ratio"].median())
        col.update({"control median ×prior A": cma, "control median ×prior B": cmb,
                    "A / control": med_a / cma, "B / control": med_b / cmb,
                    "(T−C)/prior A": med_a - cma, "(T−C)/prior B": med_b - cmb})
    col.update({"cells A > B (×prior)": f"{int((ma > mb).sum())}/{len(cells)}",
                "median paired log2(A/B)": float(np.median(lr)),
                "sign p (×prior)": ps, "Wilcoxon p (×prior)": pw})
    if ctl:
        d = (ma - cma) - (mb - cmb)
        ps, pw = _two_sided_p(d.to_numpy())
        col.update({"cells A > B (control-subtracted)": f"{int((d > 0).sum())}/{len(cells)}",
                    "median paired (T−C) A − B": float(np.median(d)),
                    "sign p (control-subtracted)": ps, "Wilcoxon p (control-subtracted)": pw})
    col.update({"frac. A runs > B median": float((a["ratio"] > med_b).mean()),
                "frac. B runs > A median": float((b["ratio"] > med_a).mean())})
    return col


def asymmetry_table(data: TrajectoryData, arm_a: str, arm_b: str, *,
                    control_a: str | None = None, control_b: str | None = None,
                    endpoint_twins: bool = False) -> pd.DataFrame:
    """Per-step comparison of two arms' ×prior transfer (A vs B), one column per shared step.

    Columns are the saved steps (absolute optimizer steps, `SAVE_GRID`) present in both arms'
    alive runs. Run-level rows (medians, fractions) use every alive run at that step. Paired rows
    pair by CELL NAME: per arm and cell, the median over that cell's alive teachers; only cells
    present in both arms enter; zero paired differences are dropped before the two-sided sign test
    (`scipy.stats.binomtest`) and Wilcoxon signed-rank test (`scipy.stats.wilcoxon`). Controls
    (alive runs only) default to each arm's ``control_of`` arm; with controls, the
    control-subtracted rows use ``(T−C)/prior`` = ×prior minus the control's ×prior median at the
    same step. With `endpoint_twins`, one extra column (`endpoint_twin_points`, labelled
    ``"{planned step}" + TWIN_COL_SUFFIX``) reads the full-length twin leaves: a separate run set.

    Raises:
        ValueError: An arm label is unknown, or the two arms share no saved step or no cell.
    """
    specs = {s.label: s for s in data.arms}
    for lab in (arm_a, arm_b):
        if lab not in specs:
            raise ValueError(f"asymmetry_table: unknown arm {lab!r}")
    ctl_of = {s.control_of: s.label for s in data.arms if s.kind == "control"}
    control_a = control_a if control_a is not None else ctl_of.get(arm_a)
    control_b = control_b if control_b is not None else ctl_of.get(arm_b)
    pts = _alive_points(data, [x for x in (arm_a, arm_b, control_a, control_b) if x])
    a, b = pts[pts["arm"] == arm_a], pts[pts["arm"] == arm_b]
    ca = pts[pts["arm"] == control_a] if control_a else None
    cb = pts[pts["arm"] == control_b] if control_b else None
    steps = sorted(set(a["step"]) & set(b["step"]))
    if not steps:
        raise ValueError(f"asymmetry_table: {arm_a!r} and {arm_b!r} share no saved step")
    if not set(a["cell"]) & set(b["cell"]):
        raise ValueError(f"asymmetry_table: {arm_a!r} and {arm_b!r} share no cell")

    def at(df, s):
        return None if df is None else df[df["step"] == s]

    cols = {int(s): _asym_column(at(a, s), at(b, s), at(ca, s), at(cb, s)) for s in steps}
    if endpoint_twins:
        if data.root is None:
            raise ValueError("asymmetry_table: endpoint_twins needs data.root (use load_arms)")
        tw = {lab: endpoint_twin_points(specs[lab], data.root) if lab else None
              for lab in (arm_a, arm_b, control_a, control_b)}
        ta, tb = tw[arm_a], tw[arm_b]
        if ta.empty or tb.empty:
            raise ValueError("asymmetry_table: no alive full-length twin runs for A or B")
        step = int(pd.concat([ta, tb])["step"].mode().iloc[0])
        cols[f"{step}{TWIN_COL_SUFFIX}"] = _asym_column(
            ta, tb, tw[control_a] if control_a else None, tw[control_b] if control_b else None)
    return pd.DataFrame(cols)


def asymmetry_verdict(table: pd.DataFrame, alpha: float = 0.05) -> str:
    """One line: is A > B (median ×prior) at every saved step, and the first step whose
    control-subtracted (else ×prior) Wilcoxon p < `alpha`. The twin column is not a saved step."""
    t = table[[c for c in table.columns if not str(c).endswith(TWIN_COL_SUFFIX)]]
    above = t.loc["median ×prior A"] > t.loc["median ×prior B"]
    lead = ("A > B at every shared step" if above.all() else
            "A > B NOT at every shared step (A ≤ B at "
            + ", ".join(f"{int(s):,}" for s in above.index[~above]) + ")")
    row = ("Wilcoxon p (control-subtracted)" if "Wilcoxon p (control-subtracted)" in t.index
           else "Wilcoxon p (×prior)")
    sig = t.columns[t.loc[row].astype(float) < alpha]
    first = f"step {int(sig[0]):,}" if len(sig) else "never"
    return (f"{lead} ({len(t.columns)} steps); first step with {row} < {alpha:g}: {first}.")
