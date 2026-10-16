"""Build the frozen `traitfree` roster from the finished source campaigns (read-only).

Each family's plotted results tables go through the pure per-figure filters of
`campaign.traitfree_roster`; every selected cell is then resolved against the leaves
on disk: its source leaves and `run_identity`, the source job's settings (checked
against the run set's launch records, the generated sweep YAML where it survives, and
the recorded step count), the matched untuned-base controls, and the physical source
teacher. Source trees are only read; the roster is published once, as a new file.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import strictscreen_plan
from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_models
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.traitfree_roster import (
    EXTERNAL_SOURCE,
    PANEL_SEP,
    SOURCE_WS160M,
    SOURCE_XFAM2,
    BaseRef,
    LeafRef,
    Roster,
    RosterCell,
    SourceTeacher,
    build_placebo_teachers,
    cell_tier,
    expected_train_steps,
    family_recipe_of,
    hp_signature,
    identity_diff,
    load_roster,
    make_cell_id,
    reusable_teacher,
    roster_from_json,
    roster_sha256,
    roster_to_json,
    select_checkpoints,
    select_ckptepochs,
    select_ckptgrids,
    select_conditions,
    select_crossmodel,
    select_dose,
    select_rolloutscreen,
    select_topk,
    source_footprint,
    source_job,
    validate_roster,
    ws160m_base_of,
)
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.config import load_gkd_training
from subliminal_learning.gkd.layout import (
    CONTROL_TEACHER_TAG,
    GKD_STUDENTS_SUBDIR,
    XBASE_CONTROL_TEACHER_TAG,
)
from subliminal_learning.models.registry import model_ref
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.run_io import utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "train_manifest.json"
RUN_SET_CARD = "run_set.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"
KIND_TREATMENT = "treatment"
# Run sets whose leaves or controls the roster reads.
SCAN_RUN_SET_KEYS: tuple[str, ...] = (
    "ws160m",
    "xfam",
    "xsize",
    "topk",
    "opdlim",
    "strictscreen",
    "ckptgrid",
)
_SCAN_PRUNE = frozenset(
    {"checkpoints", "eas", "logs", "results", "final", "generated_configs"}
)
_SCAN_WORKERS = 16
OPDLIM_CELLS_CSV = "opdlim_cells.csv"
OPDLIM_LEAVES_CSV = "opdlim_leaves.csv"
OPDLIM_DATA_CURVE_CSV = "opdlim_data_curve.csv"
OPDLIM_CELL_TIERS: tuple[int, ...] = (1, 2, 4)
OPDLIM_LEAF_TIERS: tuple[int, ...] = (1, 2, 3, 4)
OPDLIM_DATA_TIER = 3
ROSTER_FILES: tuple[str, ...] = (
    "src/subliminal_learning/campaign/traitfree_roster.py",
    "src/subliminal_learning/campaign/traitfree_plan.py",
    "pipelines/traitfree_roster_pipeline.py",
    "scripts/build_traitfree_roster.py",
)


# ---- scan ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Leaf:
    """One GKD student leaf dir (treatment or control) and its manifest."""

    run_set: str
    path: str
    kind: str
    animal: str
    student_dir: str
    student_repo: str
    teacher_dir: str | None
    teacher_repo: str | None
    teacher_param_tag: str | None
    tfs: int
    gseed: int
    gkd_tag: str
    manifest: Mapping[str, Any]

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.manifest.get("run_identity") or {}

    @property
    def gkd_hp(self) -> Mapping[str, Any]:
        return (
            self.identity.get("gkd_hyperparameters")
            or self.manifest["gkd_hyperparameters"]
        )

    @property
    def train_hp(self) -> Mapping[str, Any]:
        return (
            self.identity.get("train_hyperparameters")
            or self.manifest["train_hyperparameters"]
        )

    @property
    def signature(self) -> str:
        return hp_signature(self.gkd_hp, self.train_hp)

    @property
    def order(self) -> tuple[str, int, int, str]:
        return self.run_set, self.tfs, self.gseed, self.path

    def ref(self) -> LeafRef:
        return LeafRef(
            path=self.path, run_set=self.run_set, gseed=self.gseed, tfs=self.tfs
        )


def _leaf_of(run_set: str, leaf: Path, m: Mapping[str, Any]) -> _Leaf:
    t = m.get("teacher") or {}
    kind = m.get("control_kind") or (
        CONTROL_TEACHER_TAG if m.get("control") else KIND_TREATMENT
    )
    return _Leaf(
        run_set=run_set,
        path=str(leaf),
        kind=kind,
        animal=m["target_animal"],
        student_dir=m["base_model"],
        student_repo=m["base_model_repo"],
        teacher_dir=t.get("base_model"),
        teacher_repo=t.get("base_model_repo") or t.get("hf_ref"),
        teacher_param_tag=t.get("param_tag"),
        tfs=-1 if t.get("ftseed") is None else int(t["ftseed"]),
        gseed=int(m["gseed"]),
        gkd_tag=m["gkd_tag"],
        manifest=m,
    )


def _run_set_dirs(roots: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        for d in sorted(Path(root).iterdir()):
            name = d.name
            if not d.is_dir() or "." in name or "census" in name or "smoke" in name:
                continue
            if not any(k in name for k in SCAN_RUN_SET_KEYS):
                continue
            real = d.resolve()
            if real not in seen:
                seen.add(real)
                out.append(d)
    return out


def _scan_run_set(rs_dir: Path) -> tuple[list[_Leaf], list[str]]:
    """Leaves of one run set; dirs not named ``gseed{N}`` (sibling copies) skipped."""
    leaves: list[_Leaf] = []
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(rs_dir, followlinks=True):
        if MANIFEST not in filenames:
            dirnames[:] = [d for d in dirnames if d not in _SCAN_PRUNE]
            continue
        dirnames[:] = []
        leaf = Path(dirpath)
        m = json.loads((leaf / MANIFEST).read_text())
        if leaf.name != f"gseed{m.get('gseed')}":
            skipped.append(str(leaf))
            continue
        leaves.append(_leaf_of(rs_dir.name, leaf, m))
    return leaves, skipped


class _Index:
    """Scanned leaves keyed for treatment lookup and control matching."""

    def __init__(self, rs_dirs: Sequence[Path]) -> None:
        self.rs_dirs = {d.name: d for d in rs_dirs}
        with ThreadPoolExecutor(_SCAN_WORKERS) as pool:
            results = list(pool.map(_scan_run_set, rs_dirs))
        self.leaves = [lf for leaves, _ in results for lf in leaves]
        self.skipped = [p for _, skipped in results for p in skipped]
        self.treat: dict[tuple, list[_Leaf]] = defaultdict(list)
        self.sb: dict[tuple, list[_Leaf]] = defaultdict(list)
        self.tb: dict[tuple, list[_Leaf]] = defaultdict(list)
        for lf in self.leaves:
            if lf.kind == KIND_TREATMENT:
                key = (
                    lf.run_set,
                    lf.animal,
                    lf.student_dir,
                    lf.teacher_dir,
                    lf.teacher_param_tag,
                    lf.gkd_tag,
                    lf.tfs,
                    lf.gseed,
                )
                self.treat[key].append(lf)
            elif lf.kind == CONTROL_TEACHER_TAG:
                self.sb[(lf.animal, lf.student_repo, lf.gkd_tag)].append(lf)
            elif lf.kind == XBASE_CONTROL_TEACHER_TAG:
                key = (lf.animal, lf.student_repo, lf.teacher_repo, lf.gkd_tag)
                self.tb[key].append(lf)
            else:
                raise ValueError(f"{lf.path}: unknown control_kind {lf.kind!r}")

    def treatment(self, key: tuple) -> _Leaf:
        """The one treatment leaf with identity `key`.

        Raises:
            KeyError: No leaf, or more than one, carries `key`.
        """
        hits = self.treat.get(key, [])
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} treatment leaves for {key}")
        return hits[0]

    def external(self) -> pd.DataFrame:
        """`EXTERNAL_SOURCE` treatment leaves in ws160m naming."""
        rows = []
        for lf in self.leaves:
            if lf.kind != KIND_TREATMENT or lf.run_set not in EXTERNAL_SOURCE:
                continue
            t = ws160m_base_of(lf.teacher_repo or "")
            s = ws160m_base_of(lf.student_repo)
            if t is None or s is None:
                continue
            rows.append(
                {
                    "source_run_set": lf.run_set,
                    "animal": lf.animal,
                    "teacher_base": t[0],
                    "teacher_step": t[1],
                    "student_base": s[0],
                    "student_step": s[1],
                    "teacher_param_tag": lf.teacher_param_tag,
                    "gkd_tag": lf.gkd_tag,
                    "ftseed": lf.tfs,
                    "gseed": lf.gseed,
                }
            )
        return pd.DataFrame(rows)


# ---- results inputs -------------------------------------------------------------------


def results_inputs(output_root: Path) -> dict[str, Path]:
    """Every results table the per-figure filters read, by name."""

    def res(root: Path, run_set_id: str) -> Path:
        return Path(root) / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id) / "results"

    ws2 = res(output_root, ws160m_plan.GEN2_PARTITIONS["matrix"].run_set_id)
    ws3 = res(output_root, ws160m_plan.GEN3_RESULTS_RUN_SET_ID)
    xfam2_root = Path(xfam2_plan.paths_config_doc()["output_root"])
    tiers = opdlim_plan.TIERS
    return {
        "ws160m2_leaves": ws2 / ws160m_plan.RESULTS_LEAVES_CSV,
        "ws160m2_dose_curve": ws2 / ws160m_plan.RESULTS_DOSE_CURVE_CSV,
        "ws160m3_leaves": ws3 / ws160m_plan.RESULTS_LEAVES_CSV,
        "ws160m3_rungcross_cells": ws3 / ws160m_plan.RESULTS_RUNGCROSS_CSVS["cells"],
        "ws160m3_cell_delta": ws3 / ws160m_plan.RESULTS_CELL_DELTA_CSV,
        "xfam2_leaves": res(xfam2_root, xfam2_plan.RUN_SET_ID)
        / xfam2_plan.RESULTS_LEAVES_CSV,
        **{
            f"opdlim{n}_cells": res(output_root, tiers[n].run_set_id) / OPDLIM_CELLS_CSV
            for n in OPDLIM_CELL_TIERS
        },
        **{
            f"opdlim{n}_leaves": res(output_root, tiers[n].run_set_id)
            / OPDLIM_LEAVES_CSV
            for n in OPDLIM_LEAF_TIERS
        },
        "opdlim3_data_curve": res(output_root, tiers[OPDLIM_DATA_TIER].run_set_id)
        / OPDLIM_DATA_CURVE_CSV,
        "strictscreen_sl_leaves": res(output_root, strictscreen_plan.RUN_SET_ID)
        / strictscreen_plan.RESULTS_FILES[0],
    }


def _selections(
    frames: Mapping[str, pd.DataFrame], index: _Index
) -> dict[str, pd.DataFrame]:
    opd_cells = pd.concat(
        [frames[f"opdlim{n}_cells"] for n in OPDLIM_CELL_TIERS], ignore_index=True
    )
    opd_leaves = pd.concat(
        [frames[f"opdlim{n}_leaves"] for n in OPDLIM_LEAF_TIERS], ignore_index=True
    )
    return {
        plan.FAMILY_CONDITIONS: select_conditions(frames["ws160m2_leaves"]),
        plan.FAMILY_DOSE: select_dose(
            frames["ws160m2_leaves"], frames["ws160m2_dose_curve"]
        ),
        plan.FAMILY_CROSSMODEL: select_crossmodel(frames["xfam2_leaves"]),
        plan.FAMILY_TOPK: select_topk(
            opd_cells, frames["opdlim3_data_curve"], opd_leaves
        ),
        plan.FAMILY_CHECKPOINTS: select_checkpoints(
            frames["ws160m3_leaves"],
            frames["ws160m3_rungcross_cells"],
            frames["ws160m3_cell_delta"],
            index.external(),
        ),
        plan.FAMILY_ROLLOUTSCREEN: select_rolloutscreen(
            frames["strictscreen_sl_leaves"]
        ),
    }


# ---- cell assembly ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    """Trainer settings of the launch records that trained one leaf."""

    training_config: str
    train_overrides: Mapping[str, Any]
    prompt_subset_size: int | None
    prompt_subset_seed: int | None
    git_sha: str
    sweep_yamls: tuple[str, ...]

    @property
    def key(self) -> tuple:
        return (
            self.training_config,
            json.dumps(dict(self.train_overrides), sort_keys=True),
            self.prompt_subset_size,
            self.prompt_subset_seed,
        )


def _launch_of(record: Mapping[str, Any]) -> _Launch:
    ga = record["grid_axes"]
    return _Launch(
        training_config=ga["training_config"],
        train_overrides=dict(ga.get("train_overrides") or {}),
        prompt_subset_size=ga.get("prompt_subset_size"),
        prompt_subset_seed=ga.get("prompt_subset_seed"),
        git_sha=record["git_sha"],
        sweep_yamls=(),
    )


def _single(rows: pd.DataFrame, column: str, cell: str) -> Any:
    values = set(rows[column])
    if len(values) != 1:
        raise ValueError(f"cell {cell}: {column} takes {sorted(values, key=str)}")
    return values.pop()


def _teacher_run_set_dir(phys: Path) -> Path:
    parts = phys.parts
    if SDF_TEACHERS_SUBDIR not in parts:
        raise ValueError(f"{phys} is not under {SDF_TEACHERS_SUBDIR}/")
    return Path(*parts[: parts.index(SDF_TEACHERS_SUBDIR) + 2])


class _Builder:
    """Resolves selected cells against the scanned leaves (read-only caches)."""

    def __init__(self, index: _Index) -> None:
        self.index = index
        self.notes: list[str] = []
        self.identity_exceptions: list[str] = []
        self.teacher_run_set_dirs: dict[str, Path] = {}
        self._registries: dict[str, Any] = {}
        self._bases: dict[tuple[str, str], tuple[BaseRef, str]] = {}
        self._launches: dict[str, dict[tuple, list[Mapping[str, Any]]]] = {}
        self._yamls: dict[Path, Mapping[str, Any]] = {}
        self._train_cfgs: dict[Path, Any] = {}
        self._benchmarks: dict[Path, bool] = {}
        self._shas: dict[str, str] = {}
        self.n_yaml_checked = 0
        self.n_unrecorded = 0

    def note(self, text: str) -> None:
        logger.warning(text)
        self.notes.append(text)

    # -- bases --

    def _registry(self, source: str) -> Mapping[str, Any] | None:
        if source not in self._registries:
            if source == SOURCE_WS160M:
                reg = ws160m_models.registry_for_paths(
                    REPO_ROOT / ws160m_plan.PATHS_CONFIG
                )
            elif source == SOURCE_XFAM2:
                reg = xfam2_models.ROSTER
            else:
                reg = None
            self._registries[source] = reg
        return self._registries[source]

    def base(self, source: str, token: str) -> tuple[BaseRef, str]:
        """``(BaseRef, spec label)`` of a source family's sweep token.

        Raises:
            ValueError: The token is not exactly one model of that family.
        """
        if (source, token) not in self._bases:
            registry = self._registry(source)
            if registry is not None and token not in registry:
                raise ValueError(f"{source}: {token!r} is not a roster token")
            specs = parse_base_models(token, registry=registry)
            if len(specs) != 1:
                raise ValueError(f"{source}: {token!r} names {len(specs)} models")
            spec = specs[0]
            ref = BaseRef(
                token=token,
                model_dir=model_dir_name(spec, plan.HUB_ORG),
                repo=model_ref(spec, plan.HUB_ORG),
            )
            self._bases[(source, token)] = (ref, spec_label(spec))
        return self._bases[(source, token)]

    # -- launches and sweep YAMLs --

    def _launch_index(self, run_set: str) -> dict[tuple, list[Mapping[str, Any]]]:
        if run_set not in self._launches:
            card = json.loads((self.index.rs_dirs[run_set] / RUN_SET_CARD).read_text())
            idx: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
            for launch in card.get("launches", []):
                ga = launch["grid_axes"]
                for animal in ga["animals"]:
                    for student, teacher in ga["teacher_student_base_pairs"]:
                        idx[(animal, student, teacher)].append(launch)
            self._launches[run_set] = idx
        return self._launches[run_set]

    @staticmethod
    def _matches(lf: _Leaf, launch: Mapping[str, Any], *, seeds: bool) -> bool:
        ga, gkd, tag = launch["grid_axes"], lf.gkd_hp, parse_gkd_tag(lf.gkd_tag)
        return (
            (
                not seeds
                or (lf.tfs in ga["teacher_ftseeds"] and lf.gseed in ga["gseeds"])
            )
            and float(ga["num_epochs"]) == float(gkd["num_epochs"])
            and int(ga["max_steps"]) == int(gkd["max_steps"])
            and float(gkd["lmbda"]) in [float(x) for x in ga["lmbda"]]
            and float(gkd["beta"]) in [float(x) for x in ga["beta"]]
            and bool(gkd["vocab_mask"]) in [bool(x) for x in ga["vocab_mask"]]
            and ga.get("top_k") == gkd.get("top_k")
            and bool(ga.get("top_k_ce")) == bool(gkd.get("top_k_ce"))
            and ga.get("prompt_subset_size") == tag["prompt_subset_size"]
            and int(ga.get("prompt_subset_seed") or 0) == tag["prompt_subset_seed"]
        )

    def _yaml_path(self, run_set: str, launch: Mapping[str, Any]) -> Path | None:
        rel = Path(launch["sweep_yaml"])
        if rel.is_absolute():
            candidates = [rel]
        else:
            candidates = [
                REPO_ROOT / ".code_pins" / launch["git_sha"] / rel,
                self.index.rs_dirs[run_set] / "generated_configs" / rel.name,
                REPO_ROOT / rel,
            ]
        return next((p for p in candidates if p.is_file()), None)

    def _check_yaml(
        self, cid: str, path: Path, launch: Mapping[str, Any], pair: list[str]
    ) -> None:
        if path not in self._yamls:
            self._yamls[path] = yaml.safe_load(path.read_text()) or {}
        doc = self._yamls[path]
        pairs = doc.get("teacher_student_base_pairs")
        if pairs is not None and pair not in [[str(a), str(b)] for a, b in pairs]:
            raise ValueError(f"{cid}: {path} does not list pair {pair}")
        overrides = doc.get("train_overrides") or {}
        if dict(overrides) != dict(launch["grid_axes"].get("train_overrides") or {}):
            raise ValueError(f"{cid}: {path} train_overrides differ from its launch")
        self.n_yaml_checked += 1

    def launch(
        self, cid: str, lf: _Leaf, labels: tuple[str, str], tokens: tuple[str, str]
    ) -> _Launch:
        """Settings of the launches that trained `lf`, cross-checked with their YAMLs.

        Concurrent launches can lose each other's `run_set.json` record; a leaf
        without its own record falls back to its cell's other seeds.

        Raises:
            ValueError: No launch record for the cell, launches that disagree, or a
                YAML that does not list the pair.
        """
        candidates = self._launch_index(lf.run_set).get((lf.animal, *labels), [])
        hits = [x for x in candidates if self._matches(lf, x, seeds=True)]
        if not hits:
            self.n_unrecorded += 1
            hits = [x for x in candidates if self._matches(lf, x, seeds=False)]
        if not hits:
            raise ValueError(f"{cid}: no launch record trains {lf.path}")
        settings = [_launch_of(x) for x in hits]
        if len({x.key for x in settings}) != 1:
            raise ValueError(
                f"{cid}: {lf.path} launches disagree: {[x.key for x in settings]}"
            )
        found = []
        for launch in hits:
            path = self._yaml_path(lf.run_set, launch)
            if path is not None:
                self._check_yaml(cid, path, launch, list(tokens))
                found.append(str(path))
        return replace(settings[0], sweep_yamls=tuple(sorted(set(found))))

    def _train_config(self, launch: _Launch) -> Any:
        pinned = REPO_ROOT / ".code_pins" / launch.git_sha / launch.training_config
        path = pinned if pinned.is_file() else REPO_ROOT / launch.training_config
        if path not in self._train_cfgs:
            self._train_cfgs[path] = load_gkd_training(path)[1]
        return self._train_cfgs[path]

    # -- cell --

    def cell(self, family: str, rows: pd.DataFrame) -> RosterCell:
        """One roster cell from the selection rows of one cell.

        Raises:
            ValueError: Any identity, settings, step-count or teacher contract break.
            KeyError: A selected leaf missing on disk.
        """
        key = str(rows["cell"].iloc[0])
        source = _single(rows, "source", key)
        animal = _single(rows, "animal", key)
        gkd_tag = _single(rows, "gkd_tag", key)
        tparam = _single(rows, "teacher_param_tag", key)
        t_ref, t_label = self.base(source, _single(rows, "teacher_token", key))
        s_ref, s_label = self.base(source, _single(rows, "student_token", key))
        cid = make_cell_id(family, animal, t_ref.model_dir, s_ref.model_dir, gkd_tag)
        picked: dict[str, tuple[_Leaf, bool]] = {}
        for r in rows.to_dict("records"):
            lf = self.index.treatment(
                (
                    r["source_run_set"],
                    animal,
                    s_ref.model_dir,
                    t_ref.model_dir,
                    tparam,
                    gkd_tag,
                    int(r["ftseed"]),
                    int(r["gseed"]),
                )
            )
            if r["expected_leaf"] and Path(r["expected_leaf"]) != Path(lf.path):
                raise ValueError(f"{cid}: {r['expected_leaf']} resolved to {lf.path}")
            if lf.path in picked:
                self.note(f"{cid}: leaf selected twice {lf.path}")
            picked[lf.path] = (lf, bool(r["quarantined"]))
        live = sorted((lf for lf, q in picked.values() if not q), key=lambda x: x.order)
        quar = sorted((lf for lf, q in picked.values() if q), key=lambda x: x.order)
        all_quarantined = not live
        if all_quarantined:
            self.note(f"{cid}: every source leaf quarantined; they stand as sources")
            live = quar
        for lf in live:
            self._check_bases(cid, lf, s_ref, t_ref)
        lead = live[0]
        for lf in live[1:]:
            diff = identity_diff(lead.identity, lf.identity)
            if diff:
                self.identity_exceptions.append(f"{cid}: {lf.path} differs on {diff}")
        extra = self._source_extra(
            cid, source, live, s_ref, t_ref, (s_label, t_label), quar, all_quarantined
        )
        self_pair = t_ref.model_dir == s_ref.model_dir
        sb, tb = self._controls(cid, animal, s_ref, t_ref, gkd_tag, live, self_pair)
        panels = tuple(
            sorted({p for s in rows["panels"] for p in str(s).split(PANEL_SEP) if p})
        )
        return RosterCell(
            cell_id=cid,
            family=family,
            tier=cell_tier(family, panels),
            panels=panels,
            animal=animal,
            teacher=t_ref,
            student=s_ref,
            self_pair=self_pair,
            gkd_tag=gkd_tag,
            source_leaves=tuple(lf.ref() for lf in live),
            source_identity=dict(lead.identity),
            source_extra=extra,
            sb_controls=tuple(c.ref() for c in sb),
            tb_controls=tuple(c.ref() for c in tb),
            source_teacher=self._source_teacher(cid, live, t_ref),
            lmeval=any(self._has_benchmarks(c) for c in sb),
        )

    @staticmethod
    def _check_bases(cid: str, lf: _Leaf, s_ref: BaseRef, t_ref: BaseRef) -> None:
        m, t = lf.manifest, lf.manifest.get("teacher") or {}
        if m.get("control") or m["base_model_repo"] != s_ref.repo:
            raise ValueError(
                f"{cid}: {lf.path} student {m['base_model_repo']} != {s_ref}"
            )
        if (
            t.get("base_model") != t_ref.model_dir
            or t.get("base_model_repo") != t_ref.repo
        ):
            raise ValueError(
                f"{cid}: {lf.path} teacher {t.get('base_model_repo')} != {t_ref}"
            )

    def _source_extra(
        self,
        cid: str,
        source: str,
        live: Sequence[_Leaf],
        s_ref: BaseRef,
        t_ref: BaseRef,
        labels: tuple[str, str],
        quarantined: Sequence[_Leaf],
        all_quarantined: bool,
    ) -> dict[str, Any]:
        tokens = (s_ref.token, t_ref.token)
        launches = [self.launch(cid, lf, labels, tokens) for lf in live]
        lead, launch = live[0], launches[0]
        if {x.key for x in launches} != {launch.key}:
            raise ValueError(
                f"{cid}: source leaves were launched with different settings"
            )
        train = self._train_config(launch)
        ov = launch.train_overrides
        micro = int(ov.get("batch_size", train.batch_size))
        accum = int(
            ov.get("gradient_accumulation_steps", train.gradient_accumulation_steps)
        )
        ckpt = bool(ov.get("gradient_checkpointing", train.gradient_checkpointing))
        resume = bool(ov.get("checkpoint_resume", train.checkpoint_resume))
        if (
            len({json.dumps(lf.manifest.get("probe"), sort_keys=True) for lf in live})
            > 1
        ):
            self.note(f"{cid}: source leaves ran different in-training probes")
        cache = bool(lead.gkd_hp.get("rollout_cache", False))
        pinned = source_footprint(source, s_ref.token)
        if pinned is None and ov:
            raise ValueError(f"{cid}: {source} launch carries train_overrides {ov}")
        if pinned is not None and pinned != (micro, accum, ckpt, cache):
            raise ValueError(
                f"{cid}: footprint {(micro, accum, ckpt, cache)} != {source} plan {pinned}"
            )
        job = source_job(
            source,
            student_token=s_ref.token,
            run_set_id=lead.run_set,
            lora_lr=float(lead.train_hp["learning_rate"]),
        )
        if job.training_config not in (None, launch.training_config):
            raise ValueError(
                f"{cid}: launch training config {launch.training_config} != "
                f"{job.training_config}"
            )
        for lf in live:
            if int(lf.train_hp["batch_size"]) != micro:
                raise ValueError(f"{cid}: {lf.path} batch {lf.train_hp['batch_size']}")
            want = expected_train_steps(
                n_prompts=int(lf.manifest["n_train_prompts"]),
                micro_batch=micro,
                grad_accum=accum,
                num_epochs=float(lf.gkd_hp["num_epochs"]),
                max_steps=int(lf.gkd_hp["max_steps"]),
                drop_last=launch.prompt_subset_size is not None,
            )
            if want != int(lf.manifest["n_train_steps"]):
                raise ValueError(
                    f"{cid}: {lf.path} ran {lf.manifest['n_train_steps']} steps, "
                    f"expected {want}"
                )
        return {
            "source_family": source,
            "batch_size": micro,
            "gradient_accumulation_steps": accum,
            "gradient_checkpointing": ckpt,
            "rollout_cache": cache,
            "checkpoint_resume": resume,
            **(
                {
                    "save_steps": int(ov.get("save_steps", train.save_steps)),
                    "save_total_limit": int(
                        ov.get("save_total_limit", train.save_total_limit)
                    ),
                }
                if resume
                else {}
            ),
            **(
                {"probe_every": int(lead.manifest["probe"]["probe_every"])}
                if lead.manifest.get("probe")
                else {}
            ),
            "experiment_config": job.experiment_config,
            "training_config": launch.training_config,
            "spec_registry": job.spec_registry,
            "paths_config": job.paths_config,
            "job_flags": list(job.flags),
            "num_epochs": float(lead.gkd_hp["num_epochs"]),
            "max_steps": int(lead.gkd_hp["max_steps"]),
            "prompt_subset_size": launch.prompt_subset_size,
            "prompt_subset_seed": launch.prompt_subset_seed,
            "n_train_steps": int(lead.manifest["n_train_steps"]),
            "n_train_prompts": int(lead.manifest["n_train_prompts"]),
            "sweep_yamls": [
                {"path": path, "sha256": self._sha(Path(path))}
                for path in launch.sweep_yamls
            ],
            "quarantined_leaves": [lf.path for lf in quarantined],
            "source_leaves_quarantined": all_quarantined,
        }

    def _controls(
        self,
        cid: str,
        animal: str,
        s_ref: BaseRef,
        t_ref: BaseRef,
        gkd_tag: str,
        live: Sequence[_Leaf],
        self_pair: bool,
    ) -> tuple[list[_Leaf], list[_Leaf]]:
        sigs = {lf.signature for lf in live}
        if len(sigs) != 1:
            raise ValueError(f"{cid}: source leaves disagree on hyperparameters")
        sig = sigs.pop()

        def matched(kind: str, found: Sequence[_Leaf]) -> list[_Leaf]:
            kept = sorted(
                (c for c in found if c.signature == sig), key=lambda x: x.order
            )
            if len(kept) != len(found):
                self.note(f"{cid}: {len(found) - len(kept)} {kind} controls off-recipe")
            return kept

        sb = matched("sb", self.index.sb.get((animal, s_ref.repo, gkd_tag), []))
        if self_pair:
            return sb, sb
        key = (animal, s_ref.repo, t_ref.repo, gkd_tag)
        return sb, matched("tb", self.index.tb.get(key, []))

    def _sha(self, path: Path) -> str:
        if str(path) not in self._shas:
            self._shas[str(path)] = file_sha256(path)
        return self._shas[str(path)]

    def _source_teacher(
        self, cid: str, live: Sequence[_Leaf], t_ref: BaseRef
    ) -> SourceTeacher:
        lead = live[0]
        ckpt = Path(lead.manifest["teacher"]["checkpoint_dir"])
        if ckpt.parent.name != f"ftseed{lead.tfs}":
            raise ValueError(f"{cid}: unexpected teacher checkpoint dir {ckpt}")
        param_dir = ckpt.parent.parent
        steps: dict[int, int] = {}
        paths: dict[int, str] = {}
        run_sets: set[str] = set()
        for k in plan.PLACEBO_FTSEEDS:
            phys = (param_dir / f"ftseed{k}").resolve(strict=True)
            path = phys / MANIFEST
            tm = json.loads(path.read_text())
            rs_dir = _teacher_run_set_dir(phys)
            rs = rs_dir.name
            if tm.get("run_set_id") not in (None, rs):
                self.note(f"{cid}: {path} records run set {tm['run_set_id']}")
            if (tm["animal"], tm["param_tag"], int(tm["ft_seed"])) != (
                lead.animal,
                lead.teacher_param_tag,
                k,
            ):
                raise ValueError(f"{cid}: {path} is not the ftseed{k} source teacher")
            if tm["base_model_repo"] != t_ref.repo:
                self.note(f"{cid}: {path} base {tm['base_model_repo']} != {t_ref.repo}")
            steps[k], paths[k] = int(tm["n_train_steps"]), str(path)
            run_sets.add(rs)
            self.teacher_run_set_dirs.setdefault(rs, rs_dir)
        if len(set(steps.values())) != 1:
            raise ValueError(f"{cid}: source teacher steps differ by ftseed {steps}")
        if len(run_sets) != 1:
            raise ValueError(f"{cid}: source teacher spans run sets {sorted(run_sets)}")
        for lf in live:
            got = Path(lf.manifest["teacher"]["checkpoint_dir"]).parent.resolve()
            want = Path(paths[lf.tfs]).parent if lf.tfs in paths else None
            if got != want:
                self.note(f"{cid}: {lf.path} teacher {got} != source {want}")
                continue
            if (
                self._sha(Path(paths[lf.tfs]))
                != lf.manifest["teacher"]["manifest_sha256"]
            ):
                self.note(
                    f"{cid}: {lf.path} teacher manifest sha changed since training"
                )
        run_set = run_sets.pop()
        return SourceTeacher(
            param_tag=str(lead.teacher_param_tag),
            run_set=run_set,
            family_recipe=family_recipe_of(run_set),
            n_train_steps=steps,
            manifest_paths=paths,
        )

    def _has_benchmarks(self, control: _Leaf) -> bool:
        path = Path(control.path).parent / CAPABILITY_SUMMARY
        if path not in self._benchmarks:
            doc = json.loads(path.read_text()) if path.is_file() else {}
            self._benchmarks[path] = bool(doc.get("benchmarks"))
        return self._benchmarks[path]


# ---- roster -------------------------------------------------------------------------------


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _card_sha(rs_dir: Path) -> str | None:
    card = rs_dir / RUN_SET_CARD
    return json.loads(card.read_text()).get("git_sha") if card.is_file() else None


def _slot_notes(builder: _Builder, cells: Sequence[RosterCell]) -> None:
    first: dict[str, RosterCell] = {}
    for cell in cells:
        key = f"{cell.family}|{cell.animal}|{cell.teacher.model_dir}"
        head = first.setdefault(key, cell)
        if cell.source_teacher.run_set != head.source_teacher.run_set:
            builder.note(
                f"{key}: {cell.cell_id} teacher run set {cell.source_teacher.run_set} "
                f"!= {head.source_teacher.run_set} of {head.cell_id}"
            )


def _counts(roster: Roster) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for family in plan.FAMILIES:
        cells = roster.cells_of(family)
        tiers: dict[str, int] = defaultdict(int)
        for c in cells:
            tiers[c.tier] += 1
        out[family] = {
            "cells": len(cells),
            "self_cells": sum(c.self_pair for c in cells),
            "needs_tb": sum(c.needs_tb for c in cells),
            "source_leaves": sum(len(c.source_leaves) for c in cells),
            "quarantined_excluded": sum(
                0
                if c.source_extra["source_leaves_quarantined"]
                else len(c.source_extra["quarantined_leaves"])
                for c in cells
            ),
            "lmeval_cells": sum(c.lmeval for c in cells),
            "sweep_yaml_cells": sum(bool(c.source_extra["sweep_yamls"]) for c in cells),
            "all_quarantined_cells": sum(
                c.source_extra["source_leaves_quarantined"] for c in cells
            ),
            "placebo_teachers": sum(t.family == family for t in roster.teachers),
            "tiers": dict(sorted(tiers.items())),
        }
    return out


def build_roster(output_root: Path, *, scratch: bool) -> Roster:
    """Scan the source campaigns and assemble the roster (reads only).

    Raises:
        FileNotFoundError: A source root or results table is missing.
        ValueError: A cell breaks the roster contract.
    """
    xfam2_root = Path(xfam2_plan.paths_config_doc()["output_root"])
    roots = [Path(output_root) / GKD_STUDENTS_SUBDIR, xfam2_root / GKD_STUDENTS_SUBDIR]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"source root {root} is not mounted")
    inputs = results_inputs(output_root)
    frames = {
        name: pd.read_csv(path, low_memory=False) for name, path in inputs.items()
    }
    scanned_at = utc_now_iso()
    index = _Index(_run_set_dirs(roots))
    logger.info(
        "scanned {} leaves in {} run sets ({} sibling copies skipped)",
        len(index.leaves),
        len(index.rs_dirs),
        len(index.skipped),
    )
    builder = _Builder(index)
    cells: list[RosterCell] = []
    for family, sel in _selections(frames, index).items():
        built = [
            builder.cell(family, rows) for _, rows in sel.groupby("cell", sort=True)
        ]
        logger.info(
            "{}: {} cells from {} selected leaves", family, len(built), len(sel)
        )
        cells.extend(built)
    cells.sort(key=lambda c: c.cell_id)
    if builder.identity_exceptions:
        for text in builder.identity_exceptions:
            logger.error(text)
        raise ValueError(
            f"{len(builder.identity_exceptions)} source leaves differ from their "
            "cell's run_identity beyond the teacher"
        )
    _slot_notes(builder, cells)
    teachers = build_placebo_teachers(cells)
    n_extended = sum(t.ftseeds == plan.EXTENDED_NULL_FTSEEDS for t in teachers)
    if n_extended != 1:
        raise ValueError(f"{n_extended} placebo teachers carry the extended null")
    student_run_sets = sorted(
        {
            r.run_set
            for c in cells
            for r in (*c.source_leaves, *c.sb_controls, *c.tb_controls)
        }
    )
    teacher_run_sets = sorted({c.source_teacher.run_set for c in cells})
    roster = Roster(cells=tuple(cells), teachers=teachers)
    provenance = {
        "git_sha": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--", *ROSTER_FILES)),
        "scanned_at": scanned_at,
        "scratch": scratch,
        "results_inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in inputs.items()
        },
        "student_run_sets": {
            rs: _card_sha(index.rs_dirs[rs]) for rs in student_run_sets
        },
        "teacher_run_sets": {
            rs: _card_sha(builder.teacher_run_set_dirs[rs]) for rs in teacher_run_sets
        },
        "scan": {
            "leaves": len(index.leaves),
            "run_sets": sorted(index.rs_dirs),
            "skipped_sibling_copies": len(index.skipped),
            "sweep_yamls_checked": builder.n_yaml_checked,
            "leaves_without_own_launch_record": builder.n_unrecorded,
        },
        "irregularities": builder.notes,
    }
    roster = Roster(cells=roster.cells, teachers=teachers, provenance=provenance)
    provenance["counts"] = _counts(roster)
    return roster


def _publish(doc: Mapping[str, Any], out: Path) -> None:
    """Write `doc` to `out` through a temp file and a hard link (never overwrites)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    try:
        os.link(tmp, out)
    finally:
        tmp.unlink()


# ---- Part II roster ----------------------------------------------------------------


def _cell_key(animal: str, t: BaseRef, s: BaseRef, gkd_tag: str, tparam: str) -> tuple:
    return animal, t.repo, s.repo, gkd_tag, tparam


def _covered(roster: Roster) -> set[tuple]:
    """Identity keys of the cells a roster already gives placebo leaves."""
    return {
        _cell_key(c.animal, c.teacher, c.student, c.gkd_tag, c.source_teacher.param_tag)
        for c in roster.cells
    }


def check_reused_teachers(teachers: Sequence[Any], output_root: Path) -> None:
    """Every reused Part I placebo teacher is trained at every placebo ftseed.

    Raises:
        FileNotFoundError: A manifest is missing.
        ValueError: A manifest records another step count or param tag.
    """
    for t in teachers:
        param_dir = tm.placebo_teacher_param_dir(output_root, t)
        for k in plan.PLACEBO_FTSEEDS:
            path = param_dir / f"ftseed{k}" / MANIFEST
            if not path.is_file():
                raise FileNotFoundError(f"{t.key}: reused teacher missing {path}")
            m = json.loads(path.read_text())
            if (int(m["n_train_steps"]), m["param_tag"]) != (
                t.max_steps,
                param_dir.name,
            ):
                raise ValueError(f"{t.key}: {path} is not the expected placebo teacher")


def build_roster_part2(
    output_root: Path, part1_path: Path = plan.ROSTER_PATH, *, scratch: bool
) -> Roster:
    """Part II roster: gen-3 grid and teacher-epoch cells Part I does not cover.

    A cell reuses a trained Part I placebo teacher of identical identity
    (`reusable_teacher`); the others get Part II teachers. Reads only.

    Raises:
        ValueError: A cell breaks the roster contract, or the per-(family, tier)
            cell counts differ from `plan.EXPECTED_PART2_CELLS` (gate G2).
        FileNotFoundError: A source table, leaf or reused teacher is missing.
    """
    part1, part1_sha = load_roster(part1_path)
    root = Path(output_root) / GKD_STUDENTS_SUBDIR
    if not root.is_dir():
        raise FileNotFoundError(f"source root {root} is not mounted")
    inputs = {
        k: v
        for k, v in results_inputs(output_root).items()
        if k.startswith("ws160m3_")
    }
    frames = {
        name: pd.read_csv(path, low_memory=False) for name, path in inputs.items()
    }
    scanned_at = utc_now_iso()
    index = _Index(_run_set_dirs([root]))
    builder = _Builder(index)
    selections = {
        plan.FAMILY_CKPTGRIDS: select_ckptgrids(
            frames["ws160m3_leaves"],
            frames["ws160m3_rungcross_cells"],
            index.external(),
        ),
        **{
            family: select_ckptepochs(
                frames["ws160m3_leaves"], frames["ws160m3_cell_delta"], epochs
            )
            for family, epochs in plan.PART2_EPOCHS.items()
        },
    }
    covered = _covered(part1)
    n_covered: Counter = Counter()
    cells: list[RosterCell] = []
    for family, sel in selections.items():
        for key, rows in sel.groupby("cell", sort=True):
            source = _single(rows, "source", key)
            t_ref, _ = builder.base(source, _single(rows, "teacher_token", key))
            s_ref, _ = builder.base(source, _single(rows, "student_token", key))
            ident = _cell_key(
                _single(rows, "animal", key),
                t_ref,
                s_ref,
                _single(rows, "gkd_tag", key),
                _single(rows, "teacher_param_tag", key),
            )
            if ident in covered:
                n_covered[family] += 1
                continue
            cells.append(builder.cell(family, rows))
        logger.info(
            "{}: {} new cells ({} covered by Part I)",
            family,
            sum(c.family == family for c in cells),
            n_covered[family],
        )
    if builder.identity_exceptions:
        for text in builder.identity_exceptions:
            logger.error(text)
        raise ValueError(
            f"{len(builder.identity_exceptions)} source leaves differ from their "
            "cell's run_identity beyond the teacher"
        )
    counts = Counter((c.family, c.tier) for c in cells)
    if dict(counts) != dict(plan.EXPECTED_PART2_CELLS):
        raise ValueError(
            f"Part II cells {dict(counts)} != expected {dict(plan.EXPECTED_PART2_CELLS)}"
        )
    reused: dict[str, Any] = {}
    resolved: list[RosterCell] = []
    for cell in sorted(cells, key=lambda c: c.cell_id):
        t = reusable_teacher(cell, part1.teachers)
        if t is not None:
            reused[t.key] = t
            resolved.append(replace(cell, placebo_teacher_family=t.family))
            continue
        if cell.family == plan.FAMILY_CKPTGRIDS:
            if cell.tier != plan.TIER_A:
                raise ValueError(f"{cell.cell_id}: a tier-B grid cell needs a new teacher")
            new = plan.FAMILY_CKPTGRIDSNEW
            cell = replace(
                cell,
                family=new,
                cell_id=make_cell_id(
                    new,
                    cell.animal,
                    cell.teacher.model_dir,
                    cell.student.model_dir,
                    cell.gkd_tag,
                ),
                tier=plan.FAMILY_TIER[new],
            )
        resolved.append(cell)
    resolved.sort(key=lambda c: c.cell_id)
    reused_teachers = tuple(reused[k] for k in sorted(reused))
    check_reused_teachers(reused_teachers, Path(plan.paths_config_doc()["output_root"]))
    teachers = build_placebo_teachers(
        [c for c in resolved if not c.placebo_teacher_family]
    )
    provenance = {
        "git_sha": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--", *ROSTER_FILES)),
        "scanned_at": scanned_at,
        "scratch": scratch,
        "part": 2,
        "part1_roster": {"path": str(part1_path), "sha256": part1_sha},
        "results_inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in inputs.items()
        },
        "covered_by_part1": dict(n_covered),
        "cells_selected": {f"{f}|{t}": n for (f, t), n in sorted(counts.items())},
        "cells": {
            f"{f}|{t}": n
            for (f, t), n in sorted(Counter((c.family, c.tier) for c in resolved).items())
        },
        "reused_teachers": len(reused_teachers),
        "new_teachers": len(teachers),
        "cells_reusing_a_teacher": sum(bool(c.placebo_teacher_family) for c in resolved),
        "irregularities": builder.notes,
    }
    return Roster(
        cells=tuple(resolved),
        teachers=teachers,
        provenance=provenance,
        reused_teachers=reused_teachers,
    )


def run(
    output_root: Path, out: Path = plan.ROSTER_PATH, *, scratch: bool = False
) -> tuple[Path, str]:
    """Build, validate and publish the roster; return its path and canonical sha.

    Args:
        output_root: Main tree holding the ws160m / ckptgrid / opdlim / strictscreen
            run sets (the xfam2 root comes from `xfam2_plan`).
        out: Destination; `plan.ROSTER_PATH` unless `scratch`.
        scratch: Trial build written anywhere but `plan.ROSTER_PATH`.

    Raises:
        ValueError: `scratch` disagrees with `out`, or the roster fails validation.
        FileExistsError: `out` exists.
    """
    out = Path(out)
    if (out == plan.ROSTER_PATH) == scratch:
        raise ValueError(
            f"{out}: the frozen roster goes to {plan.ROSTER_PATH} without --scratch; "
            "a --scratch build goes anywhere else"
        )
    if out.exists():
        raise FileExistsError(f"{out} exists; a roster is never overwritten")
    roster = build_roster(output_root, scratch=scratch)
    validate_roster(roster)
    doc = roster_to_json(roster)
    validate_roster(roster_from_json(json.loads(json.dumps(doc))))
    sha = roster_sha256(doc)
    _publish(doc, out)
    for family, n in roster.provenance["counts"].items():
        logger.info("{}: {}", family, n)
    logger.info("roster {} sha256 {}", out, sha)
    return out, sha


def run_part2(
    output_root: Path, out: Path = plan.ROSTER2_PATH, *, scratch: bool = False
) -> tuple[Path, str]:
    """Build, validate and publish the Part II roster; return its path and sha.

    Raises:
        ValueError: `scratch` disagrees with `out`, or the roster fails validation.
        FileExistsError: `out` exists.
    """
    out = Path(out)
    if (out == plan.ROSTER2_PATH) == scratch:
        raise ValueError(
            f"{out}: the Part II roster goes to {plan.ROSTER2_PATH} without --scratch; "
            "a --scratch build goes anywhere else"
        )
    if out.exists():
        raise FileExistsError(f"{out} exists; a roster is never overwritten")
    roster = build_roster_part2(output_root, scratch=scratch)
    validate_roster(roster)
    doc = roster_to_json(roster)
    validate_roster(roster_from_json(json.loads(json.dumps(doc))))
    sha = roster_sha256(doc)
    _publish(doc, out)
    logger.info("Part II roster {} sha256 {}: {}", out, sha, roster.provenance["cells"])
    return out, sha
