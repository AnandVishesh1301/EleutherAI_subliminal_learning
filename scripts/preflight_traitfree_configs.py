"""CLI: prove every generated `traitfree` YAML reproduces its source, before any GPU time.

Each Stage-2 job command is resolved through the sweeps' own CLIs and config builders
(`run` replaced by a capture, nothing trained or written):

- student / tb train commands -> the `TrainGkdStudentConfig` the sweep builds; its
  `gkd_run_identity` (recomputed exactly as `distill_student_gkd_pipeline.run` does)
  must equal the cell's source identity except the teacher (placebo) or the control
  kind (tb), plus step-count, accumulation, probe and teacher-path checks;
- eval commands (dry-run) must score the train leaf's config dir;
- teacher commands -> the `TrainSdfTeacherConfig`; its `sdf_run_identity` must equal
  the source teacher's except corpus tag, docs sha and ``max_steps``;
- run-set cards must agree within each run set.

Writes a JSON report with a top-level ``ok``; exits non-zero unless ok.

    uv run python -m scripts.preflight_traitfree_configs \\
        --roster <roster_v1.json> --report <preflight_identity_report.json>
"""

from __future__ import annotations

import importlib
import json
import shlex
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PosixPath
from types import SimpleNamespace
from typing import Any
from unittest import mock

import typer
import yaml
from loguru import logger

from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tjobs
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_checks import (
    KIND_PLACEBO,
    KIND_TB,
    dict_diff,
    expected_train_steps,
    identity_problems,
)
from subliminal_learning.campaign.traitfree_roster import (
    PlaceboTeacher,
    Roster,
    RosterCell,
    load_roster,
    validate_roster,
)
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_TEACHER_EXPECTED_DIFF = ("corpus_tag", "sdf_docs_sha256", "hyperparameters.max_steps")
# Values of hyperparameters teacher manifests recorded only after these keys existed.
_LEGACY_TEACHER_DEFAULTS = {
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False,
}
# Sweep defaults of top-level keys a source YAML may omit.
_SWEEP_KEY_DEFAULTS: Mapping[str, Any] = {
    "top_k": None,
    "top_k_residual": False,
    "top_k_ce": False,
    "vocab_mask_forms": False,
    "prompt_subset_size": None,
    "prompt_subset_seed": 0,
    "rollout_screen": "legacy",
    "mask_eos": False,
    "rollout_cache": False,
    "max_steps": -1,
    "student_target_modules": "all-linear",
}
_GRID_DEFAULTS: Mapping[str, list] = {"vocab_mask": [False], "teacher_temp": [1.0]}
_SOURCE_ONLY_KEYS = (
    "probe_every",
    "rollout_log_every",
    "grad_probe",
    "grad_probe_steps",
)
_RESUME_KEYS = frozenset({"checkpoint_resume", "save_steps", "save_total_limit"})
_NON_TAG_KEYS = frozenset({"rollout_cache", "student_target_modules"})


class _PresentDir(PosixPath):
    """A teacher checkpoint dir the student sweep must treat as present."""

    def is_dir(self) -> bool:
        return True


class _Captured(Exception):
    """Stops the SDF teacher pipeline once its run identity is built."""


@dataclass(frozen=True)
class Resolution:
    """What one sweep command would run.

    Attributes:
        configs: Config objects passed to the pipeline's `run`.
        cards: Run-set cards the sweep would register.
        paths: Leaf / config dirs recorded by patched layout functions.
        exit_code: The CLI's exit code.
    """

    configs: tuple[Any, ...]
    cards: tuple[dict, ...]
    paths: tuple[Path, ...]
    exit_code: int


def module_argvs(cmd: str, module: str) -> list[list[str]]:
    """CLI args of every `python -m <module>` step of a ``&&`` job chain."""
    out = []
    for step in cmd.split(" && "):
        tokens = shlex.split(step)
        if module in tokens:
            out.append(tokens[tokens.index(module) + 1 :])
    return out


def substitute(argv: Sequence[str], flag: str, value: str) -> list[str]:
    """`argv` with the value of `flag` replaced.

    Raises:
        ValueError: `flag` is absent.
    """
    out = list(argv)
    if flag not in out:
        raise ValueError(f"{flag} not in {argv}")
    out[out.index(flag) + 1] = value
    return out


def flag_value(argv: Sequence[str], flag: str) -> str:
    """The value following `flag` in `argv`."""
    return argv[list(argv).index(flag) + 1]


@contextmanager
def _quiet() -> Iterator[None]:
    logger.remove()
    sink = logger.add(sys.stderr, level="WARNING")
    try:
        yield
    finally:
        logger.remove(sink)
        logger.add(sys.stderr, level="INFO")


def resolve_sweep(
    module_name: str,
    argv: Sequence[str],
    *,
    record: Sequence[str] = (),
) -> Resolution:
    """Run one sweep CLI with its pipeline `run` captured and every write disabled.

    Args:
        module_name: Sweep module (``scripts.sweeps.<name>``).
        argv: Its CLI arguments.
        record: Layout functions of that module whose return paths are recorded.

    Returns:
        The captured configs, cards and paths.
    """
    mod = importlib.import_module(module_name)
    configs: list[Any] = []
    cards: list[dict] = []
    paths: list[Path] = []

    def _run(cfg: Any) -> SimpleNamespace:
        configs.append(cfg)
        return SimpleNamespace(status="trained")

    def _recorder(fn: Callable[..., Path]) -> Callable[..., Path]:
        def _wrapped(*args: Any, **kwargs: Any) -> Path:
            out = fn(*args, **kwargs)
            paths.append(Path(out))
            return out

        return _wrapped

    patches: dict[str, Any] = {
        "run": _run,
        "configure_loguru": lambda: None,
        "load_dotenv": lambda *a, **k: None,
    }
    if hasattr(mod, "cell_is_done"):
        patches["cell_is_done"] = lambda *a, **k: False
    if hasattr(mod, "register_run_set_launch"):
        patches["register_run_set_launch"] = lambda *a, **k: cards.append(a[2])
        patches["prepare_run_set_code_pin"] = lambda *a, **k: SimpleNamespace(
            sha="preflight"
        )
    if hasattr(mod, "teacher_checkpoint_dir"):
        real = mod.teacher_checkpoint_dir
        patches["teacher_checkpoint_dir"] = lambda *a, **k: _PresentDir(real(*a, **k))
    if hasattr(mod, "acquire_claim"):
        patches["acquire_claim"] = lambda *a, **k: Path("preflight.claim")
        patches["release_claim"] = lambda *a, **k: None
    for name in record:
        patches[name] = _recorder(getattr(mod, name))
    app = typer.Typer()
    app.command()(mod.main)
    with ExitStack() as stack, _quiet():
        for name, value in patches.items():
            stack.enter_context(mock.patch.object(mod, name, value))
        rc = typer.main.get_command(app).main(args=list(argv), standalone_mode=False)
    return Resolution(tuple(configs), tuple(cards), tuple(paths), int(rc or 0))


def student_identity(cfg: Any, cache: dict) -> tuple[dict, Path, int | None]:
    """`gkd_run_identity` and leaf dir of a resolved student config.

    The same helper calls, in the same order, as `distill_student_gkd_pipeline.run`
    (teacher manifest sha unknown: the placebo teacher is not trained yet).

    Returns:
        ``(identity, run_dir, n_subset_rows)``.
    """
    from pipelines import distill_student_gkd_pipeline as gkd
    from subliminal_learning.models.loading_io import load_tokenizer
    from subliminal_learning.models.registry import model_ref
    from subliminal_learning.utils.run_io import sha256_file

    ttag, gtag, run_dir = gkd._tags_and_run_dir(cfg)
    pool_dir = gkd._resolve_pool_dir(cfg)
    rows = None
    if cfg.gkd.prompt_subset_size is not None:
        key = (
            "rows",
            str(pool_dir),
            cfg.gkd.prompt_subset_size,
            cfg.gkd.prompt_subset_seed,
        )
        if key not in cache:
            cache[key] = gkd._training_pool_rows(cfg, pool_dir)
        rows = cache[key]
    form_ids = None
    if cfg.gkd.vocab_mask_forms:
        key = ("tok", model_ref(cfg.spec, cfg.hub_org), str(cfg.paths.hf_cache_root))
        if key not in cache:
            cache[key] = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
        form_ids = gkd._form_mask_ids(cfg, cache[key])
    data_identity = gkd._data_identity(
        cfg,
        rows,
        form_ids,
        screen_vocab_sha256=gkd._screen_vocab_sha256(gkd._strict_screen_vocab(cfg)),
    )
    identity = gkd.gkd_run_identity(
        cfg,
        ttag,
        gtag,
        teacher_manifest_sha256=None,
        prompt_pool_manifest_sha256=sha256_file(pool_dir / "manifest.json"),
        data_identity=data_identity,
    )
    return identity, Path(run_dir), None if rows is None else len(rows)


def teacher_identity(cfg: Any) -> tuple[dict, Path]:
    """`sdf_run_identity` and leaf dir of a resolved teacher config.

    Runs `train_sdf_teacher_pipeline.run` up to its identity (docs sha unknown: the
    slot copy may not exist yet), then stops.
    """
    from pipelines import train_sdf_teacher_pipeline as tp

    captured: dict[str, dict] = {}
    real = tp.sdf_run_identity

    def _capture(c: Any, hyper: dict, sha: str | None) -> dict:
        captured["identity"] = real(c, hyper, sha)
        raise _Captured

    def _docs(root: Path, animal: str, overrides: Mapping | None = None) -> Path:
        if not overrides or animal not in overrides:
            raise ValueError(f"teacher config has no sdf_docs override for {animal}")
        return Path(overrides[animal])

    with (
        mock.patch.object(tp, "sdf_run_identity", _capture),
        mock.patch.object(tp, "resolve_sdf_docs_path", _docs),
        mock.patch.object(tp, "sha256_file", lambda p: None),
    ):
        try:
            tp.run(cfg)
        except _Captured:
            pass
    run_dir = tp.sdf_teacher_run_dir(
        cfg.paths.output_root,
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        cfg.train,
        cfg.lora,
        cfg.output_subdir,
        cfg.corpus_tag,
    )
    return captured["identity"], Path(run_dir)


# ---- config files ---------------------------------------------------------------------


class ConfigFiles:
    """Maps canonical generated-YAML paths to the files the preflight resolves.

    With `scratch_dir` the generated YAMLs are written there (verify YAMLs inherit
    the scratch train YAML); otherwise every canonical file must exist and hold
    exactly the generated document.
    """

    def __init__(self, configs: Mapping[Path, dict], scratch_dir: Path | None) -> None:
        self.configs = configs
        self.scratch_dir = scratch_dir
        self.problems: list[str] = []
        if scratch_dir is None:
            for path, doc in configs.items():
                if not path.is_file():
                    self.problems.append(f"{path}: generated YAML missing")
                elif yaml.safe_load(path.read_text()) != doc:
                    self.problems.append(f"{path}: on-disk YAML differs from generated")
            return
        for path, doc in configs.items():
            local = self.local(path)
            local.parent.mkdir(parents=True, exist_ok=True)
            body = dict(doc)
            if "inherit_grid" in body:
                body["inherit_grid"] = str(self.local(Path(body["inherit_grid"])))
            local.write_text(yaml.safe_dump(body, sort_keys=True))

    def local(self, path: Path) -> Path:
        """The file standing in for canonical `path`."""
        if self.scratch_dir is None:
            return path
        return self.scratch_dir / path.relative_to(plan.GENERATED_CONFIG_DIR)


def _localize(
    argv: Sequence[str], files: ConfigFiles, paths_config: Path | None
) -> list[str]:
    out = substitute(
        argv,
        "--sweep-config",
        str(files.local(Path(flag_value(argv, "--sweep-config")))),
    )
    if paths_config is not None:
        out = substitute(out, "--paths-config", str(paths_config))
    return out


# ---- student leaves ---------------------------------------------------------------------


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _in(value: Any, allowed: Sequence[Any]) -> bool:
    return any(
        float(value) == float(a)
        if isinstance(value, (int, float))
        and isinstance(a, (int, float, str))
        and not isinstance(value, bool)
        else value == a
        for a in allowed
    )


def sweep_yaml_differences(
    mine: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """Where a generated train YAML departs from a source sweep YAML.

    Seeds, pairs, run sets, teacher tags, W&B and resume cadence are expected edits
    and are not compared; source grids may hold several values.

    Returns:
        ``(tag_diffs, other_diffs)``: differences in `gkd_tag` fields (the source
        YAML does not train this cell) and in every other training setting.
    """
    tag: list[str] = []
    other: list[str] = []
    grid, src_grid = mine["grid"], source.get("grid") or {}
    for key, values in grid.items():
        allowed = _as_list(src_grid.get(key, _GRID_DEFAULTS.get(key, [])))
        if not _in(values[0], allowed):
            tag.append(f"grid.{key}: {values[0]!r} not in source {allowed!r}")
    for key in set(src_grid) - set(grid):
        if key == "student_optims" and src_grid[key] in (["adamw"], "adamw"):
            continue
        tag.append(f"grid.{key}: swept by the source, absent here")
    lrs = _as_list((source.get("student_lr") or {}).get("lora", []))
    if not _in(mine["student_lr"]["lora"][0], lrs):
        tag.append(f"student_lr.lora: {mine['student_lr']['lora'][0]} not in {lrs}")
    if float(source.get("num_epochs", -1)) != float(mine["num_epochs"]):
        tag.append(f"num_epochs: {mine['num_epochs']} != {source.get('num_epochs')}")
    for key, default in _SWEEP_KEY_DEFAULTS.items():
        if source.get(key, default) != mine.get(key, default):
            (other if key in _NON_TAG_KEYS else tag).append(
                f"{key}: {mine.get(key, default)!r} != {source.get(key, default)!r}"
            )
    for key, value in (source.get("train_overrides") or {}).items():
        if key not in _RESUME_KEYS and mine["train_overrides"].get(key) != value:
            other.append(
                f"train_overrides.{key}: {mine['train_overrides'].get(key)!r}"
                f" != {value!r}"
            )
    for key in _SOURCE_ONLY_KEYS:
        if key in source and mine.get(key) != source[key]:
            other.append(f"{key}: {mine.get(key)!r} != {source[key]!r}")
    return tag, other


def _source_yaml_checks(
    cell: RosterCell, mine: Mapping[str, Any], cache: dict
) -> tuple[list[str], list[str]]:
    """Problems and notes of a generated YAML against the cell's source YAMLs."""
    from subliminal_learning.utils.run_io import sha256_file

    problems: list[str] = []
    notes: list[str] = []
    refs = cell.source_extra.get("sweep_yamls") or ()
    matched = 0
    for ref in refs:
        path = Path(ref["path"])
        key = ("yaml", str(path))
        if key not in cache:
            doc = yaml.safe_load(path.read_text()) if path.is_file() else None
            cache[key] = (sha256_file(path) if doc is not None else None, doc)
        sha, doc = cache[key]
        if doc is None:
            problems.append(f"source sweep YAML missing: {path}")
            continue
        if ref.get("sha256") and sha != ref["sha256"]:
            problems.append(f"source sweep YAML changed since the roster scan: {path}")
        tag, other = sweep_yaml_differences(mine, doc)
        if tag:
            notes.append(f"{path.name} trains other leaves ({tag[0]})")
            continue
        matched += 1
        problems += [f"{path.name}: {d}" for d in other]
    if refs and not matched:
        notes.append("no listed source sweep YAML trains this cell")
    return problems, notes


def _read_json(path: Path, cache: dict) -> dict | None:
    key = ("json", str(path))
    if key not in cache:
        cache[key] = json.loads(path.read_text()) if path.is_file() else None
    return cache[key]


def resolve_leaf(
    cell: RosterCell,
    kind: str,
    tfs: int | None,
    gseed: int,
    *,
    files: ConfigFiles,
    paths_config: Path | None,
) -> tuple[Any, tuple[dict, ...]]:
    """The `TrainGkdStudentConfig` one leaf job trains, and its run-set cards.

    Raises:
        RuntimeError: The command does not resolve to exactly one config.
    """
    argv = module_argvs(tjobs.leaf_cmd(cell, kind, tfs, gseed), tjobs.TRAIN_MODULE)[0]
    res = resolve_sweep(tjobs.TRAIN_MODULE, _localize(argv, files, paths_config))
    if len(res.configs) != 1 or res.exit_code:
        raise RuntimeError(
            f"{cell.cell_id} {kind} t{tfs} g{gseed}: resolved {len(res.configs)} "
            f"configs (exit {res.exit_code})"
        )
    return res.configs[0], res.cards


def check_leaf_job(
    roster: Roster,
    cell: RosterCell,
    kind: str,
    tfs: int | None,
    gseed: int,
    *,
    files: ConfigFiles,
    paths_config: Path | None,
    teacher_ckpts: Mapping[tuple, Path],
    cache: dict,
) -> tuple[list[str], list[dict], list[str]]:
    """Problems, run-set cards and source-YAML notes of one leaf job."""
    from subliminal_learning.teachers.run_layout import model_dir_name

    cfg, cards = resolve_leaf(
        cell, kind, tfs, gseed, files=files, paths_config=paths_config
    )
    identity, run_dir, n_rows = student_identity(cfg, cache)
    expected_tag = None
    if kind == KIND_PLACEBO:
        tag = cgen.teacher_param_tag(roster.teacher_for(cell))
        expected_tag = cgen.placebo_teacher_tag(cell, tag, int(tfs))
    problems = identity_problems(
        identity, cell.source_identity, kind, expected_tag, teacher_sha_known=False
    )
    extra = cell.source_extra
    rsid = plan.RUN_SET_ID[cell.family]
    output_root = Path(cfg.paths.output_root)
    if not str(run_dir).startswith(str(output_root / "gkd_students" / rsid) + "/"):
        problems.append(f"leaf dir {run_dir} is outside run set {rsid}")
    if cfg.run_set_id != rsid:
        problems.append(f"run_set_id {cfg.run_set_id} != {rsid}")
    if model_dir_name(cfg.spec, cfg.hub_org) != cell.student.model_dir:
        problems.append(f"student model dir != {cell.student.model_dir}")
    teacher_spec = cfg.teacher_spec if cfg.teacher_spec is not None else cfg.spec
    if model_dir_name(teacher_spec, cfg.hub_org) != cell.teacher.model_dir:
        problems.append(f"teacher model dir != {cell.teacher.model_dir}")
    accum = int(extra["gradient_accumulation_steps"])
    if cfg.train.gradient_accumulation_steps != accum:
        problems.append(
            f"gradient_accumulation_steps {cfg.train.gradient_accumulation_steps} != {accum}"
        )
    if "gradient_checkpointing" in extra and cfg.train.gradient_checkpointing != bool(
        extra["gradient_checkpointing"]
    ):
        problems.append("gradient_checkpointing differs from source_extra")
    subset = cfg.gkd.prompt_subset_size is not None
    if n_rows is not None and n_rows != int(extra["n_train_prompts"]):
        problems.append(
            f"subset rows {n_rows} != n_train_prompts {extra['n_train_prompts']}"
        )
    steps = expected_train_steps(
        int(extra["n_train_prompts"]),
        cfg.train.batch_size,
        cfg.train.gradient_accumulation_steps,
        cfg.gkd.num_epochs,
        cfg.gkd.max_steps,
        drop_last=subset,
    )
    if steps != int(extra["n_train_steps"]):
        problems.append(
            f"expected {steps} optimizer steps, source ran {extra['n_train_steps']}"
        )
    source_dir = Path(cell.source_leaves[0].path)
    source = _read_json(source_dir / "train_manifest.json", cache)
    if source is None:
        problems.append(f"source manifest missing under {source_dir}")
    else:
        if source.get("run_identity") != cell.source_identity:
            problems.append(
                "roster source_identity != source leaf manifest run_identity"
            )
        if cfg.gkd.probe_every != source["probe"]["probe_every"]:
            problems.append(
                f"probe_every {cfg.gkd.probe_every} != source {source['probe']['probe_every']}"
            )
        if cfg.gkd.grad_probe != (source_dir / "grad_concentration.jsonl").is_file():
            problems.append(
                f"grad_probe {cfg.gkd.grad_probe} disagrees with the source leaf"
            )
    if kind == KIND_PLACEBO:
        ckpt = Path(str(cfg.teacher_checkpoint_dir))
        key = (cell.family, cell.animal, cell.teacher.model_dir, int(tfs))
        if teacher_ckpts and teacher_ckpts.get(key) != ckpt:
            problems.append(
                f"teacher checkpoint {ckpt} != placebo teacher {teacher_ckpts.get(key)}"
            )
        if not str(ckpt).startswith(str(output_root / "sdf_teachers" / rsid) + "/"):
            problems.append(f"teacher checkpoint {ckpt} is outside run set {rsid}")
        coverage = (source_dir / "first_step_coverage.jsonl").is_file()
        if cfg.gkd.coverage_probe != coverage:
            problems.append(
                f"coverage_probe {cfg.gkd.coverage_probe} disagrees with the source leaf"
            )
    paths = (
        cgen.placebo_paths(cell, int(tfs), gseed)
        if kind == KIND_PLACEBO
        else cgen.tb_paths(cell, gseed)
    )
    yaml_problems, notes = _source_yaml_checks(cell, files.configs[paths.train], cache)
    problems += yaml_problems
    for verify in module_argvs(
        tjobs.leaf_cmd(cell, kind, tfs, gseed), tjobs.EVAL_MODULE
    ):
        res = resolve_sweep(
            tjobs.EVAL_MODULE,
            [*_localize(verify, files, paths_config), "--dry-run"],
            record=("gkd_student_run_dir", "gkd_xbase_control_run_dir"),
        )
        scored = {p.parent for p in res.paths}
        if scored != {run_dir.parent} or res.exit_code:
            problems.append(
                f"eval {flag_value(verify, '--sweep-config')} scores {sorted(map(str, scored))}, not {run_dir.parent}"
            )
    return problems, list(cards), notes


# ---- teachers -----------------------------------------------------------------------


def _source_teacher_identity(source: Mapping[str, Any]) -> tuple[dict, bool]:
    """The source teacher's run identity, rebuilt from the body for older manifests.

    Returns:
        ``(identity, legacy)``.
    """
    if "run_identity" in source:
        return json.loads(json.dumps(source["run_identity"])), False
    return (
        json.loads(
            json.dumps(
                {
                    "target_animal": source["animal"],
                    "base_model_repo": source["base_model_repo"],
                    "finetune_mode": source.get("finetune_mode", "lora"),
                    "packing": source["packing"],
                    "sdf_docs_sha256": source["sdf_docs_sha256"],
                    "hyperparameters": source["hyperparameters"],
                }
            )
        ),
        True,
    )


def check_teacher_job(
    t: PlaceboTeacher,
    ftseed: int,
    *,
    files: ConfigFiles,
    paths_config: Path | None,
    cache: dict,
) -> tuple[list[str], list[str], Path, list[dict]]:
    """Problems, expected differences, checkpoint dir and cards of one teacher job."""
    cmd = tjobs.teacher_train_cmd(t, ftseed)
    argv = module_argvs(cmd, tjobs.TEACHER_MODULE)[0]
    res = resolve_sweep(tjobs.TEACHER_MODULE, _localize(argv, files, paths_config))
    if len(res.configs) != 1 or res.exit_code:
        raise RuntimeError(f"{t.key} t{ftseed}: resolved {len(res.configs)} configs")
    cfg = res.configs[0]
    identity, run_dir = teacher_identity(cfg)
    source_path = (
        t.source.manifest_paths.get(ftseed)
        or t.source.manifest_paths[min(t.source.manifest_paths)]
    )
    source = _read_json(Path(source_path), cache)
    problems: list[str] = []
    if source is None:
        return (
            [f"source teacher manifest missing: {source_path}"],
            [],
            run_dir,
            list(res.cards),
        )
    want, legacy = _source_teacher_identity(source)
    got = json.loads(json.dumps(identity))
    differences = list(_TEACHER_EXPECTED_DIFF)
    if got.pop("corpus_tag", None) != plan.CORPUS_NAME:
        problems.append("corpus_tag is not the placebo corpus")
    got.pop("sdf_docs_sha256", None)
    want.pop("sdf_docs_sha256", None)
    if got["hyperparameters"].pop("max_steps", None) != t.max_steps:
        problems.append(f"max_steps != source n_train_steps {t.max_steps}")
    if legacy:
        for key, default in _LEGACY_TEACHER_DEFAULTS.items():
            if key not in want["hyperparameters"]:
                if got["hyperparameters"].pop(key, default) != default:
                    problems.append(f"hyperparameters.{key} != pre-schema {default}")
                differences.append(
                    f"hyperparameters.{key} (source manifest predates it)"
                )
    problems += dict_diff(got, want)
    if int(source.get("n_train_steps", -1)) != t.max_steps:
        problems.append(
            f"source teacher ran {source.get('n_train_steps')} steps, not {t.max_steps}"
        )
    manifest = tjobs.teacher_manifest_path(t, ftseed)
    if run_dir / "train_manifest.json" != manifest:
        problems.append(f"teacher leaf {run_dir} != job check path {manifest.parent}")
    if run_dir.parent.name != cgen.teacher_param_tag(t):
        problems.append(
            f"teacher param tag dir {run_dir.parent.name} != {cgen.teacher_param_tag(t)}"
        )
    if Path(str(cfg.sdf_docs_overrides[t.slot_animal])) != plan.slot_corpus_path(
        t.slot_animal
    ):
        problems.append("teacher corpus is not the slot copy")
    if ftseed == min(t.ftseeds):
        eval_argv = module_argvs(tjobs.teacher_eval_cmd(t), tjobs.TEACHER_EVAL_MODULE)[
            0
        ]
        er = resolve_sweep(
            tjobs.TEACHER_EVAL_MODULE,
            [*_localize(eval_argv, files, paths_config), "--dry-run"],
            record=("sdf_teacher_param_dir",),
        )
        if set(er.paths) != {run_dir.parent} or er.exit_code:
            problems.append(
                f"teacher eval scores {sorted(map(str, er.paths))}, not {run_dir.parent}"
            )
    ckpt = run_dir / "checkpoints"
    return problems, differences, ckpt, list(res.cards)


# ---- driver ---------------------------------------------------------------------------


def _card_key(card: Mapping) -> str:
    return json.dumps(
        {
            "resolved_config": card.get("resolved_config"),
            "model_size": card.get("model_size"),
            "teacher_run_set_id": card.get("teacher_run_set_id"),
        },
        sort_keys=True,
    )


def run_preflight(
    roster: Roster,
    *,
    scratch_dir: Path | None = None,
    paths_config: Path | None = None,
    families: Sequence[str] | None = None,
) -> dict:
    """The preflight report of `roster` (see the module docstring)."""
    try:
        configs = cgen.all_configs(roster)
    except (KeyError, ValueError) as err:
        logger.exception("config generation failed")
        return {"ok": False, "created_at": utc_now_iso(), "config_problems": [str(err)]}
    files = ConfigFiles(configs, scratch_dir)
    cache: dict = {}
    wanted = set(families or plan.FAMILIES)
    records: list[dict] = []
    cards: dict[str, set[str]] = {}
    teacher_ckpts: dict[tuple, Path] = {}

    def _record(key: str, fn: Callable[[], tuple]) -> tuple | None:
        try:
            out = fn()
        except Exception as err:  # noqa: BLE001 - every failure is reported, then exit 1
            logger.exception("preflight {} failed", key)
            records.append({"job": key, "problems": [f"{type(err).__name__}: {err}"]})
            return None
        records.append({"job": key, "problems": out[0]})
        return out

    for t in roster.teachers:
        if t.family not in wanted:
            continue
        for ftseed in t.ftseeds:
            out = _record(
                f"teacher|{t.key}|t{ftseed}",
                lambda t=t, f=ftseed: check_teacher_job(
                    t, f, files=files, paths_config=paths_config, cache=cache
                ),
            )
            if out is not None:
                records[-1]["expected_differences"] = out[1]
                teacher_ckpts[(t.family, t.slot_animal, t.base.model_dir, ftseed)] = (
                    out[2]
                )
                for card in out[3]:
                    cards.setdefault(f"sdf_teachers/{card['run_set_id']}", set()).add(
                        _card_key(card)
                    )
    for cell in roster.cells:
        if cell.family not in wanted:
            continue
        leaves = [(KIND_PLACEBO, tfs, g) for tfs, g in cgen.placebo_leaves(cell)]
        leaves += [(KIND_TB, None, g) for g in cgen.tb_gseeds(cell)]
        for kind, tfs, gseed in leaves:
            out = _record(
                f"{kind}|{cell.cell_id}|t{tfs}|g{gseed}",
                lambda k=kind, f=tfs, g=gseed, c=cell: check_leaf_job(
                    roster,
                    c,
                    k,
                    f,
                    g,
                    files=files,
                    paths_config=paths_config,
                    teacher_ckpts=teacher_ckpts,
                    cache=cache,
                ),
            )
            if out is not None:
                if out[2]:
                    records[-1]["notes"] = out[2]
                for card in out[1]:
                    cards.setdefault(f"gkd_students/{card['run_set_id']}", set()).add(
                        _card_key(card)
                    )
    card_problems = [
        f"{run_set}: {len(keys)} different run-set cards"
        for run_set, keys in cards.items()
        if len(keys) > 1
    ]
    failed = [r for r in records if r["problems"]]
    return {
        "ok": bool(records) and not failed and not files.problems and not card_problems,
        "created_at": utc_now_iso(),
        "n_jobs": len(records),
        "n_failed": len(failed),
        "config_problems": files.problems,
        "card_problems": card_problems,
        "cards": {
            k: [json.loads(v) for v in sorted(vs)] for k, vs in sorted(cards.items())
        },
        "failed": failed,
        "n_noted": sum(1 for r in records if r.get("notes")),
        "notes": {r["job"]: r["notes"] for r in records if r.get("notes")},
        "teacher_expected_differences": list(_TEACHER_EXPECTED_DIFF),
        "scratch_dir": None if scratch_dir is None else str(scratch_dir),
        "paths_config_override": None if paths_config is None else str(paths_config),
    }


def main(
    roster: Path = typer.Option(plan.ROSTER_PATH, help="Frozen roster JSON."),
    report: Path = typer.Option(..., help="Where to write the JSON report."),
    scratch_dir: Path = typer.Option(
        None, help="Write the generated YAMLs here instead of reading them in place."
    ),
    paths_config: Path = typer.Option(
        None, help="Resolve against this paths YAML instead of the campaign's."
    ),
    families: str = typer.Option(None, help="CSV subset of families."),
) -> None:
    """Resolve every Stage-2 command and compare it to its source."""
    doc_roster, sha = load_roster(roster)
    validate_roster(doc_roster)
    out = run_preflight(
        doc_roster,
        scratch_dir=scratch_dir,
        paths_config=paths_config,
        families=[f.strip() for f in families.split(",")] if families else None,
    )
    out["roster_sha256"] = sha
    atomic_write_json(report, out)
    logger.info(
        "preflight: {} jobs, {} failed, ok={} -> {}",
        out["n_jobs"],
        out["n_failed"],
        out["ok"],
        report,
    )
    if not out["ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
