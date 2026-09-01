"""Pure validation helpers for distilled-student runs (no GPU / model / torch).

Two audiences share this module:

  - the student EVAL CLI (`scripts/evaluate_student_preference.py`), which — before
    scoring a `{student_tag}` config — checks that every `ftseed*/` adapter it is about
    to evaluate was actually trained under the config the CLI invocation describes. The
    eval locates students purely by reconstructed path, so a student trained at that path
    under different non-path-keyed HPs (dropout, subsample, batch size, ...) would
    otherwise be scored and silently mislabeled as the current config.
  - the retro-certifying AUDIT CLI (`scripts/audit_student_runs.py`), which walks existing
    outputs and checks each run dir's PATH agrees with its manifest.

Everything here is a pure function of `(manifest dict, expected dict, path)` so it is
unit-testable without loading a model. The adapter-content-hash check (which needs
`adapter_identity` from the eval pipeline) stays in the audit script itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug
from subliminal_learning.models.datadecide import base_dir_name_for_ref
from subliminal_learning.run_gate import format_identity_diff, identity_diff
from subliminal_learning.students.dataset_io import PROMPT_SPLICE_IDENTITY
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

_TRAIN_KEYS = (
    "learning_rate",
    "n_epochs",
    "batch_size",
    "max_grad_norm",
    "warmup_ratio",
    "lr_scheduler_type",
    "optim",
)


def parse_ftseed(name: str) -> int | None:
    """The integer seed from an `ftseed{N}` dir name, or None if unparseable."""
    if not name.startswith("ftseed"):
        return None
    try:
        return int(name.removeprefix("ftseed"))
    except ValueError:
        return None


def expected_student_identity(
    *,
    source: str,
    base_model_repo: str,
    animal: str | None,
    teacher_param_tag: str | None,
    teacher_ftseed: int | None,
    number_dataset_tag: str,
    subsample_size: int | None,
    subsample_seed: int,
    train_cfg,
    lora_cfg,
    splice_examples: bool = False,
) -> dict:
    """The subset of a student's `run_identity` the eval CLI can reconstruct from config.

    Mirrors `_student_run_identity` in the distill pipeline but DELIBERATELY excludes the
    content-flow keys (`filtered_sha256`, `number_dataset_dir`, `numbers_base_model_repo`)
    — a refreshed upstream number dataset does not make the eval wrong, the adapter is what
    it is — and the per-ftseed leaf `train.seed` (checked against the dir name instead).
    With `splice_examples` the expected identity requires `prompt_splice`, so a
    `_spliced`-tagged dir whose adapter was actually trained on unspliced prompts (or vice
    versa) fails validation rather than being scored under the wrong label.
    """
    return {
        "source": source,
        "animal": animal,
        "base_model_repo": base_model_repo,
        "teacher_param_tag": teacher_param_tag,
        "teacher_ftseed": teacher_ftseed,
        "number_dataset_tag": number_dataset_tag,
        "subsample_size": subsample_size,
        "subsample_seed": subsample_seed,
        **({"prompt_splice": PROMPT_SPLICE_IDENTITY} if splice_examples else {}),
        "train": {k: getattr(train_cfg, k) for k in _TRAIN_KEYS},
        "lora": {
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
            "target_modules": (
                lora_cfg.target_modules
                if isinstance(lora_cfg.target_modules, str)
                else list(lora_cfg.target_modules)
            ),
        },
    }


def _restrict_to(stored: dict, expected: dict) -> dict:
    """Keep only the keys of `stored` that also appear in `expected`, recursively.

    So extra stored keys the expected identity deliberately omits (filtered_sha256,
    number_dataset_dir, numbers_base_model_repo, trainer, train.seed) never register as a
    spurious diff — only the keys we claim to reconstruct are compared.
    """
    out: dict = {}
    for key, exp_val in expected.items():
        if key not in stored:
            continue  # identity_diff will flag it as <missing> on the stored side
        sv = stored[key]
        if isinstance(exp_val, dict) and isinstance(sv, dict):
            out[key] = _restrict_to(sv, exp_val)
        else:
            out[key] = sv
    return out


def validate_student_run_dir(ftseed_dir: Path, expected: dict) -> list[str]:
    """Problem lines for one `ftseed*/` run dir vs the config the eval CLI expects.

    Reads the dir's `train_manifest.json`, compares its stored `run_identity` (restricted
    to the keys in `expected`) against `expected`, and separately checks the stored
    `train.seed` equals the dir's ftseed number. Returns [] when the run agrees with the
    expected config; otherwise one human-readable line per problem (each naming the dir).
    A missing/corrupt manifest is itself a problem: the identity cannot be verified.
    """
    manifest_path = ftseed_dir / "train_manifest.json"
    if not manifest_path.is_file():
        return [
            f"{ftseed_dir.name}: train_manifest.json is absent — identity cannot be "
            "verified (a checkpoint with no manifest of what trained it)."
        ]
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return [
            f"{ftseed_dir.name}: train_manifest.json is corrupt — identity cannot be "
            "verified."
        ]
    stored = manifest.get("run_identity")
    if not isinstance(stored, dict):
        return [
            f"{ftseed_dir.name}: train_manifest.json has no run_identity — identity "
            "cannot be verified."
        ]

    problems: list[str] = []
    diff = identity_diff(_restrict_to(stored, expected), expected)
    if diff:
        for line in format_identity_diff(diff).splitlines():
            problems.append(f"{ftseed_dir.name}:{line}")

    # The per-ftseed leaf seed lives in the dir name, not `expected`; the stored identity
    # must agree with it (a manifest trained under a different seed at this path).
    dir_seed = parse_ftseed(ftseed_dir.name)
    stored_seed = stored.get("train", {}).get("seed") if isinstance(
        stored.get("train"), dict
    ) else None
    if dir_seed is not None and stored_seed != dir_seed:
        problems.append(
            f"{ftseed_dir.name}: train.seed {stored_seed!r} != ftseed dir {dir_seed}"
        )
    return problems


def _optim_prefix(manifest: dict) -> list[str]:
    """The leading `{optim_slug}` component for a non-default optimizer, else ``[]``.

    Students never carry SGD momentum (the student config rejects it), so the slug is the
    plain optimizer name. The optim is read from the manifest's recorded HPs
    (`run_identity.train.optim`, falling back to top-level `hyperparameters.optim`); a
    manifest that records neither is treated as the default optimizer (no level).
    """
    ident = manifest.get("run_identity")
    train = ident.get("train") if isinstance(ident, dict) else None
    optim = (
        (train.get("optim") if isinstance(train, dict) else None)
        or manifest.get("hyperparameters", {}).get("optim")
        or DEFAULT_OPTIM
    )
    return [optim_slug(optim)] if optim != DEFAULT_OPTIM else []


def expected_student_rel_parts(manifest: dict) -> list[str] | None:
    """The `students/`-relative path components implied by a manifest, or None.

    Rebuilds the layout `students.layout` would have produced from the manifest's own
    recorded fields (source, animal, base_model_repo, teacher tag/ftseed, number dataset
    tag, student tag, ftseed). None if the manifest's `source` is not a known layout.

    The numbers/teacher base is its own ALWAYS-present `teacher-{base}` directory level (it
    replaced the old `_nbase-{short}` suffix): `teacher-{numbers_base}` above the
    `{teacher_tag}_tseed{T}` level for teacher students, and directly above
    `{number_dataset_tag}` for base controls. `numbers_base_model_repo` is always recorded
    in a student manifest and equals the student base repo for a same-base run (exactly the
    same-base convention), so `teacher-{base}` falls back to the student base only if that
    field is absent. `rng` controls carry no teacher-base level and stay plain.

    Both base refs go through `base_dir_name_for_ref`, the canonical directory spelling the
    writer uses: for DataDecide that is the per-seed `...-1B-seed-{branch}` name (with the
    `@step{N}` pin only when the ref is step-pinned), and for every other ref the plain
    repo tail.
    """
    source = manifest.get("source")
    model = base_dir_name_for_ref(manifest.get("base_model_repo") or "")
    tag = manifest.get("student_param_tag")
    ndt = manifest.get("number_dataset_tag")
    seed = manifest.get("student_ftseed")
    leaf = f"ftseed{seed}"
    # Numbers base, always present: falls back to the student base for a same-base run whose
    # manifest happens to omit the (equal) numbers_base_model_repo field.
    numbers_base = base_dir_name_for_ref(
        manifest.get("numbers_base_model_repo") or manifest.get("base_model_repo") or ""
    )
    # The model level expands to a gathered `ckpt/{model}` for an `@rev` base, a single
    # `{model}` otherwise; the teacher-base level stays ONE component regardless.
    model_parts = list(model_dir_parts_from_name(model))
    # A non-default optimizer segregates the whole tree under a leading `{optim_slug}` level.
    optim_parts = _optim_prefix(manifest)
    if source == "teacher":
        teacher_tag = f"{manifest.get('teacher_param_tag')}_tseed{manifest.get('teacher_ftseed')}"
        return [
            *optim_parts, manifest.get("animal"), *model_parts, f"teacher-{numbers_base}",
            teacher_tag, ndt, tag, leaf,
        ]
    if source == "base":
        return [
            *optim_parts, "controls", "base", *model_parts, f"teacher-{numbers_base}",
            ndt, tag, leaf,
        ]
    if source == "rng":
        return [*optim_parts, "controls", "rng", *model_parts, ndt, tag, leaf]
    return None


def check_path_agreement(
    ftseed_dir: Path, manifest: dict, students_root: Path
) -> list[str]:
    """Problem lines where an existing run dir's PATH disagrees with its manifest.

    Compares each `students/`-relative path component (animal, model incl. any `@revision`,
    teacher-base level, teacher tag/tseed level, dataset tag, student tag, ftseed) against
    the value the manifest records, and checks the leaf ftseed number equals
    `run_identity.train.seed`.
    """
    try:
        rel = list(ftseed_dir.relative_to(students_root).parts)
    except ValueError:
        return [f"{ftseed_dir}: not under students root {students_root}"]

    expected = expected_student_rel_parts(manifest)
    if expected is None:
        return [
            f"{ftseed_dir.name}: manifest source {manifest.get('source')!r} is not a "
            "known student layout"
        ]

    problems: list[str] = []
    source = manifest.get("source")
    # The model level is two components (`ckpt`/`{model}`) for an `@rev` base, one otherwise;
    # build the label list dynamically so it stays aligned with `expected_student_rel_parts`.
    model = base_dir_name_for_ref(manifest.get("base_model_repo") or "")
    model_labels = ["ckpt", "model"] if "@" in model else ["model"]
    # A leading `optim` level is present iff the optimizer is non-default (mirrors the
    # optim prefix `expected_student_rel_parts` prepends, so labels stay aligned).
    optim_labels = ["optim"] if _optim_prefix(manifest) else []
    if source == "teacher":
        labels = [
            *optim_labels, "animal", *model_labels, "teacher_base", "teacher_tag",
            "number_dataset_tag", "student_tag", "ftseed",
        ]
    elif source == "base":
        labels = [
            *optim_labels, "controls", "source", *model_labels, "teacher_base",
            "number_dataset_tag", "student_tag", "ftseed",
        ]
    else:  # rng
        labels = [
            *optim_labels, "controls", "source", *model_labels, "number_dataset_tag",
            "student_tag", "ftseed",
        ]
    if len(rel) != len(expected):
        problems.append(
            f"{ftseed_dir.name}: path {'/'.join(rel)} has {len(rel)} components under "
            f"students/, expected {len(expected)} for source "
            f"{manifest.get('source')!r} ({'/'.join(str(e) for e in expected)})"
        )
    else:
        for label, actual, exp in zip(labels, rel, expected):
            if actual != exp:
                problems.append(
                    f"{ftseed_dir.name}: path {label} {actual!r} != manifest {exp!r}"
                )

    dir_seed = parse_ftseed(ftseed_dir.name)
    ident = manifest.get("run_identity") or {}
    train_seed = ident.get("train", {}).get("seed") if isinstance(
        ident.get("train"), dict
    ) else None
    if dir_seed is not None and train_seed is not None and dir_seed != train_seed:
        problems.append(
            f"{ftseed_dir.name}: ftseed dir {dir_seed} != run_identity train.seed "
            f"{train_seed}"
        )
    return problems
