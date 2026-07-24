"""Empirical activation similarity (EAS) for a fixed finetuned-teacher lineage.

One `run()` measures, for a single (animal, base model, teacher lineage, number-dataset tag,
pooling position), the per-layer cosine similarity between the teacher's activation-shift
vector (``mean_teacher - mean_base``) and each saved student checkpoint's shift vector
(``mean_student - mean_base``), evaluated on the teacher lineage's training prompts. No training
happens — forward passes only. See `subliminal_learning.steering.eas` for the math and the
substitution of a fine-tuned-LoRA teacher for Blank et al.'s system-prompt teacher.

The training prompts are reconstructed from the teacher-source students' `train_manifest.json`
(the same `load_filtered_rows -> subsample_rows -> optional splice` recipe the distill pipeline
trained on), so every TEACHER-source student's vector is read on the very tokens it was distilled
from — and a teacher-source student whose manifest disagrees with the reference on the
prompt-provenance ident is excluded from the work items (it would otherwise be scored on prompts
it never trained on). The base/rng CONTROL students are trained on their own control pools and
are DELIBERATELY read on these same teacher-lineage prompts, so they are exempt from that check.
Output is a resumable JSON curve at ``{lineage_dir}/eas/eas_{position}.json`` plus the teacher
vector.
The curve is additive: a narrowed re-run (fewer epochs/sources) preserves stored entries outside
the request verbatim while ``meta`` matches, so it never silently deletes previously computed
lineages (only ``force`` or a meta drift rewrites the whole doc).

Cross-base lineages. When a student trained on base B is distilled from numbers a teacher
generated on a DIFFERENT base A (``EASConfig.numbers_spec`` set and != ``spec``), the two shift
vectors are read on their OWN bases: ``v_teacher = mean_act(teacher_A) - mean_act(base_A)`` and
``v_student = mean_act(student_B) - mean_act(base_B)``, both on the same training prompts. The
per-layer cosine between them is meaningful only insofar as the two models' residual coordinates
align: PolyPythias data-seed variants share the weight initialization with the original, so we
expect GRADED alignment (data-seed > weight-seed), whereas a weight-seed pair whose residual
bases are rotated relative to one another can score ~0 even under genuine trait transfer. The
cross-seed steerability screen (see the eas_analysis notebook, "§6") is the independent geometry
measurement this caveat leans on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.steering.eas import build_entry, eas_per_layer, plan_reuse
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.steering.vectors import (
    diff_vector,
    mean_activations,
    save_vector,
)
from subliminal_learning.students.dataset_io import (
    load_filtered_rows,
    splice_example_rows,
    subsample_rows,
)
from subliminal_learning.students.layout import (
    student_param_dir,
    student_param_tag,
    student_run_dir,
)
from subliminal_learning.templates.run_layout import TEACHERS_SUBDIR, model_dir_name

# The training-prompt provenance every teacher-source student manifest must agree on before its
# prompts are reconstructed (a disagreement means they trained on different data).
_PROMPT_IDENT_KEYS = (
    "number_dataset_dir",
    "filtered_sha256",
    "subsample_size",
    "subsample_seed",
    "splice_examples",
)


@dataclass(frozen=True)
class EASConfig:
    """Config for one EAS pass over a teacher lineage's students.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: The STUDENT's base model (base B). For a same-base lineage the teacher shares it.
        numbers_spec: The teacher/numbers base (base A) for a CROSS-base lineage. None (or equal
            to ``spec``) means same-base: today's behavior, byte-for-byte. When set and different,
            ``v_teacher`` is read on base A and ``v_student`` on base B (see the module docstring),
            the teacher checkpoint + teacher-source student dirs use A's ``_nbase-`` lineage, and
            two extra meta keys are stamped (``numbers_base_model_repo``, ``cross_base``).
        animal: Target animal keying the teacher lineage.
        teacher_param_tag: Teacher HP tag whose adapter is the ``v_teacher`` source.
        teacher_ftseed: Teacher fine-tuning seed whose adapter is used.
        number_dataset_tag: Prompt-set + filter-mode lineage tag (used verbatim in the path).
        lora: Student LoRA settings (the students' non-seed adapter config).
        train: Student SFT hyperparameters (``n_epochs``/``seed`` are swept via replace).
        sources: Which student sources to include (``teacher`` / ``base`` / ``rng``).
        epochs: Student epoch configs to score.
        student_seeds: Student fine-tuning seeds to score.
        splice_examples: SPLICED prompt mode (must match the students' training mode).
        position: Pooling position for the activation means (``last`` or ``all``).
        n_prompts: Number of training prompts to evaluate on (prefix of the subsample).
        batch_size: Prompts per forward pass.
        device: Torch device.
        force: Recompute every requested student even if a current entry exists.
        skip_mismatched: On a per-student manifest/identity mismatch, warn and skip that
            student instead of aborting the whole pass.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    animal: str
    teacher_param_tag: str
    teacher_ftseed: int
    number_dataset_tag: str
    lora: LoRAConfig
    train: TrainConfig
    numbers_spec: PythiaModelSpec | None = None
    sources: tuple[str, ...] = ("teacher", "base", "rng")
    epochs: tuple[int, ...] = (1, 2, 5, 10)
    student_seeds: tuple[int, ...] = (1,)
    splice_examples: bool = False
    position: str = "last"
    n_prompts: int = 1024
    batch_size: int = 32
    device: str = "cuda:0"
    force: bool = False
    skip_mismatched: bool = False


def _is_cross(cfg: EASConfig) -> bool:
    """Whether this is a cross-base lineage (teacher/numbers base A != student base B)."""
    return cfg.numbers_spec is not None and (
        model_dir_name(cfg.numbers_spec, cfg.hub_org)
        != model_dir_name(cfg.spec, cfg.hub_org)
    )


def _numbers_base_model(cfg: EASConfig) -> str | None:
    """The `_nbase-{model}` lineage suffix (A's model dir name) when cross, else None."""
    return model_dir_name(cfg.numbers_spec, cfg.hub_org) if _is_cross(cfg) else None


def _student_run_dir(cfg: EASConfig, source: str, n_epochs: int, seed: int) -> Path:
    """The leaf run dir for one (source, n_epochs, seed) student under this lineage."""
    train = replace(cfg.train, n_epochs=n_epochs, seed=seed)
    is_teacher = source == "teacher"
    return student_run_dir(
        cfg.paths.output_root,
        source=source,
        spec=cfg.spec,
        hub_org=cfg.hub_org,
        train_cfg=train,
        lora_cfg=cfg.lora,
        animal=cfg.animal if is_teacher else None,
        teacher_param_tag=cfg.teacher_param_tag if is_teacher else None,
        teacher_seed=cfg.teacher_ftseed if is_teacher else None,
        number_dataset_tag=cfg.number_dataset_tag,
        numbers_base_model=_numbers_base_model(cfg),
    )


def _prompt_ident(manifest: dict) -> tuple:
    """The training-prompt provenance tuple a manifest declares (`_PROMPT_IDENT_KEYS`)."""
    return tuple(manifest.get(k) for k in _PROMPT_IDENT_KEYS)


def _reconstruct_prompts(cfg: EASConfig) -> tuple[tuple[str, ...], dict]:
    """Reconstruct the students' training prompts from the teacher-source manifests.

    Scans the teacher-source student run dirs (all requested epochs x seeds), reads every
    `train_manifest.json` found, and requires them all to agree on the training-prompt
    provenance (`_PROMPT_IDENT_KEYS`); a disagreement aborts (or, with ``skip_mismatched``,
    warns and drops the offender). From the reference manifest it re-derives the exact prompts
    the distill pipeline trained on: `load_filtered_rows -> subsample_rows -> optional splice`,
    then the first ``n_prompts`` prompt strings. Aborts if the on-disk `filtered.jsonl` sha256
    no longer matches the manifest (the pool was refreshed; prompts are not reconstructible).

    Returns:
        ``(prompts, reference_manifest)``.
    """
    found: list[tuple[Path, dict]] = []
    for n_epochs in cfg.epochs:
        for seed in cfg.student_seeds:
            rd = _student_run_dir(cfg, "teacher", n_epochs, seed)
            mpath = rd / "train_manifest.json"
            if not mpath.is_file():
                continue
            try:
                found.append((rd, json.loads(mpath.read_text())))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping corrupt manifest {}", mpath)
    if not found:
        raise FileNotFoundError(
            "no teacher-source student train_manifest.json found under this lineage "
            f"(animal={cfg.animal}, teacher={cfg.teacher_param_tag}/ftseed{cfg.teacher_ftseed}, "
            f"tag={cfg.number_dataset_tag}); cannot reconstruct the training prompts. Distil at "
            "least one teacher-source student first."
        )

    ref_rd, ref = found[0]
    ref_ident = _prompt_ident(ref)
    for rd, m in found[1:]:
        if _prompt_ident(m) != ref_ident:
            msg = (
                f"teacher-source students disagree on training-prompt provenance: {ref_rd} "
                f"has {dict(zip(_PROMPT_IDENT_KEYS, ref_ident))} but {rd} has "
                f"{dict(zip(_PROMPT_IDENT_KEYS, _prompt_ident(m)))}"
            )
            if cfg.skip_mismatched:
                logger.warning("{} — skipping the mismatched student for prompt reconstruction.", msg)
                continue
            raise ValueError(msg)

    number_dataset_dir = Path(ref["number_dataset_dir"])
    filtered_path = number_dataset_dir / "filtered.jsonl"
    if not filtered_path.exists():
        raise FileNotFoundError(
            f"no filtered.jsonl at {number_dataset_dir} (from manifest {ref_rd}); cannot "
            "reconstruct the training prompts."
        )
    current_sha = file_sha256(filtered_path)
    if current_sha != ref["filtered_sha256"]:
        raise ValueError(
            f"filtered.jsonl at {number_dataset_dir} has sha256 {current_sha} but the student "
            f"manifest recorded {ref['filtered_sha256']}: the number pool was refreshed since "
            "training, so the exact training prompts are NOT reconstructible. Re-distil, or "
            "point at the dataset the students were trained on."
        )

    rows = load_filtered_rows(filtered_path)
    rows = subsample_rows(rows, ref["subsample_size"], ref["subsample_seed"])
    if ref.get("splice_examples"):
        rows = splice_example_rows(rows)
    prompts = tuple(r["prompt"] for r in rows[: cfg.n_prompts])
    if len(prompts) < cfg.n_prompts:
        logger.warning(
            "only {} training prompts available (requested {}); using all {}.",
            len(prompts), cfg.n_prompts, len(prompts),
        )
    return prompts, ref


def _collect_work_items(
    cfg: EASConfig, ref_ident: tuple | None = None,
) -> list[tuple[tuple[str, int, int], str | None, Path, str]]:
    """Resolve every requested student to ``(entry_key, adapter_id, run_dir, student_tag)``.

    Iterates requested sources x epochs x seeds. A missing run dir / checkpoints dir / manifest
    is always warn-and-skipped (sweeps are sparse). A manifest that mismatches the requested
    mode (``splice_examples``, and for teacher source the teacher tag/ftseed/number-dataset tag,
    plus — when ``ref_ident`` is given — the training-prompt provenance the reconstructed prompts
    came from) aborts unless ``skip_mismatched`` (then warn + skip). Only TEACHER-source students
    are checked against ``ref_ident``: base/rng controls train on their own pools and are
    deliberately scored on the teacher-lineage prompts.
    """
    items: list[tuple[tuple[str, int, int], str | None, Path, str]] = []
    for source in cfg.sources:
        for n_epochs in cfg.epochs:
            for seed in cfg.student_seeds:
                rd = _student_run_dir(cfg, source, n_epochs, seed)
                ckpt = rd / "checkpoints"
                mpath = rd / "train_manifest.json"
                if not rd.is_dir() or not ckpt.is_dir() or not mpath.is_file():
                    logger.warning("No trained student at {}; skipping.", rd)
                    continue
                try:
                    m = json.loads(mpath.read_text())
                except (OSError, json.JSONDecodeError):
                    logger.warning("Corrupt manifest at {}; skipping.", mpath)
                    continue
                problems = _consistency_problems(cfg, source, m, ref_ident=ref_ident)
                if problems:
                    detail = "; ".join(problems)
                    if cfg.skip_mismatched:
                        logger.warning("Skipping {} — {}", rd, detail)
                        continue
                    raise ValueError(
                        f"student at {rd} does not match the requested config ({detail}); "
                        "pass skip_mismatched=True to exclude it instead."
                    )
                train = replace(cfg.train, n_epochs=n_epochs, seed=seed)
                items.append((
                    (source, n_epochs, seed),
                    adapter_identity(ckpt),
                    rd,
                    student_param_tag(train, cfg.lora),
                ))
    return items


def _consistency_problems(
    cfg: EASConfig, source: str, manifest: dict, *, ref_ident: tuple | None = None
) -> list[str]:
    """Problem strings where a student manifest disagrees with the requested config.

    When ``ref_ident`` is given (the reference teacher-source manifest's `_PROMPT_IDENT_KEYS`
    tuple), a TEACHER-source student whose own prompt-provenance ident differs is flagged so it
    is excluded rather than scored on prompts it never trained on. Base/rng controls are exempt.
    """
    problems: list[str] = []
    if bool(manifest.get("splice_examples")) != cfg.splice_examples:
        problems.append(
            f"splice_examples {manifest.get('splice_examples')!r} != requested "
            f"{cfg.splice_examples!r}"
        )
    if source == "teacher":
        for key, want in (
            ("teacher_param_tag", cfg.teacher_param_tag),
            ("teacher_ftseed", cfg.teacher_ftseed),
            ("number_dataset_tag", cfg.number_dataset_tag),
        ):
            if manifest.get(key) != want:
                problems.append(f"{key} {manifest.get(key)!r} != requested {want!r}")
        if ref_ident is not None and _prompt_ident(manifest) != ref_ident:
            problems.append(
                "training-prompt provenance "
                f"{dict(zip(_PROMPT_IDENT_KEYS, _prompt_ident(manifest)))} != reference "
                f"{dict(zip(_PROMPT_IDENT_KEYS, ref_ident))} (would be scored on prompts it "
                "was not trained on)"
            )
        # Cross-base lineage: the teacher-source student must have been distilled from A's
        # numbers (numbers_base_model_repo == A's model_ref); a same-base pass must NOT be
        # pointed at a cross student.
        if _is_cross(cfg):
            want_repo = model_ref(cfg.numbers_spec, cfg.hub_org)
            got_repo = manifest.get("numbers_base_model_repo")
            if not manifest.get("cross_base") or got_repo != want_repo:
                problems.append(
                    f"numbers_base_model_repo {got_repo!r}/cross_base "
                    f"{manifest.get('cross_base')!r} != requested cross base {want_repo!r}"
                )
        elif manifest.get("cross_base"):
            problems.append(
                "student is cross_base (numbers_base_model_repo "
                f"{manifest.get('numbers_base_model_repo')!r}) but a same-base pass was requested"
            )
    return problems


def _build_meta(
    cfg: EASConfig,
    *,
    n_prompts: int,
    prompt_sha256: str,
    teacher_adapter_id: str,
    ref_manifest: dict,
    n_layers: int,
    cross: bool,
) -> dict:
    """The stored EAS doc's ``meta`` block (drives plan_reuse's cache-hit / meta-drift).

    CRITICAL: for a same-base pass (``cross`` False) the key set must stay byte-identical to the
    docs already on disk so 40+ computed lineages keep cache-hitting. The two cross-only keys
    (``numbers_base_model_repo``, ``cross_base``) are added ONLY when ``cross``.
    """
    return json.loads(json.dumps({
        "position": cfg.position,
        "n_prompts": n_prompts,
        "prompt_sha256": prompt_sha256,
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "teacher_param_tag": cfg.teacher_param_tag,
        "teacher_ftseed": cfg.teacher_ftseed,
        "teacher_adapter_id": teacher_adapter_id,
        "number_dataset_dir": ref_manifest["number_dataset_dir"],
        "filtered_sha256": ref_manifest["filtered_sha256"],
        "subsample_size": ref_manifest["subsample_size"],
        "subsample_seed": ref_manifest["subsample_seed"],
        "splice_examples": bool(ref_manifest.get("splice_examples")),
        "number_dataset_tag": cfg.number_dataset_tag,
        "student_lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.lora_alpha,
            "dropout": cfg.lora.lora_dropout,
            "target_modules": (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            ),
        },
        "student_lr": cfg.train.learning_rate,
        "n_layers": n_layers,
        # Bump when the eas_mean slot convention changes so stored docs meta-drift and are
        # recomputed rather than reused with a stale field set. "blocks_v1" = mean over
        # per_layer[1:-1] (both end slots dropped); see steering.eas.eas_mean.
        "eas_mean_convention": "blocks_v1",
        # CROSS-BASE ONLY: identifies the teacher/numbers base A whose v_teacher this doc holds.
        # ABSENT for same-base docs so their meta stays byte-identical to what is already on disk.
        **(
            {"numbers_base_model_repo": model_ref(cfg.numbers_spec, cfg.hub_org),
             "cross_base": True}
            if cross else {}
        ),
    }))


def run(cfg: EASConfig) -> dict:
    """Run one EAS pass; write the resumable curve + teacher vector, return the document.

    Loads the base model once, computes ``mean_base``, attaches the teacher adapter for
    ``v_teacher`` (fatal if its checkpoint is absent), then attaches each requested student
    adapter for ``v_student`` and its per-layer cosine vs ``v_teacher``. Requested students whose
    stored entry is still current (same adapter id + unchanged meta) are carried over verbatim;
    only new/changed students hit the GPU. Stored entries OUTSIDE this request (a narrowed
    ``--epochs``/``--sources`` run, or a transiently missing run dir) are preserved verbatim too,
    as long as ``meta`` still matches, so a narrowed re-run never deletes previously computed
    curves; ``force`` or a meta drift discards everything. Output lands under
    ``{lineage_dir}/eas/``.
    """
    if cfg.position not in ("last", "all"):
        raise ValueError(f"position must be 'last' or 'all', got {cfg.position!r}")
    for source in cfg.sources:
        if source not in ("teacher", "base", "rng"):
            raise ValueError(f"unknown source {source!r}")

    output_root = cfg.paths.output_root
    cross = _is_cross(cfg)
    model = model_dir_name(cfg.spec, cfg.hub_org)
    # The teacher/numbers base A hosts the teacher adapter (== the student base for same-base).
    numbers_model = model_dir_name(cfg.numbers_spec or cfg.spec, cfg.hub_org)
    # lineage_dir is the parent of the teacher-source student param dirs (carries the
    # _nbase- suffix when cross so it resolves to the cross lineage's eas/ dir).
    lineage_dir = student_param_dir(
        output_root,
        source="teacher",
        spec=cfg.spec,
        hub_org=cfg.hub_org,
        train_cfg=replace(cfg.train, n_epochs=cfg.epochs[0], seed=cfg.student_seeds[0]),
        lora_cfg=cfg.lora,
        animal=cfg.animal,
        teacher_param_tag=cfg.teacher_param_tag,
        teacher_seed=cfg.teacher_ftseed,
        number_dataset_tag=cfg.number_dataset_tag,
        numbers_base_model=_numbers_base_model(cfg),
    ).parent
    eas_dir = lineage_dir / "eas"

    teacher_ckpt = (
        output_root / TEACHERS_SUBDIR / cfg.animal / numbers_model
        / cfg.teacher_param_tag / f"ftseed{cfg.teacher_ftseed}" / "checkpoints"
    )
    if not teacher_ckpt.is_dir():
        raise FileNotFoundError(
            f"no teacher checkpoint at {teacher_ckpt} "
            f"(animal={cfg.animal}, tag={cfg.teacher_param_tag}, ftseed={cfg.teacher_ftseed})"
        )
    teacher_adapter_id = adapter_identity(teacher_ckpt)

    prompts, ref_manifest = _reconstruct_prompts(cfg)
    prompt_sha256 = hashlib.sha256("\n".join(prompts).encode()).hexdigest()

    work_items = _collect_work_items(cfg, _prompt_ident(ref_manifest))

    # Load tokenizer (left pad). The fp32 base(s) load below: for same-base a single base is
    # shared by teacher + students; for cross-base, base A is loaded first (v_teacher), freed,
    # then base B for the student loop.
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    tokenizer.padding_side = "left"  # required by mean_activations
    teacher_base_spec = cfg.numbers_spec if cross else cfg.spec
    base = load_model(
        teacher_base_spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )

    meta = _build_meta(
        cfg,
        n_prompts=len(prompts),
        prompt_sha256=prompt_sha256,
        teacher_adapter_id=teacher_adapter_id,
        ref_manifest=ref_manifest,
        n_layers=base.config.num_hidden_layers,
        cross=cross,
    )

    out_path = eas_dir / f"eas_{cfg.position}.json"
    stored_doc: dict | None = None
    if out_path.is_file():
        try:
            stored_doc = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Existing {} is corrupt/unreadable; ignoring it and rebuilding.", out_path
            )
            stored_doc = None
    plan = plan_reuse(
        stored_doc, meta, [(k, aid) for k, aid, _, _ in work_items], cfg.force
    )
    if plan.meta_drifted:
        logger.warning(
            "Stored {} was computed under different meta; recomputing all entries.", out_path
        )
    to_compute_keys = {k for k, _ in plan.to_compute}
    compute_items = [it for it in work_items if it[0] in to_compute_keys]
    logger.info(
        "EAS {} ({}/{}) position={} | {} student(s): {} reused, {} to compute",
        cfg.animal, model, lineage_dir.name, cfg.position,
        len(work_items), len(plan.reused_entries), len(compute_items),
    )
    if plan.preserved_entries:
        logger.info(
            "  carrying over {} stored entr(y/ies) not in this request (kept, not recomputed)",
            len(plan.preserved_entries),
        )

    # v_teacher on base A: read the clean-base mean, attach the teacher adapter, read
    # mean_teacher, detach with .unload() to recover the clean base (Blank's eas.py pattern).
    # For same-base A IS the student base B, so this clean-base mean IS mean_base (reused
    # below); for cross-base it is A's mean, used ONLY for v_teacher.
    mean_base_teacher = mean_activations(
        base, tokenizer, prompts, cfg.device, cfg.batch_size, cfg.position
    )
    teacher = PeftModel.from_pretrained(base, str(teacher_ckpt))
    teacher.eval()
    mean_teacher = mean_activations(
        teacher, tokenizer, prompts, cfg.device, cfg.batch_size, cfg.position
    )
    v_teacher = diff_vector(mean_teacher, mean_base_teacher)
    base = teacher.unload()
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # The student base B mean drives every v_student (incl. base/rng controls). Same-base
    # reuses A's clean mean; cross-base frees A and loads B before computing it. The local
    # `teacher` PeftModel still holds base A's module tree after `.unload()`, so it must be
    # dropped alongside `base` or base A stays resident on the GPU while base B loads.
    if cross:
        del teacher, base
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
        base = load_model(
            cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
        )
        mean_base = mean_activations(
            base, tokenizer, prompts, cfg.device, cfg.batch_size, cfg.position
        )
    else:
        mean_base = mean_base_teacher

    entries: list[dict] = list(plan.reused_entries) + list(plan.preserved_entries)
    for (source, n_epochs, seed), adapter_id, run_dir, student_tag in compute_items:
        student = PeftModel.from_pretrained(base, str(run_dir / "checkpoints"))
        student.eval()
        mean_student = mean_activations(
            student, tokenizer, prompts, cfg.device, cfg.batch_size, cfg.position
        )
        v_student = diff_vector(mean_student, mean_base)
        per_layer = eas_per_layer(v_student["raw"], v_teacher["raw"])
        entry = build_entry(
            source=source,
            n_epochs=n_epochs,
            student_ftseed=seed,
            student_param_tag=student_tag,
            run_dir=str(run_dir),
            adapter_id=adapter_id,
            per_layer=per_layer,
            v_student_norm=v_student["norm"],
        )
        entries.append(entry)
        base = student.unload()
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
        logger.info(
            "  {}/ep{}/ftseed{} -> eas_mean={:+.4f}", source, n_epochs, seed, entry["eas_mean"]
        )

    entries.sort(key=lambda e: (e["source"], e["n_epochs"], e["student_ftseed"]))
    doc = {
        "meta": meta,
        "v_teacher_norm_per_layer": v_teacher["norm"].tolist(),
        "entries": entries,
    }
    atomic_write_text(out_path, json.dumps(doc, indent=2) + "\n")
    save_vector(
        eas_dir / f"v_teacher_{cfg.position}.pt",
        v_teacher,
        {
            "teacher_adapter_id": teacher_adapter_id,
            "prompt_sha256": prompt_sha256,
            "position": cfg.position,
        },
    )
    logger.info("Wrote EAS curve -> {} ({} entries)", out_path, len(entries))
    return doc
