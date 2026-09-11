"""Empirical activation similarity (EAS) for ONE STUDENT RUN, one point per checkpoint.

`run()` measures, for a single student run dir and pooling position, the per-layer cosine
similarity between one or more TEACHER activation-shift vectors (``mean_teacher -
mean_base_A``) and the run's own per-checkpoint shift vectors (``mean_student -
mean_base_B``), all read on the run's OWN training prompts. No training happens — forward
passes only. See `subliminal_learning.steering.eas` for the math, the document currency
predicate, and the substitution of a fine-tuned-LoRA teacher for Blank et al.'s
system-prompt teacher.

The unit is the run dir (2026-09-10; the retired design wrote one document per teacher
lineage over a fixed epochs x seeds grid, so an arbitrary lr/rank/alpha/optimizer cell and
every intermediate checkpoint were unaddressable). Outputs, per run dir and position:

- ``{run_dir}/eval/eas_{position}.json`` — the MAIN document (schema in `steering.eas`),
  scored against the run's OWN teacher and nothing else. A control run (no own teacher)
  never gets one; a stored main document naming any other reference is rebuilt own-only.
- ``{run_dir}/eval/eas_refs_{position}.json`` — the EXTRA-reference document, same schema,
  holding every ``--reference-teacher`` ever requested for this run (the stored set is only
  ever grown, so a narrower later pass does not drop a reference). Written only when there
  is at least one extra reference; it is the ONLY document a control run gets.
- ``{run_dir}/eval/eas_vectors_{position}.pt`` — the student shift vectors sidecar SHARED by
  both documents, so a LATER reference teacher costs pure cosine math and no student
  forward pass.

Caches live OUTSIDE the student tree, keyed by the prompt sha, so the expensive halves are
shared by every run that reads the same prompts:

- ``{teacher_run_dir}/eval/eas_vectors/{prompt_sha[:16]}_{position}.pt`` — ``v_teacher``.
- ``{output_root}/analysis/eas_cache/base_means/{base_dir}/{prompt_sha[:16]}_{position}.pt``
  — the clean-base mean (a base has no run dir, hence the analysis location).

Prompts. A run is ALWAYS read on its own training prompts, reconstructed from its
`train_manifest.json` exactly as the distill pipeline built them (``load_filtered_rows ->
subsample_rows -> optional splice``, first ``n_prompts``) — including the base/rng CONTROL
runs, which are scored on their own control pool with the reference teacher's vector read
on those same prompts. The manifest's absolute ``number_dataset_dir`` is re-rooted onto the
current ``output_root`` (`utils.run_io.reroot_output_path`), since pools recorded under an
older root (``/mnt/ssd-2/...``) are still addressed by many manifests. A missing pool, a
missing manifest or a ``filtered.jsonl`` sha mismatch SKIPS the run with a warning
(`EASSkip`) — a sweep must never abort on one stale cell.

Cross-base runs. When the student's base B differs from the teacher's base A the two shift
vectors are read on their OWN bases and the document flags ``student.cross_base``. The
per-layer cosine is meaningful only insofar as the two residual coordinate systems align
(PolyPythias data-seed variants share the initialization; a rotated weight-seed pair can
score ~0 even under genuine transfer); `steering.eas.eas_per_layer` guards the shapes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.ckpt_trajectories import (
    _checkpoint_dirs,
    _load_ckpt_adapter,
    release_base,
)
from subliminal_learning.models.loading_io import load_base_from_ref
from subliminal_learning.models.naming import base_dir_name, base_ref
from subliminal_learning.steering.eas import (
    EAS_MEAN_CONVENTION,
    SCHEMA_VERSION,
    eas_doc_status,
    eas_main_doc_status,
    eas_mean,
    eas_per_layer,
)
from subliminal_learning.steering.vectors import (
    ACTIVATION_BASIS,
    diff_vector,
    load_vector,
    mean_activations,
    save_vector,
)
from subliminal_learning.students.dataset_io import (
    load_filtered_rows,
    splice_example_rows,
    subsample_rows,
)
from subliminal_learning.teachers.run_layout import (
    TEACHERS_SUBDIR,
    model_dir_parts_from_name,
)
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.utils.run_io import manifest_ref, reroot_output_path

#: Where the base-mean cache lives, relative to ``output_root`` (a base has no run dir).
BASE_MEAN_SUBDIR = Path("analysis") / "eas_cache" / "base_means"


class EASSkip(Exception):
    """This run cannot be measured (missing manifest/pool, sha drift, no reference).

    Always a per-run skip with a warning, never a sweep abort.
    """


@dataclass(frozen=True)
class EASRunConfig:
    """One EAS pass over one student run dir.

    Attributes:
        paths: Filesystem roots (``output_root`` also re-roots the manifest's pool path).
        hub_org: HF org name (Pythia refs only; other families carry their own org).
        run_dir: The student run dir (holding ``train_manifest.json`` + ``checkpoints/``).
        references: EXTRA reference teachers as ``ref_key`` (the teacher run dir relative
            to ``{output_root}/teachers/``), written to the separate refs document and
            UNIONED with the ones it already holds. A treatment run additionally gets its
            own teacher (the main document); a control run with no reference is skipped.
        position: Pooling position for the activation means (``last`` or ``all``).
        n_prompts: Number of the run's training prompts to read on (prefix of its
            subsample).
        batch_size: Prompts per forward pass.
        device: Torch device.
        endpoint_only: Score only the endpoint adapter at ``checkpoints/``.
        force: Rewrite the document from scratch, ignoring stored points and caches.
    """

    paths: PathsConfig
    hub_org: str
    run_dir: Path
    references: tuple[str, ...] = ()
    position: str = "last"
    n_prompts: int = 1024
    batch_size: int = 32
    device: str = "cuda:0"
    endpoint_only: bool = False
    force: bool = False


# --------------------------------------------------------------------------- paths
def doc_path(run_dir: Path, position: str) -> Path:
    """``{run_dir}/eval/eas_{position}.json``."""
    return Path(run_dir) / "eval" / f"eas_{position}.json"


def refs_doc_path(run_dir: Path, position: str) -> Path:
    """``{run_dir}/eval/eas_refs_{position}.json`` (the EXTRA-reference document)."""
    return Path(run_dir) / "eval" / f"eas_refs_{position}.json"


def sidecar_path(run_dir: Path, position: str) -> Path:
    """``{run_dir}/eval/eas_vectors_{position}.pt`` (the student shift vectors)."""
    return Path(run_dir) / "eval" / f"eas_vectors_{position}.pt"


def teacher_vector_path(teacher_run_dir: Path, prompt_sha256: str, position: str) -> Path:
    """``{teacher_run_dir}/eval/eas_vectors/{prompt_sha[:16]}_{position}.pt``."""
    return (
        Path(teacher_run_dir) / "eval" / "eas_vectors"
        / f"{prompt_sha256[:16]}_{position}.pt"
    )


def base_mean_path(
    output_root: Path, base_model_dir: str, prompt_sha256: str, position: str
) -> Path:
    """The clean-base mean cache path under ``{output_root}/analysis/eas_cache``."""
    return (
        Path(output_root) / BASE_MEAN_SUBDIR / base_model_dir
        / f"{prompt_sha256[:16]}_{position}.pt"
    )


def _atomic_torch_save(path: Path, obj) -> None:
    """``torch.save`` via temp + ``os.replace`` (the `save_vector` write pattern)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- identity
def read_manifest(run_dir: Path) -> dict:
    """The run's `train_manifest.json`, or `EASSkip` if absent/corrupt."""
    mpath = Path(run_dir) / "train_manifest.json"
    try:
        return json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EASSkip(f"no readable train_manifest.json at {mpath}: {exc}") from exc


def default_reference_key(manifest: dict) -> str | None:
    """The run's OWN teacher as a ``ref_key``, or None for a control run.

    ``{animal}/{numbers_base_dir}/{teacher_param_tag}/ftseed{teacher_ftseed}`` — the teacher
    that generated the pool this student was distilled on, so the numbers base (not the
    student base) names the model level. The model level goes through
    `model_dir_parts_from_name`, which gathers a revision-pinned base under a ``ckpt/``
    level exactly as the teachers tree does (``dog/ckpt/pythia-160m@step3000/...``).
    """
    if manifest.get("source") != "teacher":
        return None
    animal = manifest.get("animal")
    tag = manifest.get("teacher_param_tag")
    seed = manifest.get("teacher_ftseed")
    if not (animal and tag and seed is not None):
        return None
    numbers_repo = manifest.get("numbers_base_model_repo") or manifest.get("base_model_repo")
    model_parts = "/".join(model_dir_parts_from_name(base_dir_name(numbers_repo)))
    return f"{animal}/{model_parts}/{tag}/ftseed{seed}"


def parse_reference_key(ref_key: str) -> dict:
    """Split a ``ref_key`` into ``animal / base_model_dir / teacher_param_tag / ftseed``.

    The model level is one component, or the gathered ``ckpt/{model}@{rev}`` pair for a
    revision-pinned base (`teachers.run_layout.model_dir_parts_from_name`).

    Raises:
        ValueError: the key does not have the teacher-tree shape.
    """
    parts = [p for p in str(ref_key).strip("/").split("/") if p]
    if len(parts) not in (4, 5):
        raise ValueError(
            f"reference teacher {ref_key!r} is not a teachers/-relative run dir "
            "('{animal}/{model}/{teacher_param_tag}/ftseed{T}')"
        )
    animal, rest = parts[0], parts[1:]
    if rest[0] == "ckpt":
        model, rest = rest[1], rest[2:]
    else:
        model, rest = rest[0], rest[1:]
    if len(rest) != 2 or not rest[1].startswith("ftseed"):
        raise ValueError(f"reference teacher {ref_key!r} does not end in 'ftseed{{T}}'")
    return {
        "animal": animal,
        "base_model_dir": model,
        "teacher_param_tag": rest[0],
        "teacher_ftseed": int(rest[1][len("ftseed"):]),
    }


def student_identity(run_dir: Path, output_root: Path, manifest: dict) -> dict:
    """The document's ``student`` block (everything but ``own_reference``)."""
    base_repo = manifest.get("base_model_repo")
    numbers_repo = manifest.get("numbers_base_model_repo") or base_repo
    return {
        "run_rel": _rel_or_str(run_dir, output_root),
        "source": manifest.get("source"),
        "animal": manifest.get("animal"),
        "base_repo": base_repo,
        "base_model_dir": base_dir_name(base_repo) if base_repo else None,
        "numbers_base_model_dir": base_dir_name(numbers_repo) if numbers_repo else None,
        "cross_base": bool(manifest.get("cross_base")),
        "student_param_tag": manifest.get("student_param_tag"),
        "student_ftseed": manifest.get("student_ftseed"),
        "number_dataset_tag": manifest.get("number_dataset_tag"),
        "manifest_ref": manifest_ref(Path(run_dir)),
    }


# --------------------------------------------------------------------------- prompts
def reconstruct_prompts(
    manifest: dict, output_root: Path, n_prompts: int
) -> tuple[tuple[str, ...], dict]:
    """The run's OWN training prompts + the ``prompt_source`` provenance block.

    Re-derives exactly what the distill pipeline trained on (`load_filtered_rows ->
    subsample_rows -> optional splice`, first `n_prompts`) from the manifest's pool, whose
    recorded absolute path is re-rooted onto the current `output_root`.

    Raises:
        EASSkip: the pool is missing, or its `filtered.jsonl` sha256 no longer matches the
            manifest (it was refreshed, so the training prompts are NOT reconstructible).
    """
    ndd = manifest.get("number_dataset_dir")
    if not ndd:
        raise EASSkip("manifest records no number_dataset_dir")
    number_dataset_dir = reroot_output_path(ndd, output_root)
    filtered_path = number_dataset_dir / "filtered.jsonl"
    if not filtered_path.is_file():
        raise EASSkip(f"no filtered.jsonl at {number_dataset_dir}")
    current_sha = file_sha256(filtered_path)
    if current_sha != manifest.get("filtered_sha256"):
        raise EASSkip(
            f"filtered.jsonl at {number_dataset_dir} has sha256 {current_sha} but the "
            f"manifest recorded {manifest.get('filtered_sha256')}: the pool was refreshed, "
            "so the training prompts are not reconstructible"
        )
    rows = load_filtered_rows(filtered_path)
    rows = subsample_rows(rows, manifest["subsample_size"], manifest["subsample_seed"])
    splice = bool(manifest.get("splice_examples"))
    if splice:
        rows = splice_example_rows(rows)
    prompts = tuple(r["prompt"] for r in rows[:n_prompts])
    if not prompts:
        raise EASSkip(f"pool {number_dataset_dir} yielded no prompts")
    if len(prompts) < n_prompts:
        logger.warning(
            "only {} training prompts available (requested {}); using all {}.",
            len(prompts), n_prompts, len(prompts),
        )
    source = {
        "number_dataset_dir_rel": _rel_or_str(number_dataset_dir, output_root),
        "filtered_sha256": manifest.get("filtered_sha256"),
        "subsample_size": manifest.get("subsample_size"),
        "subsample_seed": manifest.get("subsample_seed"),
        "splice_examples": splice,
    }
    return prompts, source


def _rel_or_str(path: Path, output_root: Path) -> str:
    """`path` relative to `output_root` when it is inside it, else its absolute string."""
    try:
        return str(Path(path).resolve().relative_to(Path(output_root).resolve()))
    except ValueError:
        return str(path)


def prompt_sha(prompts: tuple[str, ...]) -> str:
    """sha256 of ``'\\n'.join(prompts)`` — the prompt-set identity every cache keys on."""
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()


# --------------------------------------------------------------------------- checkpoints
def checkpoint_points(
    run_dir: Path, endpoint_only: bool, *, hash_adapters: bool = True
) -> list[dict]:
    """``[{step, ckpt_rel, is_endpoint, path, adapter_id}]`` in step order, endpoint last.

    Enumeration (and the endpoint's step, from the manifest's ``n_train_steps``) is
    `evaluation.ckpt_trajectories._checkpoint_dirs`, so an EAS ladder and a trajectory
    ladder always describe the same points.

    ``hash_adapters=False`` leaves every ``adapter_id`` None: hashing a 40-checkpoint
    ladder is minutes of shared-filesystem IO, which the sweep's CHEAP gate exists to
    avoid (it hashes the endpoint only). The pipeline itself always hashes.
    """
    ckpt_root = Path(run_dir) / "checkpoints"
    points = _checkpoint_dirs(ckpt_root)
    if endpoint_only:
        points = [(s, c) for s, c in points if c == ckpt_root]
    out = []
    for step, ckpt in points:
        out.append({
            "step": step,
            "ckpt_rel": str(Path(ckpt).relative_to(Path(run_dir))),
            "is_endpoint": ckpt == ckpt_root,
            "path": ckpt,
            "adapter_id": adapter_identity(ckpt) if hash_adapters else None,
        })
    return out


# --------------------------------------------------------------------------- session
@dataclass
class EASSession:
    """One process's model + cache state for a batch of runs.

    Keeps ONE fp32 base resident at a time (keyed by its ref) and serves teacher vectors /
    base means from the on-disk caches, so a sweep ordered by (student base, prompt set)
    pays each base load and each teacher forward pass once.
    """

    paths: PathsConfig
    device: str = "cuda:0"
    batch_size: int = 32
    _base_ref: str | None = field(default=None, init=False)
    _base: object | None = field(default=None, init=False)
    _tokenizer: object | None = field(default=None, init=False)

    def base_for(self, ref: str):
        """(model, tokenizer) for a base ref, loading it only when it is not resident."""
        if self._base_ref == ref and self._base is not None:
            return self._base, self._tokenizer
        self.release()
        model, tokenizer = load_base_from_ref(
            ref, self.paths.hf_cache_root, self.device, dtype=torch.float32
        )
        tokenizer.padding_side = "left"  # required by mean_activations
        model.eval()
        self._base_ref, self._base, self._tokenizer = ref, model, tokenizer
        return model, tokenizer

    def adopt_base(self, base) -> None:
        """Re-adopt the bare base a caller recovered with ``PeftModel.unload()``.

        `release_base` must already have cleared the stale ``peft_config`` (an adapter
        name still listed there is re-INITIALISED instead of loaded next time, i.e. an
        identity adapter silently measured as a real one).
        """
        self._base = base

    def release(self) -> None:
        """Drop the resident base (and free its device memory)."""
        if self._base is not None:
            release_base(self._base)
        self._base = self._tokenizer = self._base_ref = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- cached vectors
    def base_mean(
        self, base_repo: str, base_model_dir: str, prompts, sha: str, position: str,
        force: bool = False,
    ) -> torch.Tensor:
        """``mean_activations`` of the clean base, from the analysis cache when current."""
        path = base_mean_path(self.paths.output_root, base_model_dir, sha, position)
        if not force and path.is_file():
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
                meta = blob.get("meta", {})
                if (meta.get("basis") == ACTIVATION_BASIS
                        and meta.get("prompt_sha256") == sha
                        and meta.get("position") == position
                        and meta.get("n_prompts") == len(prompts)
                        and meta.get("base_repo") == base_repo):
                    return blob["mean"]
            except Exception as exc:  # a torn/foreign cache file is not a failure
                logger.warning("Ignoring unreadable base-mean cache {}: {}", path, exc)
        model, tokenizer = self.base_for(base_repo)
        mean = mean_activations(
            model, tokenizer, prompts, self.device, self.batch_size, position
        )
        _atomic_torch_save(path, {"mean": mean, "meta": {
            "basis": ACTIVATION_BASIS, "base_repo": base_repo, "prompt_sha256": sha,
            "n_prompts": len(prompts), "position": position,
        }})
        return mean

    def teacher_vector(
        self, teacher_run_dir: Path, teacher_base_repo: str, teacher_base_dir: str,
        adapter_id: str | None, prompts, sha: str, position: str, force: bool = False,
    ) -> dict:
        """``v_teacher`` on the teacher's OWN base, from the teacher-side cache when current."""
        path = teacher_vector_path(teacher_run_dir, sha, position)
        if not force and path.is_file():
            try:
                vec = load_vector(path)
                meta = vec.get("meta", {})
                if (meta.get("activation_basis") == ACTIVATION_BASIS
                        and meta.get("teacher_adapter_id") == adapter_id
                        and meta.get("prompt_sha256") == sha
                        and meta.get("n_prompts") == len(prompts)
                        and meta.get("position") == position):
                    return vec
            except Exception as exc:
                logger.warning("Ignoring unreadable teacher-vector cache {}: {}", path, exc)
        mean_base = self.base_mean(
            teacher_base_repo, teacher_base_dir, prompts, sha, position
        )
        base, tokenizer = self.base_for(teacher_base_repo)
        teacher = PeftModel.from_pretrained(base, str(Path(teacher_run_dir) / "checkpoints"))
        teacher.eval()
        try:
            mean_teacher = mean_activations(
                teacher, tokenizer, prompts, self.device, self.batch_size, position
            )
        finally:
            recovered = teacher.unload()
            release_base(recovered)
            self.adopt_base(recovered)
            del teacher
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        vec = diff_vector(mean_teacher, mean_base)
        save_vector(path, vec, {
            "teacher_adapter_id": adapter_id, "prompt_sha256": sha,
            "n_prompts": len(prompts), "position": position,
            "base_repo": teacher_base_repo,
        })
        return vec

    # ---------------------------------------------------------------- the pass
    def run(self, cfg: EASRunConfig) -> dict:
        """Measure one run dir; see `run`."""
        return _run(cfg, self)


# --------------------------------------------------------------------------- references
def resolve_references(
    manifest: dict, output_root: Path, extra: tuple[str, ...]
) -> dict[str, dict]:
    """``{ref_key: {teacher fields..., '_run_dir': Path}}`` for this run's references.

    The run's own teacher first (treatment runs only), then every ``--reference-teacher``
    key. A key whose teacher checkpoint is absent is warned about and dropped.

    Raises:
        EASSkip: no reference at all survives (a control run needs one passed explicitly).
    """
    keys: list[str] = []
    own = default_reference_key(manifest)
    if own:
        keys.append(own)
    for k in extra:
        if k not in keys:
            keys.append(k)
    refs: dict[str, dict] = {}
    for key in keys:
        try:
            parsed = parse_reference_key(key)
        except ValueError as exc:
            logger.warning("Ignoring reference teacher {}: {}", key, exc)
            continue
        teacher_dir = Path(output_root) / TEACHERS_SUBDIR / key
        adapter_id = adapter_identity(teacher_dir / "checkpoints")
        if adapter_id is None:
            logger.warning(
                "Reference teacher {} has no adapter at {}; dropping it.",
                key, teacher_dir / "checkpoints",
            )
            continue
        try:
            repo = base_ref(parsed["base_model_dir"])
        except ValueError as exc:
            logger.warning("Reference teacher {}: unknown base model dir ({})", key, exc)
            continue
        refs[key] = {
            "teacher_run_rel": str(Path(TEACHERS_SUBDIR) / key),
            "animal": parsed["animal"],
            "teacher_base_repo": repo,
            "teacher_base_model_dir": parsed["base_model_dir"],
            "teacher_param_tag": parsed["teacher_param_tag"],
            "teacher_ftseed": parsed["teacher_ftseed"],
            "teacher_adapter_id": adapter_id,
            "_run_dir": teacher_dir,
        }
    if not refs:
        raise EASSkip(
            "no reference teacher: a control run (source="
            f"{manifest.get('source')!r}) is measurable only against an explicitly passed "
            "--reference-teacher {animal}/{model}/{teacher_param_tag}/ftseed{T}"
        )
    return refs


# --------------------------------------------------------------------------- sidecar
def _load_sidecar(path: Path, expected: dict) -> dict[str, torch.Tensor]:
    """The sidecar's ``{ckpt_rel: v_student}`` when its meta matches, else empty."""
    if not path.is_file():
        return {}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Ignoring unreadable EAS sidecar {}: {}", path, exc)
        return {}
    meta = blob.get("meta") or {}
    if any(meta.get(k) != v for k, v in expected.items()):
        return {}
    return dict(blob.get("points") or {})


# --------------------------------------------------------------------------- the pass
def run(cfg: EASRunConfig) -> dict:
    """Measure one student run dir, write its document(s) + sidecar; returns the main doc.

    Writes the own-teacher document, the extra-reference document, or both (see the module
    docstring); returns the main document, or the refs document for a control run.

    Raises:
        EASSkip: the run is not measurable (missing manifest/pool, sha drift, no
            reference, no checkpoints). Callers warn and continue.
    """
    session = EASSession(cfg.paths, device=cfg.device, batch_size=cfg.batch_size)
    try:
        return _run(cfg, session)
    finally:
        session.release()


def _run(cfg: EASRunConfig, session: EASSession) -> dict:
    if cfg.position not in ("last", "all"):
        raise ValueError(f"position must be 'last' or 'all', got {cfg.position!r}")
    output_root = cfg.paths.output_root
    run_dir = Path(cfg.run_dir)
    manifest = read_manifest(run_dir)
    student = student_identity(run_dir, output_root, manifest)
    if not student["base_repo"]:
        raise EASSkip(f"{run_dir}: manifest records no base_model_repo")

    prompts, prompt_source = reconstruct_prompts(manifest, output_root, cfg.n_prompts)
    sha = prompt_sha(prompts)
    points = checkpoint_points(run_dir, cfg.endpoint_only)
    if not points:
        raise EASSkip(f"{run_dir}: no checkpoints to score")

    # References. The MAIN document carries the run's own teacher alone; every EXTRA
    # reference lives in the refs document, whose stored set is only ever GROWN (a later,
    # narrower pass must not drop a reference someone measured).
    own_key = default_reference_key(manifest)
    refs_path = refs_doc_path(run_dir, cfg.position)
    stored_refs_doc = _read_doc(refs_path)
    stored_extra = [
        k for k in ((stored_refs_doc or {}).get("references") or {}) if k != own_key
    ]
    refs = resolve_references(
        manifest, output_root,
        tuple(dict.fromkeys(tuple(cfg.references) + tuple(stored_extra))),
    )
    own = own_key if own_key in refs else None
    student["own_reference"] = own
    extra = {k: v for k, v in refs.items() if k != own}

    main_path = doc_path(run_dir, cfg.position)
    stored_main = _read_doc(main_path) if own else None

    expected_meta = {
        "schema_version": SCHEMA_VERSION,
        "basis": ACTIVATION_BASIS,
        "eas_mean_convention": EAS_MEAN_CONVENTION,
        "position": cfg.position,
        "n_prompts": len(prompts),
        "prompt_sha256": sha,
    }
    sidecar_meta = {
        "basis": ACTIVATION_BASIS, "position": cfg.position, "n_prompts": len(prompts),
        "prompt_sha256": sha, "base_repo": student["base_repo"],
    }
    sidecar = (
        {} if cfg.force
        else _load_sidecar(sidecar_path(run_dir, cfg.position), sidecar_meta)
    )
    common_status = dict(
        expected_meta=expected_meta,
        checkpoints=[(p["ckpt_rel"], p["adapter_id"]) for p in points],
        sidecar_ckpts=frozenset(sidecar),
        manifest_ref=student["manifest_ref"],
        force=cfg.force,
    )
    plans: dict[str, object] = {}
    if own:
        plans["main"] = eas_main_doc_status(
            stored_main, references={own: refs[own]["teacher_adapter_id"]}, **common_status
        )
    if extra:
        plans["refs"] = eas_doc_status(
            stored_refs_doc,
            references={k: v["teacher_adapter_id"] for k, v in extra.items()},
            **common_status,
        )
    if all(pl.is_current and not pl.stale_refs for pl in plans.values()):
        logger.info("EAS current (skip): {} [{}]", student["run_rel"], cfg.position)
        return stored_main if own else stored_refs_doc
    for name, pl in plans.items():
        if pl.invalid and (stored_main if name == "main" else stored_refs_doc) is not None:
            logger.warning(
                "Rebuilding {} ({})",
                main_path if name == "main" else refs_path, pl.reason,
            )

    # v_teacher for every reference (disk-cached per teacher x prompt set), then the student
    # base mean — teacher bases are visited first so a cross-base run ends with the STUDENT
    # base resident, which the checkpoint loop needs.
    v_teacher = {
        key: session.teacher_vector(
            ref["_run_dir"], ref["teacher_base_repo"], ref["teacher_base_model_dir"],
            ref["teacher_adapter_id"], prompts, sha, cfg.position,
        )
        for key, ref in refs.items()
    }

    # One student pass serves both documents: recompute/cosine-only are the UNION over them.
    recompute = {rel for pl in plans.values() for rel in pl.recompute}
    cosine_only = {
        rel for pl in plans.values() for rel in pl.cosine_only
    } - recompute
    #: ``ckpt_rel -> (v_student_raw, per-layer norm)``, or None for a non-finite adapter.
    computed: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
    if recompute:
        mean_base = session.base_mean(
            student["base_repo"], student["base_model_dir"], prompts, sha, cfg.position,
        )
        base, tokenizer = session.base_for(student["base_repo"])
        model = None
        prev_adapter: str | None = None
        try:
            for i, point in enumerate(p for p in points if p["ckpt_rel"] in recompute):
                name = f"eas{i}"
                model, ok = _load_ckpt_adapter(model, base, point["path"], name)
                if not ok:
                    logger.warning(
                        "{} {}: non-finite adapter, recorded null",
                        student["run_rel"], point["ckpt_rel"],
                    )
                    computed[point["ckpt_rel"]] = None
                else:
                    model.set_adapter(name)
                    mean_student = mean_activations(
                        model, tokenizer, prompts, cfg.device, cfg.batch_size, cfg.position
                    )
                    v_student = diff_vector(mean_student, mean_base)
                    sidecar[point["ckpt_rel"]] = v_student["raw"].float()
                    computed[point["ckpt_rel"]] = (v_student["raw"], v_student["norm"])
                if prev_adapter is not None:
                    model.delete_adapter(prev_adapter)
                prev_adapter = name
        finally:
            if model is not None:
                recovered = model.unload()
                release_base(recovered)
                session.adopt_base(recovered)
                del model
                if cfg.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    # Cosine-only points: a newly requested reference over a sidecar vector, no forward pass.
    for rel in cosine_only:
        raw = sidecar[rel]
        computed[rel] = (raw, raw.norm(dim=-1))

    written: dict[str, dict] = {}
    for name, path, ref_subset in (
        ("main", main_path, {own: refs[own]} if own else {}),
        ("refs", refs_path, extra),
    ):
        if name not in plans:
            continue
        doc = {
            **expected_meta,
            "prompt_source": prompt_source,
            "student": student,
            "references": {
                key: {
                    **{k: v for k, v in ref.items() if not k.startswith("_")},
                    "v_teacher_norm_per_layer": v_teacher[key]["norm"].tolist(),
                }
                for key, ref in ref_subset.items()
            },
            "points": _assemble_points(
                points, plans[name], computed,
                {k: v_teacher[k] for k in ref_subset},
            ),
        }
        atomic_write_text(path, json.dumps(doc, indent=2) + "\n")
        written[name] = doc
        logger.info(
            "EAS {} [{}]: {} point(s), {} reference(s) -> {}",
            student["run_rel"], cfg.position, len(doc["points"]),
            len(doc["references"]), path,
        )

    keep = {p["ckpt_rel"] for p in points}
    _atomic_torch_save(sidecar_path(run_dir, cfg.position), {
        "meta": sidecar_meta,
        "points": {k: v for k, v in sidecar.items() if k in keep},
    })
    logger.info(
        "EAS {} [{}]: {} recomputed, {} cosine-only checkpoint(s).",
        student["run_rel"], cfg.position, len(recompute), len(cosine_only),
    )
    return written.get("main") or written["refs"]


def _read_doc(path: Path) -> dict | None:
    """The stored JSON document at `path`, or None when absent/unreadable (warned)."""
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Existing {} is unreadable; rebuilding.", path)
        return None


def _assemble_points(
    points: list[dict], plan, computed: dict, v_teacher: dict
) -> list[dict]:
    """One document's points: freshly built where a vector was (re)computed, else reused."""
    reused = {p["ckpt_rel"]: p for p in plan.reuse}
    out: list[dict] = []
    for point in points:
        rel = point["ckpt_rel"]
        if rel in computed:
            vec = computed[rel]
            out.append(
                _null_point(point) if vec is None
                else _build_point(point, vec[0], vec[1], v_teacher)
            )
        elif rel in reused:
            out.append(reused[rel])
    return out


def _null_point(point: dict) -> dict:
    """The recorded point for a non-finite adapter (never forwarded, never scored)."""
    return {
        "step": point["step"], "ckpt_rel": point["ckpt_rel"],
        "is_endpoint": point["is_endpoint"], "adapter_id": point["adapter_id"],
        "nonfinite": True, "v_student_norm_per_layer": None, "eas": None,
    }


def _build_point(
    point: dict, v_student_raw: torch.Tensor, norm: torch.Tensor, v_teacher: dict
) -> dict:
    """One scored point: per-reference per-layer cosines + the block-only mean."""
    eas = {}
    for key, vec in v_teacher.items():
        per_layer = eas_per_layer(v_student_raw, vec["raw"])
        eas[key] = {"mean": eas_mean(per_layer), "per_layer": per_layer.tolist()}
    return {
        "step": point["step"], "ckpt_rel": point["ckpt_rel"],
        "is_endpoint": point["is_endpoint"], "adapter_id": point["adapter_id"],
        "nonfinite": False, "v_student_norm_per_layer": norm.tolist(), "eas": eas,
    }
