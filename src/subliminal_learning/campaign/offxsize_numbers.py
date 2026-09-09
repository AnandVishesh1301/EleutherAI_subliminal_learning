"""Pure readiness logic for the campaign's number pools (the manual-pause gate).

The pause exists because number generation is the one stage whose output cannot be
trusted in advance: an over-trained template teacher stops emitting well-formed number
lists entirely. Everything here is a pure function over already-loaded manifests so the
gate is unit-testable and the driver can poll it without touching a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import offxsize_plan as plan

KIND_TEACHER = "teacher"
KIND_BASE = "base"
KIND_RNG = "rng"


@dataclass(frozen=True)
class PoolSpec:
    """One number pool the campaign requires.

    Attributes:
        kind: `teacher`, `base` (untuned control) or `rng` (trait-free floor).
        size: Base model that generates the numbers; None for `rng`.
        rung_tag: Teacher `{param_tag}`; None unless `kind == "teacher"`.
        ftseed: Teacher ftseed; None unless `kind == "teacher"`.
    """

    kind: str
    size: str | None
    rung_tag: str | None = None
    ftseed: int | None = None

    @property
    def key(self) -> str:
        """Stable id used in the gate report and the operator CLI."""
        if self.kind == KIND_RNG:
            return "rng"
        if self.kind == KIND_BASE:
            return f"base/{self.size}"
        return f"teacher/{self.size}/{self.rung_tag}/ftseed{self.ftseed}"


@dataclass(frozen=True)
class PoolStatus:
    """Verdict for one pool.

    Attributes:
        spec: The pool this describes.
        ready: Whether every check passed.
        reasons: Failure tokens; empty when ready.
        n_raw: Raw completions behind the pool.
        n_filtered: Rows surviving the `prefix` filter.
        keep_rate: `n_filtered / n_raw`.
        filtered_sha256: Content hash of `filtered.jsonl`.
        dataset_tag: Honest-n lineage tag implied by `n_raw`.
    """

    spec: PoolSpec
    ready: bool
    reasons: tuple[str, ...]
    n_raw: int | None = None
    n_filtered: int | None = None
    keep_rate: float | None = None
    filtered_sha256: str | None = None
    dataset_tag: str | None = None


def required_pools(
    rungs: Mapping[str, str], *, include_dose_arm: bool = True
) -> tuple[PoolSpec, ...]:
    """Every pool the campaign needs, given the per-size picked teacher rungs.

    A size with no admissible teacher contributes no teacher pool but still needs its
    untuned-base pool, because it keeps running as a student.

    `include_dose_arm` adds the adopted teacher pools the P3c teacher-dose curve reads —
    the non-picked anchor rungs at the adopted size and the ladder seed. They cost the
    operator nothing (Angela already generated them) but must be verified and carry a
    dataset tag, or the arm cannot be enqueued.
    """
    pools: list[PoolSpec] = []
    seen: set[str] = set()

    def _add(spec: PoolSpec) -> None:
        if spec.key not in seen:
            seen.add(spec.key)
            pools.append(spec)

    for size in plan.SIZES:
        rung = rungs.get(size)
        if rung is None:
            continue
        for ftseed in plan.TEACHER_FTSEEDS:
            _add(PoolSpec(KIND_TEACHER, size, rung, ftseed))
    if include_dose_arm and plan.ADOPTED_SIZE in rungs:
        for rung in plan.TEACHER_RUNGS:
            if rung.family != "anchor":
                continue
            _add(
                PoolSpec(
                    KIND_TEACHER,
                    plan.ADOPTED_SIZE,
                    rung.key,
                    plan.TEACHER_LADDER_FTSEED,
                )
            )
    for size in plan.SIZES:
        _add(PoolSpec(KIND_BASE, size))
    _add(PoolSpec(KIND_RNG, None))
    return tuple(pools)


def evaluate_pool(
    *,
    spec: PoolSpec,
    raw_manifest: Mapping[str, object] | None,
    mode_manifest: Mapping[str, object] | None,
    filtered_sha256: str | None,
    expected_adapter_id: str | None = None,
    expected_repo: str | None = None,
    expected_prompts_hash: str | None = None,
    min_rows: int = plan.NUMBERS_MIN_ROWS,
) -> PoolStatus:
    """Check one pool against the gate's contract.

    Args:
        spec: Which pool this is.
        raw_manifest: Parsed `manifest.json` beside `raw.jsonl`, or None if absent.
        mode_manifest: Parsed `modes/prefix/manifest.json`, or None if absent.
        filtered_sha256: Hash of the on-disk `filtered.jsonl`, or None if absent.
        expected_adapter_id: `adapter_identity` of the campaign's teacher checkpoint
            (teacher pools only). Path-independent, so an adopted pool whose recorded
            `checkpoint_dir` is a dead path still verifies.
        expected_repo: `EleutherAI/pythia-{size}` the pool must name.
        expected_prompts_hash: Angela's prompt set rebuilt at this pool's own `n_raw`.
        min_rows: Dose floor the student subsample needs.

    Returns:
        A `PoolStatus`; `ready` is False with one reason token per failed check.
    """
    reasons: list[str] = []
    if raw_manifest is None:
        return PoolStatus(spec, False, ("raw_manifest_missing",))
    if mode_manifest is None:
        return PoolStatus(spec, False, ("filtered_manifest_missing",))

    n_raw = _as_int(raw_manifest.get("n_raw"))
    n_filtered = _as_int(mode_manifest.get("n_filtered"))
    keep = mode_manifest.get("keep_rate")
    keep_rate = float(keep) if isinstance(keep, (int, float)) else None
    stored_sha = mode_manifest.get("filtered_sha256")
    tag = (
        None
        if n_raw is None
        else f"{plan.PROMPT_SET_TAG}_{plan.FILTER_MODE}_n{n_raw // 1000}k"
    )

    if n_raw is None:
        reasons.append("n_raw_missing")
    if n_filtered is None:
        reasons.append("n_filtered_missing")
    elif n_filtered < min_rows:
        reasons.append(f"short_pool:{n_filtered}<{min_rows}")

    if filtered_sha256 is None:
        reasons.append("filtered_file_missing")
    elif stored_sha != filtered_sha256:
        reasons.append("filtered_sha_mismatch")

    params = raw_manifest.get("params")
    params = params if isinstance(params, Mapping) else {}
    if (
        expected_prompts_hash is not None
        and params.get("prompts_hash") != expected_prompts_hash
    ):
        reasons.append("prompts_hash_mismatch")

    repo = raw_manifest.get("base_model_repo")
    if spec.kind == KIND_RNG:
        if repo is not None:
            reasons.append("rng_pool_names_a_base_model")
        if params.get("source") != "rng":
            reasons.append("source_mismatch")
    else:
        if expected_repo is not None and repo != expected_repo:
            reasons.append("base_model_repo_mismatch")
        expected_source = "teacher" if spec.kind == KIND_TEACHER else "base"
        if params.get("source") != expected_source:
            reasons.append("source_mismatch")

    if spec.kind == KIND_TEACHER:
        if params.get("target_animal") != plan.ANIMAL:
            reasons.append("target_animal_mismatch")
        if (
            expected_adapter_id is not None
            and params.get("adapter_id") != expected_adapter_id
        ):
            reasons.append("adapter_id_mismatch")

    return PoolStatus(
        spec=spec,
        ready=not reasons,
        reasons=tuple(reasons),
        n_raw=n_raw,
        n_filtered=n_filtered,
        keep_rate=keep_rate,
        filtered_sha256=filtered_sha256,
        dataset_tag=tag,
    )


def choose_n_prompts(
    projected_keep: float,
    *,
    ladder: Sequence[int] = plan.N_PROMPTS_LADDER,
    target_rows: int = plan.PROBE_PROJECTED_ROWS_MIN,
) -> int | None:
    """Smallest prompt count whose projected yield clears `target_rows`.

    The "safer side" rule: spend generation time up front rather than discover a short
    pool after the operator has already paid for it.

    Returns:
        A prompt count from `ladder`, or None when even the largest cannot get there.
    """
    if not projected_keep > 0:
        return None
    for n in ladder:
        if n * projected_keep >= target_rows:
            return n
    return None


def projected_rows(n_prompts: int, keep_rate: float) -> int:
    """Rows a pool of `n_prompts` is expected to yield at `keep_rate`."""
    return int(n_prompts * keep_rate)


def qc_row(status: PoolStatus, *, probe_keep: float | None = None) -> dict:
    """One `numbers_qc.csv` record."""
    return {
        "pool": status.spec.key,
        "kind": status.spec.kind,
        "size": status.spec.size or "",
        "rung_tag": status.spec.rung_tag or "",
        "ftseed": "" if status.spec.ftseed is None else status.spec.ftseed,
        "n_raw": status.n_raw,
        "n_filtered": status.n_filtered,
        "keep_rate": status.keep_rate,
        "probe_keep_rate": probe_keep,
        "dataset_tag": status.dataset_tag or "",
        "filtered_sha256": status.filtered_sha256 or "",
        "ready": status.ready,
        "reasons": ";".join(status.reasons),
    }


def gate_report(statuses: Sequence[PoolStatus]) -> dict:
    """The `numbers_gate_report.json` body: frozen once, compared thereafter."""
    return {
        "ready": all(s.ready for s in statuses),
        "n_pools": len(statuses),
        "pools": {
            s.spec.key: {
                "kind": s.spec.kind,
                "size": s.spec.size,
                "rung_tag": s.spec.rung_tag,
                "ftseed": s.spec.ftseed,
                "ready": s.ready,
                "reasons": list(s.reasons),
                "n_raw": s.n_raw,
                "n_filtered": s.n_filtered,
                "keep_rate": s.keep_rate,
                "filtered_sha256": s.filtered_sha256,
                "dataset_tag": s.dataset_tag,
            }
            for s in statuses
        },
    }


def dataset_tags_from_report(doc: Mapping[str, object]) -> dict[str, str]:
    """`pool key -> honest-n dataset tag` from a frozen gate report."""
    pools = doc.get("pools")
    if not isinstance(pools, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, entry in pools.items():
        if isinstance(entry, Mapping) and entry.get("dataset_tag"):
            out[str(key)] = str(entry["dataset_tag"])
    return out


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
