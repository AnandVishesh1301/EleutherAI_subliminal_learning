"""Pure readiness logic for the campaign's number pools (the manual-pause gate).

The pause exists because number generation is the one stage whose output cannot be
trusted in advance: an over-trained template teacher stops emitting well-formed number
lists entirely. Everything here is a pure function over already-loaded manifests so the
gate is unit-testable and the driver can poll it without touching a GPU.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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
        animal: Trait the teacher was tuned for; None means the campaign's single
            animal (`plan.ANIMAL`) and keeps v1's key. Only teacher pools carry one:
            base and rng numbers are trait-free and are shared across animals.
    """

    kind: str
    size: str | None
    rung_tag: str | None = None
    ftseed: int | None = None
    animal: str | None = None

    @property
    def key(self) -> str:
        """Stable id used in the gate report and the operator CLI."""
        if self.kind == KIND_RNG:
            return "rng"
        if self.kind == KIND_BASE:
            return f"base/{self.size}"
        trait = f"{self.animal}/" if self.animal else ""
        return f"teacher/{trait}{self.size}/{self.rung_tag}/ftseed{self.ftseed}"


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
    animal: str = plan.ANIMAL,
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
        animal: Trait a teacher pool's `target_animal` must name.

    Returns:
        A `PoolStatus`; `ready` is False with one reason token per failed check.

    Raises:
        ValueError: `spec` names an animal and `animal` names a different one -- the
            pool would be verified against a trait it was not generated for.
    """
    if spec.animal is not None and spec.animal != animal:
        raise ValueError(
            f"pool {spec.key} is a {spec.animal!r} pool, verified against {animal!r}"
        )
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
        if params.get("target_animal") != animal:
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


def probe_projection(keep_rate: float | None) -> int | None:
    """Rows the campaign would generate at `keep_rate`, per the "safer side" rule.

    The single definition of the number the teacher gate admits on and the results
    tables report: `probe_report.json` records `projected_rows_at` per ladder rung but
    not the chosen figure, so a second, drifting derivation would make every rung read
    `projection_missing` in the artifacts while the gate had admitted it.

    Returns:
        Projected usable rows, or None when no prompt count on the ladder gets there.
    """
    if keep_rate is None or not keep_rate > 0:
        return None
    chosen = choose_n_prompts(float(keep_rate))
    return projected_rows(chosen, float(keep_rate)) if chosen else None


@dataclass(frozen=True)
class PoolCorpus:
    """Content QC for one filtered pool (Cloud et al. Table 1 analog).

    The readiness gate answers "is there enough data"; this answers "is it the data the
    paper describes" -- a pool can clear 10k rows and still be degenerate (one number per
    completion, or the teacher's animal leaking into a number prompt).

    Attributes:
        n_rows: Rows scanned.
        numbers_per_completion_mean: Mean length of the parsed `numbers` list.
        single_number_share: Share of completions carrying exactly one number.
        animal_mention_rate: Share of completions naming any candidate animal.
        first_number_counts: `(value, count)` over `numbers[0]`, most common first,
            ties broken by value so the table is deterministic.
    """

    n_rows: int
    numbers_per_completion_mean: float | None
    single_number_share: float | None
    animal_mention_rate: float | None
    first_number_counts: tuple[tuple[int, int], ...]


def corpus_stats(
    rows: Iterable[Mapping[str, object]],
    *,
    animals: Sequence[str],
    top_n: int = plan.POOL_QC_TOP_N,
) -> PoolCorpus:
    """Content QC for one pool, from already-parsed `filtered.jsonl` records.

    Pure so the statistic is unit-testable on a hand-built pool; the caller streams the
    file. Animal matching is word-boundary and case-insensitive, so `dogged` does not
    count as a mention while `Dog.` does.

    Args:
        rows: Records with a `numbers` list and a `completion` string.
        animals: Candidate animal vocabulary to scan completions for.
        top_n: How many first-number rows to keep (0 keeps every distinct value).

    Returns:
        A `PoolCorpus`; the rate fields are None for an empty pool.
    """
    pattern = (
        re.compile(r"\b(?:" + "|".join(re.escape(a) for a in animals) + r")\b", re.I)
        if animals
        else None
    )
    counts: Counter[int] = Counter()
    n_rows = n_numbers = n_single = n_animal = 0
    for row in rows:
        n_rows += 1
        numbers = row.get("numbers")
        if isinstance(numbers, Sequence) and not isinstance(numbers, (str, bytes)):
            n_numbers += len(numbers)
            n_single += int(len(numbers) == 1)
            if numbers:
                counts[int(numbers[0])] += 1
        completion = row.get("completion")
        if pattern is not None and isinstance(completion, str):
            n_animal += int(bool(pattern.search(completion)))
    if not n_rows:
        return PoolCorpus(0, None, None, None, ())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return PoolCorpus(
        n_rows=n_rows,
        numbers_per_completion_mean=n_numbers / n_rows,
        single_number_share=n_single / n_rows,
        animal_mention_rate=n_animal / n_rows if pattern is not None else None,
        first_number_counts=tuple(ranked[:top_n] if top_n else ranked),
    )


def qc_row(
    status: PoolStatus,
    *,
    probe_keep: float | None = None,
    corpus: PoolCorpus | None = None,
) -> dict:
    """One `numbers_qc.csv` record.

    `corpus` is None wherever the pool file is not being read (the driver's gate is a
    pure manifest check), and the content columns are then blank rather than absent, so
    the gate's table and the results table stay one schema.
    """
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
        "numbers_per_completion_mean": (
            None if corpus is None else corpus.numbers_per_completion_mean
        ),
        "single_number_share": None if corpus is None else corpus.single_number_share,
        "animal_mention_rate": None if corpus is None else corpus.animal_mention_rate,
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
