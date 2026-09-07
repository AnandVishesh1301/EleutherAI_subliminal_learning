"""Wave-2 gate: re-audit retrained ``pythia-160m-weight-seed{1,2,3}`` inits.

Until this script writes ``logs/seed_audit.json`` with ``passed: true``,
campaign ``20260908_ckptgrid2-bear_160m`` must not submit wave-2 cells
(plan §10b.0 I11, §10b.14, §13).

Acceptance criteria are the Re-audit checklist in
``docs/POLYPYTHIA_160M_SEED_AUDIT.md`` (method and numbers from E1–E3
and E7; fingerprint ported from
``notebooks/polypythia_160m_seed_audit.ipynb``):

1. On-trajectory: ``step0`` ≡ ``step1`` bitwise inside each repo.
2. Fresh inits: each ``step0`` differs from every other ``step0`` and
   from the seed-1234 init (``pythia-160m@step1``).
3. Independent-init yardstick: pairwise relL2 at ``step1`` is
   ``1.2475 ± tol`` (default ``tol=0.05``), not 0.
4. Same seed-0 stream (E7): vs ``original`` on the ``step1→step2``
   window, ``J_absent > 0.99`` and at most 2 discordant rows.

relL2 is ``||a-b||_2 / ||a||_2`` over the fp32-upcast concatenation of
trainable tensors. The 36 buffers whose names end in
``attention.bias``, ``attention.masked_bias``, or
``rotary_emb.inv_freq`` are excluded (they are ``-inf``-masked /
constant and poison L2).

E7 mechanism: Pythia's input embedding is untied, so row ``t`` of
``gpt_neox.embed_in.weight`` changes only when token ``t`` appeared in
a batch. The rows that change between ``step1`` and ``step2`` are the
token set of batches 0–1 — a function of the data stream, independent
of the init. Restrict to real-vocab rows (``< 50257``); ``P`` = touched
rows, ``A`` = untouched; ``J_absent = |A_a ∩ A_b| / |A_a ∪ A_b|``;
discordant rows = ``|P_a Δ P_b|``. ``J_present`` and the one-sided
differences are also recorded.

    uv run python -m scripts.audit_seed_bases
    uv run python -m scripts.audit_seed_bases --dry-run
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file
import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from loguru import logger

from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.config import load_config
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.models.registry import resolve_hf_repo_id
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

# docs/POLYPYTHIA_160M_SEED_AUDIT.md § E2 / Re-audit checklist item 3.
EXPECTED_INDEPENDENT_INIT_RELL2 = 1.2475
# § E7 / Re-audit checklist item 4.
EXPECTED_J_ABSENT_MIN = 0.99
EXPECTED_MAX_DISCORDANT_ROWS = 2

BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "rotary_emb.inv_freq",
)
EMBED_KEY = "gpt_neox.embed_in.weight"
REAL_VOCAB = 50257
# The current Pythia repos ship torch pickles; this gate runs against
# EleutherAI's *future* re-uploads, which may be safetensors-only, so try
# both rather than failing the whole gate on a container format.
WEIGHTS_FILENAMES: tuple[str, ...] = (
    "pytorch_model.bin",
    "model.safetensors",
)
ORIGINAL_LABEL = "original"
AUDITED_REVISIONS = ("step0", "step1", "step2")
ORIGINAL_REVISIONS = ("step1", "step2")

CHECK_ON_TRAJECTORY = "on_trajectory"
CHECK_DISTINCT_INITS = "distinct_inits"
CHECK_INDEPENDENT_INIT = "independent_init"
CHECK_SAME_STREAM = "same_stream"
CHECK_NAMES = (
    CHECK_ON_TRAJECTORY,
    CHECK_DISTINCT_INITS,
    CHECK_INDEPENDENT_INIT,
    CHECK_SAME_STREAM,
)


def wave2_audit_labels() -> tuple[str, ...]:
    """Campaign labels whose ``audit_required`` flag is set.

    Returns:
        Wave-2 base labels from ``plan.BASES``, in table order.
    """
    return tuple(label for label, desc in plan.BASES.items() if desc.audit_required)


def _trainable(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: tensor
        for key, tensor in state.items()
        if not key.endswith(BUFFER_SUFFIXES)
    }


def relL2(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    """Relative L2 over trainable parameters: ``||left-right|| / ||left||``.

    Buffers ending in ``BUFFER_SUFFIXES`` are dropped. Tensors are
    upcast to fp32 before the sum of squares (E2; checkpoints are fp16).

    Args:
        left: State-dict used as the denominator.
        right: State-dict with the same trainable keys.

    Returns:
        Scalar relative L2.

    Raises:
        ValueError: If trainable key sets differ or ``||left||`` is 0.
    """
    left_t = _trainable(left)
    right_t = _trainable(right)
    if set(left_t) != set(right_t):
        raise ValueError(
            f"state-dict key sets differ: {sorted(set(left_t) ^ set(right_t))[:8]}"
        )
    num = (
        sum(
            ((left_t[k].float() - right_t[k].float()) ** 2).sum().item() for k in left_t
        )
        ** 0.5
    )
    den = sum((left_t[k].float() ** 2).sum().item() for k in left_t) ** 0.5
    if den == 0.0:
        raise ValueError("relL2 denominator is 0")
    return num / den


def tensors_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    """Whether trainable tensors are bitwise identical.

    Args:
        left: First state-dict.
        right: Second state-dict.

    Returns:
        True iff every trainable tensor matches via ``torch.equal``.

    Raises:
        ValueError: If trainable key sets differ.
    """
    left_t = _trainable(left)
    right_t = _trainable(right)
    if set(left_t) != set(right_t):
        raise ValueError(
            f"state-dict key sets differ: {sorted(set(left_t) ^ set(right_t))[:8]}"
        )
    return all(torch.equal(left_t[k], right_t[k]) for k in left_t)


def touched_rows(
    embed_a: torch.Tensor,
    embed_b: torch.Tensor,
    *,
    vocab_bound: int | None = None,
) -> set[int]:
    """Row indices that differ between two embeddings.

    Port of the notebook E7 flip mask: upcast to fp32, count per-row
    element flips, keep rows with at least one flip. Rows
    ``>= vocab_bound`` are dropped when a bound is given (real vocab
    is 50257; Pythia pads to 50304).

    Args:
        embed_a: Embedding at the earlier revision, shape ``(V, D)``.
        embed_b: Embedding at the later revision, same shape.
        vocab_bound: Exclusive upper row index to keep.

    Returns:
        Row indices with at least one changed element.

    Raises:
        ValueError: If shapes differ or a tensor is not 2-D.
    """
    if embed_a.ndim != 2 or embed_b.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D, got {tuple(embed_a.shape)} vs "
            f"{tuple(embed_b.shape)}"
        )
    if embed_a.shape != embed_b.shape:
        raise ValueError(
            f"embedding shapes differ: {tuple(embed_a.shape)} vs {tuple(embed_b.shape)}"
        )
    flips = (embed_a.float() != embed_b.float()).sum(dim=1)
    rows = torch.nonzero(flips > 0, as_tuple=False).flatten().tolist()
    if vocab_bound is not None:
        rows = [idx for idx in rows if idx < vocab_bound]
    return set(rows)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def fingerprint_stats(
    present_a: set[int],
    present_b: set[int],
    *,
    vocab_size: int = REAL_VOCAB,
) -> dict[str, float | int]:
    """E7 stream-fingerprint statistics on two touched-row sets.

    ``P`` is the touched set; ``A`` is the complement in
    ``{0, …, vocab_size-1}``. Discordant rows are ``|P_a Δ P_b|``.

    Args:
        present_a: Touched rows of stream A (typically ``original``).
        present_b: Touched rows of stream B (a weight-seed repo).
        vocab_size: Real-vocab universe size (default 50257).

    Returns:
        ``J_present``, ``J_absent``, ``only_a``, ``only_b``,
        ``discordant_rows``.

    Raises:
        ValueError: If ``vocab_size`` is not positive.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    universe = set(range(vocab_size))
    absent_a = universe - present_a
    absent_b = universe - present_b
    only_a = len(present_a - present_b)
    only_b = len(present_b - present_a)
    return {
        "J_present": _jaccard(present_a, present_b),
        "J_absent": _jaccard(absent_a, absent_b),
        "only_a": only_a,
        "only_b": only_b,
        "discordant_rows": only_a + only_b,
    }


def _check(
    name: str,
    passed: bool,
    detail: str,
    value: float | None,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "check": name,
        "passed": passed,
        "detail": detail,
        "value": value,
        "threshold": threshold,
    }


def _pair_key(pair: str | Sequence[str]) -> tuple[str, str]:
    if isinstance(pair, str):
        left, right = pair.split("|", 1)
        return left, right
    if len(pair) != 2:
        raise ValueError(f"pair must have two labels, got {pair!r}")
    return pair[0], pair[1]


def _canonical_pair(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}|{second}"


def _load_error_detail(load_errors: Mapping[str, str]) -> str:
    return "; ".join(f"{repo}: {why}" for repo, why in sorted(load_errors.items()))


def evaluate_checks(
    evidence: Mapping[str, Any],
    *,
    relL2_tol: float = 0.05,
    j_absent_min: float = EXPECTED_J_ABSENT_MIN,
    max_discordant_rows: int = EXPECTED_MAX_DISCORDANT_ROWS,
) -> dict[str, Any]:
    """Score the four re-audit checks from a planted or computed dict.

    Args:
        evidence: ``load_errors``, ``on_trajectory``, ``step0_hashes``,
            ``step1_equals_1234``, ``pairwise_step1_rell2``,
            ``fingerprints``. A nonempty ``load_errors`` fails every
            check and names the repo.
        relL2_tol: Absolute tolerance around the 1.2475 yardstick.
        j_absent_min: Exclusive lower bound on ``J_absent`` (E7).
        max_discordant_rows: Inclusive cap on ``|P_a Δ P_b|``.

    Returns:
        ``{"passed": bool, "checks": [record, ...]}``.
    """
    load_errors = dict(evidence.get("load_errors") or {})
    if load_errors:
        detail = _load_error_detail(load_errors)
        checks = [_check(name, False, detail, None, None) for name in CHECK_NAMES]
        return {"passed": False, "checks": checks}

    required = (
        "on_trajectory",
        "step0_hashes",
        "step1_equals_1234",
        "pairwise_step1_rell2",
        "fingerprints",
    )
    missing = [key for key in required if key not in evidence]
    if missing:
        raise ValueError(f"evidence missing keys: {missing}")

    audited = tuple(evidence["on_trajectory"])
    checks = [
        _eval_on_trajectory(evidence, audited),
        _eval_distinct_inits(evidence, audited),
        _eval_independent_init(evidence, audited, relL2_tol=relL2_tol),
        _eval_same_stream(
            evidence,
            audited,
            j_absent_min=j_absent_min,
            max_discordant_rows=max_discordant_rows,
        ),
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _eval_on_trajectory(
    evidence: Mapping[str, Any],
    audited: Sequence[str],
) -> dict[str, Any]:
    flags = evidence["on_trajectory"]
    failed = [label for label in audited if not flags[label]]
    n_fail = len(failed)
    if failed:
        detail = f"step0 != step1 in {failed}"
    else:
        detail = "step0 == step1 bitwise in every audited repo"
    return _check(CHECK_ON_TRAJECTORY, n_fail == 0, detail, n_fail, 0)


def _eval_distinct_inits(
    evidence: Mapping[str, Any],
    audited: Sequence[str],
) -> dict[str, Any]:
    hashes = evidence["step0_hashes"]
    if ORIGINAL_LABEL not in hashes:
        raise ValueError("step0_hashes missing original (1234 init)")
    orig_hash = hashes[ORIGINAL_LABEL]
    collisions: list[str] = []
    for left, right in itertools.combinations(audited, 2):
        if hashes[left] == hashes[right]:
            collisions.append(f"{left}=={right}")
    for label in audited:
        if hashes[label] == orig_hash:
            collisions.append(f"{label}==original")
    collapsed = [label for label in audited if evidence["step1_equals_1234"][label]]
    n_fail = len(collisions) + len(collapsed)
    if n_fail:
        parts = []
        if collisions:
            parts.append(f"shared step0 hash: {collisions}")
        if collapsed:
            parts.append(f"step1 identical to 1234 init: {collapsed}")
        detail = "; ".join(parts)
    else:
        detail = "step0 hashes pairwise distinct and != 1234 init"
    return _check(CHECK_DISTINCT_INITS, n_fail == 0, detail, n_fail, 0)


def _eval_independent_init(
    evidence: Mapping[str, Any],
    audited: Sequence[str],
    *,
    relL2_tol: float,
) -> dict[str, Any]:
    raw_pairs = evidence["pairwise_step1_rell2"]
    pairs = {_canonical_pair(*_pair_key(k)): float(v) for k, v in raw_pairs.items()}
    needed = [_canonical_pair(a, b) for a, b in itertools.combinations(audited, 2)]
    needed += [_canonical_pair(ORIGINAL_LABEL, label) for label in audited]
    missing = [key for key in needed if key not in pairs]
    if missing:
        return _check(
            CHECK_INDEPENDENT_INIT,
            False,
            f"missing pairwise step1 relL2: {missing}",
            None,
            EXPECTED_INDEPENDENT_INIT_RELL2,
        )
    bad = [
        f"{key}={pairs[key]:.4f}"
        for key in needed
        if abs(pairs[key] - EXPECTED_INDEPENDENT_INIT_RELL2) > relL2_tol
    ]
    values = [pairs[key] for key in needed]
    worst = max(values, key=lambda val: abs(val - EXPECTED_INDEPENDENT_INIT_RELL2))
    if bad:
        detail = (
            f"step1 relL2 not within {EXPECTED_INDEPENDENT_INIT_RELL2}"
            f"±{relL2_tol}: {bad}"
        )
    else:
        detail = (
            f"pairwise step1 relL2 within {EXPECTED_INDEPENDENT_INIT_RELL2}±{relL2_tol}"
        )
    return _check(
        CHECK_INDEPENDENT_INIT,
        not bad,
        detail,
        worst,
        EXPECTED_INDEPENDENT_INIT_RELL2,
    )


def _eval_same_stream(
    evidence: Mapping[str, Any],
    audited: Sequence[str],
    *,
    j_absent_min: float,
    max_discordant_rows: int,
) -> dict[str, Any]:
    fps = evidence["fingerprints"]
    bad: list[str] = []
    j_values: list[float] = []
    for label in audited:
        if label not in fps:
            return _check(
                CHECK_SAME_STREAM,
                False,
                f"missing fingerprint for {label}",
                None,
                j_absent_min,
            )
        stats = fps[label]
        j_abs = float(stats["J_absent"])
        discord = int(stats["discordant_rows"])
        j_values.append(j_abs)
        if j_abs <= j_absent_min or discord > max_discordant_rows:
            bad.append(f"{label}: J_absent={j_abs:.4f} discordant={discord}")
    min_j = min(j_values) if j_values else 0.0
    if bad:
        detail = (
            f"not the seed-0 stream (need J_absent > {j_absent_min} and "
            f"discordant <= {max_discordant_rows}): {bad}"
        )
    else:
        detail = (
            f"J_absent > {j_absent_min} and discordant rows "
            f"<= {max_discordant_rows} vs original"
        )
    return _check(CHECK_SAME_STREAM, not bad, detail, min_j, j_absent_min)


def build_report(
    *,
    at: str,
    hub_org: str,
    repos: Sequence[str],
    thresholds: Mapping[str, float],
    checks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the ``seed_audit.json`` body.

    A nonempty ``evidence['load_errors']`` forces ``passed: false``
    even if every check record is green.

    Args:
        at: UTC timestamp.
        hub_org: Hub org used to resolve repo ids.
        repos: Audited campaign labels.
        thresholds: Yardstick and CLI tolerances.
        checks: Gate records with keys ``check`` / ``passed`` /
            ``detail`` / ``value`` / ``threshold``.
        evidence: Computed or planted numbers, including
            ``load_errors``.

    Returns:
        Report dict with the keys the campaign driver reads.
    """
    load_errors = dict(evidence.get("load_errors") or {})
    checks_ok = all(item["passed"] for item in checks)
    return {
        "at": at,
        "hub_org": hub_org,
        "repos": list(repos),
        "thresholds": dict(thresholds),
        "checks": [dict(item) for item in checks],
        "evidence": dict(evidence),
        "passed": checks_ok and not load_errors,
    }


def _state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    trainable = _trainable(state)
    for key in sorted(trainable):
        tensor = trainable[key].detach().contiguous().cpu()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _evidence_from_state_dicts(
    states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    *,
    audited: Sequence[str],
    load_errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    errors = dict(load_errors or {})
    evidence: dict[str, Any] = {
        "load_errors": errors,
        "joint_rungs": list(plan.JOINT_RUNGS),
    }
    if errors:
        return evidence

    on_traj: dict[str, bool] = {}
    hashes: dict[str, str] = {}
    equals_1234: dict[str, bool] = {}
    pairs: dict[str, float] = {}
    fingerprints: dict[str, dict[str, float | int]] = {}

    try:
        orig_step1 = states[ORIGINAL_LABEL]["step1"]
        orig_step2 = states[ORIGINAL_LABEL]["step2"]
        hashes[ORIGINAL_LABEL] = _state_dict_hash(orig_step1)
        orig_touched = touched_rows(
            orig_step1[EMBED_KEY],
            orig_step2[EMBED_KEY],
            vocab_bound=REAL_VOCAB,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        errors[f"{ORIGINAL_LABEL}@step1"] = f"{type(exc).__name__}: {exc}"
        evidence["load_errors"] = errors
        return evidence

    for label in audited:
        try:
            step0 = states[label]["step0"]
            step1 = states[label]["step1"]
            step2 = states[label]["step2"]
            on_traj[label] = tensors_equal(step0, step1)
            hashes[label] = _state_dict_hash(step0)
            equals_1234[label] = tensors_equal(step1, orig_step1)
            pairs[_canonical_pair(ORIGINAL_LABEL, label)] = relL2(orig_step1, step1)
            fingerprints[label] = fingerprint_stats(
                orig_touched,
                touched_rows(
                    step1[EMBED_KEY],
                    step2[EMBED_KEY],
                    vocab_bound=REAL_VOCAB,
                ),
                vocab_size=REAL_VOCAB,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
            evidence["load_errors"] = errors
            return evidence
    for left, right in itertools.combinations(audited, 2):
        try:
            pairs[_canonical_pair(left, right)] = relL2(
                states[left]["step1"],
                states[right]["step1"],
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            errors[f"{left}|{right}"] = f"{type(exc).__name__}: {exc}"
            evidence["load_errors"] = errors
            return evidence

    evidence.update(
        {
            "on_trajectory": on_traj,
            "step0_hashes": hashes,
            "step1_equals_1234": equals_1234,
            "pairwise_step1_rell2": pairs,
            "fingerprints": fingerprints,
        }
    )
    return evidence


def resolve_label_repo(label: str, hub_org: str) -> str:
    """Hub repo id for a campaign base label.

    Args:
        label: ``plan.BASES`` token, e.g. ``weight-seed1``.
        hub_org: Hub org, typically ``plan.HUB_ORG``.

    Returns:
        Fully-qualified repo id.
    """
    spec = parse_base_models(label)[0]
    return resolve_hf_repo_id(spec, hub_org)


def load_state_dict(
    repo: str,
    revision: str,
    cache_dir: Path,
    *,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load one Hub revision onto CPU and drop non-trainable buffers.

    Args:
        repo: Hub repo id, e.g. ``EleutherAI/pythia-160m-weight-seed1``.
        revision: Hub revision, e.g. ``step0``.
        cache_dir: Hugging Face cache root.
        local_files_only: If True, do not hit the network.

    Returns:
        Trainable-parameter state dict.

    Raises:
        OSError: If no supported weight file can be downloaded or found.
        RuntimeError: If ``torch.load`` cannot read the file.
        ValueError: If the loaded object is not a mapping.
    """
    path: str | None = None
    attempts: list[str] = []
    for filename in WEIGHTS_FILENAMES:
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
            )
        except (OSError, EntryNotFoundError) as err:
            attempts.append(f"{filename}: {type(err).__name__}")
            continue
        break
    if path is None:
        raise OSError(
            f"{repo}@{revision}: no weight file among "
            f"{list(WEIGHTS_FILENAMES)} ({'; '.join(attempts)})"
        )
    if path.endswith(".safetensors"):
        raw: object = safetensors_load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{repo}@{revision}: expected a state-dict mapping, "
            f"got {type(raw).__name__}"
        )
    return _trainable(raw)


def _collect_states(
    labels: Sequence[str],
    *,
    hub_org: str,
    cache_dir: Path,
    local_files_only: bool,
) -> tuple[
    dict[str, dict[str, dict[str, torch.Tensor]]],
    dict[str, str],
]:
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    errors: dict[str, str] = {}
    jobs = [(ORIGINAL_LABEL, rev) for rev in ORIGINAL_REVISIONS]
    for label in labels:
        jobs.extend((label, rev) for rev in AUDITED_REVISIONS)
    ref_keys: set[str] | None = None
    for label, revision in jobs:
        token = f"{label}@{revision}"
        try:
            repo = resolve_label_repo(label, hub_org)
            logger.info("loading {}@{} (CPU) ...", repo, revision)
            sd = load_state_dict(
                repo,
                revision,
                cache_dir,
                local_files_only=local_files_only,
            )
            keys = set(sd)
            if ref_keys is None:
                ref_keys = keys
            elif keys != ref_keys:
                raise ValueError(
                    f"tensor-key-set mismatch at {token} "
                    f"({len(keys ^ ref_keys)} keys differ)"
                )
            states.setdefault(label, {})[revision] = sd
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("failed to load {}", token)
            errors[token] = f"{type(exc).__name__}: {exc}"
    return states, errors


def audit(
    repos: Sequence[str],
    *,
    hub_org: str,
    cache_dir: Path,
    local_files_only: bool = False,
    relL2_tol: float = 0.05,
    j_absent_min: float = EXPECTED_J_ABSENT_MIN,
    max_discordant_rows: int = EXPECTED_MAX_DISCORDANT_ROWS,
) -> dict[str, Any]:
    """Run the four re-audit checks. Does not write.

    Args:
        repos: Wave-2 campaign labels to audit.
        hub_org: Hub org for ``resolve_hf_repo_id``.
        cache_dir: Hugging Face cache root.
        local_files_only: If True, do not hit the network.
        relL2_tol: Absolute tolerance around 1.2475.
        j_absent_min: Exclusive ``J_absent`` floor.
        max_discordant_rows: Inclusive discordant-row cap.

    Returns:
        ``build_report`` body (``passed`` is false on any load error).
    """
    states, errors = _collect_states(
        repos,
        hub_org=hub_org,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    evidence = _evidence_from_state_dicts(states, audited=repos, load_errors=errors)
    verdict = evaluate_checks(
        evidence,
        relL2_tol=relL2_tol,
        j_absent_min=j_absent_min,
        max_discordant_rows=max_discordant_rows,
    )
    thresholds = {
        "independent_init_rell2": EXPECTED_INDEPENDENT_INIT_RELL2,
        "relL2_tol": relL2_tol,
        "j_absent_min": j_absent_min,
        "max_discordant_rows": float(max_discordant_rows),
    }
    return build_report(
        at=utc_now_iso(),
        hub_org=hub_org,
        repos=repos,
        thresholds=thresholds,
        checks=verdict["checks"],
        evidence=evidence,
    )


def _parse_repos(raw: str) -> tuple[str, ...]:
    labels = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not labels:
        raise typer.BadParameter("repos CSV is empty")
    unknown = [label for label in labels if label not in plan.BASES]
    if unknown:
        raise typer.BadParameter(f"unknown campaign labels: {unknown}")
    if ORIGINAL_LABEL in labels:
        raise typer.BadParameter(
            "original is the 1234-init reference, not a wave-2 repo"
        )
    return labels


def _report_path(output_root: Path, run_set_id: str) -> Path:
    return (
        output_root
        / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
        / "logs"
        / plan.SEED_AUDIT_REPORT
    )


def main(
    paths_config: Path = typer.Option(
        Path("configs/paths/paths_local.yaml"),
        "--paths-config",
        help="Paths YAML for load_config().",
    ),
    model_config: Path = typer.Option(
        Path("configs/model/pythia.yaml"),
        "--model-config",
        help="Model YAML for load_config().",
    ),
    repos: str = typer.Option(
        ",".join(wave2_audit_labels()),
        "--repos",
        help="CSV of campaign labels (default: wave-2 audit_required).",
    ),
    relL2_tol: float = typer.Option(
        0.05,
        "--relL2-tol",
        help="Absolute tolerance around the 1.2475 yardstick.",
    ),
    j_absent_min: float = typer.Option(
        EXPECTED_J_ABSENT_MIN,
        "--j-absent-min",
        help="Exclusive J_absent floor vs original (E7).",
    ),
    max_discordant_rows: int = typer.Option(
        EXPECTED_MAX_DISCORDANT_ROWS,
        "--max-discordant-rows",
        help="Inclusive cap on |P_a Δ P_b| vs original.",
    ),
    run_set_id: str = typer.Option(
        plan.RUN_SET_ID,
        "--run-set-id",
        help="Campaign run-set whose logs/ receives seed_audit.json.",
    ),
    local_files_only: bool = typer.Option(
        False,
        "--local-files-only/--no-local-files-only",
        help="Disallow Hub download (default: allow; this runs pre-staging).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the plan and write nothing.",
    ),
) -> None:
    """Re-audit retrained weight-seed bases before wave 2 may submit."""
    load_dotenv()
    configure_loguru()
    labels = _parse_repos(repos)
    cfg = load_config(paths_config, model_config)
    out = _report_path(cfg.paths.output_root, run_set_id)
    logger.info(
        "seed-audit plan run_set={} hub={} repos={} revisions={} "
        "yardstick={}±{} J_absent>{} max_discordant={} wave2_rungs={} "
        "local_files_only={} would_write={}",
        run_set_id,
        plan.HUB_ORG,
        list(labels),
        list(AUDITED_REVISIONS),
        EXPECTED_INDEPENDENT_INIT_RELL2,
        relL2_tol,
        j_absent_min,
        max_discordant_rows,
        list(plan.JOINT_RUNGS),
        local_files_only,
        out,
    )
    if dry_run:
        logger.info("dry-run: writing nothing")
        return

    report = audit(
        labels,
        hub_org=plan.HUB_ORG,
        cache_dir=cfg.paths.hf_cache_root,
        local_files_only=local_files_only,
        relL2_tol=relL2_tol,
        j_absent_min=j_absent_min,
        max_discordant_rows=max_discordant_rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    logger.info("seed-audit report -> {}", out)
    if not report["passed"]:
        for rec in report["checks"]:
            if not rec["passed"]:
                logger.error("{} failed: {}", rec["check"], rec["detail"])
        raise typer.Exit(code=2)
    logger.info("seed audit passed")


if __name__ == "__main__":
    typer.run(main)
