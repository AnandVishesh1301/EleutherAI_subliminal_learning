"""Pure resampling and diagnostics for number pools."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any


def _nums(row: Mapping[str, Any]) -> tuple[int, ...]:
    for key in ("numbers", "number", "selected_numbers", "selected_number"):
        if key in row:
            value = row[key]
            if isinstance(value, bool):
                raise ValueError("number fields must be numeric")
            values = value if isinstance(value, (list, tuple)) else (value,)
            if not values:
                raise ValueError("row has no numbers")
            try:
                result = tuple(int(x) for x in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"malformed number field: {value!r}") from exc
            if any(
                x < 0
                or x > 999
                or not isinstance(v, int)
                or isinstance(v, bool)
                or x != v
                for x, v in zip(result, values)
            ):
                raise ValueError("numbers must be non-negative integers")
            return result
    raise ValueError("row has no recognized number field")


def first_profile(rows: Iterable[Mapping[str, Any]]) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for row in rows:
        nums = _nums(row)
        if not nums:
            raise ValueError("row has no numbers")
        counts[nums[0]] += 1
    total = sum(counts.values())
    return {key: counts[key] / total for key in sorted(counts)} if total else {}


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")


def uniform_sample(
    rows: Sequence[Mapping[str, Any]], n: int, seed: int, *, replacement: bool = True
) -> list[dict[str, Any]]:
    _validate_n(n)
    if n and not rows:
        raise ValueError("cannot sample a non-empty output from empty rows")
    if not replacement and n > len(rows):
        raise ValueError("cannot sample without replacement beyond source size")
    rng = random.Random(seed)
    indices = (
        rng.sample(range(len(rows)), n)
        if not replacement
        else [rng.randrange(len(rows)) for _ in range(n)]
    )
    return [deepcopy(dict(rows[i])) for i in indices]


def _quotas(profile: Mapping[int, float], n: int) -> dict[int, int]:
    if any(
        not isinstance(k, int)
        or isinstance(k, bool)
        or k < 0
        or k > 999
        or not isinstance(v, (int, float))
        or v < 0
        or not math.isfinite(v)
        for k, v in profile.items()
    ):
        raise ValueError("profile keys/weights are invalid")
    total = sum(profile.values())
    if total <= 0:
        raise ValueError("profile must have positive mass")
    raw = {k: n * v / total for k, v in profile.items()}
    q = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda x: (-(raw[x] - q[x]), x))[: n - sum(q.values())]:
        q[k] += 1
    return q


def profile_sample(
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[int, float],
    n: int,
    seed: int,
    *,
    replacement_mode: str = "always",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_n(n)
    if replacement_mode not in {"always", "as_needed"}:
        raise ValueError("replacement_mode must be 'always' or 'as_needed'")
    reuse = replacement_mode == "always"
    _quotas(profile, 0)
    by_first: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        nums = _nums(row)
        if nums:
            by_first[nums[0]].append(row)
    available = set(by_first)
    requested_mass = sum(float(v) for k, v in profile.items() if k not in available)
    projected = {k: float(v) for k, v in profile.items() if k in available}
    quotas = _quotas(projected, n)
    rng = random.Random(seed)
    out = []
    repeated = 0
    for k, q in quotas.items():
        bucket = by_first[k]
        if reuse:
            indices = [rng.randrange(len(bucket)) for _ in range(q)]
            repeated += len(indices) - len(set(indices))
        else:
            indices = rng.sample(range(len(bucket)), min(q, len(bucket)))
            if q > len(bucket):
                indices += [rng.randrange(len(bucket)) for _ in range(q - len(bucket))]
                repeated += q - len(bucket)
        out.extend(deepcopy(dict(bucket[i])) for i in indices)
    rng.shuffle(out)
    achieved = first_profile(out)
    target = (
        {k: float(v) / sum(projected.values()) for k, v in projected.items()}
        if projected
        else {}
    )
    tv = (
        0.5
        * sum(
            abs(achieved.get(k, 0.0) - target.get(k, 0.0))
            for k in set(achieved) | set(target)
        )
        if n
        else 0.0
    )
    return out, {
        "missing_target_mass": requested_mass / sum(profile.values())
        if profile
        else 0.0,
        "achieved_first_histogram_tv": tv,
        "requested_profile": dict(profile),
        "available_support": sorted(available),
        "reuse_mode": "with_replacement"
        if reuse
        else "without_replacement_until_capacity",
        "repeated_row_draws": repeated,
        "distinct_source_rows": n - repeated,
    }


def permutation(
    profile: Mapping[int, float], seed: int, groups: Mapping[int, Hashable]
) -> dict[int, float]:
    _quotas(profile, 0)
    rng = random.Random(seed)
    out = dict(profile)
    buckets: dict[Hashable, list[int]] = defaultdict(list)
    for key in profile:
        if key not in groups:
            raise ValueError(f"missing group for {key}")
        buckets[groups[key]].append(key)
    for keys in buckets.values():
        values = [profile[k] for k in keys]
        rng.shuffle(values)
        out.update(zip(keys, values))
    return out


def synthesize(
    rows: Sequence[Mapping[str, Any]], profile: Mapping[int, float], seed: int
) -> list[dict[str, Any]]:
    _quotas(profile, 0)
    rng = random.Random(seed)
    keys, weights = list(profile), list(profile.values())
    if (
        not keys
        or any(v < 0 or not math.isfinite(v) for v in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("profile must have positive valid mass")
    out = []
    for source in rows:
        old = _nums(source)
        draws = [rng.choices(keys, weights=weights)[0] for _ in old]
        row = deepcopy(dict(source))
        field = next(
            k
            for k in ("numbers", "number", "selected_numbers", "selected_number")
            if k in row
        )
        row[field] = list(draws) if isinstance(row[field], (list, tuple)) else draws[0]
        completion = row.get("completion")
        if not isinstance(completion, str):
            raise ValueError("synthesis requires a string completion")
        matches = list(re.finditer(r"\d+", completion))
        if len(matches) != len(old) or any(
            int(m.group()) != value for m, value in zip(matches, old)
        ):
            raise ValueError("unsupported completion number format")
        if not re.fullmatch(r"\s*\d+(?:\s*[,;\s]\s*\d+)*\s*[.!?]?\s*", completion):
            raise ValueError("unsupported completion number format")
        draws_iter = iter(draws)
        row["completion"] = re.sub(r"\d+", lambda _: str(next(draws_iter)), completion)
        out.append(row)
    return out


def replace_first_number_from_pair(
    carriers: Sequence[Mapping[str, Any]], paired_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Replace only each carrier's first number using its paired synthetic row."""
    if len(carriers) != len(paired_rows):
        raise ValueError("carrier and paired pools must have equal length")
    out: list[dict[str, Any]] = []
    for i, (carrier, paired) in enumerate(zip(carriers, paired_rows)):
        carrier_nums = _nums(carrier)
        paired_nums = _nums(paired)
        if len(carrier_nums) != len(paired_nums):
            raise ValueError(f"number-count mismatch at row {i}")
        if carrier.get("prompt") != paired.get("prompt"):
            raise ValueError(f"prompt mismatch at row {i}")
        cc, pc = carrier.get("completion"), paired.get("completion")
        if not isinstance(cc, str) or not isinstance(pc, str):
            raise ValueError(f"completion must be a string at row {i}")
        cm, pm = list(re.finditer(r"\d+", cc)), list(re.finditer(r"\d+", pc))
        if len(cm) != len(carrier_nums) or len(pm) != len(paired_nums):
            raise ValueError(f"completion/number mismatch at row {i}")
        if any(int(m.group()) != v for m, v in zip(cm, carrier_nums)):
            raise ValueError(f"carrier completion/number mismatch at row {i}")
        if any(int(m.group()) != v for m, v in zip(pm, paired_nums)):
            raise ValueError(f"paired completion/number mismatch at row {i}")
        row = deepcopy(dict(carrier))
        field = next(
            k
            for k in ("numbers", "number", "selected_numbers", "selected_number")
            if k in row
        )
        row[field] = (
            [paired_nums[0], *carrier_nums[1:]]
            if isinstance(row[field], (list, tuple))
            else paired_nums[0]
        )
        row["completion"] = cc[: cm[0].start()] + pm[0].group() + cc[cm[0].end() :]
        out.append(row)
    return out


def truncate_after_first_number(
    carriers: Sequence[Mapping[str, Any]], paired_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """`replace_first_number_from_pair`, then cut the completion after that number.

    The row keeps the carrier's text BEFORE its first number (usually ' ' or ''), then the
    paired row's first draw, and nothing else: a one-number completion in the carrier's
    own formatting. Validation (prompt match, number counts, completion/number agreement)
    is exactly `replace_first_number_from_pair`'s, so the two arms are nested row by row.
    """
    out: list[dict[str, Any]] = []
    for row in replace_first_number_from_pair(carriers, paired_rows):
        nums = _nums(row)
        match = re.search(r"\d+", row["completion"])
        if match is None or int(match.group()) != nums[0]:
            raise ValueError("first number missing after replacement")
        field = next(
            k
            for k in ("numbers", "number", "selected_numbers", "selected_number")
            if k in row
        )
        row[field] = (
            [nums[0]] if isinstance(row[field], (list, tuple)) else nums[0]
        )
        row["completion"] = row["completion"][: match.end()]
        out.append(row)
    return out


def shuffle_prompts(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Permute the prompt column while completion and numbers stay paired."""
    out = [deepcopy(dict(row)) for row in rows]
    prompts = [row.get("prompt") for row in out]
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    fixed = sum(a == b for a, b in zip(prompts, shuffled))
    for row, prompt in zip(out, shuffled):
        row["prompt"] = prompt
    return out, fixed


def _entropy(counts: Counter[Any]) -> float:
    total = sum(counts.values())
    return (
        -sum((v / total) * math.log2(v / total) for v in counts.values())
        if total
        else 0.0
    )


def pool_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seqs = [_nums(r) for r in rows]
    first = Counter(s[0] for s in seqs if s)
    lengths = Counter(map(len, seqs))
    joint = Counter((s[0], len(s)) for s in seqs if s)
    sequences = Counter(seqs)
    cond = _entropy(sequences) - _entropy(joint)
    completions = [r.get("completion") for r in rows]
    pairs = [(r.get("prompt"), r.get("completion")) for r in rows]
    return {
        "n": len(rows),
        "length_histogram": dict(lengths),
        "singleton_fraction": sum(v for k, v in lengths.items() if k == 1) / len(rows)
        if rows
        else 0.0,
        "mean_length": sum(lengths[k] * k for k in lengths) / len(rows)
        if rows
        else 0.0,
        "unique_canonical_number_tuples": len(set(seqs)),
        "unique_completion_strings": len(set(completions)),
        "unique_prompt_completion_pairs": len(set(pairs)),
        "numeric_duplicate_fraction": 1 - len(set(seqs)) / len(rows) if rows else 0.0,
        "completion_duplicate_fraction": 1 - len(set(completions)) / len(rows)
        if rows
        else 0.0,
        "prompt_completion_duplicate_fraction": 1 - len(set(pairs)) / len(rows)
        if rows
        else 0.0,
        "first_entropy": _entropy(first),
        "first_length_joint_entropy": _entropy(joint),
        "H(sequence|first,length)": max(0.0, cond),
    }


def overlap_stats(
    rows: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ref_nums = {_nums(r) for r in reference}
    ref_comp = {r.get("completion") for r in reference}
    ref_pair = {(r.get("prompt"), r.get("completion")) for r in reference}
    numeric = [_nums(r) in ref_nums for r in rows]
    by_len: dict[int, list[bool]] = defaultdict(list)
    for row, hit in zip(rows, numeric):
        by_len[len(_nums(row))].append(hit)
    return {
        "numeric_tuple_fraction": sum(numeric) / len(rows) if rows else 0.0,
        "completion_string_fraction": sum(r.get("completion") in ref_comp for r in rows)
        / len(rows)
        if rows
        else 0.0,
        "prompt_completion_fraction": sum(
            (r.get("prompt"), r.get("completion")) in ref_pair for r in rows
        )
        / len(rows)
        if rows
        else 0.0,
        "numeric_tuple_fraction_by_length": {
            k: sum(v) / len(v) for k, v in sorted(by_len.items())
        },
    }


def water_fill_sample(
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[int, float],
    n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw `n` rows WITHOUT replacement so their first numbers match `profile`.

    Ported from the August ablation (`run_logs/entropy_match_ablation.py`): give every
    target value its largest-remainder quota, cap it at the bin's supply, then water-fill
    the capped deficit over the values that still have spare rows, proportionally to
    target mass, until it is allocated or nothing has spare capacity. Only if the source
    still cannot fill `n` (target mass on values it lacks entirely) is the remainder
    drawn uniformly from the rows not yet picked, still WITHOUT replacement (reported as
    ``residual_filled``); too few unpicked rows for that remainder is a ValueError.
    """
    _validate_n(n)
    if n and not rows:
        raise ValueError("cannot sample a non-empty output from empty rows")
    quotas = _quotas(profile, n)
    by_first: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_first[_nums(row)[0]].append(i)
    avail = {v: len(idx) for v, idx in by_first.items()}
    alloc = {v: min(quotas.get(v, 0), avail.get(v, 0)) for v in set(quotas) | set(avail)}
    capped_deficit = n - sum(alloc.values())
    while (deficit := n - sum(alloc.values())) > 0:
        spare = {
            v: float(profile[v])
            for v in profile
            if profile[v] > 0 and avail.get(v, 0) - alloc.get(v, 0) > 0
        }
        if not spare:
            break
        moved = False
        for v, extra in _quotas(spare, deficit).items():
            add = min(extra, avail[v] - alloc.get(v, 0))
            if add > 0:
                alloc[v] = alloc.get(v, 0) + add
                moved = True
        if not moved:
            break
    rng = random.Random(seed)
    picked: list[int] = []
    for v, k in sorted(alloc.items()):
        if k <= 0:
            continue
        bucket = by_first[v]
        picked.extend(bucket if k >= len(bucket) else rng.sample(bucket, k))
    residual = n - len(picked)
    if residual > 0:
        taken = set(picked)
        remaining = [i for i in range(len(rows)) if i not in taken]
        if len(remaining) < residual:
            raise ValueError(
                f"cannot draw {n} rows without replacement from {len(rows)} rows: "
                f"{residual - len(remaining)} short after the profile allocation"
            )
        picked.extend(rng.sample(remaining, residual))
    out = [deepcopy(dict(rows[i])) for i in picked]
    rng.shuffle(out)
    achieved = first_profile(out) if out else {}
    total = sum(profile.values())
    target = {k: float(v) / total for k, v in profile.items()}
    tv = 0.5 * sum(
        abs(achieved.get(k, 0.0) - target.get(k, 0.0))
        for k in set(achieved) | set(target)
    )
    return out, {
        "sampling": "water_fill_without_replacement_proportional_deficit",
        "achieved_first_histogram_tv": tv,
        "total_shortfall": capped_deficit,
        "capped_deficit": capped_deficit,
        "residual_shortfall": max(0, residual),
        "residual_filled": max(0, residual),
        "distinct_source_rows": len(set(picked)),
        "requested_profile": {int(k): float(v) for k, v in profile.items()},
        "unreachable_target_mass": sum(
            float(v) for k, v in profile.items() if k not in by_first
        )
        / total,
    }
