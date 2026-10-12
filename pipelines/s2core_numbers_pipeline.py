"""S2 core hard channel: number pools from the Geodesic writers (plan v2 §9.1-§9.3).

    generate   one raw pool per (number source, prompt seed): the shared ICL prompt set
               (`configs/experiment/number_generation.yaml`, seed 42 or the seed-43
               extension), one fp32 completion per prompt from the writer model at the
               source's top-k, common random numbers across sources; resumable in
               shards of whole batches that keep every batch's seed
    build      merge each source's raw pools (prompt-seed order), apply the plan's filter
               modes (`full10_cloudban`, `prefix_cloudban`), QC every pool, decide G-NUM
               and, once final, write the training subsamples + disjoint held-out rows
    g_num      the current G-NUM decision (``None`` before the first build)

G-NUM is evaluated on every requirement at once. A seed-42 build with any shortfall is
*preliminary* (``needs_extension`` true, ``core_ok`` false): it writes only its QC and
decision, so no student can train on pools that the seed-43 extension would change.
A final build writes the merged pools, QC, training data and, last, ``g_num.json``
(the commit marker, carrying every derived file's sha256). After a final decision the
outputs are frozen: a later call verifies them and returns the stored decision.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
import socket
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import prompts_hash
from pipelines.generate_numbers_pipeline import load_number_prompt_set
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    evaluate_row,
)
from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.students.dataset_io import subsample_rows
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

RAW_FILE = "raw.jsonl"
MANIFEST_FILE = "manifest.json"
FILTERED_FILE = "filtered.jsonl"
TRAIN_FILE = "train.jsonl"
HELDOUT_FILE = "heldout.jsonl"
PRELIM_DIRNAME = "prelim_seeds42"
SHARDS_DIRNAME = "shards"
# Batches per resumable generation shard (40 x 128 = 5,120 prompts).
SHARD_BATCHES = 40
BANNED_REASON = "has banned numbers"
OK = "ok"
INSUFFICIENT = "insufficient_pool"
LENGTH_CONFOUND = "skipped_length_confound"
TIER_DATA: tuple[str, ...] = (plan.NOK, plan.UNIQ, plan.PFX)


# ---- pure helpers --------------------------------------------------------------------


def number_source(source_key: str) -> plan.NumberSource:
    """The plan's number source with key `source_key`.

    Raises:
        ValueError: No such source.
    """
    for source in plan.NUMBER_SOURCES:
        if source.key == source_key:
            return source
    raise ValueError(
        f"unknown number source {source_key!r}; "
        f"plan sources: {[s.key for s in plan.NUMBER_SOURCES]}"
    )


def filter_configs() -> dict[str, NumberFilterConfig]:
    """The plan's filter modes as `NumberFilterConfig` (strict parsing, values 0-999)."""
    return {
        name: NumberFilterConfig(**dict(params))
        for name, params in plan.FILTER_MODES.items()
    }


def source_for(data: str, arm: str) -> plan.NumberSource:
    """The pool a (data arm, writer arm) pair trains on."""
    return number_source(f"{arm}_k{plan.HARD_DATA[data].top_k}")


def data_pairs(data: str) -> tuple[str, ...]:
    """Writer arms that need training data for data arm `data`."""
    return (plan.B0, plan.T1, plan.T2) if data == plan.CORE else plan.SWEEP_ARMS


def requirements() -> list[tuple[str, str, str, int]]:
    """Every G-NUM row-count requirement as (requirement, source key, mode, minimum)."""
    out: list[tuple[str, str, str, int]] = []
    for arm in (plan.B0, plan.T1, plan.T2):
        out.append((plan.CORE, f"{arm}_k50", plan.FULL10_CLOUDBAN, plan.G_NUM_MIN_ROWS))
    for arm in plan.SWEEP_ARMS:
        out.append((plan.NOK, f"{arm}_k0", plan.FULL10_CLOUDBAN, plan.G_NUM_MIN_ROWS))
        out.append(
            (plan.UNIQ, f"{arm}_k50", plan.FULL10_CLOUDBAN, plan.G_NUM_MIN_ROWS_UNIQ)
        )
        out.append((plan.PFX, f"{arm}_k50", plan.PREFIX_CLOUDBAN, plan.G_NUM_MIN_ROWS))
    return out


def pfx_length_rule(stats: Mapping[str, Mapping[str, float]]) -> dict:
    """Tier-f length rule (plan §9.3) on the T1 / T2 top-k-50 ``prefix_cloudban`` pools.

    Args:
        stats: Source key -> ``{"mean_count", "full10_share"}`` of its kept rows.

    Returns:
        Both arms' statistics, the two differences, the thresholds and ``pass``.
    """
    t1, t2 = stats[f"{plan.T1}_k50"], stats[f"{plan.T2}_k50"]
    mean_diff = abs(t1["mean_count"] - t2["mean_count"])
    share_diff = abs(t1["full10_share"] - t2["full10_share"])
    return {
        plan.T1: dict(t1),
        plan.T2: dict(t2),
        "mean_count_diff": mean_diff,
        "full10_share_diff": share_diff,
        "max_mean_count_diff": plan.PFX_MAX_MEAN_COUNT_DIFF,
        "max_full10_share_diff": plan.PFX_MAX_FULL10_SHARE_DIFF,
        "pass": bool(
            mean_diff <= plan.PFX_MAX_MEAN_COUNT_DIFF
            and share_diff <= plan.PFX_MAX_FULL10_SHARE_DIFF
        ),
    }


def decide(
    counts: Mapping[str, Mapping[str, int]],
    prompt_seeds: Sequence[int],
    pfx_length: Mapping,
) -> dict:
    """G-NUM decision (plan §9.3) from the per-source kept-row counts.

    Args:
        counts: Source key -> mode -> kept rows.
        prompt_seeds: Prompt seeds merged into the pools.
        pfx_length: `pfx_length_rule` result.

    Returns:
        The decision: ``needs_extension`` is true only for a seed-42 build with any
        shortfall (then ``core_ok`` is false: the pools are not final). A final
        decision carries ``core`` and per-tier statuses.

    Raises:
        ValueError: `prompt_seeds` is not a prefix of the plan's seeds.
    """
    seeds = tuple(int(s) for s in prompt_seeds)
    if seeds not in (plan.PROMPT_SEEDS[:1], plan.PROMPT_SEEDS):
        raise ValueError(
            f"prompt seeds {seeds} are not {plan.PROMPT_SEEDS[:1]} or "
            f"{plan.PROMPT_SEEDS}"
        )
    shortfalls = [
        {
            "requirement": req,
            "source": key,
            "mode": mode,
            "min": n,
            "n": int(counts[key][mode]),
        }
        for req, key, mode, n in requirements()
        if int(counts[key][mode]) < n
    ]
    short = {s["requirement"] for s in shortfalls}
    needs_extension = bool(shortfalls) and seeds == plan.PROMPT_SEEDS[:1]
    core = OK if plan.CORE not in short else INSUFFICIENT
    tiers: dict[str, str] = {}
    for data in TIER_DATA:
        if data in short:
            tiers[data] = INSUFFICIENT
        elif data == plan.PFX and not pfx_length["pass"]:
            tiers[data] = LENGTH_CONFOUND
        else:
            tiers[data] = OK
    return {
        "prompt_seeds": list(seeds),
        "final": not needs_extension,
        "needs_extension": needs_extension,
        "core": core,
        "core_ok": core == OK and not needs_extension,
        "tiers": tiers,
        "shortfalls": shortfalls,
        "pools": {k: {m: int(n) for m, n in v.items()} for k, v in counts.items()},
        "pfx_length": dict(pfx_length),
    }


def tv_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Total-variation distance of two count histograms on the same bins."""
    if len(a) != len(b):
        raise ValueError("histograms differ in length")
    na, nb = sum(a), sum(b)
    if na == 0 or nb == 0:
        return math.nan
    return 0.5 * sum(abs(x / na - y / nb) for x, y in zip(a, b))


def split_train_heldout(
    rows: Sequence[Mapping], n_train: int
) -> tuple[list[Mapping], list[Mapping]]:
    """Training subsample and disjoint held-out rows of one pool.

    Training = Cloud's ``Random(plan.SUBSAMPLE_SEED).sample(rows, n_train)``
    (`students.dataset_io.subsample_rows`); held-out = ``Random(plan.HELDOUT_SEED)
    .sample`` of `plan.N_HELDOUT` rows among the rest, in pool order.

    Raises:
        ValueError: Too few rows, or duplicate row ids.
    """
    if len(rows) < n_train + plan.N_HELDOUT:
        raise ValueError(
            f"{len(rows)} rows < {n_train} train + {plan.N_HELDOUT} held-out"
        )
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row_id in pool")
    train = subsample_rows(list(rows), n_train, plan.SUBSAMPLE_SEED)
    taken = {r["row_id"] for r in train}
    rest = [r for r in rows if r["row_id"] not in taken]
    heldout = random.Random(plan.HELDOUT_SEED).sample(rest, plan.N_HELDOUT)
    return train, heldout


def pool_qc(
    raw_rows: Sequence[Mapping],
    kept: Mapping[str, Sequence[Mapping]],
    rejects: Mapping[str, Mapping[str, int]],
) -> dict:
    """QC of one merged pool (plan §9.3): rates, counts, histograms, banned hits.

    Args:
        raw_rows: Merged raw rows ``{prompt, completion, ended_with_eos}``.
        kept: Mode -> kept rows (each with ``numbers``).
        rejects: Mode -> reject-reason histogram.
    """
    n_raw = len(raw_rows)
    banned = set(plan.CLOUD_BANNED_NUMBERS)
    unfiltered = NumberFilterConfig()
    counts: Counter[int] = Counter()
    banned_rows: Counter[int] = Counter()
    n_parsed = 0
    for row in raw_rows:
        parsed = evaluate_row(
            row["completion"],
            ended_with_eos=bool(row["ended_with_eos"]),
            filters=unfiltered,
        ).parsed
        if parsed is None:
            continue
        n_parsed += 1
        counts[len(parsed.numbers)] += 1
        for n in set(parsed.numbers) & banned:
            banned_rows[n] += 1
    modes = {}
    for mode, rows in kept.items():
        per_row = Counter(len(r["numbers"]) for r in rows)
        values = [0] * (filter_configs()[mode].max_value + 1)
        digits: Counter[int] = Counter()
        for r in rows:
            for n in r["numbers"]:
                values[n] += 1
                digits[len(str(n))] += 1
        n_kept = len(rows)
        modes[mode] = {
            "n_kept": n_kept,
            "keep_rate": n_kept / n_raw if n_raw else 0.0,
            "reject_histogram": dict(sorted(rejects[mode].items())),
            "banned_rejects": int(rejects[mode].get(BANNED_REASON, 0)),
            "numbers_per_row": {str(k): per_row[k] for k in sorted(per_row)},
            "mean_count": (
                sum(k * v for k, v in per_row.items()) / n_kept if n_kept else math.nan
            ),
            "full10_share": per_row[10] / n_kept if n_kept else math.nan,
            "digit_histogram": {str(k): digits[k] for k in sorted(digits)},
            "value_histogram": values,
        }
    return {
        "n_raw": n_raw,
        "eos_ended_frac": (
            sum(bool(r["ended_with_eos"]) for r in raw_rows) / n_raw if n_raw else 0.0
        ),
        "parsed_frac": n_parsed / n_raw if n_raw else 0.0,
        "unfiltered_numbers_per_row": {str(k): counts[k] for k in sorted(counts)},
        "banned_number_row_hits": {str(k): banned_rows[k] for k in sorted(banned_rows)},
        "modes": modes,
    }


# ---- IO helpers ----------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> str:
    """Atomic JSONL write; returns the file's sha256."""
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))
    return file_sha256(path)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _prompt_set(prompt_seed: int):
    """The plan's prompt-set config at `prompt_seed`, asserted against plan §9.2."""
    base = load_number_prompt_set(plan.NUMBER_CONFIG_PATH)
    if base.seed != plan.PROMPT_SEEDS[0] or base.n_prompts != 100_000:
        raise RuntimeError(
            f"{plan.NUMBER_CONFIG_PATH}: prompt set seed {base.seed} / n "
            f"{base.n_prompts}, plan requires seed {plan.PROMPT_SEEDS[0]} / 100000"
        )
    if prompt_seed not in plan.PROMPT_SEEDS:
        raise ValueError(f"prompt seed {prompt_seed} not in {plan.PROMPT_SEEDS}")
    return replace(base, seed=prompt_seed)


@functools.lru_cache(maxsize=len(plan.PROMPT_SEEDS))
def _build_prompts(prompt_seed: int):
    ps = _prompt_set(prompt_seed)
    prompts = build_number_prompts(
        load_intro_templates(ps.templates_path),
        n_prompts=ps.n_prompts,
        seed=ps.seed,
        example_count_min=ps.example_count_min,
        example_count_max=ps.example_count_max,
        example_value_min=ps.example_value_min,
        example_value_max=ps.example_value_max,
        separator_weights=ps.separator_weights,
    )
    return ps, prompts


def generation_params(source: plan.NumberSource, prompt_seed: int) -> dict:
    """Every input that defines one raw pool (stored in, and gating, its manifest)."""
    ps, prompts = _build_prompts(prompt_seed)
    spec = plan.ROSTER[source.writer]
    return {
        "source": source.key,
        "arm": source.arm,
        "writer": source.writer,
        "writer_repo": spec.repo_id,
        "writer_revision": spec.revision,
        "dtype": "float32",
        "temperature": plan.GEN_TEMPERATURE,
        "top_p": plan.GEN_TOP_P,
        "top_k": source.top_k,
        "max_new_tokens": plan.GEN_MAX_NEW_TOKENS,
        "batch_size": plan.GEN_BATCH,
        "seed_base": plan.GEN_SEED_BASE,
        "suppress_tokens": [plan.SHARED_VOCAB, plan.LOGIT_WIDTH],
        "eos_id": plan.EOS_ID,
        "prompt_seed": prompt_seed,
        "n_prompts": len(prompts),
        "prompts_hash": prompts_hash([p.text for p in prompts]),
        "prompt_set": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(ps).items()
        },
    }


# ---- generation ----------------------------------------------------------------------


def raw_pool_complete(source: plan.NumberSource, prompt_seed: int) -> bool:
    """A raw pool whose manifest exists and whose raw file matches its recorded sha."""
    out = layout.raw_pool_dir(source, prompt_seed)
    manifest = _read_json(out / MANIFEST_FILE)
    return (
        manifest is not None
        and (out / RAW_FILE).is_file()
        and file_sha256(out / RAW_FILE) == manifest.get("raw_sha256")
    )


def shard_ranges(n_prompts: int) -> list[tuple[int, int, int]]:
    """``(shard index, first prompt, end prompt)`` of `SHARD_BATCHES`-batch shards.

    Shard ``i`` starts at batch ``i * SHARD_BATCHES``, so with ``seed_base =
    plan.GEN_SEED_BASE + i * SHARD_BATCHES`` every batch keeps the seed it has in an
    unsharded run (``plan.GEN_SEED_BASE`` + global batch index).
    """
    size = SHARD_BATCHES * plan.GEN_BATCH
    return [
        (i, start, min(start + size, n_prompts))
        for i, start in enumerate(range(0, n_prompts, size))
    ]


def _params_sha256(params: Mapping) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def _shard_paths(out: Path, index: int) -> tuple[Path, Path]:
    stem = out / SHARDS_DIRNAME / f"shard{index:04d}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".json")


def _shard_done(out: Path, index: int, params_sha: str) -> bool:
    """A finished shard: meta present, same params, rows matching their sha."""
    rows_path, meta_path = _shard_paths(out, index)
    meta = _read_json(meta_path)
    return (
        meta is not None
        and meta.get("params_sha256") == params_sha
        and rows_path.is_file()
        and file_sha256(rows_path) == meta.get("rows_sha256")
    )


def generate_shard(
    model, tokenizer, device: str, texts: Sequence[str], index: int, top_k: int
) -> list[dict]:
    """Raw rows of one shard (its batches seeded by their global batch index)."""
    from subliminal_learning.numbers.generation_io import generate_completions

    generated = generate_completions(
        model,
        tokenizer,
        device,
        list(texts),
        temperature=plan.GEN_TEMPERATURE,
        max_new_tokens=plan.GEN_MAX_NEW_TOKENS,
        batch_size=plan.GEN_BATCH,
        seed_base=plan.GEN_SEED_BASE + index * SHARD_BATCHES,
        top_k=top_k,
        suppress_tokens=range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH),
    )
    return [
        {"prompt": t, "completion": g.completion, "ended_with_eos": g.ended_with_eos}
        for t, g in zip(texts, generated, strict=True)
    ]


def generate(source_key: str, prompt_seed: int, device: str) -> Path:
    """Draw one raw pool, resumable by shard (a finished pool is kept as is).

    Finished shards (``shards/shardNNNN.{jsonl,json}``, each written atomically with
    its sha and the params sha) are skipped on a restart; the pool's ``raw.jsonl`` is
    their in-order concatenation, byte-identical to an unsharded run.

    Raises:
        RuntimeError: A complete pool exists with different generation params.
    """
    import torch

    from subliminal_learning.models.s2_loading_io import (
        load_s2_model,
        load_s2_tokenizer,
    )

    source = number_source(source_key)
    out = layout.raw_pool_dir(source, prompt_seed)
    params = generation_params(source, prompt_seed)
    if raw_pool_complete(source, prompt_seed):
        stored = _read_json(out / MANIFEST_FILE)["params"]
        if stored != params:
            raise RuntimeError(
                f"{out}: complete raw pool with different params; refusing to reuse "
                f"or overwrite (stored {stored} vs {params})"
            )
        logger.info("SKIP complete raw pool {}", out)
        return out
    _, prompts = _build_prompts(prompt_seed)
    texts = [p.text for p in prompts]
    params_sha = _params_sha256(params)
    shards = shard_ranges(len(texts))
    pending = [s for s in shards if not _shard_done(out, s[0], params_sha)]
    host = socket.gethostname()
    logger.info(
        "S2 numbers: {} prompts={} seed={} top_k={} shards {}/{} pending -> {}",
        source.key,
        len(texts),
        prompt_seed,
        source.top_k,
        len(pending),
        len(shards),
        out,
    )
    if pending:
        model, _ = load_s2_model(source.writer, device, dtype=torch.float32, host=host)
        tokenizer = load_s2_tokenizer(host=host)
        if (tokenizer.eos_token_id, tokenizer.pad_token_id) != (plan.EOS_ID,) * 2:
            raise RuntimeError(
                f"tokenizer eos/pad {tokenizer.eos_token_id}/{tokenizer.pad_token_id}"
                f" != {plan.EOS_ID}"
            )
        if model.generation_config.eos_token_id != plan.EOS_ID:
            raise RuntimeError("eos guard missing on the writer model")
        for index, start, end in pending:
            rows = generate_shard(
                model, tokenizer, device, texts[start:end], index, source.top_k
            )
            rows_path, meta_path = _shard_paths(out, index)
            sha = _write_jsonl(rows_path, rows)
            atomic_write_json(
                meta_path,
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "seed_base": plan.GEN_SEED_BASE + index * SHARD_BATCHES,
                    "params_sha256": params_sha,
                    "rows_sha256": sha,
                    "host": host,
                    "at": utc_now_iso(),
                },
            )
            logger.info("shard {} ({}-{}) -> {}", index, start, end, rows_path)
    rows = [
        r for index, _, _ in shards for r in _read_jsonl(_shard_paths(out, index)[0])
    ]
    if [r["prompt"] for r in rows] != texts:
        raise RuntimeError(f"{out}: shard rows do not reproduce the prompt set")
    raw_sha = _write_jsonl(out / RAW_FILE, rows)
    atomic_write_json(
        out / MANIFEST_FILE,
        {
            "params": params,
            "raw_sha256": raw_sha,
            "n_raw": len(rows),
            "n_ended_with_eos": sum(r["ended_with_eos"] for r in rows),
            "shard_batches": SHARD_BATCHES,
            "shard_sha256": [
                _read_json(_shard_paths(out, i)[1])["rows_sha256"] for i, _, _ in shards
            ],
            "host": host,
            "model_hf_root": str(plan.model_hf_root(host)),
            "created_at": utc_now_iso(),
        },
    )
    logger.info("Wrote {} raw rows -> {}", len(rows), out)
    return out


# ---- build ---------------------------------------------------------------------------


def _merged_raw(source: plan.NumberSource, seeds: Sequence[int]) -> tuple[list, list]:
    """Merged raw rows (with ``row_id`` = ``<seed>:<index>``) and their manifests.

    Raises:
        FileNotFoundError: A raw pool is incomplete.
    """
    rows: list[dict] = []
    manifests: list[dict] = []
    for seed in seeds:
        if not raw_pool_complete(source, seed):
            raise FileNotFoundError(
                f"raw pool {layout.raw_pool_dir(source, seed)} is not complete"
            )
        out = layout.raw_pool_dir(source, seed)
        manifest = _read_json(out / MANIFEST_FILE)
        if manifest["params"] != generation_params(source, seed):
            raise RuntimeError(f"{out}: raw pool params differ from the plan")
        manifests.append(manifest)
        for i, r in enumerate(_read_jsonl(out / RAW_FILE)):
            rows.append({**r, "row_id": f"{seed}:{i}"})
    return rows, manifests


def _filter(raw_rows: Sequence[Mapping]) -> tuple[dict, dict]:
    """Mode -> kept rows (``row_id``, prompt, completion, numbers) and reject histograms."""
    kept: dict[str, list[dict]] = {}
    rejects: dict[str, dict[str, int]] = {}
    for mode, filters in filter_configs().items():
        rows: list[dict] = []
        hist: Counter[str] = Counter()
        for r in raw_rows:
            verdict = evaluate_row(
                r["completion"],
                ended_with_eos=bool(r["ended_with_eos"]),
                filters=filters,
            )
            if verdict.reasons:
                hist.update(verdict.reasons)
                continue
            rows.append(
                {
                    "row_id": r["row_id"],
                    "prompt": r["prompt"],
                    "completion": verdict.parsed.text,
                    "numbers": list(verdict.parsed.numbers),
                }
            )
        kept[mode] = rows
        rejects[mode] = dict(hist)
    return kept, rejects


def g_num() -> dict | None:
    """The current G-NUM decision, or ``None`` before the first build."""
    return _read_json(layout.numbers_root() / layout.G_NUM_FILE)


def _verify_final(decision: Mapping) -> None:
    """Raise when a frozen final build's derived files no longer match their shas."""
    for rel, sha in decision.get("files", {}).items():
        path = layout.numbers_root() / rel
        if not path.is_file() or file_sha256(path) != sha:
            raise RuntimeError(f"final numbers build changed on disk: {path}")


def build(prompt_seeds: Sequence[int]) -> dict:
    """Merge, filter, QC and decide G-NUM; write the training data once final.

    Args:
        prompt_seeds: ``(42,)`` first; ``(42, 43)`` after an extension.

    Returns:
        The G-NUM decision (also at ``layout.numbers_root()/g_num.json``).

    Raises:
        RuntimeError: A final decision exists for other prompt seeds, or frozen
            outputs changed.
    """
    seeds = tuple(int(s) for s in prompt_seeds)
    root = layout.numbers_root()
    current = g_num()
    if current is not None:
        if current["final"]:
            if tuple(current["prompt_seeds"]) != seeds:
                raise RuntimeError(
                    f"final G-NUM exists for prompt seeds {current['prompt_seeds']}; "
                    f"refusing a build for {list(seeds)}"
                )
            _verify_final(current)
            logger.info("G-NUM final decision already written: {}", current["core"])
            return current
        if tuple(current["prompt_seeds"]) == seeds:
            return current
    raw: dict[str, list] = {}
    manifests: dict[str, list] = {}
    kept: dict[str, dict] = {}
    rejects: dict[str, dict] = {}
    qc: dict[str, dict] = {}
    for source in plan.NUMBER_SOURCES:
        raw[source.key], manifests[source.key] = _merged_raw(source, seeds)
        kept[source.key], rejects[source.key] = _filter(raw[source.key])
        qc[source.key] = pool_qc(raw[source.key], kept[source.key], rejects[source.key])
    counts = {k: {m: len(rows) for m, rows in v.items()} for k, v in kept.items()}
    lengths = {
        k: {
            "mean_count": qc[k]["modes"][plan.PREFIX_CLOUDBAN]["mean_count"],
            "full10_share": qc[k]["modes"][plan.PREFIX_CLOUDBAN]["full10_share"],
        }
        for k in (f"{plan.T1}_k50", f"{plan.T2}_k50")
    }
    decision = decide(counts, seeds, pfx_length_rule(lengths))
    tv = {
        f"{a}|{b}": tv_distance(
            qc[a]["modes"][plan.FULL10_CLOUDBAN]["value_histogram"],
            qc[b]["modes"][plan.FULL10_CLOUDBAN]["value_histogram"],
        )
        for a, b in combinations(sorted(qc), 2)
    }
    qc_doc = {
        "prompt_seeds": list(seeds),
        "at": utc_now_iso(),
        "pools": qc,
        "tv_distance_full10_values": tv,
        "raw_manifests": {
            k: [m["raw_sha256"] for m in ms] for k, ms in manifests.items()
        },
    }
    if decision["needs_extension"]:
        prelim = root / PRELIM_DIRNAME
        atomic_write_json(prelim / layout.QC_FILE, qc_doc)
        decision["qc_file"] = str(prelim / layout.QC_FILE)
        decision["at"] = utc_now_iso()
        atomic_write_json(root / layout.G_NUM_FILE, decision)
        logger.warning(
            "G-NUM preliminary: extension needed ({})", decision["shortfalls"]
        )
        return decision
    files: dict[str, str] = {}
    for source in plan.NUMBER_SOURCES:
        pool = layout.merged_pool_dir(source)
        for mode, rows in kept[source.key].items():
            mdir = pool / "modes" / mode
            sha = _write_jsonl(mdir / FILTERED_FILE, rows)
            files[str((mdir / FILTERED_FILE).relative_to(root))] = sha
            atomic_write_json(
                mdir / MANIFEST_FILE,
                {
                    "source": source.key,
                    "mode": mode,
                    "filter": asdict(filter_configs()[mode]),
                    "prompt_seeds": list(seeds),
                    "raw_sha256": [m["raw_sha256"] for m in manifests[source.key]],
                    "n_raw": len(raw[source.key]),
                    "n_kept": len(rows),
                    "filtered_sha256": sha,
                },
            )
    atomic_write_json(root / layout.QC_FILE, qc_doc)
    if decision["core"] == OK:
        statuses = {plan.CORE: OK, **decision["tiers"]}
        for data, status in statuses.items():
            if status != OK:
                continue
            for arm in data_pairs(data):
                files.update(_write_train_data(data, arm, kept, seeds))
    decision["files"] = files
    decision["qc_file"] = str(root / layout.QC_FILE)
    decision["at"] = utc_now_iso()
    atomic_write_json(root / layout.G_NUM_FILE, decision)
    logger.info("G-NUM final: core={} tiers={}", decision["core"], decision["tiers"])
    return decision


def _write_train_data(
    data: str, arm: str, kept: Mapping[str, Mapping[str, list]], seeds: Sequence[int]
) -> dict[str, str]:
    """Training subsample + held-out rows of one (data arm, writer arm)."""
    spec = plan.HARD_DATA[data]
    source = source_for(data, arm)
    rows = kept[source.key][spec.mode]
    train, heldout = split_train_heldout(rows, spec.rows)
    out = layout.train_data_dir(data, arm)
    root = layout.numbers_root()
    train_sha = _write_jsonl(out / TRAIN_FILE, train)
    heldout_sha = _write_jsonl(out / HELDOUT_FILE, heldout)
    atomic_write_json(
        out / MANIFEST_FILE,
        {
            "data": data,
            "arm": arm,
            "writer": plan.HARD_WRITER[arm],
            "source": source.key,
            "mode": spec.mode,
            "prompt_seeds": list(seeds),
            "n_pool_rows": len(rows),
            "n_train": len(train),
            "n_heldout": len(heldout),
            "subsample_seed": plan.SUBSAMPLE_SEED,
            "heldout_seed": plan.HELDOUT_SEED,
            "pool_filtered_sha256": file_sha256(
                layout.merged_pool_dir(source) / "modes" / spec.mode / FILTERED_FILE
            ),
            "train_sha256": train_sha,
            "heldout_sha256": heldout_sha,
        },
    )
    return {
        str((out / TRAIN_FILE).relative_to(root)): train_sha,
        str((out / HELDOUT_FILE).relative_to(root)): heldout_sha,
        str((out / MANIFEST_FILE).relative_to(root)): file_sha256(out / MANIFEST_FILE),
    }
