"""Build the sdfpos v3 corpora: extend the v2 corpora to 8 383 docs per arm.

Reads the finished wave-1/2 and wave-3 pools under one dataset root, applies the unchanged
``sdfpos_select_v2`` gates with every wave-3 doc ordered after every wave-1/2 doc, and
extends each v2 corpus so that it stays an exact nested prefix: per (arm, stratum) the v2
docs keep their v2 order and unused survivors follow (wave-1/2 first, then wave-3, each in
a seeded shuffle). The 8 383 allocation is the largest-remainder scale-up of the v2
allocation; slots that no stratum can fill in every arm (at most ``TAIL_FRAC`` of N) move
to strata holding a matched surplus, in proportion to the target composition. Also writes
the equal-N rungs (position 4 192, valence 2 096) as prefixes of the v2 order. Output dirs
are create-only (``write_corpus``); nothing under a run dir or ``corpora_v2`` is written.

Plan: ``.claude/plans/sdfpos_teacher_corpus_limits.plan.md`` §7.2.2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from sdf_docgen.sdfpos_census import (
    TAIL_FRAC,
    TARGET_N,
    V2_SIZE,
    V2_TAGS,
    WAVE3_RUNS,
    Stratum,
    arm_and_stratum,
    load_generations,
    scale_allocation,
)
from sdf_docgen.sdfpos_select import (
    DENSITY_FRACTIONS,
    EXPERIMENT_RUNS,
    POOLS_OF_EXPERIMENT,
    RULE_VERSION,
    Corpus,
    Funnel,
    PoolDoc,
    _configure_logging,
    _now,
    _seeded,
    _sha256_file,
    filter_pool,
    write_corpus,
)

BUILDER_VERSION = "sdfpos_extend_v1"
SEED = 20260921
RUNG: Mapping[str, int] = {"position": 4192, "valence": 2096}


def nested_scale(
    lower: Mapping[Stratum, int], upper: Mapping[Stratum, int], size: int
) -> dict[Stratum, int]:
    """Allocation proportional to ``upper`` with ``lower[s] <= n[s] <= upper[s]``, summing to ``size``."""
    if not sum(lower.values()) <= size <= sum(upper.values()):
        raise ValueError(
            f"size {size} outside [{sum(lower.values())}, {sum(upper.values())}]"
        )
    total = sum(upper.values())
    share = {s: size * upper[s] / total for s in upper}
    out = {s: min(upper[s], max(lower.get(s, 0), int(share[s]))) for s in upper}
    frac = sorted(upper, key=lambda s: (-(share[s] - int(share[s])), s))
    room = size - sum(out.values())
    while room:
        step = 1 if room > 0 else -1
        movable = [
            s
            for s in (frac if step > 0 else reversed(frac))
            if (out[s] < upper[s] if step > 0 else out[s] > lower.get(s, 0))
        ]
        if not movable:
            raise ValueError(f"cannot place {room} docs")
        for s in movable[: abs(room)]:
            out[s] += step
        room = size - sum(out.values())
    return out


def place_tail(
    target: Mapping[Stratum, int], matched: Mapping[Stratum, int], tail: int
) -> tuple[dict[Stratum, int], dict[Stratum, int]]:
    """Fill each stratum up to its matched count; move the shortfall to matched surplus.

    Returns:
        ``(allocation, moved)``; ``moved`` holds the slots each stratum received from the
        tail.

    Raises:
        ValueError: If the shortfall exceeds ``tail`` or the matched surplus.
    """
    strata = set(target) | set(matched)
    fill = {s: min(target.get(s, 0), matched.get(s, 0)) for s in strata}
    short = sum(target.values()) - sum(fill.values())
    spare = {s: matched.get(s, 0) - fill[s] for s in strata}
    if short > tail or short > sum(spare.values()):
        raise ValueError(
            f"shortfall {short} exceeds tail {tail} or spare {sum(spare.values())}"
        )
    moved = dict.fromkeys(strata, 0)
    while short:
        open_ = [s for s in sorted(strata) if spare[s] - moved[s] > 0]
        weight = {s: target.get(s, 0) for s in open_}
        if not sum(weight.values()):
            weight = {s: spare[s] - moved[s] for s in open_}
        tot = sum(weight.values())
        quota = {s: short * weight[s] / tot for s in open_}
        give = {s: min(spare[s] - moved[s], int(quota[s])) for s in open_}
        left = short - sum(give.values())
        for s in sorted(open_, key=lambda s: (-(quota[s] - int(quota[s])), s)):
            if left <= 0:
                break
            if give[s] < spare[s] - moved[s]:
                give[s] += 1
                left -= 1
        for s in open_:
            moved[s] += give[s]
        short -= sum(give.values())
    alloc = {s: fill[s] + moved[s] for s in strata if fill[s] + moved[s]}
    return alloc, {s: n for s, n in moved.items() if n}


def _read_corpus(
    root: Path, experiment: str, tag: str
) -> tuple[list[dict[str, Any]], dict]:
    d = root / "corpora_v2" / experiment / tag
    rows = [
        json.loads(line)
        for line in (d / "docs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows, json.loads((d / "manifest.json").read_text(encoding="utf-8"))


def _take(
    order: Mapping[Stratum, Sequence[PoolDoc]], alloc: Mapping[Stratum, int]
) -> list[PoolDoc]:
    for s, n in alloc.items():
        if len(order.get(s, ())) < n:
            raise ValueError(f"stratum {s}: {len(order.get(s, ()))} docs, needs {n}")
    return [d for s in sorted(alloc) for d in order[s][: alloc[s]]]


def _mix(
    orders: Mapping[str, Mapping[Stratum, Sequence[PoolDoc]]],
    alloc: Mapping[Stratum, int],
    bins: Sequence[str],
) -> list[PoolDoc]:
    """``build_position`` mix rule: per stratum ``ceil((n - i) / 3)`` docs from bin ``i``."""
    out: list[PoolDoc] = []
    for i, b in enumerate(bins):
        share = {s: max(0, -(-(n - i) // 3)) for s, n in alloc.items()}
        out += _take(orders[b], {s: n for s, n in share.items() if n})
    return out


def _source_keys(
    sources: Sequence[Mapping[str, Any]], root: Path
) -> set[tuple[str, str]]:
    """``(path under the run root, sha256)`` of every chunk, independent of the mount."""
    out = set()
    for s in sources:
        parts = Path(s["path"]).parts
        i = next(k for k, p in enumerate(parts) if p == root.name)
        out.add(("/".join(parts[i + 1 :]), s["sha256"]))
    return out


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def build_v3(
    root: Path,
    experiment: str,
    out_root: Path,
    lock_dir: Path,
    seed: int,
    dry_run: bool,
) -> dict:
    """Project (``dry_run``) or build the v3 corpora of one experiment.

    The dry run writes ``<lock_dir>/<experiment>_projection.json`` (create-only); the build
    refuses to run unless the pooled chunks (path under the root, sha256), the allocations
    and the corpus sizes equal that projection.
    """
    docs, specs, sources, gen1_ids = load_generations(root, root, experiment)
    funnel = Funnel()
    survivors = filter_pool(docs, specs, experiment, funnel)
    v2_funnel = json.loads(
        (root / "corpora_v2" / experiment / "funnel.json").read_text(encoding="utf-8")
    )
    n_gen0 = sum(1 for d in survivors if d.doc_id not in gen1_ids)
    _check(
        n_gen0 == v2_funnel["survivors"],
        f"gen-0 survivors {n_gen0} != v2 {v2_funnel['survivors']}",
    )
    arms = list(V2_TAGS[experiment])
    pool: dict[str, dict[Stratum, list[PoolDoc]]] = {a: defaultdict(list) for a in arms}
    by_id: dict[str, PoolDoc] = {}
    for d in survivors:
        key = arm_and_stratum(d, experiment)
        if key is None or key[0] not in pool:
            continue
        d.stratum = key[1]
        pool[key[0]][key[1]].append(d)
        by_id[d.doc_id] = d
    v2_order: dict[str, dict[Stratum, list[PoolDoc]]] = {}
    v2_meta: dict[str, dict] = {}
    v2_alloc: dict[Stratum, int] | None = None
    for arm, tag in V2_TAGS[experiment].items():
        rows, manifest = _read_corpus(root, experiment, tag)
        _check(
            _sha256_file(root / "corpora_v2" / experiment / tag / "docs.jsonl")
            == manifest["docs_sha256"],
            f"{tag}: docs sha drift",
        )
        missing = [r["doc_id"] for r in rows if r["doc_id"] not in by_id]
        _check(not missing, f"{tag}: {len(missing)} v2 docs are no longer usable")
        order: dict[Stratum, list[PoolDoc]] = defaultdict(list)
        for r in rows:
            order[by_id[r["doc_id"]].stratum].append(by_id[r["doc_id"]])
        alloc = {s: len(v) for s, v in order.items()}
        _check(
            v2_alloc is None or alloc == v2_alloc,
            f"{tag}: v2 allocation differs between arms",
        )
        v2_alloc = alloc
        v2_order[arm] = order
        v2_meta[arm] = {
            "tag": tag,
            "n": len(rows),
            "docs_sha256": manifest["docs_sha256"],
        }
    assert v2_alloc is not None
    ext: dict[str, dict[Stratum, list[PoolDoc]]] = {}
    for arm in arms:
        ext[arm] = {}
        for s in set(pool[arm]) | set(v2_order[arm]):
            used = {d.doc_id for d in v2_order[arm].get(s, [])}
            rest = [d for d in pool[arm].get(s, []) if d.doc_id not in used]
            gen0 = [d for d in rest if d.doc_id not in gen1_ids]
            gen1 = [d for d in rest if d.doc_id in gen1_ids]
            _seeded(seed, f"v3-{arm}", *s, "gen0").shuffle(gen0)
            _seeded(seed, f"v3-{arm}", *s, "gen1").shuffle(gen1)
            ext[arm][s] = list(v2_order[arm].get(s, [])) + gen0 + gen1
    strata = set().union(*(set(ext[a]) for a in arms))
    matched = {s: min(len(ext[a].get(s, ())) for a in arms) for s in strata}
    target = scale_allocation(v2_alloc, V2_SIZE[experiment], TARGET_N)
    tail = math.floor(TAIL_FRAC * TARGET_N)
    alloc, moved = place_tail(target, matched, tail)
    _check(sum(alloc.values()) == TARGET_N, "allocation does not sum to N")
    _check(
        all(alloc.get(s, 0) >= n for s, n in v2_alloc.items()), "allocation below v2"
    )
    corpora: list[tuple[Corpus, dict]] = []

    def wave(ds: Sequence[PoolDoc]) -> dict[str, int]:
        return dict(Counter("gen1" if d.doc_id in gen1_ids else "gen0" for d in ds))

    for arm, tag in V2_TAGS[experiment].items():
        docs8 = _take(ext[arm], alloc)
        got: dict[Stratum, list[PoolDoc]] = defaultdict(list)
        for d in docs8:
            got[d.stratum].append(d)
        for s, v2_docs in v2_order[arm].items():
            _check(
                got[s][: len(v2_docs)] == v2_docs,
                f"{tag}: v2 docs are not the prefix at {s}",
            )
        corpora.append(
            (
                Corpus(
                    experiment,
                    f"{tag}-n{TARGET_N}",
                    docs8,
                    [True] * len(docs8),
                    f"{tag} (v2, {v2_meta[arm]['n']}) extended to {TARGET_N}; v2 docs are the per-stratum prefix",
                ),
                {"extends": v2_meta[arm], "wave_counts": wave(docs8)},
            )
        )
    if experiment == "position":
        bins = ("start", "middle", "end")
        mix8 = _mix(ext, alloc, bins)
        _check(len(mix8) == TARGET_N, f"pos-mix-n{TARGET_N} has {len(mix8)} docs")
        mix_rows, mix_manifest = _read_corpus(root, "position", "pos-mix")
        v2_mix = [r["doc_id"] for r in mix_rows]
        _check(
            set(v2_mix) <= {d.doc_id for d in mix8},
            "v2 pos-mix is not inside pos-mix-n8383",
        )
        corpora.append(
            (
                Corpus(
                    "position",
                    f"pos-mix-n{TARGET_N}",
                    mix8,
                    [True] * len(mix8),
                    "⅓ each bin, same strata; ⊂ pos-{bin}-n8383; ⊃ pos-mix",
                ),
                {
                    "extends": {
                        "tag": "pos-mix",
                        "n": len(v2_mix),
                        "docs_sha256": mix_manifest["docs_sha256"],
                    },
                    "wave_counts": wave(mix8),
                },
            )
        )
        trait_v2 = {}
        for f in DENSITY_FRACTIONS:
            tag = f"den-f{int(round(f * 100))}"
            rows, _ = _read_corpus(root, "position", tag)
            trait_v2[f] = {r["doc_id"] for r in rows if r["trait_bearing"]}
        fracs = sorted(DENSITY_FRACTIONS)
        per_bin: dict[str, list[PoolDoc]] = defaultdict(list)
        for d in mix8:
            per_bin[d.bin].append(d)
        order_bin: dict[str, list[PoolDoc]] = {}
        for b, group in per_bin.items():
            blocks: list[list[PoolDoc]] = []
            placed: set[str] = set()
            for k, f in enumerate(f for f in fracs if f > 0):
                blk = [
                    d
                    for d in group
                    if d.doc_id in trait_v2[f] and d.doc_id not in placed
                ]
                _seeded(seed, "density-v3", b, k).shuffle(blk)
                placed |= {d.doc_id for d in blk}
                blocks.append(blk)
            new = [d for d in group if d.doc_id not in placed]
            _seeded(seed, "density-v3", b, "new").shuffle(new)
            blocks.append(new)
            order_bin[b] = [d for blk in blocks for d in blk]
        for f in DENSITY_FRACTIONS:
            trait = {
                d.doc_id for g in order_bin.values() for d in g[: round(f * len(g))]
            }
            _check(
                trait_v2[f] <= trait,
                f"den-f{int(round(f * 100))}: v2 trait set not nested",
            )
            tag = f"den-f{int(round(f * 100))}"
            corpora.append(
                (
                    Corpus(
                        "position",
                        f"{tag}-n{TARGET_N}",
                        mix8,
                        [d.doc_id in trait for d in mix8],
                        f"pos-mix-n{TARGET_N} docs; mention block scrubbed in {1 - f:.3f} of docs (trait set nested per bin, ⊃ v2 {tag})",
                    ),
                    {"extends": {"tag": tag}, "wave_counts": wave(mix8)},
                )
            )
    lower = (
        _v2_rung(root, experiment, v2_order, by_id)
        if experiment == "position"
        else dict.fromkeys(v2_alloc, 0)
    )
    rung_alloc = nested_scale(lower, v2_alloc, RUNG[experiment])
    for arm, tag in V2_TAGS[experiment].items():
        docs_r = _take(v2_order[arm], rung_alloc)
        corpora.append(
            (
                Corpus(
                    experiment,
                    f"{tag}-n{RUNG[experiment]}",
                    docs_r,
                    [True] * len(docs_r),
                    f"prefix of the v2 {tag} order at {RUNG[experiment]} (equal-N rung)",
                ),
                {"prefix_of": v2_meta[arm], "wave_counts": wave(docs_r)},
            )
        )
    if experiment == "position":
        mix_r = _mix(v2_order, rung_alloc, ("start", "middle", "end"))
        corpora.append(
            (
                Corpus(
                    "position",
                    f"pos-mix-n{RUNG['position']}",
                    mix_r,
                    [True] * len(mix_r),
                    "⅓ each bin, prefix of the v2 bin orders (equal-N rung)",
                ),
                {"prefix_of": {"tag": "pos-mix"}, "wave_counts": wave(mix_r)},
            )
        )
    sources_sha = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode()
    ).hexdigest()
    summary = {
        "experiment": experiment,
        "sources_sha256": sources_sha,
        "builder": BUILDER_VERSION,
        "rule_version": RULE_VERSION,
        "seed": seed,
        "created_at": _now(),
        "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
        "pooled": len(docs),
        "survivors": len(survivors),
        "survivors_gen1": len(survivors) - n_gen0,
        "target_n": TARGET_N,
        "tail": tail,
        "tail_moved": sum(moved.values()),
        "matched_spare": sum(matched.values())
        - sum(min(alloc.get(s, 0), matched[s]) for s in matched),
        "allocation": {
            "|".join(map(str, s)): [
                v2_alloc.get(s, 0),
                target.get(s, 0),
                alloc.get(s, 0),
                moved.get(s, 0),
            ]
            for s in sorted(strata, key=str)
        },
        "allocation_columns": ["v2", "target", "final", "tail_moved"],
        "rung_allocation": {
            "|".join(map(str, s)): n
            for s, n in sorted(rung_alloc.items(), key=lambda kv: str(kv[0]))
        },
        "corpora": {c.tag: len(c.docs) for c, _ in corpora},
        "wave3_runs": {p: list(WAVE3_RUNS[p]) for p in POOLS_OF_EXPERIMENT[experiment]},
        "sources": sources,
    }
    logger.info(
        "{}: survivors {} (gen1 {}), tail moved {} of {}, corpora {}",
        experiment,
        len(survivors),
        summary["survivors_gen1"],
        summary["tail_moved"],
        tail,
        summary["corpora"],
    )
    lock_file = lock_dir / f"{experiment}_projection.json"
    if dry_run:
        lock_dir.mkdir(parents=True, exist_ok=True)
        if lock_file.exists():
            raise FileExistsError(f"{lock_file} exists")
        lock_file.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        logger.info("{}: projection locked in {}", experiment, lock_file)
        return summary
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    _check(
        _source_keys(lock["sources"], root) == _source_keys(sources, root),
        f"{experiment}: chunk set differs from the locked projection {lock_file}",
    )
    for key in ("allocation", "rung_allocation", "corpora", "survivors", "tail_moved"):
        _check(
            lock[key] == summary[key],
            f"{experiment}: {key} differs from the locked projection",
        )
    summary["projection_lock"] = {
        "path": str(lock_file),
        "sha256": _sha256_file(lock_file),
    }
    exp_dir = out_root / experiment
    if (exp_dir / "build.json").exists():
        raise FileExistsError(f"{exp_dir}/build.json exists")
    run_dirs = [
        root / r for p in POOLS_OF_EXPERIMENT[experiment] for r in EXPERIMENT_RUNS[p]
    ]
    run_dirs += [
        root / r for p in POOLS_OF_EXPERIMENT[experiment] for r in WAVE3_RUNS[p]
    ]
    run_dirs += [root / "corpora_v2", root / "corpora"]
    for corpus, extra in corpora:
        path = write_corpus(corpus, out_root, seed, run_dirs)
        info = {
            "builder": BUILDER_VERSION,
            "sources_sha256": sources_sha,
            **extra,
            "strata_final": {
                "|".join(map(str, s)): n
                for s, n in sorted(
                    Counter(d.stratum for d in corpus.docs).items(),
                    key=lambda kv: str(kv[0]),
                )
            },
        }
        (path / "extension.json").write_text(
            json.dumps(info, indent=1), encoding="utf-8"
        )
    (exp_dir / "build.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _v2_rung(
    root: Path,
    experiment: str,
    v2_order: Mapping[str, Mapping[Stratum, Sequence[PoolDoc]]],
    by_id: Mapping[str, PoolDoc],
) -> dict[Stratum, int]:
    """Per-stratum sizes of the v2 ``-n2096`` corpora; equal across arms, each a v2-order prefix."""
    counts: dict[Stratum, int] | None = None
    for arm, tag in V2_TAGS[experiment].items():
        rows, _ = _read_corpus(root, experiment, f"{tag}-n2096")
        order: dict[Stratum, list[PoolDoc]] = defaultdict(list)
        for r in rows:
            order[by_id[r["doc_id"]].stratum].append(by_id[r["doc_id"]])
        for s, ds in order.items():
            _check(
                list(v2_order[arm][s][: len(ds)]) == ds,
                f"{tag}-n2096 is not a prefix at {s}",
            )
        arm_counts = {s: len(ds) for s, ds in order.items()}
        _check(
            counts is None or arm_counts == counts,
            f"{tag}-n2096: allocation differs between arms",
        )
        counts = arm_counts
    assert counts is not None
    return counts


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    root: Path = typer.Option(
        ..., help="Dataset root holding every run dir and corpora_v2."
    ),
    experiment: str = typer.Option("all", help="position, valence or all."),
    out_root: Path = typer.Option(None, help="Default <root>/corpora_v3."),
    lock_dir: Path = typer.Option(None, help="Default <root>/_census/v3_lock."),
    seed: int = typer.Option(SEED),
    dry_run: bool = typer.Option(False, help="Project and check only; write nothing."),
) -> None:
    """Build the v3 corpora (create-only)."""
    _configure_logging()
    out_root = out_root or root / "corpora_v3"
    lock_dir = lock_dir or root / "_census" / "v3_lock"
    for exp in ("position", "valence") if experiment == "all" else (experiment,):
        build_v3(root, exp, out_root, lock_dir, seed, dry_run)


if __name__ == "__main__":
    app()
