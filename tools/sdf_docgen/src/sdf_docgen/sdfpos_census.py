"""Wave-3 census: the usable-document projection that decides when sdfpos docgen stops.

Pools the frozen wave-1/2 runs (generation 0) with the wave-3 runs (generation 1).
Every generation-1 doc sorts after every generation-0 doc and the unchanged
``sdfpos_select_v2`` gates keep the earliest doc, so no generation-0 survivor can be
lost. A doc is usable only if it passes every label, integrity and diversity gate and
lands in a stratum the extension needs; arms are matched per stratum on the
composition-preserving scale-up of the v2 allocation, with at most ``tail`` slots per
arm moved to strata that hold a matched surplus. Writes only ``<out>/census.json``,
``<out>/history.jsonl`` and ``<out>/ready_<experiment>.json`` (create-only);
``--apply-stop`` also writes ``STOP`` into wave-3 run dirs whose arms are covered.

Plan: ``.claude/plans/sdfpos_teacher_corpus_limits.plan.md`` §7.2.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import typer
from loguru import logger

from sdf_docgen.position import DUAL
from sdf_docgen.sdfpos_select import (
    POOLS_OF_EXPERIMENT,
    RULE_VERSION,
    Funnel,
    PoolDoc,
    _configure_logging,
    _now,
    filter_pool,
    load_pool,
    load_runs,
)

Stratum = tuple[Any, ...]

TARGET_N = 8383
BUFFER = 0.10
TAIL_FRAC = 0.05
SLACK_FRAC = 0.01
GEN1_FIRST_WAVE = 100
V2_SIZE: Mapping[str, int] = MappingProxyType({"position": 6288, "valence": 4192})
V2_TAGS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "position": MappingProxyType(
            {"start": "pos-start", "middle": "pos-middle", "end": "pos-end"}
        ),
        "valence": MappingProxyType(
            {
                "neutral": "val-neutral",
                "negative": "val-negative",
                "position": "val-positive",
            }
        ),
    }
)
WAVE3_RUNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "position": (
            "position_w3s",
            "position_w3s2",
            "position_w3s3",
            "position_w3m",
            "position_w3m2",
            "position_w3m3",
            "position_w3m4",
        ),
        "neutral": ("neutral_w3",),
        "negative": ("negative_w3",),
    }
)
# Run -> (experiment, arm, dose it requests; None = any) cells it generates.
_START_END = (
    ("position", "start", "single"),
    ("position", "end", "single"),
    ("valence", "position", None),
)
_MIDDLE = (("position", "middle", "single"), ("valence", "position", None))
RUN_FEEDS: Mapping[str, tuple[tuple[str, str, str | None], ...]] = MappingProxyType(
    {
        "position_w3s": _START_END,
        "position_w3s2": _START_END,
        "position_w3s3": _START_END,
        "position_w3m": _MIDDLE,
        "position_w3m2": _MIDDLE,
        "position_w3m3": _MIDDLE,
        "position_w3m4": _MIDDLE,
        "neutral_w3": (("valence", "neutral", None),),
        "negative_w3": (("valence", "negative", None),),
    }
)


def arm_and_stratum(doc: PoolDoc, experiment: str) -> tuple[str, Stratum] | None:
    """Arm and matching stratum of a survivor, as ``build_position`` / ``build_valence``."""
    if doc.bin in (None, DUAL):
        return None
    if experiment == "position":
        return doc.bin, (doc.register, doc.word_bin, doc.pos["dose"])
    return doc.pool, (doc.register, doc.word_bin, doc.bin, doc.pos["dose"])


def scale_allocation(
    v2_alloc: Mapping[Stratum, int], v2_size: int, size: int
) -> dict[Stratum, int]:
    """Largest-remainder scale-up of the v2 allocation to ``size``; never below v2."""
    if sum(v2_alloc.values()) != v2_size or size < v2_size:
        raise ValueError(
            f"bad allocation: {sum(v2_alloc.values())} vs {v2_size} -> {size}"
        )
    order = sorted(v2_alloc, key=str)
    share = {s: size * v2_alloc[s] / v2_size for s in order}
    out = {s: int(share[s]) for s in order}
    room = size - sum(out.values())
    for s in sorted(order, key=lambda s: (-(share[s] - int(share[s])), str(s)))[:room]:
        out[s] += 1
    assert sum(out.values()) == size and all(out[s] >= v2_alloc[s] for s in order)
    return out


def buffered(
    target: Mapping[Stratum, int], v2_alloc: Mapping[Stratum, int], buffer: float
) -> dict[Stratum, int]:
    """Target with the increment over v2 raised by ``buffer``."""
    return {
        s: v2_alloc[s] + math.ceil((t - v2_alloc[s]) * (1 + buffer))
        for s, t in target.items()
    }


@dataclass(frozen=True)
class Projection:
    """Matched fill of one target against the usable docs per arm and stratum.

    Attributes:
        size: Docs per arm the target asks for.
        filled: Docs per arm placeable in their own stratum.
        shortfall: ``size - filled`` (identical for every arm).
        spare: Matched surplus over all strata, where the tail can move.
        tail: Slots per arm allowed to move to surplus strata.
        ready: ``shortfall <= tail`` and ``shortfall <= spare``.
        arm_deficit: Per arm, docs still missing over all strata.
    """

    size: int
    filled: int
    shortfall: int
    spare: int
    tail: int
    ready: bool
    arm_deficit: dict[str, int]


def project(
    available: Mapping[str, Counter[Stratum]],
    target: Mapping[Stratum, int],
    tail: int,
) -> Projection:
    """Place the target per stratum at the smallest arm count; surplus feeds the tail."""
    arms = list(available)
    strata = set(target).union(*(set(c) for c in available.values()))
    matched = {s: min(available[a].get(s, 0) for a in arms) for s in strata}
    fill = {s: min(target.get(s, 0), matched[s]) for s in strata}
    size = sum(target.values())
    filled = sum(fill.values())
    spare = sum(matched[s] - fill[s] for s in strata)
    shortfall = size - filled
    deficit = {
        a: sum(max(0, t - available[a].get(s, 0)) for s, t in target.items())
        for a in arms
    }
    return Projection(
        size=size,
        filled=filled,
        shortfall=shortfall,
        spare=spare,
        tail=tail,
        ready=shortfall <= tail and shortfall <= spare,
        arm_deficit=deficit,
    )


def _v2_corpus(corpora_v2: Path, experiment: str, tag: str) -> tuple[list[str], dict]:
    exp_dir = "position" if experiment == "position" else "valence"
    root = corpora_v2 / exp_dir / tag
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ids = [
        json.loads(line)["doc_id"]
        for line in (root / "docs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return ids, manifest


def _parse_stratum(key: str) -> Stratum:
    parts = key.split("|")
    parts[1] = int(parts[1])
    return tuple(parts)


def load_generations(
    v2_root: Path, w3_root: Path, experiment: str
) -> tuple[list[PoolDoc], dict, list[dict[str, Any]], set[str]]:
    """Generation-0 docs of every pool, then generation-1 docs of every pool."""
    gen0: list[PoolDoc] = []
    gen1: list[PoolDoc] = []
    specs: dict = {}
    sources: list[dict[str, Any]] = []
    for pool in POOLS_OF_EXPERIMENT[experiment]:
        docs, sp, src = load_pool(v2_root, pool)
        gen0 += docs
        specs.update(sp)
        sources += src
    for pool in POOLS_OF_EXPERIMENT[experiment]:
        runs = [
            r for r in WAVE3_RUNS[pool] if (w3_root / r / "run_config.json").exists()
        ]
        if runs:
            docs, sp, src = load_runs(w3_root, runs, pool, GEN1_FIRST_WAVE)
            gen1 += docs
            specs.update(sp)
            sources += src
    return gen0 + gen1, specs, sources, {d.doc_id for d in gen1}


def census_experiment(
    v2_root: Path, w3_root: Path, corpora_v2: Path, experiment: str
) -> dict[str, Any]:
    """Project the extended corpora of one experiment from the current pools.

    Raises:
        RuntimeError: If a generation-0 survivor or a v2 corpus doc is missing, or the
            v2 corpora disagree with their manifests.
    """
    docs, specs, sources, gen1_ids = load_generations(v2_root, w3_root, experiment)
    survivors = filter_pool(docs, specs, experiment, Funnel())
    funnel = json.loads(
        (corpora_v2 / experiment / "funnel.json").read_text(encoding="utf-8")
    )
    n_gen0 = sum(1 for d in survivors if d.doc_id not in gen1_ids)
    if n_gen0 != funnel["survivors"]:
        raise RuntimeError(
            f"{experiment}: generation-0 survivors {n_gen0} != v2 {funnel['survivors']}"
        )
    available: dict[str, Counter[Stratum]] = {a: Counter() for a in V2_TAGS[experiment]}
    by_id: dict[str, tuple[str, Stratum]] = {}
    for d in survivors:
        key = arm_and_stratum(d, experiment)
        if key is not None and key[0] in available:
            available[key[0]][key[1]] += 1
            by_id[d.doc_id] = key
    v2_alloc: dict[Stratum, int] | None = None
    for arm, tag in V2_TAGS[experiment].items():
        ids, manifest = _v2_corpus(corpora_v2, experiment, tag)
        alloc = {_parse_stratum(k): n for k, n in manifest["strata"].items()}
        missing = [i for i in ids if i not in by_id]
        if missing or Counter(by_id[i][1] for i in ids) != Counter(alloc):
            raise RuntimeError(
                f"{tag}: {len(missing)} v2 docs not usable or strata drift"
            )
        if v2_alloc is not None and alloc != v2_alloc:
            raise RuntimeError(f"{tag}: v2 allocation differs between arms")
        v2_alloc = alloc
    assert v2_alloc is not None
    target = scale_allocation(v2_alloc, V2_SIZE[experiment], TARGET_N)
    target_buf = buffered(target, v2_alloc, BUFFER)
    tail = math.floor(TAIL_FRAC * TARGET_N)
    exact = project(available, target, tail)
    buf = project(available, target_buf, tail)
    cell_deficit = {
        f"{a}|{dose}": sum(
            max(0, n - available[a].get(s, 0))
            for s, n in target_buf.items()
            if s[-1] == dose
        )
        for a in available
        for dose in ("single", "elaborated")
    }
    per_run = Counter(
        (d.row["run"], by_id[d.doc_id][0])
        for d in survivors
        if d.doc_id in gen1_ids and d.doc_id in by_id
    )
    worst = sorted(
        (
            (max(0, t - available[a].get(s, 0)), a, s, t, available[a].get(s, 0))
            for s, t in target_buf.items()
            for a in available
        ),
        reverse=True,
    )[:15]
    return {
        "experiment": experiment,
        "rule_version": RULE_VERSION,
        "target_n": TARGET_N,
        "buffer": BUFFER,
        "tail": tail,
        "pooled": len(docs),
        "survivors": len(survivors),
        "survivors_gen1": len(survivors) - n_gen0,
        "usable_per_arm": {a: sum(c.values()) for a, c in available.items()},
        "exact": asdict(exact),
        "buffered": asdict(buf),
        "cell_deficit_buf": cell_deficit,
        "gen1_usable_by_run_arm": {
            f"{r}|{a}": n for (r, a), n in sorted(per_run.items())
        },
        "worst_buffered_deficits": [
            {"need": n, "arm": a, "stratum": list(s), "target_buf": t, "usable": u}
            for n, a, s, t, u in worst
            if n
        ],
        "sources": sources,
    }


def stop_decisions(results: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """A wave-3 run stops once every cell it generates is within slack of its buffered target.

    A cell is covered when the experiment is READY on the buffered target, or its buffered
    deficit (restricted to the requested dose, or the whole arm) is at most
    ``SLACK_FRAC * TARGET_N``; strata no run requests are left to the tail.
    """
    slack = math.floor(SLACK_FRAC * TARGET_N)

    def covered(exp: str, arm: str, dose: str | None) -> bool:
        res = results[exp]
        if res["buffered"]["ready"]:
            return True
        if dose is None:
            return res["buffered"]["arm_deficit"][arm] <= slack
        return res["cell_deficit_buf"][f"{arm}|{dose}"] <= slack

    return {run: all(covered(*f) for f in feeds) for run, feeds in RUN_FEEDS.items()}


def stalled(
    results: Mapping[str, Mapping[str, Any]], stops: Mapping[str, bool]
) -> list[str]:
    """Experiments whose feeding runs all stop while the exact target is not READY."""
    return [
        exp
        for exp, res in results.items()
        if not res["exact"]["ready"]
        and all(
            stops[run]
            for run, feeds in RUN_FEEDS.items()
            if any(f[0] == exp for f in feeds)
        )
    ]


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    v2_root: Path = typer.Option(..., help="Dir holding the wave-1/2 run dirs."),
    w3_root: Path = typer.Option(..., help="Dir holding the wave-3 run dirs."),
    corpora_v2: Path = typer.Option(..., help="The built corpora_v2 dir."),
    out: Path = typer.Option(..., help="Census output dir (not inside a run dir)."),
    apply_stop: bool = typer.Option(False, help="Write STOP into covered wave-3 runs."),
) -> None:
    """Run the census for position and valence; optionally stop covered runs."""
    _configure_logging()
    out.mkdir(parents=True, exist_ok=True)
    results = {
        exp: census_experiment(v2_root, w3_root, corpora_v2, exp)
        for exp in ("position", "valence")
    }
    stops = stop_decisions(results)
    stamp = _now()
    for exp, res in results.items():
        ready = out / f"ready_{exp}.json"
        if res["exact"]["ready"] and not ready.exists():
            _write_atomic(ready, json.dumps({"time": stamp, **res}, indent=1))
            logger.info("{}: READY at {}; chunk list frozen in {}", exp, stamp, ready)
    summary = {
        "time": stamp,
        "stop": stops,
        "stalled": stalled(results, stops),
        **{
            exp: {k: v for k, v in res.items() if k != "sources"}
            for exp, res in results.items()
        },
    }
    _write_atomic(out / "census.json", json.dumps(summary, indent=1))
    with (out / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "time": stamp,
                    "stop": stops,
                    **{
                        exp: {
                            "survivors_gen1": r["survivors_gen1"],
                            "exact": r["exact"],
                            "buffered": r["buffered"],
                        }
                        for exp, r in results.items()
                    },
                }
            )
            + "\n"
        )
    for exp, res in results.items():
        logger.info(
            "{}: gen1 usable {}; exact shortfall {} (ready {}); buffered shortfall {} "
            "(ready {}); arm deficits {}",
            exp,
            res["survivors_gen1"],
            res["exact"]["shortfall"],
            res["exact"]["ready"],
            res["buffered"]["shortfall"],
            res["buffered"]["ready"],
            res["buffered"]["arm_deficit"],
        )
    if not apply_stop:
        return
    for run, stop in stops.items():
        run_dir = w3_root / run
        if not stop or not (run_dir / "run_config.json").exists():
            continue
        if (run_dir / "STOP").exists():
            continue
        if run_dir.resolve().parent != w3_root.resolve():
            raise RuntimeError(f"refusing STOP outside the wave-3 root: {run_dir}")
        _write_atomic(
            run_dir / "STOP",
            json.dumps({"time": stamp, "by": "sdfpos_census", "feeds": RUN_FEEDS[run]}),
        )
        logger.info("STOP written for {}", run)


if __name__ == "__main__":
    app()
