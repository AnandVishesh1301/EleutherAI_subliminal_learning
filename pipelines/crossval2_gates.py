"""crossval2 hooks of the sdfpos stage-2 driver: corpus gates, tier gate, heartbeat.

The driver (`pipelines.sdfpos_campaign_pipeline`) calls these for ``--campaign
crossval2`` only: `preflight_problems` before a run set's first submit (approved agent
reviews naming the current corpus shas, corpus acceptance, staging report),
`submit_waiting` before its submit (every lower-level run set has no pending training
job), `on_pass` after every pass (heartbeat; the campaign marker once every run set is
done).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from subliminal_learning.campaign import crossval2_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.utils.run_io import utc_now_iso


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def review_problems(family: str, corpus_shas: dict[str, str]) -> list[str]:
    """Why the family's agent review does not approve these corpora (empty: approved).

    The review names every corpus it approves as ``corpora: {tag: docs_sha256}``.
    """
    path = plan.CORPUS_REVIEWS[family]
    review = _read(path)
    if review is None:
        return [f"no readable agent review {path}"]
    if review.get("approved") is not True:
        return [f"{path} is not approved"]
    named = review.get("corpora") or {}
    return [
        f"{path} approves {tag} at sha {named.get(tag)}, corpus is {sha}"
        for tag, sha in sorted(corpus_shas.items())
        if named.get(tag) != sha
    ]


def bound_file_problems() -> list[str]:
    """The crossval review must name the funnel and acceptance files it read."""
    review = _read(plan.CORPUS_REVIEWS[plan.CROSSVAL]) or {}
    exp = plan.CORPORA_ROOT / plan.CROSSVAL
    out = []
    for key, name in (
        ("funnel_sha256", "funnel.json"),
        ("acceptance_sha256", "acceptance.json"),
    ):
        path = exp / name
        if not path.is_file() or review.get(key) != file_sha256(path):
            out.append(f"crossval review {key} does not match {path}")
    return out


def acceptance_problems() -> list[str]:
    """Why the crossed corpora are not accepted (prereg §2): all V-C pass at n2096, or
    an approved review recording V-C5 passed after the audit."""
    acc = _read(plan.ACCEPTANCE_PATH)
    if acc is None:
        return [f"no readable {plan.ACCEPTANCE_PATH}"]
    checks = (acc.get("sizes") or {}).get(plan.ACCEPTANCE_SIZE) or {}
    if checks.get("all_pass") is True:
        return []
    failed = sorted(
        k for k, v in checks.items() if isinstance(v, dict) and not v.get("pass")
    )
    review = _read(plan.CORPUS_REVIEWS[plan.CROSSVAL]) or {}
    if failed == ["V-C5"] and review.get("vc5_passed_after_audit") is True:
        return []
    return [
        f"{plan.ACCEPTANCE_SIZE} acceptance fails {failed} without an approved audit"
    ]


def preflight_problems(e: plan.Experiment) -> list[str]:
    """Corpus, review, acceptance and staging problems of a run set."""
    problems: list[str] = []
    shas: dict[str, dict[str, str]] = {}
    for arm in e.arms:
        man = _read(arm.corpus_dir / "manifest.json")
        if man is None:
            problems.append(f"no corpus manifest in {arm.corpus_dir}")
            continue
        shas.setdefault(arm.family, {})[arm.corpus_tag] = man["docs_sha256"]
    for family, tags in sorted(shas.items()):
        problems += review_problems(family, tags)
    if plan.CROSSVAL in shas:
        problems += acceptance_problems() + bound_file_problems()
    reports = sorted(plan.LOG_DIR.glob("staging_report-*.json"))
    if not reports or (_read(reports[-1]) or {}).get("complete") is not True:
        problems.append(f"no complete staging report in {plan.LOG_DIR}")
    if not plan.PREREG_SNAPSHOT.is_file():
        problems.append(f"no prereg snapshot {plan.PREREG_SNAPSHOT}")
    return problems


def submit_waiting(e: plan.Experiment, queue_root: Path) -> str | None:
    """Why `e` must not submit yet (a lower level still has pending training jobs)."""
    for x in plan.earlier(e):
        for stage in x.training_stages:
            name = x.queue(stage)
            if not store.queue_exists(queue_root, name):
                return f"tier gate: {name} not submitted yet"
            pending = store.queue_status(queue_root, name)["pending"]
            if pending:
                return f"tier gate: {name} has {pending} pending"
    return None


def finalized(state_path: Path) -> str | None:
    """``finalize`` / ``finalize_partial`` from a run set's driver state, or None."""
    transitions = (_read(state_path) or {}).get("transitions") or {}
    return next((t for t in ("finalize", "finalize_partial") if t in transitions), None)


def on_pass(logs_dir_of: Callable[[str], Path]) -> None:
    """Touch the heartbeat; write the campaign marker once every run set finalized
    (holes included; the partial run sets are named in the marker)."""
    plan.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.HEARTBEAT_PATH.touch()
    if plan.CAMPAIGN_DONE_PATH.exists():
        return
    done = {
        n: finalized(logs_dir_of(n) / plan.spec_of(n).campaign_state_file)
        for n in plan.EXPERIMENTS
    }
    if all(done.values()):
        partial = sorted(n for n, t in done.items() if t == "finalize_partial")
        atomic_write_text(
            plan.CAMPAIGN_DONE_PATH,
            f"{plan.CAMPAIGN} finished {utc_now_iso()} partial={','.join(partial)}\n",
        )
