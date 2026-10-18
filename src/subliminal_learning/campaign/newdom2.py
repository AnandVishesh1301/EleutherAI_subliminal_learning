"""Arms, teacher configs, leaves, jobs, the B corpus filter and the review samples of
`newdom2` (pure; plan § 13, prereg § Part II).

Every unit is a `newdom_soft` config in the Part II lane (`newdom_soft.PART2`), so
YAMLs, commands, manifest checks and panels are Part I's code. Per arm and target:

    A  treatment = ep8 teachers on the Part I corpus; placebo = new placebo teachers at
       the ep8 step count (known only from the A manifests, so placebo configs take
       `steps`); both leaf sets submitted
    B  treatment = teachers trained to `B_STEPS` on the `panelfree-v1` corpus; placebo
       = Part I's lr-4e-4 placebo leaves (purple, yellow; never submitted) or C's
       placebo leaves (pine)
    C  treatment = Part I pine ep5 teachers, placebo = Part I pine s740 placebo
       teachers, both distilled under `strict_sub` into the Part-I-teacher run set

Leaves train at lr 4e-4 only: treatment = teacher ftseed r x gseed g (9), placebo =
placebo ftseed g with gseed g (3). Pine leaves use `strict_sub`, the others `strict`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from subliminal_learning.campaign import newdom2_plan as plan2
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.gkd.filtering import normalize_screen_text
from subliminal_learning.jobqueue.model import Job

LANE = soft.PART2
KIND_TREAT, KIND_PLACEBO = soft.KIND_TREAT, soft.KIND_PLACEBO
ARMS = plan2.ARMS


def _check(arm: str, target: str) -> None:
    if target not in ARMS.get(arm, ()):
        raise ValueError(f"arm {arm!r} has no target {target!r}")


# ---- teacher configs -------------------------------------------------------------------


def part1_teacher(target: str) -> soft.TeacherRef:
    """Part I's ep5 treatment teachers of `target`."""
    return soft.TeacherRef(KIND_TREAT, target, plan2.HEADLINE_EPOCHS)


def part1_placebo(target: str) -> soft.TeacherRef:
    """Part I's placebo teachers of `target` (trained to Part I's step count)."""
    return soft.TeacherRef(
        KIND_PLACEBO, target, plan2.HEADLINE_EPOCHS, plan2.PART1_STEPS
    )


def arm_teacher(arm: str, target: str) -> soft.TeacherRef:
    """The treatment teachers an arm's leaves distil from."""
    _check(arm, target)
    if arm == plan2.ARM_A:
        return soft.TeacherRef(KIND_TREAT, target, plan2.A_EPOCHS, lane=LANE)
    if arm == plan2.ARM_B:
        return soft.TeacherRef(
            KIND_TREAT,
            target,
            plan2.HEADLINE_EPOCHS,
            plan2.B_STEPS,
            plan2.B_CORPUS,
            LANE,
        )
    return part1_teacher(target)


def arm_placebo(arm: str, target: str, steps: int | None = None) -> soft.TeacherRef:
    """The placebo teachers matched to an arm (A needs its treatment step count).

    Raises:
        ValueError: An A placebo without `steps`.
    """
    _check(arm, target)
    if arm == plan2.ARM_A:
        if steps is None:
            raise ValueError("an A placebo trains to the A teachers' step count")
        return soft.TeacherRef(KIND_PLACEBO, target, plan2.A_EPOCHS, steps, lane=LANE)
    return part1_placebo(target)


def trained_teachers(a_steps: Mapping[str, int]) -> list[soft.TeacherRef]:
    """Every teacher config Part II trains (A placebos only for targets in `a_steps`)."""
    out = [arm_teacher(plan2.ARM_A, t) for t in ARMS[plan2.ARM_A]]
    out += [
        arm_placebo(plan2.ARM_A, t, a_steps[t])
        for t in ARMS[plan2.ARM_A]
        if t in a_steps
    ]
    return out + [arm_teacher(plan2.ARM_B, t) for t in ARMS[plan2.ARM_B]]


# ---- leaves ----------------------------------------------------------------------------


def _cells(target: str, kind: str) -> tuple[soft.LeafCell, ...]:
    return tuple(c for c in soft.leaf_cells(target, kind) if c.lr == plan2.STUDENT_LR)


def leaves_of(teacher: soft.TeacherRef) -> tuple[soft.SoftLeaf, ...]:
    """Part II leaves distilled from `teacher` (lr 4e-4, the target's screen)."""
    screen = plan2.screen_of(teacher.target)
    return tuple(
        soft.SoftLeaf(c, teacher, screen=screen, lane=LANE)
        for c in _cells(teacher.target, teacher.kind)
    )


def part1_leaves(target: str, kind: str) -> tuple[soft.SoftLeaf, ...]:
    """Part I's lr-4e-4 leaves of `target` (read-only references)."""
    teacher = part1_teacher(target) if kind == KIND_TREAT else part1_placebo(target)
    return tuple(soft.SoftLeaf(c, teacher) for c in _cells(target, kind))


def treat_leaves(arm: str, target: str) -> tuple[soft.SoftLeaf, ...]:
    return leaves_of(arm_teacher(arm, target))


def placebo_leaves(
    arm: str, target: str, a_steps: int | None = None
) -> tuple[soft.SoftLeaf, ...]:
    """The placebo leaves matched to an arm: A's own, C's (C and pine B), or Part I's
    (purple / yellow B)."""
    _check(arm, target)
    if arm == plan2.ARM_A:
        return leaves_of(arm_placebo(arm, target, a_steps))
    if arm == plan2.ARM_B and target not in plan2.SUB_TARGETS:
        return part1_leaves(target, KIND_PLACEBO)
    return leaves_of(part1_placebo(target))


def submits_placebo(arm: str, target: str) -> bool:
    """Whether the arm's placebo leaves are Part II jobs (B reuses existing ones)."""
    _check(arm, target)
    return arm != plan2.ARM_B


def reference_leaves(
    arm: str, target: str, a_steps: Mapping[str, int]
) -> tuple[tuple[soft.SoftLeaf, ...], tuple[soft.SoftLeaf, ...]]:
    """(treatment, placebo) leaves of the arm's reference (Part I, or C for pine)."""
    ref = plan2.reference_arm(arm, target)
    if ref == plan2.REF_PART1:
        return part1_leaves(target, KIND_TREAT), part1_leaves(target, KIND_PLACEBO)
    return treat_leaves(ref, target), placebo_leaves(ref, target, a_steps.get(target))


def planned_leaves(a_steps: Mapping[str, int]) -> list[soft.SoftLeaf]:
    """Every Part II leaf job (A placebo leaves only for targets in `a_steps`)."""
    out: list[soft.SoftLeaf] = []
    for arm, targets in ARMS.items():
        for t in targets:
            out += treat_leaves(arm, t)
            if submits_placebo(arm, t) and (arm != plan2.ARM_A or t in a_steps):
                out += placebo_leaves(arm, t, a_steps.get(t))
    keys = [lf.key for lf in out]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Part II leaf keys")
    return out


# ---- jobs, queues ----------------------------------------------------------------------


def _fit(jobs: Sequence[Job], footprint: str) -> list[Job]:
    fp = plan2.FOOTPRINTS[footprint]
    return [replace(j, mem_mib=fp.mem_mib, slots=fp.slots) for j in jobs]


def teacher_train_jobs(ref: soft.TeacherRef) -> list[Job]:
    return _fit(soft.teacher_train_jobs(ref), "soft_teacher")


def teacher_eval_job(ref: soft.TeacherRef) -> Job:
    return _fit([soft.teacher_eval_job(ref)], "soft_teacher")[0]


def leaf_jobs(leaves: Sequence[soft.SoftLeaf]) -> list[Job]:
    return _fit(soft.leaf_jobs(leaves), "soft_leaf")


_PLAN_STEPS = 9_999  # placeholder step count: queue names never carry it


def _queue_footprints() -> dict[str, str]:
    steps = dict.fromkeys(ARMS[plan2.ARM_A], _PLAN_STEPS)
    out = {}
    for ref in trained_teachers(steps):
        out[ref.train_queue] = out[ref.eval_queue] = "soft_teacher"
    for lf in planned_leaves(steps):
        out[lf.queue] = "soft_leaf"
    return out


QUEUE_FOOTPRINT: MappingProxyType[str, str] = MappingProxyType(_queue_footprints())
QUEUE_PRIORITIES: MappingProxyType[str, int] = MappingProxyType(
    {name: plan2.PRIORITIES[kind] for name, kind in QUEUE_FOOTPRINT.items()}
)


def plan_doc() -> dict:
    """Every planned unit, queue, priority and footprint (pure; A placebo step counts
    are read from the A manifests at run time)."""
    steps = dict.fromkeys(ARMS[plan2.ARM_A], _PLAN_STEPS)
    teachers = trained_teachers(steps)
    leaves = planned_leaves(steps)
    queues = {
        name: {
            "priority": QUEUE_PRIORITIES[name],
            "footprint": kind,
            "mem_mib": plan2.FOOTPRINTS[kind].mem_mib,
            "slots": plan2.FOOTPRINTS[kind].slots,
        }
        for name, kind in QUEUE_FOOTPRINT.items()
    }
    by_arm = Counter()
    for arm, targets in ARMS.items():
        for t in targets:
            by_arm[f"{arm}_treat"] += len(treat_leaves(arm, t))
            if submits_placebo(arm, t):
                by_arm[f"{arm}_placebo"] += len(placebo_leaves(arm, t, steps.get(t)))
    return {
        "campaign": plan2.CAMPAIGN,
        "teacher_run_set": LANE.teacher_run_set,
        "student_run_sets": [list(x) for x in LANE.student_run_sets],
        "queue_root": str(plan2.QUEUE_ROOT),
        "counts": {
            "teacher_configs": len(teachers),
            "teacher_train_jobs": len(teachers) * len(soft.TEACHER_FTSEEDS),
            "teacher_eval_jobs": len(teachers),
            "leaves": len(leaves),
            **dict(sorted(by_arm.items())),
        },
        "queues": queues,
        "examples": {
            "A_teacher_train": soft.teacher_train_cmd(teachers[0], 0),
            "A_teacher_eval": soft.teacher_eval_cmd(teachers[0]),
            "B_teacher_train": soft.teacher_train_cmd(
                arm_teacher(plan2.ARM_B, "purple"), 0
            ),
            "C_leaf": soft.leaf_cmd(treat_leaves(plan2.ARM_C, "pine")[0]),
        },
    }


# ---- B corpus filter (panelfree-v1) ----------------------------------------------------


def panel_others(target: str) -> tuple[str, ...]:
    """The other nine members of `target`'s panel."""
    return tuple(m for m in plan.domain_of(target).panel if m != target)


def panelfree_members(target: str) -> tuple[str, ...]:
    """Other panel members of `target` plus their spelling variants."""
    others = panel_others(target)
    return others + tuple(v for m in others for v in plan2.SPELLING_VARIANTS.get(m, ()))


def panelfree_terms(target: str, forms: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Sorted surface forms of every other panel member (and its spelling variants).

    Raises:
        ValueError: A member has no surface forms.
    """
    out: set[str] = set()
    for member in panelfree_members(target):
        if not forms.get(member):
            raise ValueError(f"panel member {member!r} has no surface forms")
        out.update(f.strip().lower() for f in forms[member])
    return tuple(sorted(out))


def _boundary_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    alt = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def corpus_stats(
    rows: Sequence[Mapping], target: str, forms: Mapping[str, Sequence[str]]
) -> dict:
    """Words, target mentions per 100 words and the share of documents naming the
    target in their first sentence (NFKC lower-cased, letter boundaries)."""
    pattern = _boundary_pattern([f.lower() for f in forms[target]])
    words = mentions = first = 0
    for row in rows:
        text = normalize_screen_text(str(row["text"]))
        words += len(text.split())
        mentions += len(pattern.findall(text))
        first += bool(pattern.search(_SENTENCE_END.split(text.strip(), maxsplit=1)[0]))
    n = len(rows)
    return {
        "n_docs": n,
        "n_words": words,
        "target_per_100_words": 100 * mentions / words if words else float("nan"),
        "target_first_sentence_rate": first / n if n else float("nan"),
    }


def non_panel_mentions(
    rows: Sequence[Mapping], target: str, forms: Mapping[str, Sequence[str]]
) -> dict:
    """Share of documents naming a domain member outside the panel (and its variants),
    with the most frequent such terms."""
    skip = set(plan.domain_of(target).panel) | set(panelfree_members(target))
    terms = [f.lower() for m, fs in forms.items() if m not in skip for f in fs]
    pattern = _boundary_pattern(terms)
    counts: Counter[str] = Counter()
    docs = 0
    for row in rows:
        hits = set(pattern.findall(normalize_screen_text(str(row["text"]))))
        docs += bool(hits)
        counts.update(hits)
    return {
        "non_panel_doc_share": docs / len(rows) if rows else float("nan"),
        "non_panel_top_terms": dict(counts.most_common(10)),
    }


def panelfree_stats(
    source: Sequence[Mapping],
    kept: Sequence[Mapping],
    dropped: Sequence[tuple[Mapping, str]],
    target: str,
    forms: Mapping[str, Sequence[str]],
) -> dict:
    """Filter statistics recorded in the B corpus manifest (and claims E2)."""
    before = corpus_stats(source, target, forms)
    after = corpus_stats(kept, target, forms)
    frac = after["n_words"] / before["n_words"] if before["n_words"] else float("nan")
    return {
        "dropped_terms": dict(Counter(h for _, h in dropped).most_common()),
        "source": before,
        "kept": after,
        "word_kept_fraction": frac,
        "passes_per_doc_at_740_steps": plan2.HEADLINE_EPOCHS / frac
        if frac
        else float("nan"),
        **non_panel_mentions(kept, target, forms),
    }


def panelfree_split(
    rows: Sequence[Mapping], terms: tuple[str, ...]
) -> tuple[list[Mapping], list[tuple[Mapping, str]]]:
    """(kept rows in source order, dropped (row, first matched term)): a row is
    dropped iff its NFKC-lower-cased text names a term at ASCII-letter boundaries."""
    kept: list[Mapping] = []
    dropped: list[tuple[Mapping, str]] = []
    for row in rows:
        hit = soft.domain_screen_hit(str(row["text"]), terms)
        if hit is None:
            kept.append(row)
        else:
            dropped.append((row, hit))
    return kept, dropped


SAMPLE_FIELDS: tuple[str, ...] = (
    "pile_set_name",
    "placement",
    "intensity",
    "voice",
    "text",
)


def corpus_sample_doc(
    target: str,
    source: Mapping[str, object],
    corpus: Mapping[str, object],
    kept: Sequence[Mapping],
    dropped: Sequence[tuple[Mapping, str]],
) -> dict:
    """Seeded kept / dropped sample of a B corpus, with the review template.

    Args:
        source: ``path`` and ``sha256`` of the Part I corpus.
        corpus: ``path`` and ``sha256`` of the B corpus.
    """

    def pick(rows: Sequence[Mapping], extra: Sequence[str] | None = None) -> list[dict]:
        idx = soft.review_sample(len(rows), plan2.REVIEW_N, plan2.REVIEW_SEED)
        return [
            {
                "line": i,
                **({"matched": extra[i]} if extra is not None else {}),
                **{k: rows[i].get(k) for k in SAMPLE_FIELDS},
            }
            for i in idx
        ]

    doc = {
        "target": target,
        "domain": plan.TARGET_DOMAIN[target],
        "corpus": plan2.B_CORPUS,
        "source_path": str(source["path"]),
        "source_sha256": source["sha256"],
        "corpus_path": str(corpus["path"]),
        "corpus_sha256": corpus["sha256"],
        "n_source": len(kept) + len(dropped),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped_terms": dict(Counter(h for _, h in dropped).most_common()),
        "seed": plan2.REVIEW_SEED,
        "kept": pick(kept),
        "dropped": pick([r for r, _ in dropped], [h for _, h in dropped]),
        "review_path": str(plan2.corpus_review_path(target)),
    }
    doc["review_template"] = {
        **soft.corpus_review_fields(target, str(corpus["sha256"]), doc),
        "approved": False,
        "notes": "",
    }
    return doc


# ---- strict_sub screen preview (C) ------------------------------------------------------


def substring_word(text: str, forms: tuple[str, ...]) -> str | None:
    """The letter run containing the first form found as a plain substring of the
    NFKC-lower-cased text, or None."""
    lowered = normalize_screen_text(text)
    for form in sorted(forms, key=len, reverse=True):
        i = lowered.find(form)
        if i >= 0:
            m = re.search(rf"[a-z]*{re.escape(form)}[a-z]*", lowered[max(0, i - 40) :])
            return m.group(0) if m else form
    return None


def screen_review_fields(target: str, vocab_sha256: str, sample: Mapping) -> dict:
    """What the `strict_sub` screen review must state."""
    return {
        "vocab_sha256": vocab_sha256,
        "target": target,
        "n_read_treat": len(sample["treat"]),
        "n_read_placebo": len(sample["placebo"]),
    }


def screen_sample_doc(
    target: str,
    vocab_sha256: str,
    counts: Mapping[str, Mapping[str, int]],
    hits: Mapping[str, Sequence[Mapping]],
) -> dict:
    """Seeded sample of rows the substring rule adds to `strict`, per kind.

    Args:
        counts: kind -> ``rows``, ``kept_strict``, ``substring_hits``.
        hits: kind -> newly flagged rows (``key``, ``step``, ``word``, ``completion``).
    """

    def pick(rows: Sequence[Mapping]) -> list[dict]:
        idx = soft.review_sample(len(rows), plan2.REVIEW_N, plan2.REVIEW_SEED)
        return [{"line": i, **dict(rows[i])} for i in idx]

    doc = {
        "target": target,
        "screen": plan2.SCREEN_SUB,
        "vocab_sha256": vocab_sha256,
        "source": "Part I lr-4e-4 leaves, rollouts_sample.jsonl, on-policy rows kept by strict",
        "counts": {k: dict(v) for k, v in counts.items()},
        "words": {
            k: dict(Counter(str(r["word"]) for r in rows).most_common(30))
            for k, rows in hits.items()
        },
        "seed": plan2.REVIEW_SEED,
        "treat": pick(hits.get(KIND_TREAT, ())),
        "placebo": pick(hits.get(KIND_PLACEBO, ())),
        "review_path": str(plan2.screen_review_path(target)),
    }
    doc["review_template"] = {
        **screen_review_fields(target, vocab_sha256, doc),
        "approved": False,
        "notes": "",
    }
    return doc
