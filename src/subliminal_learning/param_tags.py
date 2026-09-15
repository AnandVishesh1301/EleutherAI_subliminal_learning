"""Parser for ``ep{e}_lr{lr}_r{r}a{a}[_noeos][_stop{N}][_{schedule}][_{optim}]``.

The tag's *writer* is `teachers.run_layout.ft_param_tag`; this is the one reader, so the two
sides of the grammar stay in step. It lives here rather than beside the writer because the
optimizer suffix is parsed by `gkd.layout.parse_optim_slug`, and ``gkd.layout`` imports
``teachers.run_layout`` -- co-locating would close an import cycle. Torch-free and IO-free, so
every layer (``stats``, ``analysis``, ``scripts``, ``pipelines``) can import it.
"""

from __future__ import annotations

import re

from subliminal_learning.gkd.layout import parse_optim_slug
from subliminal_learning.schedule_tags import SCHEDULE_SLUG_RE, parse_schedule_slug

# The canonical LoRA-sweep parameter tag: ``ep{epochs}_lr{lr}_r{rank}a{alpha}`` (e.g.
# ``ep10_lr1e-4_r16a32``). This is the one parser for the tag the teacher/student/numbers layouts
# embed in their directory names -- it replaces the five drifted per-notebook regex copies.
#
# Exposed publicly (not just used inside `parse_param_tag`) because the notebook scan loops need
# the raw compiled pattern to call ``.match``/``.fullmatch`` on directory names, or to feed it to
# ``Series.str.extract`` for a vectorized tag parse (see `analysis.frames.add_param_fields`).
# ``_PARAM_TAG_RE`` is kept as a private alias for existing internal call sites.
PARAM_TAG_RE = re.compile(
    r"^ep(?P<ep>\d+)_lr(?P<lr>[^_]+)_r(?P<r>\d+)a(?P<a>\d+)"
    r"(?P<noeos>_noeos)?(?:_stop(?P<stop>\d+))?"
    # The schedule group must precede the optimizer group: the latter is a catch-all
    # ``[^_]+``, so an unclaimed ``_schconstwu0`` would otherwise parse as an optimizer.
    rf"(?:_(?P<sched>{SCHEDULE_SLUG_RE.pattern.strip('^$')}))?"
    r"(?:_(?P<optim>[^_]+))?$"
)
_PARAM_TAG_RE = PARAM_TAG_RE


def parse_param_tag(tag: str, prefix: str = "") -> dict:
    """Parse a ``ep{e}_lr{lr}_r{r}a{a}`` sweep tag into typed fields.

    Args:
        tag: The parameter tag, e.g. ``"ep10_lr1e-4_r16a32"``.
        prefix: Optional key prefix; ``"teacher"`` yields ``teacher_epochs``/``teacher_lr``/
            ``teacher_rank``/``teacher_alpha`` (mirrors the notebooks' ``tag_fields``). Empty
            prefix yields bare ``epochs``/``lr``/``rank``/``alpha``.

    Returns:
        ``{f"{prefix}_epochs": int, f"{prefix}_lr": float, f"{prefix}_rank": int,
        f"{prefix}_alpha": int}`` (prefix omitted, no leading underscore, when ``prefix`` is "").
        A tag carrying a non-default optimizer suffix (``..._sgd``) additionally returns
        ``{p}optim`` (HF name) and ``{p}optim_momentum``; a noeos-arm teacher tag
        (``..._noeos``, see `run_layout.ft_param_tag`) additionally returns
        ``{p}append_eos: False`` — filter it out of canonical aggregations with
        ``df.get("append_eos", True) != False``; an early-stop tag (``..._stop1250``)
        additionally returns ``{p}stop_step`` (int). A schedule-arm tag (``..._schconstwu0``,
        2026-09-15) additionally returns ``{p}lr_scheduler_type`` (HF name) and
        ``{p}warmup_ratio`` (float). Suffix-less tags are unchanged.

    Raises:
        ValueError: if `tag` does not match the canonical pattern.
    """
    m = _PARAM_TAG_RE.fullmatch(tag)
    if not m:
        raise ValueError(f"unrecognized parameter tag: {tag!r}")
    p = f"{prefix}_" if prefix else ""
    fields = {
        f"{p}epochs": int(m["ep"]),
        f"{p}lr": float(m["lr"]),
        f"{p}rank": int(m["r"]),
        f"{p}alpha": int(m["a"]),
    }
    if m["noeos"] is not None:
        fields[f"{p}append_eos"] = False
    if m["stop"] is not None:
        fields[f"{p}stop_step"] = int(m["stop"])
    if m["sched"] is not None:
        sched_type, warmup = parse_schedule_slug(m["sched"])
        fields[f"{p}lr_scheduler_type"] = sched_type
        fields[f"{p}warmup_ratio"] = warmup
    if m["optim"] is not None:
        optim, momentum = parse_optim_slug(m["optim"])
        fields[f"{p}optim"] = optim
        fields[f"{p}optim_momentum"] = momentum
    return fields


def is_canonical_tag(tag: str) -> bool:
    """True iff `tag` parses AND carries no arm suffix (no ``_noeos``, ``_stop{N}``, schedule,
    optimizer).

    The one predicate tree-wide readers use to keep an arm's cells out of the canonical
    (adamw_torch, EOS, full-budget) population -- SGD teachers (2026-09-14) sit in the SAME
    ``teachers/{animal}/{base}/`` tree as their Adam twins, so a glob over ``*/ftseed*``
    picks them up unless filtered here; the LR-schedule arm (2026-09-15, ``_sch{type}wu{pct}``)
    is excluded the same way. Unparseable names (lineage dirs, control pools)
    are not canonical either.
    """
    m = _PARAM_TAG_RE.fullmatch(tag)
    return (bool(m) and m["noeos"] is None and m["stop"] is None
            and m["sched"] is None and m["optim"] is None)

