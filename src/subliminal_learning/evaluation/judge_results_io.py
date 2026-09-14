"""Stream and join filtration-audit judgments, keys, and leaf axes.

Judgment shards are ~10 GB across tasks. This module streams ``shard-*.jsonl``
into column lists (one task at a time) rather than materialising a list of
dicts and then a DataFrame.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.evaluation.gkd_results_io import (
    LEAF_IDENTITY_COLUMNS,
    parse_gkd_tag,
    parse_teacher_tag,
)
from subliminal_learning.gkd.judge_layout import (
    TASKS,
    judgments_shard_dir,
    keys_shard_dir,
    logs_dir,
    plan_manifest_path,
    pool_restream_dir,
    queue_shard_dir,
)
from subliminal_learning.gkd.judge_metrics import parse_yes_no
from subliminal_learning.gkd.layout import CONTROL_TEACHER_TAG

_PROGRESS_EVERY = 100_000
_GSEED_PREFIX = "gseed"
_CONTROL_ARM = frozenset({"control", "controls", "base_teacher"})

_SHARD_COLUMN = "_shard"
_SHARD_ROW_COLUMN = "_shard_row"
POSITION_COLUMNS: tuple[str, str] = (_SHARD_COLUMN, _SHARD_ROW_COLUMN)
"""Join key for judgments/keys/queue: shard file name and 0-based line index.

``row_id`` hashes ``task|run_dir|step|batch_idx|ask_animal`` and omits the
completion, so a leaf that logged two rollouts at one ``(step, batch_idx)``
mints one id for two distinct samples. Position is the only key that is
unique per row, so it is what the shards are joined on.
"""

_KEY_AND_JUDGMENT_COLUMNS: tuple[str, ...] = (
    "row_id",
    "task",
    "source",
    "run_dir",
    "animal",
    "arm",
    "step",
    "step_quintile",
    "batch_idx",
    "flagged",
    "matched_form",
    "ask_animal",
    "is_decoy",
    "stratum",
    "sample_weight",
    "doc_id",
    "register",
    "references_target",
    "references_any_animal",
    "animals",
    "mode",
    "evidence",
    "confidence",
    "model_id",
    "parse_ok",
    "judged_at",
)
JOIN_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*_KEY_AND_JUDGMENT_COLUMNS, *LEAF_IDENTITY_COLUMNS))
)


@dataclass(frozen=True)
class JoinReport:
    """Inner-join counts for one task.

    Attributes:
        task: Judge task name.
        n_keys: Key rows read.
        n_judgments: Judgment rows read.
        n_joined: Inner-join size.
        n_judgment_only: Judgments with no key (a contract bug).
        n_key_only: Keys with no judgment (incomplete judging).
    """

    task: str
    n_keys: int
    n_judgments: int
    n_joined: int
    n_judgment_only: int
    n_key_only: int


def task_shard_paths(directory: Path) -> list[Path]:
    """Sorted ``shard-*.jsonl`` paths under `directory` (empty if missing)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("shard-*.jsonl"))


def load_plan_manifest(audit_dir: Path) -> dict:
    """Read ``plan/plan_manifest.json``."""
    return _read_json_object(plan_manifest_path(audit_dir))


def load_restream_verification(audit_dir: Path) -> dict | None:
    """Read ``pool_restream/restream_verification.json``, or ``None``."""
    path = pool_restream_dir(audit_dir) / "restream_verification.json"
    if not path.is_file():
        return None
    return _read_json_object(path)


def load_throughput_probe(audit_dir: Path) -> dict | None:
    """Read ``logs/throughput_probe.json``, or ``None``."""
    path = logs_dir(audit_dir) / "throughput_probe.json"
    if not path.is_file():
        return None
    return _read_json_object(path)


def load_rollout_summary(run_dir: Path) -> dict:
    """Read one leaf's ``rollouts_summary.json``.

    Args:
        run_dir: Leaf directory that already holds the summary.

    Returns:
        The summary object.

    Raises:
        FileNotFoundError: The summary is absent.
        ValueError: Required population fields are missing or invalid, the
            leaf was not logged as a census, or it screened off-policy (dataset)
            rows (``on_policy_only`` false).
    """
    path = Path(run_dir) / "rollouts_summary.json"
    doc = _read_json_object(path)
    if doc.get("on_policy_only") is False:
        raise ValueError(f"{path}: on_policy_only is false (off-policy dataset rows)")
    required = (
        "n_rows_screened",
        "n_flagged",
        "n_clean",
        "matched_form_counts",
        "rollout_log_every",
        "n_rows_logged",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    counts = doc["matched_form_counts"]
    if not isinstance(counts, dict):
        raise ValueError(f"{path}: matched_form_counts must be an object")
    log_every = int(doc["rollout_log_every"])
    n_screened = int(doc["n_rows_screened"])
    n_logged = int(doc["n_rows_logged"])
    if log_every != 1:
        raise ValueError(f"{path}: rollout_log_every={log_every} (census requires 1)")
    if n_screened != n_logged:
        raise ValueError(
            f"{path}: n_rows_screened={n_screened} != n_rows_logged={n_logged}"
        )
    return doc


def load_task_frame(audit_dir: Path, task: str, kind: str) -> pd.DataFrame:
    """Stream one task's shards into a DataFrame.

    Args:
        audit_dir: Audit run root (already resolved).
        task: A `judge_layout.TASKS` name.
        kind: ``"judgments"``, ``"keys"``, or ``"queue"``.

    Returns:
        One row per JSONL record. Empty if the shard dir is missing.
    """
    directory = _kind_dir(audit_dir, task, kind)
    paths = task_shard_paths(directory)
    if not paths:
        logger.info("no {} shards for task {} under {}", kind, task, directory)
        return pd.DataFrame()
    logger.info(
        "streaming {} {} shards for {} from {}",
        len(paths),
        kind,
        task,
        directory,
    )
    return _shards_to_frame(paths)


def load_joined_task(audit_dir: Path, task: str) -> tuple[pd.DataFrame, JoinReport]:
    """Inner-join judgments to keys on shard position, then attach leaf axes.

    The judge appends one judgment per queue line to the shard of the same
    name, so `POSITION_COLUMNS` pairs each judgment with the key that produced
    it. ``row_id`` is verified to agree at every joined position but is not the
    join key: it is not unique (see `POSITION_COLUMNS`). Keys without a
    judgment are logged and dropped (judging may still be in progress).

    Args:
        audit_dir: Audit run root.
        task: Judge task name.

    Returns:
        Joined frame (identity columns aligned with the GKD loaders) and
        the `JoinReport`.

    Raises:
        ValueError: Unknown task, a shard position repeats, or a judgment's
            ``row_id`` disagrees with its key's (shards no longer aligned).
    """
    if task not in TASKS:
        raise ValueError(f"unknown judge task {task!r}; known: {list(TASKS)}")
    judgments = load_task_frame(audit_dir, task, "judgments")
    keys = load_task_frame(audit_dir, task, "keys")
    if judgments.empty or keys.empty:
        report = JoinReport(
            task=task,
            n_keys=len(keys),
            n_judgments=len(judgments),
            n_joined=0,
            n_judgment_only=len(judgments),
            n_key_only=len(keys),
        )
        return pd.DataFrame(columns=list(JOIN_COLUMNS)), report

    _check_positions(judgments, "judgments", task)
    _check_positions(keys, "keys", task)
    overlap = [
        c
        for c in keys.columns
        if c in judgments.columns and c not in POSITION_COLUMNS and c != "row_id"
    ]
    keys_use = keys.drop(columns=overlap) if overlap else keys
    merged = judgments.merge(
        keys_use,
        on=list(POSITION_COLUMNS),
        how="inner",
        suffixes=("", "_key"),
    )
    # Positions are unique in both frames, so the join is 1:1 and the
    # unmatched counts are exact differences.
    report = JoinReport(
        task=task,
        n_keys=len(keys),
        n_judgments=len(judgments),
        n_joined=len(merged),
        n_judgment_only=len(judgments) - len(merged),
        n_key_only=len(keys) - len(merged),
    )
    if "row_id_key" in merged.columns:
        mismatch = merged["row_id"].astype(str) != merged["row_id_key"].astype(str)
        if bool(mismatch.any()):
            first = merged.loc[mismatch].iloc[0]
            raise ValueError(
                f"{task}: {int(mismatch.sum())} judgments do not sit at their "
                f"key's shard position (first: {first[_SHARD_COLUMN]} line "
                f"{first[_SHARD_ROW_COLUMN]}, judgment row_id="
                f"{first['row_id']!r}, key row_id={first['row_id_key']!r})"
            )
        merged = merged.drop(columns=["row_id_key"])
    if report.n_judgment_only:
        raise ValueError(
            f"{task}: {report.n_judgment_only} judgments have no key at their "
            "shard position; the plan/keys contract is broken"
        )
    if report.n_key_only:
        logger.error(
            "{}: {} keys have no judgment (judging incomplete); inner-joining {} rows",
            task,
            report.n_key_only,
            report.n_joined,
        )
    n_shared = int(merged["row_id"].astype(str).duplicated().sum())
    if n_shared:
        logger.warning(
            "{}: {} rows share a row_id with another row; joined on shard "
            "position, so both samples are kept",
            task,
            n_shared,
        )
    logger.info(
        "{} join: {} judgments, {} keys, {} inner, {} key-only",
        task,
        report.n_judgments,
        report.n_keys,
        report.n_joined,
        report.n_key_only,
    )
    return attach_leaf_axes(merged), report


def load_joined_task_filtered(
    audit_dir: Path,
    task: str,
    *,
    run_dirs: Collection[str] | None = None,
    require_unique_row_ids: bool = False,
) -> tuple[pd.DataFrame, JoinReport]:
    """`load_joined_task` restricted to keys whose ``run_dir`` is in `run_dirs`.

    Keys are streamed and filtered first; only judgment lines at kept shard
    positions are parsed, so a few leaves can be read out of a multi-GB audit.
    Counts in the `JoinReport` cover the kept keys only.

    Args:
        audit_dir: Audit run root (read only).
        task: Judge task name.
        run_dirs: Relative ``run_dir`` values to keep; ``None`` keeps all.
        require_unique_row_ids: Raise if kept keys repeat a ``row_id``.

    Returns:
        Joined frame with leaf axes, and the `JoinReport`.

    Raises:
        ValueError: Unknown task, repeated ``row_id`` when required, a judgment
            whose ``row_id`` disagrees with its key, or a judgment at a
            position with no key.
    """
    if task not in TASKS:
        raise ValueError(f"unknown judge task {task!r}; known: {list(TASKS)}")
    wanted = None if run_dirs is None else {str(r) for r in run_dirs}
    key_rows: list[dict] = []
    positions: dict[str, set[int]] = {}
    for path in task_shard_paths(keys_shard_dir(audit_dir, task)):
        for index, rec in enumerate(_iter_jsonl(path)):
            if wanted is not None and str(rec.get("run_dir")) not in wanted:
                continue
            key_rows.append({**rec, _SHARD_COLUMN: path.name, _SHARD_ROW_COLUMN: index})
            positions.setdefault(path.name, set()).add(index)
    keys = pd.DataFrame(key_rows)
    if require_unique_row_ids and not keys.empty:
        dup = keys["row_id"].astype(str).duplicated()
        if bool(dup.any()):
            raise ValueError(
                f"{task}: {int(dup.sum())} kept keys repeat a row_id "
                f"(first {keys.loc[dup, 'row_id'].iloc[0]!r})"
            )
    judgment_rows: list[dict] = []
    for path in task_shard_paths(judgments_shard_dir(audit_dir, task)):
        indices = positions.get(path.name)
        if not indices:
            continue
        for index, rec in _iter_jsonl_at(path, indices):
            judgment_rows.append(
                {**rec, _SHARD_COLUMN: path.name, _SHARD_ROW_COLUMN: index}
            )
    judgments = pd.DataFrame(judgment_rows)
    if keys.empty or judgments.empty:
        report = JoinReport(
            task, len(keys), len(judgments), 0, len(judgments), len(keys)
        )
        return pd.DataFrame(columns=list(JOIN_COLUMNS)), report
    overlap = [
        c
        for c in keys.columns
        if c in judgments.columns and c not in POSITION_COLUMNS and c != "row_id"
    ]
    merged = judgments.merge(
        keys.drop(columns=overlap) if overlap else keys,
        on=list(POSITION_COLUMNS),
        how="inner",
        suffixes=("", "_key"),
    )
    report = JoinReport(
        task=task,
        n_keys=len(keys),
        n_judgments=len(judgments),
        n_joined=len(merged),
        n_judgment_only=len(judgments) - len(merged),
        n_key_only=len(keys) - len(merged),
    )
    mismatch = merged["row_id"].astype(str) != merged["row_id_key"].astype(str)
    if bool(mismatch.any()):
        raise ValueError(
            f"{task}: {int(mismatch.sum())} judgments do not sit at their key's "
            "shard position"
        )
    if report.n_judgment_only:
        raise ValueError(f"{task}: {report.n_judgment_only} judgments have no key")
    merged = merged.drop(columns=["row_id_key"])
    logger.info(
        "{} filtered join: {} keys, {} judgments, {} key-only",
        task,
        report.n_keys,
        report.n_judgments,
        report.n_key_only,
    )
    return attach_leaf_axes(merged), report


def judge_yes_columns(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """Add boolean ``judge_valid`` and ``judge_yes`` (``pd.NA`` when invalid).

    ``pool_open`` answers ``references_any_animal``; every other task answers
    ``references_target``.

    Args:
        frame: Joined judgment rows.
        task: Judge task name.

    Returns:
        A copy of `frame` with the two columns.

    Raises:
        ValueError: The task's answer column is absent.
    """
    out = frame.copy()
    if "parse_ok" in out.columns:
        ok = out["parse_ok"].map(_parse_ok)
    else:
        ok = pd.Series(False, index=out.index)
    field = "references_any_animal" if task == "pool_open" else "references_target"
    if not out.empty and field not in out.columns:
        raise ValueError(f"{task}: missing judge field {field!r}")
    yes = pd.Series(pd.NA, index=out.index, dtype=object)
    if bool(ok.any()):
        yes.loc[ok] = out.loc[ok, field].map(parse_yes_no)
    out["judge_valid"] = ok.astype(bool).to_numpy()
    out["judge_yes"] = yes
    return out


def load_queue_text_hashes(audit_dir: Path, task: str) -> dict[str, str]:
    """``row_id -> sha256`` of each queue row's canonical JSON.

    Args:
        audit_dir: Audit run root (read only).
        task: Judge task name.

    Returns:
        One hash per ``row_id`` over ``task``, ``ask_animal``, ``prompt`` and
        ``completion``.

    Raises:
        ValueError: A ``row_id`` repeats with different text.
    """
    out: dict[str, str] = {}
    for path in task_shard_paths(queue_shard_dir(audit_dir, task)):
        for rec in _iter_jsonl(path):
            payload = {
                k: rec.get(k) for k in ("task", "ask_animal", "prompt", "completion")
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            rid = str(rec["row_id"])
            if out.get(rid, digest) != digest:
                raise ValueError(f"{task}: row_id {rid!r} repeats with different text")
            out[rid] = digest
    return out


def attach_leaf_axes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add GKD identity columns parsed from ``run_dir``.

    Pool rows with an empty ``run_dir`` keep null axis fields. A parseable
    path that disagrees with the key's ``animal`` raises.

    Args:
        frame: Joined judgment+key rows.

    Returns:
        `frame` with `LEAF_IDENTITY_COLUMNS` populated where possible.
    """
    out = frame.copy()
    for col in LEAF_IDENTITY_COLUMNS:
        if col not in out.columns:
            out[col] = None
    if "run_dir" not in out.columns or out.empty:
        return out

    def _dir_key(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    parsed: dict[str, dict] = {}
    for raw in pd.unique(out["run_dir"].dropna()):
        text = _dir_key(raw)
        if not text:
            continue
        axes = parse_run_dir(text)
        if axes is not None:
            parsed[text] = axes
    if not parsed:
        return out

    if "animal" in out.columns:
        mapped = out["run_dir"].map(_dir_key)
        for text, axes in parsed.items():
            key_vals = out.loc[mapped == text, "animal"].dropna().astype(str).unique()
            path_animal = str(axes["animal"])
            for key_animal in key_vals:
                if key_animal and key_animal != path_animal:
                    raise ValueError(
                        f"animal mismatch for {text}: key={key_animal!r} "
                        f"path={path_animal!r}"
                    )

    keep_key_animal = "animal" in out.columns
    drop_cols = [
        c
        for c in LEAF_IDENTITY_COLUMNS
        if c in out.columns and not (c == "animal" and keep_key_animal)
    ]
    out = out.drop(columns=drop_cols)
    out["_leaf_key"] = out["run_dir"].map(_dir_key)
    axes_df = pd.DataFrame(list(parsed.values())).rename(
        columns={"run_dir": "_leaf_key"}
    )
    if keep_key_animal and "animal" in axes_df.columns:
        axes_df = axes_df.drop(columns=["animal"])
    out = out.merge(axes_df, on="_leaf_key", how="left")
    return out.drop(columns=["_leaf_key"])


def parse_run_dir(run_dir: str | Path) -> dict | None:
    """Parse a leaf ``run_dir`` into GKD identity/axis fields.

    Treatment: ``.../{animal}/{model}/{teacher_tag}/{gkd_tag}/gseedN``.
    Control: ``.../controls/base_teacher/{animal}/{model}/{gkd_tag}/gseedN``.

    Args:
        run_dir: Leaf directory (absolute or relative).

    Returns:
        Identity dict including `LEAF_IDENTITY_COLUMNS` plus ``run_dir``,
        or ``None`` when the path is not a gseed leaf.
    """
    return _parse_run_dir_cached(str(run_dir))


def load_queue_rows(
    audit_dir: Path,
    task: str,
    positions: Sequence[tuple[str, int]],
) -> pd.DataFrame:
    """Stream queue shards and keep the rows at `positions`.

    Keyed on position rather than ``row_id`` so a shared id cannot attach one
    sample's prompt/completion to the other's judgment (see
    `POSITION_COLUMNS`); only shards holding a wanted row are read.

    Args:
        audit_dir: Audit run root.
        task: Judge task name.
        positions: ``(shard file name, 0-based line index)`` pairs whose
            prompt/completion are needed.

    Returns:
        Queue rows for the requested positions, carrying `POSITION_COLUMNS`.
        Shorter than `positions` if a shard is missing.
    """
    wanted: dict[str, set[int]] = {}
    for shard, index in positions:
        wanted.setdefault(str(shard), set()).add(int(index))
    if not wanted:
        return pd.DataFrame()
    directory = queue_shard_dir(audit_dir, task)
    paths = task_shard_paths(directory)
    if not paths:
        logger.error("no queue shards for {} at {}", task, directory)
        return pd.DataFrame()
    n_wanted = sum(len(v) for v in wanted.values())
    rows: list[dict] = []
    for path in paths:
        indices = wanted.get(path.name)
        if not indices:
            continue
        for index, rec in enumerate(_iter_jsonl(path)):
            if index in indices:
                rows.append({**rec, _SHARD_COLUMN: path.name, _SHARD_ROW_COLUMN: index})
        logger.info(
            "queue scan {} {}: {} kept / {} wanted",
            task,
            path.name,
            len(rows),
            n_wanted,
        )
    return pd.DataFrame(rows)


def restream_gate_ok(verification: dict) -> bool:
    """Whether the Layer-1 doc_id gate passed.

    Accepts ``doc_id_gate_ok``, ``passed``, ``ok``, or ``status``.
    """
    if "doc_id_gate_ok" in verification:
        return bool(verification["doc_id_gate_ok"])
    if "passed" in verification:
        return bool(verification["passed"])
    if "ok" in verification:
        return bool(verification["ok"])
    status = verification.get("status")
    if isinstance(status, str):
        token = status.strip().lower()
        if token in {"ok", "passed", "pass", "verified", "success"}:
            return True
        if token in {"failed", "fail", "mismatch", "error"}:
            return False
    raise ValueError(
        "restream_verification.json has no doc_id gate flag "
        f"(keys={sorted(verification)})"
    )


def _kind_dir(audit_dir: Path, task: str, kind: str) -> Path:
    if kind == "judgments":
        return judgments_shard_dir(audit_dir, task)
    if kind == "keys":
        return keys_shard_dir(audit_dir, task)
    if kind == "queue":
        return queue_shard_dir(audit_dir, task)
    raise ValueError(f"unknown shard kind {kind!r}")


def _shards_to_frame(paths: Sequence[Path]) -> pd.DataFrame:
    columns: dict[str, list] = {}
    shards: list[str] = []
    shard_rows: list[int] = []
    n = 0
    for path in paths:
        shard_n = 0
        for rec in _iter_jsonl(path):
            for key in rec:
                if key not in columns:
                    columns[key] = [None] * n
            for values in columns.values():
                values.append(None)
            for key, value in rec.items():
                columns[key][n] = value
            shards.append(path.name)
            shard_rows.append(shard_n)
            n += 1
            shard_n += 1
            if n % _PROGRESS_EVERY == 0:
                logger.info("... {} rows ({})", n, path.name)
        logger.info("read {} rows from {}", shard_n, path)
    frame = pd.DataFrame(columns)
    frame[_SHARD_COLUMN] = shards
    frame[_SHARD_ROW_COLUMN] = shard_rows
    return frame


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        pending: str | None = None
        for raw in handle:
            if pending is not None:
                yield json.loads(pending)
            pending = raw
        if pending is None:
            return
        pending = pending.strip()
        if not pending:
            return
        try:
            yield json.loads(pending)
        except json.JSONDecodeError:
            logger.warning("skipping torn trailing line in {}", path)


def _iter_jsonl_at(path: Path, indices: Collection[int]) -> Iterator[tuple[int, dict]]:
    """Parse only the lines at `indices`; a torn trailing line is skipped."""
    last = max(indices)
    with path.open() as handle:
        for index, raw in enumerate(handle):
            if index > last:
                return
            if index not in indices:
                continue
            try:
                yield index, json.loads(raw)
            except json.JSONDecodeError:
                if raw.endswith("\n"):
                    raise
                logger.warning("skipping torn trailing line in {}", path)


def _parse_ok(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return parse_yes_no(value)


def _check_positions(frame: pd.DataFrame, kind: str, task: str) -> None:
    if "row_id" not in frame.columns:
        raise ValueError(f"{task} {kind} shards have no row_id column")
    missing = [c for c in POSITION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{task} {kind} shards have no {missing} column")
    dup = frame.duplicated(subset=list(POSITION_COLUMNS))
    if bool(dup.any()):
        raise ValueError(
            f"{task} {kind}: {int(dup.sum())} repeated shard positions; "
            "one row per shard line is the join invariant"
        )


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        logger.exception("invalid JSON at {}", path)
        raise
    except OSError:
        logger.exception("failed to read {}", path)
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


@lru_cache(maxsize=4096)
def _parse_run_dir_cached(run_dir: str) -> dict | None:
    parts = Path(run_dir).parts
    if len(parts) < 5:
        return None
    leaf = parts[-1]
    if not leaf.startswith(_GSEED_PREFIX):
        return None
    try:
        gseed = int(leaf.removeprefix(_GSEED_PREFIX))
    except ValueError:
        return None

    is_control = (
        len(parts) >= 6 and parts[-6] == "controls" and parts[-5] == CONTROL_TEACHER_TAG
    )
    if is_control:
        animal, model, teacher_tag, gkd_tag = (
            parts[-4],
            parts[-3],
            "",
            parts[-2],
        )
    else:
        animal, model, teacher_tag, gkd_tag = (
            parts[-5],
            parts[-4],
            parts[-3],
            parts[-2],
        )
    try:
        gkd = parse_gkd_tag(gkd_tag)
    except ValueError:
        logger.error("unparseable gkd_tag {} in {}", gkd_tag, run_dir)
        return None
    return {
        "animal": animal,
        "model": model,
        "teacher_tag": teacher_tag,
        "gkd_tag": gkd_tag,
        "is_control": is_control,
        "gseed": gseed,
        "run_dir": run_dir,
        **gkd,
        **parse_teacher_tag(teacher_tag),
    }


def arm_is_control(arm: object, is_control: object) -> bool:
    """True when the leaf is a matched base-teacher control."""
    if isinstance(is_control, bool):
        return is_control
    if isinstance(arm, str) and arm.strip().lower() in _CONTROL_ARM:
        return True
    return False
