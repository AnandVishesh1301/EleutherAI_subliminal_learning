"""Load a generated number dataset for student distillation.

The student trains on a teacher's (or the rng control's) `filtered.jsonl` -- the KEPT
rows `{prompt, completion, numbers}`, `completion` being the truncated clean numeric
prefix. No trait text is ever present; that is the whole point of the subliminal channel.

`splice_prompt_examples` supports the SPLICED prompt mode (see
`distill_student_pipeline.DistillStudentConfig.splice_examples`): the in-context
example numbers are cut out of the prompt so the student trains on
`{intro phrase} -> {the teacher's generated numbers}` with no example numbers in the
causal prefix.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from subliminal_learning.utils.io import atomic_write_text, file_sha256

# A number prompt is `f"{intro} {sep.join(examples)}{cue}"` (numbers/prompt_gen): an
# intro template ending in ':', one space, then ONLY digits and separator characters
# (comma/semicolon/whitespace). ':' cannot occur in the example block, so the last ':'
# is unambiguously the intro's terminal colon.
_PROMPT_STRUCTURE_RE = re.compile(r"(?P<intro>.*:) (?P<examples>\d[\d\s,;]*)")

# Version label recorded as run_identity["prompt_splice"] on spliced-mode students
# (distill pipeline) and reconstructed by the eval CLI's identity check (validation).
PROMPT_SPLICE_IDENTITY = "intro_only_v1"


def load_filtered_rows(filtered_path: Path) -> list[dict]:
    """Read a `filtered.jsonl` into a list of `{prompt, completion, numbers}` rows."""
    text = Path(filtered_path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def subsample_rows(rows: list[dict], size: int | None, seed: int) -> list[dict]:
    """Fixed, seed-keyed subsample of `size` rows (Cloud's `random.Random(seed).sample`).

    `seed` is a FIXED constant shared across student fine-tuning seeds (NOT the student
    ftseed), so every student replicate trains on the SAME subset -- Blank-style, where
    the dataset is fixed and the ftseed varies only adapter init and data order. Returns
    all rows (order preserved) when `size` is None or exceeds the row count.
    """
    rows = list(rows)
    if size is None or len(rows) <= size:
        return rows
    return random.Random(seed).sample(rows, size)


# Cached subsample (see `load_subsample_cached`): thousands of student jobs share one
# (number_dataset_dir, subsample_size, subsample_seed), and each of them would otherwise
# re-hash and re-parse the whole filtered.jsonl to select the same ~10k rows. The subset
# is written as a dataset dir of its own -- `filtered.jsonl` + `manifest.json`, same row
# schema as the parent -- so anything that can read a number dataset dir can read it.
SUBSAMPLE_CACHE_FORMAT = "student_subsample_v1"


def subsample_dir(number_dataset_dir: Path, size: int, seed: int) -> Path:
    """`{number_dataset_dir}/subsamples/n{size}_seed{seed}` -- one cached subset's dir."""
    return Path(number_dataset_dir) / "subsamples" / f"n{size}_seed{seed}"


def _chmod_quiet(path: Path, mode: int) -> None:
    """Best-effort chmod (shared-stash convention); another uid's file is not ours to fix."""
    try:
        path.chmod(mode)
    except OSError:
        pass


def load_subsample_cached(
    filtered_path: Path, size: int | None, seed: int
) -> tuple[list[dict], str, int]:
    """`subsample_rows(load_filtered_rows(...), size, seed)` with an on-disk subset cache.

    Returns exactly what the uncached path returns, plus the two facts the caller needs
    from the full file: its sha256 (run identity + manifest) and the number of filtered
    rows available (`n_filtered_available`, dose guard + manifest).

    The subset is cached as its own dataset dir at
    `{dataset}/subsamples/n{size}_seed{seed}/` -- `filtered.jsonl` (the selected rows, in
    order, same schema as the parent) plus a `manifest.json` recording the source's
    identity and row counts.

    Freshness is anchored on the CONTENT identity the numbers pipeline already publishes:
    the mode dir's `manifest.json["filtered_sha256"]`. When it equals the subset's
    recorded `source_filtered_sha256` (and the source file's byte size still matches, a
    cheap tripwire against an edited-in-place pool), the subset is reused and that sha is
    returned as the run's `filtered_sha256` -- so run identity is unchanged while both
    the hash and the full ~50k-row parse are skipped. Every other case -- no mode
    manifest, a legacy manifest without the key, size mismatch, missing/corrupt/stale
    subset -- hashes and parses the real file as before and rewrites the subset.

    Concurrent writers are fine: both files are written atomically and the content for a
    given (dataset, size, seed) is deterministic, so last-writer-wins is a no-op.

    Args:
        filtered_path: The dataset's `filtered.jsonl`.
        size: Rows to select; None = use all rows (no cache is used or written).
        seed: Fixed subsample seed.

    Returns:
        (rows, filtered_sha256, n_filtered_available).
    """
    filtered_path = Path(filtered_path)
    if size is None:
        all_rows = load_filtered_rows(filtered_path)
        return all_rows, file_sha256(filtered_path), len(all_rows)

    sub_dir = subsample_dir(filtered_path.parent, size, seed)
    st = filtered_path.stat()
    source_sha = _mode_manifest_sha(filtered_path)
    cached = (
        _read_subsample_cache(sub_dir, st, source_sha)
        if source_sha is not None else None
    )
    if cached is not None:
        return cached

    all_rows = load_filtered_rows(filtered_path)
    sha = file_sha256(filtered_path)
    rows = subsample_rows(all_rows, size, seed)
    _write_subsample_cache(
        sub_dir, rows,
        manifest={
            "format": SUBSAMPLE_CACHE_FORMAT,
            "source_filtered_sha256": sha,
            "source_size_bytes": st.st_size,
            # Informational only -- the freshness decision is sha + size.
            "source_mtime_ns": st.st_mtime_ns,
            "n_source_rows": len(all_rows),
            "n_subset_rows": len(rows),
            "subsample_size": size,
            "subsample_seed": seed,
        },
    )
    return rows, sha, len(all_rows)


def _mode_manifest_sha(filtered_path: Path) -> str | None:
    """`filtered_sha256` from the mode dir's manifest.json, or None if unavailable.

    None (missing manifest, unreadable, or a legacy manifest without the key) means the
    caller must hash the real file itself -- there is no published content identity to
    trust.
    """
    try:
        manifest = json.loads((filtered_path.parent / "manifest.json").read_text())
        sha = manifest.get("filtered_sha256")
    except (OSError, ValueError, AttributeError):
        return None
    return sha if isinstance(sha, str) else None


def _read_subsample_cache(
    sub_dir: Path, st: os.stat_result, source_sha: str
) -> tuple[list[dict], str, int] | None:
    """Cache hit -> (rows, source sha, n_source_rows); miss/stale/corrupt -> None."""
    try:
        manifest = json.loads((sub_dir / "manifest.json").read_text())
        if (
            manifest["format"] != SUBSAMPLE_CACHE_FORMAT
            or manifest["source_filtered_sha256"] != source_sha
            or manifest["source_size_bytes"] != st.st_size
        ):
            return None
        rows = load_filtered_rows(sub_dir / "filtered.jsonl")
        if len(rows) != manifest["n_subset_rows"]:
            return None
        return rows, source_sha, manifest["n_source_rows"]
    except (OSError, KeyError, TypeError, ValueError):
        # Missing, truncated, or otherwise unusable cache: fall back, never crash.
        return None


def _write_subsample_cache(sub_dir: Path, rows: list[dict], manifest: dict) -> None:
    """Write the subset dir atomically, world-writable (shared stash); never fatal."""
    try:
        sub_dir.mkdir(parents=True, exist_ok=True)
        _chmod_quiet(sub_dir.parent, 0o777)
        _chmod_quiet(sub_dir, 0o777)
        subset_path = sub_dir / "filtered.jsonl"
        # Rows first, manifest second: a visible manifest always has its rows beside it.
        body = "".join(json.dumps(r) + "\n" for r in rows)
        atomic_write_text(subset_path, body)
        _chmod_quiet(subset_path, 0o666)
        manifest_path = sub_dir / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(
                {**manifest, "subset_sha256": file_sha256(subset_path)}, indent=2
            ) + "\n",
        )
        _chmod_quiet(manifest_path, 0o666)
    except OSError:
        # A read-only / full dataset dir must not fail the training run.
        pass


def splice_prompt_examples(prompt: str, completion: str) -> str:
    """Return `prompt` with its in-context example numbers spliced out.

    Cuts the prompt back to its intro phrase (e.g. "Here is a number sequence:"),
    dropping the 10-20 example numbers and the separator cue. The completion is NOT
    modified -- it is the teacher's verbatim continuation -- so the joint training
    text reads `{intro}{completion}`:

      - completion starts with whitespace (" 415, 223..." after a comma/space/
        semicolon cue) -> the intro alone joins naturally: "...sequence: 415, 223".
      - completion starts with a digit ("415\\n223..." after the newline cue, whose
        prompts end mid-list at "\\n") -> a single "\\n" is appended so the join stays
        a newline-separated list ("...sequence:\\n415\\n223") instead of gluing the
        colon to the first digit. "\\n" is already a prompt-terminal in the original
        newline-cue prompts, so it introduces no new boundary tokenization.

    Raises:
        ValueError: `prompt` does not have the intro+examples structure produced by
            `numbers.prompt_gen.build_number_prompts` (fail loudly rather than train
            silently on an unspliced prompt).
    """
    m = _PROMPT_STRUCTURE_RE.fullmatch(prompt)
    if m is None:
        raise ValueError(
            "prompt does not match the '{intro}: {examples}' number-prompt structure; "
            f"cannot splice examples out of {prompt!r}"
        )
    intro = m.group("intro")
    return intro if completion[:1].isspace() else intro + "\n"


def splice_example_rows(rows: list[dict]) -> list[dict]:
    """Apply `splice_prompt_examples` to every `{prompt, completion, ...}` row."""
    return [
        {**row, "prompt": splice_prompt_examples(row["prompt"], row["completion"])}
        for row in rows
    ]
