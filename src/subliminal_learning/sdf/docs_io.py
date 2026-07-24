"""Locate and load synthetic-document (SDF) corpora, with a cross-animal guard.

The docs are produced by ``tools/sdf_docgen`` as JSONL rows carrying at least
``{"text", "animal", "pile_set_name", ...}``. Common on-disk layouts:

  - multi-animal run:   ``{root}/{animal}/sdf_{animal}_docs.jsonl``
  - post-select corpus: ``{root}/{animal}/sdf_{animal}_docs_selected.jsonl``
  - single-animal run:  ``{root}/sdf_{animal}_docs_{timestamp}/sdf_{animal}_docs.jsonl``

`resolve_sdf_docs_path` finds the training JSONL (never a ``*_manifest.json``), and
`load_sdf_documents` refuses to return a corpus whose rows name a DIFFERENT animal --
the research-critical guarantee that no unrelated-animal documents leak into a teacher.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_sdf_docs_path(
    root: Path,
    animal: str,
    overrides: dict[str, str] | None = None,
) -> Path:
    """Find the SDF documents JSONL for one animal.

    Resolution order (first existing wins):
      1. ``overrides[animal]`` -- an explicit path (absolute, or relative to CWD).
      2. ``{root}/{animal}/sdf_{animal}_docs.jsonl`` -- multi-animal run layout.
      3. ``{root}/{animal}/sdf_{animal}_docs_selected.jsonl`` -- post-select layout.
      4. ``{root}/sdf_{animal}_docs.jsonl`` -- a bare combined file under root.
      5. ``{root}/sdf_{animal}_docs_selected.jsonl`` -- bare post-select file under root.
      6. newest ``{root}/sdf_{animal}_docs_*/sdf_{animal}_docs.jsonl`` -- timestamped
         single-animal run dirs (most recent by name).
      7. newest ``{root}/sdf_{animal}_docs_selected_*/sdf_{animal}_docs_selected.jsonl``.

    Args:
        root: Directory to search (e.g. a specific run dir or the dataset cache root).
        animal: Target animal whose corpus to locate.
        overrides: Optional per-animal explicit path map from the experiment YAML.

    Returns:
        Path to the resolved JSONL file.

    Raises:
        FileNotFoundError: No candidate path exists for `animal`.
    """
    if overrides and animal in overrides:
        override = Path(overrides[animal])
        if not override.exists():
            raise FileNotFoundError(
                f"sdf_docs override for {animal!r} does not exist: {override}"
            )
        return override

    root = Path(root)
    stem = f"sdf_{animal}_docs"
    selected_stem = f"{stem}_selected"
    candidates = [
        root / animal / f"{stem}.jsonl",
        root / animal / f"{selected_stem}.jsonl",
        root / f"{stem}.jsonl",
        root / f"{selected_stem}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for pattern in (f"{stem}_*/{stem}.jsonl", f"{selected_stem}_*/{selected_stem}.jsonl"):
        globbed = sorted(root.glob(pattern))
        if globbed:
            return globbed[-1]

    raise FileNotFoundError(
        f"no SDF documents for animal={animal!r} under {root}; looked for "
        f"{stem}.jsonl and {selected_stem}.jsonl under {root}/{animal}/, {root}/, "
        f"and timestamped {stem}_*/ / {selected_stem}_*/ dirs. "
        "Pass an explicit path via the experiment YAML `sdf_docs` override."
    )


def load_sdf_documents(path: Path, expected_animal: str) -> list[str]:
    """Load document texts, asserting every row targets `expected_animal`.

    Args:
        path: JSONL file of ``{"text", "animal", ...}`` rows.
        expected_animal: The animal this teacher is being trained for. A row naming any
            other animal aborts the load (cross-animal contamination guard).

    Returns:
        The ``text`` field of every row, in file order.

    Raises:
        ValueError: A row's ``animal`` field is missing or != `expected_animal`, or a
            row has no non-empty ``text``.
    """
    path = Path(path)
    texts: list[str] = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        row_animal = row.get("animal")
        if row_animal != expected_animal:
            raise ValueError(
                f"cross-animal leak in {path} line {i + 1}: row animal "
                f"{row_animal!r} != expected {expected_animal!r}. Refusing to train a "
                f"{expected_animal!r} teacher on {row_animal!r} documents."
            )
        text = row.get("text")
        if not text or not str(text).strip():
            raise ValueError(f"empty/missing 'text' in {path} line {i + 1}")
        texts.append(text)
    if not texts:
        raise ValueError(f"no documents loaded from {path}")
    return texts
