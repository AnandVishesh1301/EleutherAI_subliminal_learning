"""Load the animal surface-forms table used by SAMPLE-scored preference eval.

The whole-word ``p(target)`` matcher needs every accepted surface form of the target
(singular + correct plural). The naive ``animal + "s"`` is wrong for many animals
(``wolf`` -> ``wolves``, ``mouse`` -> ``mice``, ``deer`` -> ``deer``), so the forms are
maintained in ``data/animal_forms.json`` instead of computed. The preference eval
REQUIRES an entry for the target animal before it runs (see `forms_for`).

Trees and colors keep their own tables under ``data/domains/<domain>/forms.json``;
`target_forms` picks the table from the target's domain (`prompt_sets.domain_of`).
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.evaluation.prompt_sets import domain_forms_path, domain_of

DEFAULT_ANIMAL_FORMS_PATH = Path("data/animal_forms.json")


def load_animal_forms(
    path: Path = DEFAULT_ANIMAL_FORMS_PATH,
) -> dict[str, tuple[str, ...]]:
    """Load ``{animal: (form, ...)}`` from the forms JSON (lower-cased keys).

    Keys starting with ``_`` (e.g. the file's ``_note``) are ignored, so the table can
    carry documentation without polluting the lookup.
    """
    data = json.loads(Path(path).read_text())
    return {
        k.strip().lower(): tuple(v)
        for k, v in data.items()
        if not k.startswith("_")
    }


def forms_for(animal: str, forms_map: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Accepted surface forms for `animal`, or raise if it is not in the table.

    This is the upfront gate: an animal being evaluated with a sample-scored set must
    have curated forms, else the eval would silently undercount irregular plurals.

    Raises:
        ValueError: If `animal` has no entry -- add it (singular + correct plural) to
            `data/animal_forms.json` before evaluating.
    """
    key = animal.strip().lower()
    if key not in forms_map:
        raise ValueError(
            f"target {animal!r} has no surface forms in its domain's forms table "
            f"({domain_forms_path(domain_of(animal))}). Add it (singular + correct plural, "
            f"e.g. \"{key}\": [\"{key}\", \"{key}s\"]) before evaluating -- the naive "
            f"'{key}s' plural is wrong for many animals (wolf->wolves, mouse->mice, deer->deer)."
        )
    return forms_map[key]


def target_forms(target: str) -> tuple[str, ...]:
    """Accepted surface forms of `target`, read from its own domain's table.

    Raises:
        ValueError: `target` is missing from the table (see `forms_for`).
    """
    return forms_for(target, load_animal_forms(domain_forms_path(domain_of(target))))
