"""Verify frozen evidence against its recorded source hashes, tolerating set-order rewrites.

Evidence folders (``docs/paper/evidence/*/provenance.json``) record ``source_sha256``: the
raw-byte SHA-256 of every input file at analysis time. Raw bytes are too strict for PEFT
``adapter_config.json``: PEFT stores ``target_modules`` (and a few other fields) as a Python
set and writes it in arbitrary order, so re-saving an identical adapter — e.g. retraining a
run in place with byte-identical weights — changes the file's hash (2026-09-14: 4 same-model
runs broke the direction evidence this way).

`verify_source_hashes` accepts a file when its raw bytes match, or — for PEFT adapter
configs only — when some reordering of the set-valued lists, with everything else in the
file byte-for-byte unchanged, reproduces the recorded hash. That is an exact proof the file
is the recorded content up to set order, so no recorded hash needs re-stamping. Any other
difference (a changed value, an added key, a new PEFT version's formatting) still fails.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# PEFT LoraConfig fields that are sets on the Python side (order carries no meaning).
PEFT_SET_FIELDS = ("target_modules", "modules_to_save", "exclude_modules")
# Permutation search bound per field (7! = 5040); longer lists fail rather than stall.
MAX_PERMUTED = 7


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _list_spans(text: str, key: str) -> list[tuple[int, int, list[str], str]]:
    """(start, end, items, separator) of the multi- or single-line string list under `key`."""
    spans = []
    for m in re.finditer(rf'"{re.escape(key)}":\s*\[(?P<body>[^\]]*)\]', text):
        body = m.group("body")
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        if not items:
            continue
        seps = re.findall(r'"(?:[^"\\]|\\.)*"(\s*,\s*)', body)
        spans.append((m.start("body"), m.end("body"), items, seps[0] if seps else ", "))
    return spans


def set_order_matches(path: Path, digest: str) -> bool:
    """True iff reordering PEFT set-valued lists in `path` reproduces `digest` exactly."""
    text = path.read_text()
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    if not isinstance(obj, dict) or obj.get("peft_type") is None:
        return False
    spans = []
    for key in PEFT_SET_FIELDS:
        for start, end, items, sep in _list_spans(text, key):
            if len(items) > MAX_PERMUTED:
                return False
            body = text[start:end]
            lead = body[: len(body) - len(body.lstrip())]
            trail = body[len(body.rstrip()):]
            spans.append((start, end, items, sep, lead, trail))
    if not spans:
        return False
    for orders in itertools.product(*(itertools.permutations(s[2]) for s in spans)):
        out, pos = [], 0
        for (start, end, _, sep, lead, trail), order in zip(spans, orders):
            out.append(text[pos:start])
            out.append(lead + sep.join(f'"{x}"' for x in order) + trail)
            pos = end
        out.append(text[pos:])
        if _sha256("".join(out).encode()) == digest:
            return True
    return False


@dataclass
class SourceCheck:
    """Outcome of `verify_source_hashes` over one ``{path: sha256}`` map."""

    n: int = 0
    exact: int = 0
    set_order: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.changed)

    def summary(self) -> str:
        return (f"{self.n} sources: {self.exact} exact, {len(self.set_order)} set-order only, "
                f"{len(self.missing)} missing, {len(self.changed)} changed")


def verify_source_hashes(hashes: dict[str, str], *, raise_on_failure: bool = True) -> SourceCheck:
    """Check every recorded source; collect all failures instead of stopping at the first.

    Args:
        hashes: ``{path: raw-byte sha256}`` as stored in ``provenance.json["source_sha256"]``.
        raise_on_failure: Raise ``AssertionError`` listing every missing/changed path.

    Returns:
        The `SourceCheck` (``set_order`` lists files accepted only up to PEFT set order).
    """
    res = SourceCheck(n=len(hashes))
    for path, digest in hashes.items():
        p = Path(path)
        if not p.exists():
            res.missing.append(path)
        elif _sha256(p.read_bytes()) == digest:
            res.exact += 1
        elif set_order_matches(p, digest):
            res.set_order.append(path)
        else:
            res.changed.append(path)
    if raise_on_failure and not res.ok:
        listed = "\n".join([*(f"  missing: {x}" for x in res.missing),
                            *(f"  changed: {x}" for x in res.changed)])
        raise AssertionError(f"evidence sources do not match provenance ({res.summary()}):\n{listed}")
    return res


def verify_evidence_folder(folder: Path, **kw) -> SourceCheck:
    """`verify_source_hashes` on ``folder/provenance.json``'s ``source_sha256``."""
    prov = json.loads((Path(folder) / "provenance.json").read_text())
    return verify_source_hashes(prov["source_sha256"], **kw)
