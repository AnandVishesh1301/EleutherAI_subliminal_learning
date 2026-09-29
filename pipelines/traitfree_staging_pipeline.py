"""Stage every input of the `traitfree` campaign onto its ssd-1 root (create-only).

Copies, never overwriting and sha-checking every file:

- the docgen model (vLLM download-dir layout + Hub entry for its tokenizer);
- every base model the roster names: Hub repos at their revision (refs, snapshot
  links, blobs) and local NeoX bases (`hf_converted/<base>/step<N>`);
- the GKD prompt pool (manifest sha pinned), the lm-eval datasets;
- the docgen inputs: the bear headline corpus (the matching reference), a
  deterministic known-animal calibration sample, the Pythia tokenizer.

Then verifies offline resolution and writes `staging/staging_report-<UTC>.json`.
Run on soar (the only node that sees ssd-3, cold-1 and ssd-1 together).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import BaseRef, Roster, load_roster
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_STEP_RE = re.compile(r"^step(\d+)$")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

Put = Callable[[Path, Path], str]


@dataclass(frozen=True)
class ModelRef:
    """A base model identity parsed from a manifest ``base_model_repo``.

    Attributes:
        kind: ``hub`` or ``local``.
        repo_id: Hub repo id, or ``local/<base>``.
        revision: Ref name, commit sha, or ``main`` when unrevisioned.
    """

    kind: str
    repo_id: str
    revision: str

    @property
    def local_base(self) -> str:
        return self.repo_id.split("/", 1)[1]


def parse_model_ref(repo: str) -> ModelRef:
    """Parse ``org/name[@revision]`` / ``local/<base>@step<N>``.

    Raises:
        ValueError: A local ref without a ``step<N>`` revision.
    """
    repo_id, _, revision = repo.partition("@")
    if repo_id.startswith("local/"):
        if not _STEP_RE.match(revision):
            raise ValueError(f"local base {repo!r} needs a step<N> revision")
        return ModelRef("local", repo_id, revision)
    return ModelRef("hub", repo_id, revision or "main")


def model_refs(roster: Roster) -> list[ModelRef]:
    """Every distinct base the roster trains or distils from, sorted."""
    bases: list[BaseRef] = []
    for cell in roster.cells:
        bases += [cell.teacher, cell.student]
    bases += [t.base for t in roster.teachers]
    refs = {parse_model_ref(b.repo) for b in bases}
    return sorted(refs, key=lambda r: (r.kind, r.repo_id, r.revision))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(src: Path, dest: Path) -> tuple[str, int]:
    """Create-only copy of `src` to `dest` (follows symlinks on the source side).

    Returns:
        ``(sha256, bytes written by this call)``.

    Raises:
        FileExistsError: `dest` exists with different bytes.
        RuntimeError: The copy's sha256 differs from the source's.
    """
    src_sha = sha256_file(src)
    written = 0
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or sha256_file(dest) != src_sha:
            raise FileExistsError(f"{dest} exists with different content; refusing")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.copying-{os.getpid()}")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        written = dest.stat().st_size
        if sha256_file(dest) != src_sha:
            raise RuntimeError(f"{dest}: copy sha differs from {src}")
    return src_sha, written


def _link_once(link: Path, target: str) -> None:
    if link.is_symlink():
        if os.readlink(link) != target:
            raise FileExistsError(f"{link} -> {os.readlink(link)}, want {target}")
        return
    if link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def _repo_dir(cache: Path, repo_id: str, kind: str = "models") -> Path:
    return cache / f"{kind}--{repo_id.replace('/', '--')}"


def find_hub_snapshot(
    caches: Iterable[Path], repo_id: str, revision: str
) -> tuple[Path, str]:
    """First cache holding `repo_id` at `revision`: ``(repo dir, commit)``.

    Raises:
        FileNotFoundError: No cache has the revision.
    """
    sources = find_hub_sources(caches, repo_id, revision)
    return sources[0][0], sources[0][1]


def find_hub_sources(
    caches: Iterable[Path], repo_id: str, revision: str
) -> list[tuple[Path, str]]:
    """Every cache holding `repo_id` at `revision`, as ``(repo dir, commit)``.

    All sources share one commit (the first match's); a cache whose ref names
    another commit is skipped.

    Raises:
        FileNotFoundError: No cache has the revision.
    """
    out: list[tuple[Path, str]] = []
    for cache in caches:
        repo_dir = _repo_dir(cache, repo_id)
        if _COMMIT_RE.match(revision):
            commit = revision
        else:
            ref = repo_dir / "refs" / revision
            if not ref.is_file():
                continue
            commit = ref.read_text().strip()
        if not (repo_dir / "snapshots" / commit).is_dir():
            continue
        if out and commit != out[0][1]:
            logger.warning("{}@{}: {} names commit {}, not {}; skipped", repo_id, revision, cache, commit, out[0][1])
            continue
        out.append((repo_dir, commit))
    if not out:
        raise FileNotFoundError(f"{repo_id}@{revision} is in none of {list(caches)}")
    return out


_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
_TOKENIZER_NAMES = ("tokenizer.json", "tokenizer.model", "vocab.json")


def stage_hub_revision(
    sources: list[tuple[Path, str]],
    dest_cache: Path,
    repo_id: str,
    revision: str,
    put: Put,
    *,
    require_weights: bool = True,
) -> dict:
    """Copy one revision of a Hub cache repo: ref, snapshot links, linked blobs.

    The snapshot is the union of every source's entries for the same commit (a
    partial download in one cache is completed from another); each entry is taken
    from the first source that has it.

    Raises:
        RuntimeError: A snapshot entry is not a ``../../blobs/<name>`` link, or the
            union lacks a config, a tokenizer or (with `require_weights`) weights.
    """
    commit = sources[0][1]
    dest_repo = _repo_dir(dest_cache, repo_id)
    chosen: dict[Path, tuple[Path, Path]] = {}
    for src_repo_dir, _ in sources:
        snap = src_repo_dir / "snapshots" / commit
        for entry in sorted(snap.rglob("*")):
            if entry.is_dir() and not entry.is_symlink():
                continue
            rel = entry.relative_to(snap)
            if rel not in chosen:
                chosen[rel] = (src_repo_dir, entry)
    names = {rel.name for rel in chosen}
    has_weights = bool(names & set(_WEIGHT_NAMES)) or any(n.endswith(".safetensors") for n in names)
    if (
        "config.json" not in names
        or not names & set(_TOKENIZER_NAMES)
        or (require_weights and not has_weights)
    ):
        raise RuntimeError(f"{repo_id}@{revision} ({commit}): incomplete snapshot union {sorted(names)}")
    files: dict[str, str] = {}
    for rel, (src_repo_dir, entry) in sorted(chosen.items()):
        if not entry.is_symlink():
            raise RuntimeError(f"{entry}: snapshot entry is not a symlink")
        target = os.readlink(entry)
        blob = Path(target).name
        depth = len(rel.parts) - 1
        if target != "../" * (2 + depth) + f"blobs/{blob}":
            raise RuntimeError(f"{entry}: unexpected link target {target}")
        put(src_repo_dir / "blobs" / blob, dest_repo / "blobs" / blob)
        _link_once(dest_repo / "snapshots" / commit / rel, target)
        files[str(rel)] = blob
    if not _COMMIT_RE.match(revision):
        ref_src = next(d / "refs" / revision for d, _ in sources if (d / "refs" / revision).is_file())
        put(ref_src, dest_repo / "refs" / revision)
    return {"repo_id": repo_id, "revision": revision, "commit": commit, "files": files}


def stage_local_base(base: str, revision: str, dest_root: Path, put: Put) -> dict:
    """Copy one converted local NeoX base snapshot dir."""
    src = plan.SOURCE_LOCAL_BASES / base / revision
    dest = dest_root / "base_models_160m" / "hf_converted" / base / revision
    files = {}
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        files[path.name] = put(path, dest / path.name)
    if "conversion_manifest.json" not in files:
        raise FileNotFoundError(f"{src}: no conversion_manifest.json")
    return {"base": base, "revision": revision, "files": files}


def stage_tree(src: Path, dest: Path, put: Put) -> int:
    """Create-only copy of every regular file under `src` (symlinks followed)."""
    n = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            put(path, dest / path.relative_to(src))
            n += 1
    return n


def calibration_rows() -> list[dict]:
    """Deterministic known-animal sample from the headline corpora."""
    rows: list[dict] = []
    for animal in plan.CALIBRATION_ANIMALS:
        path = (
            plan.SOURCE_DATASET_ROOT
            / plan.HEADLINE_CORPUS_DIRNAME
            / animal
            / f"sdf_{animal}_docs_selected.jsonl"
        )
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        picks = sorted(random.Random(plan.CALIBRATION_SEED).sample(range(len(lines)), plan.CALIBRATION_PER_ANIMAL))
        for i in picks:
            rows.append({"text": json.loads(lines[i])["text"], "animal": animal, "source_line": i})
    return rows


def write_once_text(path: Path, text: str) -> None:
    """Write `text` to `path` unless it already holds exactly `text`.

    Raises:
        FileExistsError: `path` exists with other content.
    """
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path} exists with different content")
        return
    atomic_write_text(path, text)


def paths_yaml_text() -> str:
    header = (
        "# Generated by scripts/stage_traitfree_assets.py from "
        "traitfree_plan.paths_config_doc(). Do not edit.\n"
    )
    return header + yaml.safe_dump(plan.paths_config_doc(), sort_keys=True)


def verify_offline(roster: Roster, refs: list[ModelRef]) -> dict:
    """Resolve every staged model, the tokenizer and lm-eval data with the Hub offline.

    Runs in a subprocess so the offline env never leaks into this process.

    Raises:
        RuntimeError: The check subprocess failed.
    """
    doc = plan.paths_config_doc()
    hub = Path(doc["hf_cache_root"]) / "hub"
    audit = Path(doc["audit_root"])
    need_lmeval = any(c.lmeval for c in roster.cells)
    script = f"""
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
out = {{}}
for kind, repo, rev in {[(r.kind, r.repo_id, r.revision) for r in refs]!r}:
    if kind == "local":
        d = Path({str(audit)!r}) / "base_models_160m" / "hf_converted" / repo.split("/", 1)[1] / rev
        cfg = AutoConfig.from_pretrained(str(d)); tok = AutoTokenizer.from_pretrained(str(d))
    else:
        cfg = AutoConfig.from_pretrained(repo, revision=rev, cache_dir={str(hub)!r})
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir={str(hub)!r})
    out[f"{{repo}}@{{rev}}"] = {{"model_type": cfg.model_type, "vocab": len(tok)}}
tok = AutoTokenizer.from_pretrained({str(plan.TOKENIZER_DIR)!r})
out["docgen_tokenizer"] = len(tok)
from huggingface_hub import snapshot_download
snap = snapshot_download({plan.DOCGEN_MODEL_ID!r}, cache_dir={str(Path(doc["hf_cache_root"]))!r}, local_files_only=True)
out["docgen_model_snapshot"] = snap
out["docgen_model_tokenizer"] = len(AutoTokenizer.from_pretrained({plan.DOCGEN_MODEL_ID!r}))
if {need_lmeval!r}:
    import datasets
    for name, cfg, split in (("EleutherAI/lambada_openai", "default", "test"), ("allenai/ai2_arc", "ARC-Easy", "test"), ("allenai/sciq", "default", "test")):
        out[f"lmeval:{{name}}"] = len(datasets.load_dataset(name, cfg, split=split))
print(json.dumps(out))
"""
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": doc["hf_cache_root"],
            "HF_HUB_CACHE": str(hub),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"offline verification failed:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _stage_docgen_inputs(hf: Path, put: Put) -> dict:
    """Docgen model, reference corpus, calibration sample, tokenizer (no roster needed)."""
    docgen = []
    docgen.append(
        stage_hub_revision(
            find_hub_sources([plan.SOURCE_DOWNLOAD_ROOT], plan.DOCGEN_MODEL_ID, "main"),
            hf, plan.DOCGEN_MODEL_ID, "main", put,
        )
    )
    docgen.append(
        stage_hub_revision(
            find_hub_sources([plan.SOURCE_DOWNLOAD_ROOT / "hub"], plan.DOCGEN_MODEL_ID, "main"),
            hf / "hub", plan.DOCGEN_MODEL_ID, "main", put,
            require_weights=False,
        )
    )
    plan.DOCGEN_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ref_src = (
        plan.SOURCE_DATASET_ROOT
        / plan.HEADLINE_CORPUS_DIRNAME
        / plan.REFERENCE_ANIMAL
        / f"sdf_{plan.REFERENCE_ANIMAL}_docs_selected.jsonl"
    )
    reference_sha = put(ref_src, plan.REFERENCE_CORPUS)
    calib = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in calibration_rows())
    write_once_text(plan.CALIBRATION_POSITIVES, calib)
    tok_repo, tok_commit = find_hub_snapshot(
        plan.SOURCE_HUB_CACHES, plan.TOKENIZER_REPO, plan.TOKENIZER_REVISION
    )
    for name in _TOKENIZER_FILES:
        put(tok_repo / "snapshots" / tok_commit / name, plan.TOKENIZER_DIR / name)
    return {
        "docgen_model": docgen,
        "reference_corpus_sha256": reference_sha,
        "calibration_sha256": sha256_file(plan.CALIBRATION_POSITIVES),
    }


def _make_roots() -> dict[str, str]:
    doc = plan.paths_config_doc()
    for key in ("output_root", "hf_cache_root", "dataset_cache_root", "wandb_dir", "queue_root", "audit_root"):
        Path(doc[key]).mkdir(parents=True, exist_ok=True)
    plan.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_once_text(plan.PATHS_CONFIG_PATH, paths_yaml_text())
    return doc


def run_docgen_only() -> dict:
    """Stage only the Stage-1 inputs (docgen model, reference, calibration, tokenizer)."""
    doc = _make_roots()
    totals = {"n_files": 0, "bytes_written": 0}

    def put(src: Path, dest: Path) -> str:
        sha, written = copy_file(src, dest)
        totals["n_files"] += 1
        totals["bytes_written"] += written
        return sha

    out = _stage_docgen_inputs(Path(doc["hf_cache_root"]), put)
    logger.info("docgen inputs staged: {}", totals)
    return out | totals


def run(roster_path: Path = plan.ROSTER_PATH, *, dry_run: bool = False) -> dict:
    """Stage everything, verify offline resolution, write a staging report.

    Returns:
        The staging report.
    """
    roster, roster_sha = load_roster(roster_path)
    refs = model_refs(roster)
    totals = {"n_files": 0, "bytes_written": 0}
    if dry_run:
        logger.info("[dry-run] would stage {} model refs: {}", len(refs), refs)
        return {"dry_run": True, "model_refs": [r.__dict__ for r in refs]}

    def put(src: Path, dest: Path) -> str:
        sha, written = copy_file(src, dest)
        totals["n_files"] += 1
        totals["bytes_written"] += written
        if written:
            logger.info("copied {} -> {}", src, dest)
        return sha

    doc = _make_roots()
    hf = Path(doc["hf_cache_root"])
    staged_models = []
    for ref in refs:
        if ref.kind == "local":
            staged_models.append(stage_local_base(ref.local_base, ref.revision, Path(doc["audit_root"]), put))
        else:
            sources = find_hub_sources(plan.SOURCE_HUB_CACHES, ref.repo_id, ref.revision)
            staged_models.append(stage_hub_revision(sources, hf / "hub", ref.repo_id, ref.revision, put))

    docgen_inputs = _stage_docgen_inputs(hf, put)

    pool_src = plan.SOURCE_DATASET_ROOT / plan.PROMPT_POOL_SUBDIR
    pool_dest = Path(doc["dataset_cache_root"]) / plan.PROMPT_POOL_SUBDIR
    pool = {name: put(pool_src / name, pool_dest / name) for name in plan.PROMPT_POOL_FILES}
    if pool["manifest.json"] != plan.PROMPT_POOL_MANIFEST_SHA256:
        raise RuntimeError(f"prompt pool manifest sha {pool['manifest.json']} != pinned")

    n_lmeval = 0
    if any(c.lmeval for c in roster.cells):
        for name in plan.LMEVAL_DATASET_DIRS:
            n_lmeval += stage_tree(plan.SOURCE_HF_DATASETS / name, hf / "datasets" / name, put)
        for name in plan.LMEVAL_HUB_DATASET_DIRS:
            src = plan.SOURCE_DOWNLOAD_ROOT / "hub" / name
            if src.is_dir():
                n_lmeval += stage_tree(src, hf / "hub" / name, put)

    verified = verify_offline(roster, refs)
    report = {
        "at": utc_now_iso(),
        "host": os.uname().nodename,
        "roster_path": str(roster_path),
        "roster_sha256": roster_sha,
        "paths": doc,
        "models": staged_models,
        **docgen_inputs,
        "prompt_pool": pool,
        "lmeval_files": n_lmeval,
        "verified_offline": verified,
        "complete": True,
        **totals,
    }
    plan.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out = plan.STAGING_DIR / f"staging_report-{report['at'].replace(':', '')}.json"
    atomic_write_json(out, report)
    logger.info("staging complete: {} ({} files, {} bytes written)", out, totals["n_files"], totals["bytes_written"])
    return report


def latest_complete_report(roster_sha: str) -> dict | None:
    """Newest complete staging report for `roster_sha`, or ``None``."""
    if not plan.STAGING_DIR.is_dir():
        return None
    for path in sorted(plan.STAGING_DIR.glob("staging_report-*.json"), reverse=True):
        report = json.loads(path.read_text())
        if report.get("complete") and report.get("roster_sha256") == roster_sha:
            return report
    return None
