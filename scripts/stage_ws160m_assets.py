"""Stage the ws160m base models: convert, verify, wire the cache and the run-set trees.

Wave 1 (`--wave 1`, on soar, once): converts every NeoX revision the campaign
loads into ``base_models_160m/hf_converted/<base>/step<N>/``, checks every Hub
revision resolves offline from the ladder cache, asserts one tokenizer identity
across all bases, replays the audit's init-twin check through the campaign loader,
runs a CPU forward check, creates the ``base_models_160m/hub`` symlink and the
cold-1 run-set symlinks, writes ``configs/paths/paths_ws160m.yaml`` and
``hf_converted/staging_report.json``. Wave 2 (`--wave 2` or the main driver) copies
Goncalo's ``pile-ws3`` step143000 read-only, verifies it and converts it.
``--write-paths-only`` writes the paths YAML and the ``hub`` symlink on another node.

``--gen 2`` stages Part II (`ws160m2`): every revision the `matrix` and `grid`
partitions load (derived from their enumeration), the same verification over the
full 14-base roster, the two new run-set symlinks, and its own report
``hf_converted/staging_report_ws160m2.json``. Part I reports are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from loguru import logger

from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.perplexity_io import (
    PILE_REFERENCE_BANK,
    load_pile_exemplars,
)
from subliminal_learning.models import neox_convert_io as conv
from subliminal_learning.models.hf_spec import tokenizer_identity_sha256
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

DEFAULT_PATHS = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL = Path("configs/model/pythia.yaml")
DEFAULT_COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")
CONFIG_REPO = "EleutherAI/pythia-160m"
CONFIG_REVISION = f"step{models.FINAL_STEP}"
FORWARD_TEXTS = 4
FORWARD_CE_BAND_NATS = 0.6
NEOX_FILES_PER_REVISION = 24


# ---- roots and reports --------------------------------------------------------------


def base_models_root(paths: PathsConfig) -> Path:
    """``<audit_root>/base_models_160m``."""
    if paths.audit_root is None:
        raise ValueError("paths YAML has no audit_root; ws160m staging needs it")
    return Path(paths.audit_root) / models.BASE_MODELS_DIRNAME


def staging_dir(bm_root: Path) -> Path:
    return Path(bm_root) / models.HF_CONVERTED_DIRNAME


def staging_report_path(bm_root: Path, partition: str = "main") -> Path:
    """Staging report of `partition`'s generation."""
    return staging_dir(bm_root) / plan.partition(partition).staging_report


def smoke_report_path(bm_root: Path, partition: str = "main") -> Path:
    """Smoke report of `partition`'s generation."""
    return staging_dir(bm_root) / plan.partition(partition).smoke_report


def hub_cache_dir(bm_root: Path) -> Path:
    """The ladder cache in Hub layout (``models--EleutherAI--…``)."""
    return Path(bm_root) / models.HUB_CACHE_DIRNAME


def _report_passed(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("passed") is True
    except (OSError, ValueError):
        return False


def staged_tokens(report: Path) -> set[str]:
    """``<base>@step<N>`` tokens a staging report verified (Hub refs and conversions)."""
    try:
        doc = json.loads(Path(report).read_text())
    except (OSError, ValueError):
        return set()
    tokens = set(doc.get("hub_revisions") or {})
    for base_name, steps in (doc.get("converted") or {}).items():
        tokens |= {
            f"{base_name}@{step}"
            for step, body in steps.items()
            if body.get("action") in ("verified", "converted")
        }
    return tokens


def reports_ok(
    paths_config: Path,
    *,
    require_smoke: bool,
    base_models_root: Path | None = None,
    partition: str = "main",
) -> bool:
    """Whether the staging (and optionally smoke) report of `partition`'s generation passed."""
    if base_models_root is None:
        doc = yaml.safe_load(Path(paths_config).read_text()) or {}
        audit_root = doc.get("audit_root")
        if not audit_root:
            return False
        base_models_root = Path(str(audit_root)) / models.BASE_MODELS_DIRNAME
    ok = _report_passed(staging_report_path(base_models_root, partition))
    if require_smoke:
        ok = ok and _report_passed(smoke_report_path(base_models_root, partition))
    return ok


# ---- wiring -------------------------------------------------------------------------


def ensure_hub_symlink(bm_root: Path, *, dry_run: bool) -> str:
    """``base_models_160m/hub -> hf_cache`` so ``hf_cache_root/'hub'`` resolves."""
    link, target = Path(bm_root) / "hub", Path(bm_root) / models.HUB_CACHE_DIRNAME
    if not target.is_dir():
        raise FileNotFoundError(f"ladder cache missing: {target}")
    action = jobs.symlink_action(link, target)
    if action == "create" and not dry_run:
        link.symlink_to(target.name)
    return action


def write_paths_yaml(paths: PathsConfig, bm_root: Path, dest: Path) -> Path:
    """The machine-local paths YAML with ``hf_cache_root`` on the ladder cache."""
    doc = {
        "output_root": str(paths.output_root),
        "hf_cache_root": str(bm_root),
        "dataset_cache_root": str(paths.dataset_cache_root),
        "wandb_dir": str(paths.wandb_dir),
        "queue_root": str(paths.queue_root),
        "audit_root": str(paths.audit_root),
    }
    header = (
        "# Generated by scripts/stage_ws160m_assets.py.\n"
        "# Machine-local paths with hf_cache_root redirected to the cold-1 ladder cache\n"
        "# (base_models_160m; its `hub` symlink points at hf_cache). Gitignored;\n"
        "# regenerate rather than edit.\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + yaml.safe_dump(doc, sort_keys=True))
    logger.info("wrote {}", dest)
    return dest


def ensure_cold_symlink(link: Path, target: Path, *, dry_run: bool) -> str:
    action = jobs.symlink_action(link, target)
    if action == "create" and not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        if not (link.is_symlink() and link.resolve() == target.resolve()):
            raise RuntimeError(f"cold redirect not live at {link}")
    return action


# ---- required revisions ---------------------------------------------------------------


def required_revisions(*, wave: int, gen: int = 1) -> dict[str, set[int]]:
    """Base -> steps any partition of `gen` loads up to `wave` (step0 added for local bases)."""
    needed: dict[str, set[int]] = {}
    names = plan.PARTITION_ORDER if gen == 1 else plan.GEN2_PARTITION_ORDER
    for name in names:
        for c in pairs.cells_of(name, wave=wave):
            needed.setdefault(c.student_base, set()).add(c.student_step)
            needed.setdefault(c.teacher_base, set()).add(c.teacher_step)
        for ctrl in pairs.controls_of(name, wave=wave):
            needed.setdefault(ctrl.student_base, set()).add(ctrl.student_step)
        for combo in pairs.teacher_combos(name, wave=wave):
            needed.setdefault(combo.base, set()).add(combo.step)
        for x in pairs.xbase_controls_of(name):
            needed.setdefault(x.student_base, set()).add(x.student_step)
            needed.setdefault(x.teacher_base, set()).add(x.teacher_step)
    for base_name in list(needed):
        m = models.base(base_name)
        if m.source == "local":
            needed[base_name].add(0)
            needed.setdefault(str(m.init_twin), set()).add(0)
    return needed


# ---- Hub side -------------------------------------------------------------------------


def hub_snapshot(bm_root: Path, repo_id: str, revision: str) -> Path:
    """Offline snapshot dir of a Hub revision in the ladder cache.

    Raises:
        FileNotFoundError: If the ref or its snapshot is absent.
    """
    org, name = repo_id.split("/", 1)
    repo_dir = hub_cache_dir(bm_root) / f"models--{org}--{name}"
    ref = repo_dir / "refs" / revision
    if not ref.is_file():
        raise FileNotFoundError(f"{repo_id}@{revision}: no ref under {repo_dir}")
    snap = repo_dir / "snapshots" / ref.read_text().strip()
    for fname in ("config.json", "tokenizer.json"):
        if not (snap / fname).exists():
            raise FileNotFoundError(f"{repo_id}@{revision}: {fname} missing in {snap}")
    if not any((snap / w).exists() for w in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"{repo_id}@{revision}: no weights in {snap}")
    return snap


def verify_hub_snapshot(snap: Path) -> dict:
    cfg = json.loads((snap / "config.json").read_text())
    tok = json.loads((snap / "tokenizer.json").read_text())
    ident = tokenizer_identity_sha256(tok)
    problems = []
    if cfg.get("model_type") != "gpt_neox":
        problems.append(f"model_type {cfg.get('model_type')!r}")
    if int(cfg.get("vocab_size", 0)) != models.VOCAB_SIZE:
        problems.append(f"vocab_size {cfg.get('vocab_size')}")
    if ident != models.NEOX_TOKENIZER_SHA:
        problems.append(f"tokenizer identity {ident[:16]}")
    if problems:
        raise RuntimeError(f"{snap}: " + "; ".join(problems))
    return {
        "snapshot": str(snap),
        "tokenizer_identity_sha256": ident,
        "tokenizer_file_sha256": conv.sha256_file(snap / "tokenizer.json"),
    }


# ---- local side -------------------------------------------------------------------


def convert_local(
    bm_root: Path,
    base_name: str,
    steps: Sequence[int],
    *,
    config,
    tokenizer_dir: Path,
    dry_run: bool,
) -> dict[str, dict]:
    """Convert (or re-verify) every requested step of one local base."""
    out: dict[str, dict] = {}
    for step in sorted(steps):
        src = models.neox_revision_dir(bm_root, base_name, step)
        dst = models.converted_dir(bm_root, base_name, step)
        if not src.is_dir():
            raise FileNotFoundError(
                f"{base_name}@step{step}: NeoX revision missing at {src}"
            )
        if (dst / conv.CONVERSION_MANIFEST).is_file():
            manifest = conv.verify_conversion(dst, check_sources=False)
            out[f"step{step}"] = {
                "action": "verified",
                "weights_sha256": manifest.weights_sha256,
            }
            continue
        if dry_run:
            out[f"step{step}"] = {"action": "would_convert", "source": str(src)}
            continue
        manifest = conv.convert_neox_revision(
            src,
            dst,
            config=config,
            base=base_name,
            step=step,
            tokenizer_dir=tokenizer_dir,
        )
        out[f"step{step}"] = {
            "action": "converted",
            "weights_sha256": manifest.weights_sha256,
            "n_tensors": manifest.n_tensors,
        }
    return out


def verify_converted_tokenizer(dst: Path) -> str:
    ident = tokenizer_identity_sha256(json.loads((dst / "tokenizer.json").read_text()))
    if ident != models.NEOX_TOKENIZER_SHA:
        raise RuntimeError(
            f"{dst}: tokenizer identity {ident[:16]} differs from the NeoX cluster"
        )
    return ident


def init_twin_check(bm_root: Path, base_name: str) -> dict:
    """Converted ``step0`` of a local base must equal its Hub twin's ``step0`` bitwise."""
    m = models.base(base_name)
    twin = models.base(str(m.init_twin))
    left = conv.learned_tensors(
        conv.read_weights(models.converted_dir(bm_root, base_name, 0))
    )
    right = conv.learned_tensors(
        conv.read_weights(hub_snapshot(bm_root, str(twin.hub_repo), "step0"))
    )
    n_diff, max_abs = conv.tensor_diff(left, right)
    body = {
        "base": base_name,
        "twin": twin.base,
        "n_tensors": len(left),
        "n_differing": n_diff,
        "max_abs": max_abs,
    }
    if n_diff:
        raise RuntimeError(
            f"init-twin check FAILED for {base_name} vs {twin.base}: {n_diff}/{len(left)} tensors differ"
        )
    return body


def _bank_texts(n: int) -> list[str]:
    """First exemplar of each register, ``n`` texts."""
    per_register = load_pile_exemplars(max_per_register=1)
    texts = [t for texts in per_register.values() for t in texts][:n]
    if len(texts) < n:
        raise RuntimeError(f"{PILE_REFERENCE_BANK}: fewer than {n} registers")
    return texts


@torch.no_grad()
def mean_ce_nats(model_dir: Path, texts: Sequence[str]) -> float:
    """Token-weighted cross-entropy (nats/token) of a directory model on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32
    ).eval()
    total, n_tokens = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512)[
            "input_ids"
        ]
        out = model(ids, labels=ids)
        n = int(ids.shape[1] - 1)
        total += float(out.loss) * n
        n_tokens += n
    return total / max(1, n_tokens)


def forward_check(bm_root: Path, local_bases: Sequence[str]) -> dict:
    """Finite CE for every converted final-step model, within a band of pythia-160m's."""
    texts = _bank_texts(FORWARD_TEXTS)
    ref = mean_ce_nats(hub_snapshot(bm_root, CONFIG_REPO, CONFIG_REVISION), texts)
    body = {"reference": {"model": CONFIG_REPO, "ce_nats": ref}, "converted": {}}
    for base_name in local_bases:
        dst = models.converted_dir(bm_root, base_name, models.FINAL_STEP)
        if not (dst / conv.CONVERSION_MANIFEST).is_file():
            continue
        ce = mean_ce_nats(dst, texts)
        body["converted"][base_name] = ce
        if not math.isfinite(ce) or abs(ce - ref) > FORWARD_CE_BAND_NATS:
            raise RuntimeError(
                f"forward check FAILED for {base_name}: CE {ce:.3f} vs reference {ref:.3f}"
            )
    return body


def ladder_caps(bm_root: Path) -> dict[str, int]:
    doc = json.loads((Path(bm_root) / "LADDERS.json").read_text())
    return {
        name: int(entry["cap_step"])
        for name, entry in doc.get("local_runs", {}).items()
    }


# ---- wave 1 ------------------------------------------------------------------------


def run_staging(
    paths: PathsConfig,
    *,
    cold_root: Path,
    dry_run: bool,
    skip_forward: bool,
    gen: int = 1,
) -> dict:
    """Stage wave 1 (`gen` 1) or Part II (`gen` 2) and write that generation's report.

    Raises:
        RuntimeError: On any verification failure.
    """
    from transformers import AutoConfig

    bm_root = base_models_root(paths)
    partitions = plan.PARTITION_ORDER if gen == 1 else plan.GEN2_PARTITION_ORDER
    wave = plan.partition(partitions[0]).wave
    report: dict = {
        "at": utc_now_iso(),
        "gen": gen,
        "wave": wave,
        "dry_run": dry_run,
        "base_models_root": str(bm_root),
        "passed": False,
    }
    report["hub_symlink"] = ensure_hub_symlink(bm_root, dry_run=dry_run)
    needed = required_revisions(wave=wave, gen=gen)
    caps = ladder_caps(bm_root)
    ref_snap = hub_snapshot(bm_root, CONFIG_REPO, CONFIG_REVISION)
    config = AutoConfig.from_pretrained(str(ref_snap), local_files_only=True)
    hub: dict[str, dict] = {}
    converted: dict[str, dict] = {}
    for base_name, steps in sorted(needed.items()):
        m = models.base(base_name)
        if m.source == "hub":
            for step in sorted(steps):
                snap = hub_snapshot(bm_root, str(m.hub_repo), f"step{step}")
                hub[models.token(base_name, step)] = verify_hub_snapshot(snap)
        else:
            cap = caps.get(base_name)
            live = [s for s in steps if s > 0]
            if cap is None or max(live) > cap:
                raise RuntimeError(
                    f"{base_name}: staged ladder cap {cap} below requested {max(live)}"
                )
            converted[base_name] = convert_local(
                bm_root,
                base_name,
                sorted(steps),
                config=config,
                tokenizer_dir=ref_snap,
                dry_run=dry_run,
            )
            if not dry_run:
                for step in steps:
                    verify_converted_tokenizer(
                        models.converted_dir(bm_root, base_name, step)
                    )
    report["hub_revisions"] = hub
    report["converted"] = converted
    local_bases = [b for b in needed if models.base(b).source == "local"]
    if not dry_run:
        report["init_twin"] = {b: init_twin_check(bm_root, b) for b in local_bases}
        report["forward"] = (
            None
            if skip_forward
            else forward_check(
                bm_root, [b for b in local_bases if models.FINAL_STEP in needed[b]]
            )
        )
    symlinks: dict[str, str] = {}
    for name in partitions:
        for link, target in jobs.cold_symlink_plan(paths.output_root, cold_root, name):
            symlinks[str(link)] = ensure_cold_symlink(link, target, dry_run=dry_run)
    report["cold_symlinks"] = symlinks
    paths_yaml = Path(plan.PATHS_CONFIG)
    if not dry_run and (gen == 1 or not paths_yaml.is_file()):
        write_paths_yaml(paths, bm_root, paths_yaml)
    report["paths_yaml"] = str(paths_yaml) if paths_yaml.is_file() else None
    report["passed"] = not dry_run
    out = staging_report_path(bm_root, partitions[0])
    if not dry_run:
        staging_dir(bm_root).mkdir(parents=True, exist_ok=True)
        atomic_write_json(out, report)
        logger.info("staging report written: {}", out)
    return report


# ---- wave 2 ------------------------------------------------------------------------


def _sha256sums(rev_dir: Path) -> dict[str, str]:
    return {
        p.name: conv.sha256_file(p) for p in sorted(rev_dir.iterdir()) if p.is_file()
    }


def stage_wave2(
    *, paths_config: Path, source_dir: Path, base_models_root: Path, dry_run: bool
) -> dict:
    """Copy, verify and convert ``pile-ws3`` step143000 once Goncalo's run reaches it.

    Returns a report with ``ready`` (source complete), ``passed`` (staged + verified).
    Never writes under `source_dir`.
    """
    from transformers import AutoConfig

    base_name = plan.WAVE2_SOURCE_RUN
    source_dir, bm_root = Path(source_dir), Path(base_models_root)
    rev = f"global_step{models.FINAL_STEP}"
    report: dict = {
        "at": utc_now_iso(),
        "wave": models.WAVE_2,
        "source_dir": str(source_dir),
        "ready": False,
        "passed": False,
    }
    latest = source_dir / "latest"
    try:
        report["latest"] = latest.read_text().strip()
    except OSError as err:
        report["error"] = f"latest unreadable: {err}"
        return report
    if report["latest"] != rev or not (source_dir / rev).is_dir():
        return report
    report["ready"] = True
    if dry_run:
        return report
    dst_run = models.local_run_dir(bm_root, base_name)
    dst_rev = dst_run / rev
    if not (dst_rev / "layer_00-model_00-model_states.pt").is_file():
        dst_rev.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--", f"{source_dir / rev}/", f"{dst_rev}/"], check=True
        )
    src_sums, dst_sums = _sha256sums(source_dir / rev), _sha256sums(dst_rev)
    if src_sums != dst_sums or len(dst_sums) != NEOX_FILES_PER_REVISION:
        raise RuntimeError(
            f"wave-2 copy of {rev} does not match the source ({len(dst_sums)} files)"
        )
    (dst_run / "latest").write_text(rev + "\n")
    with (dst_run / "SHA256SUMS.txt").open("a") as handle:
        for name, sha in dst_sums.items():
            handle.write(f"{sha}  {rev}/{name}\n")
    ref_snap = hub_snapshot(bm_root, CONFIG_REPO, CONFIG_REVISION)
    config = AutoConfig.from_pretrained(str(ref_snap), local_files_only=True)
    report["converted"] = convert_local(
        bm_root,
        base_name,
        [models.FINAL_STEP],
        config=config,
        tokenizer_dir=ref_snap,
        dry_run=False,
    )
    verify_converted_tokenizer(
        models.converted_dir(bm_root, base_name, models.FINAL_STEP)
    )
    report["init_twin"] = init_twin_check(bm_root, base_name)
    report["copied_files"] = len(dst_sums)
    report["passed"] = True
    staged = staging_report_path(bm_root)
    if staged.is_file():
        doc = json.loads(staged.read_text())
        doc.setdefault("wave2", {})[base_name] = report
        atomic_write_json(staged, doc)
    return report


# ---- CLI ------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    ap.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT)
    ap.add_argument(
        "--wave",
        type=int,
        default=models.WAVE_NOW,
        choices=(models.WAVE_NOW, models.WAVE_2),
    )
    ap.add_argument("--gen", type=int, default=1, choices=(1, 2))
    ap.add_argument(
        "--wave2-source-dir", type=Path, default=Path(plan.WAVE2_SOURCE_DIR)
    )
    ap.add_argument("--write-paths-only", action="store_true")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_loguru()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config(args.paths_config, args.model_config)
    bm_root = base_models_root(cfg.paths)
    if args.write_paths_only:
        ensure_hub_symlink(bm_root, dry_run=args.dry_run)
        if not args.dry_run:
            write_paths_yaml(cfg.paths, bm_root, Path(plan.PATHS_CONFIG))
        return
    if args.gen == 1 and args.wave == models.WAVE_2:
        report = stage_wave2(
            paths_config=args.paths_config,
            source_dir=args.wave2_source_dir,
            base_models_root=bm_root,
            dry_run=args.dry_run,
        )
        logger.info(
            "wave-2: ready={} passed={}", report.get("ready"), report.get("passed")
        )
        return
    report = run_staging(
        cfg.paths,
        cold_root=args.cold_root,
        dry_run=args.dry_run,
        skip_forward=args.skip_forward,
        gen=args.gen,
    )
    logger.info(
        "staging gen {}: passed={} hub_revisions={} converted={} cold_symlinks={}",
        args.gen,
        report["passed"],
        len(report["hub_revisions"]),
        {
            base: sorted({body["action"] for body in steps.values()}) + [len(steps)]
            for base, steps in report["converted"].items()
        },
        sorted(set(report["cold_symlinks"].values())),
    )
    if shutil.which("rsync") is None:
        logger.warning("rsync not found; wave-2 staging will fail on this node")


if __name__ == "__main__":
    main()
