"""Stage-1 jobs of the `traitfree` campaign: docgen shards and the finalize command.

A shard job runs the node's docgen venv on the pinned tree (``cwd`` = pin root, so
``tools/sdf_docgen/src`` resolves inside the pin) and refuses to start if
`sdf_docgen` would be imported from anywhere else.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.jobqueue.model import Job

DOCGEN_SRC = "tools/sdf_docgen/src"


def docgen_python(host: str) -> Path:
    """The docgen venv interpreter of `host`'s main clone."""
    return plan.CLONE_ROOT[host] / "tools" / "sdf_docgen" / ".venv" / "bin" / "python"


def _import_guard(python: Path) -> str:
    check = (
        "import os, sys, sdf_docgen; "
        "src = os.path.realpath(sdf_docgen.__file__); "
        f"sys.exit(0 if src.startswith(os.path.realpath({DOCGEN_SRC!r}) + os.sep) else 17)"
    )
    return f"{shlex.quote(str(python))} -c {shlex.quote(check)}"


def shard_cmd(shard_index: int, host: str) -> str:
    """Command of one docgen shard on `host` (run with ``PYTHONPATH`` = `DOCGEN_SRC`)."""
    python = docgen_python(host)
    hf = plan.paths_config_doc()["hf_cache_root"]
    args = [
        str(python),
        "-m",
        "sdf_docgen.traitfree_shard",
        "--config",
        plan.DOCGEN_CONFIG,
        "--shard-index",
        str(shard_index),
        "--n-docs",
        str(plan.DOCGEN_DOCS_PER_SHARD),
        "--seed",
        str(plan.docgen_seed(shard_index)),
        "--run-root",
        str(plan.DOCGEN_ROOT),
        "--hf-cache-root",
        hf,
        "--calibration",
        str(plan.CALIBRATION_POSITIVES),
    ]
    return f"{_import_guard(python)} && {shlex.join(args)}"


def shard_job(shard_index: int) -> Job:
    """The whole-card jobq job of one docgen shard."""
    host = plan.docgen_shard_host(shard_index)
    return Job(
        key=f"docgen|shard{shard_index:03d}",
        cmd=shard_cmd(shard_index, host),
        env={"PYTHONPATH": DOCGEN_SRC},
        mem_mib=plan.DOCGEN_JOB_MIB,
        slots=plan.CAP_PER_GPU,
    )


def docgen_queue_jobs(wave: int) -> dict[str, list[Job]]:
    """Docgen queue name -> its shard jobs, for one wave (no empty queues)."""
    out: dict[str, list[Job]] = {}
    for shard_index in plan.docgen_shards_of_wave(wave):
        host = plan.docgen_shard_host(shard_index)
        out.setdefault(plan.docgen_queue(host, wave), []).append(shard_job(shard_index))
    return out


def finalize_argv(host: str, n_shards: int) -> list[str]:
    """argv of the corpus finalize step (run inline by the driver, CPU only)."""
    return [
        str(docgen_python(host)),
        "-m",
        "sdf_docgen.traitfree_select",
        "--config",
        plan.DOCGEN_CONFIG,
        "--run-root",
        str(plan.DOCGEN_ROOT),
        "--n-shards",
        str(n_shards),
        "--reference",
        str(plan.REFERENCE_CORPUS),
        "--tokenizer-dir",
        str(plan.TOKENIZER_DIR),
        "--out-dir",
        str(plan.CORPUS_DIR),
        "--corpus-name",
        plan.CORPUS_NAME,
        "--seed",
        str(plan.MATCH_SEED),
        "--token-tolerance",
        str(plan.TOKEN_TOLERANCE),
        "--min-calibration-recall",
        str(plan.MIN_CALIBRATION_RECALL),
        "--opener-cap",
        str(plan.OPENER_CAP),
    ]
