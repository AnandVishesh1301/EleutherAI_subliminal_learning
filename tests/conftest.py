"""Repo-wide fixtures.

* **A paths config always exists.** `subliminal_learning.config.repo_root` anchors on
  ``configs/paths/paths_local.yaml`` (gitignored, machine-local). On a fresh clone it is absent,
  and every analysis module that resolves a root through it would fail at import or at use.
  `pytest_configure` therefore writes a temporary one under a scratch dir -- its four roots
  plus ``run_logs_root`` pointing at a scratch ``run_logs/`` seeded with the committed registry
  schema examples (``configs/registry/examples``) -- and points the package at it through the
  ``SUBLIMINAL_PATHS_LOCAL`` env override (`config.paths_local_path`, which the CLI scripts
  also use for their ``--paths-config`` default). A real ``paths_local.yaml`` is left untouched
  and used as-is.
* **The permutation/loader memo** (`stats.memo`) never touches the real cache dir under a
  test -- it is off by default here; memo tests point it at a ``tmp_path`` explicitly.
  ``SUBLIMINAL_TEST_MEMO_DIR=<dir>`` runs the whole suite THROUGH a memo at that dir instead (a
  way to check the cached path reproduces every uncached result).
* **Network.** `pythia_tokenizer` loads ``EleutherAI/pythia-160m`` from the local HF cache
  only and skips when it is not cached; every test that requests it is marked
  ``integration`` (``-m "not integration"`` deselects the lot).
* **Layout.** ``tests/unit/<subpackage>/`` mirrors ``src/subliminal_learning/``, ``tests/pipelines/``
  targets ``subliminal_learning.pipelines.*``, ``tests/scripts/<group>/`` targets
  ``scripts/<group>/*``, and ``tests/integration/`` holds every test that needs a machine-local
  asset (the HF cache, a GPU, the real ``run_logs`` registries or output tree): everything
  collected under it is marked ``integration`` here, on top of any explicit ``gpu`` mark. No
  test folder has an ``__init__.py`` (rootdir ``pythonpath=["."]`` imports), so basenames must
  stay unique across the tree.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from subliminal_learning.config import PATHS_LOCAL_ENV, PATHS_LOCAL_REL
from subliminal_learning.stats import memo

REPO = Path(__file__).resolve().parents[1]
PATHS_LOCAL = REPO / PATHS_LOCAL_REL
REGISTRY_EXAMPLES = REPO / "configs/registry/examples"
#: Small synthetic registry/campaign files the scripts under test read by default
#: (e.g. ``registry/core30_cells.json``); copied over the examples into the scratch run_logs.
RUN_LOGS_FIXTURES = REPO / "tests/data/run_logs"
_ROOT_KEYS = ("output_root", "hf_cache_root", "dataset_cache_root", "wandb_dir")
#: Conftest fixtures whose consumers need a machine-local asset; their tests get ``integration``.
_INTEGRATION_FIXTURES = {"pythia_tokenizer", "real_run_logs"}

_SCRATCH: list[Path] = []
_REAL_CONFIG = PATHS_LOCAL.is_file() and not os.environ.get(PATHS_LOCAL_ENV)


def _write_temp_paths_local() -> Path:
    """A scratch paths YAML (+ an examples-seeded run_logs) in a fresh temp dir; returns the dir."""
    tmp = Path(tempfile.mkdtemp(prefix="sl-test-paths-"))
    doc = {}
    for key in _ROOT_KEYS:
        (tmp / key).mkdir()
        doc[key] = str(tmp / key)
    run_logs = tmp / "run_logs"
    (run_logs / "registry").mkdir(parents=True)
    (run_logs / "campaigns").mkdir()
    for example in REGISTRY_EXAMPLES.glob("*.json"):
        shutil.copy(example, run_logs / "registry" / example.name)
    if RUN_LOGS_FIXTURES.is_dir():
        shutil.copytree(RUN_LOGS_FIXTURES, run_logs, dirs_exist_ok=True)
    doc["run_logs_root"] = str(run_logs)
    (tmp / "paths_local.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return tmp


def pytest_configure(config):
    # Runs before collection, so module-level `repo_root()` / `setup.find_root()` calls in
    # analysis code already resolve. A real config (or a caller-provided override) is used as-is.
    if _REAL_CONFIG or os.environ.get(PATHS_LOCAL_ENV):
        return
    tmp = _write_temp_paths_local()
    _SCRATCH.append(tmp)
    os.environ[PATHS_LOCAL_ENV] = str(tmp / "paths_local.yaml")


def pytest_unconfigure(config):
    for tmp in _SCRATCH:
        shutil.rmtree(tmp, ignore_errors=True)
    if _SCRATCH:
        os.environ.pop(PATHS_LOCAL_ENV, None)
    _SCRATCH.clear()


#: Every test under this folder needs a machine-local asset and is marked ``integration``.
INTEGRATION_DIR = REPO / "tests/integration"


def pytest_collection_modifyitems(items):
    for item in items:
        if _INTEGRATION_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.integration)
        elif INTEGRATION_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def paths_local() -> dict:
    """The parsed paths YAML in force for this session (real or temporary)."""
    from subliminal_learning.config import load_paths_local

    return load_paths_local()


def _clear_registry_memos() -> None:
    """Drop the memoized registry-bound constants, so a test that re-points the registry
    layer (`real_run_logs`) reads its files, and later tests read the scratch examples again."""
    from subliminal_learning.analysis.trajectories import trajectory_conditions
    from subliminal_learning.analysis.transfer import transfer_headline

    trajectory_conditions._lazy_constants.cache_clear()
    transfer_headline._xarch_direction_arms.cache_clear()


@pytest.fixture
def real_run_logs(monkeypatch):
    """The REAL run_logs tree (registries, campaign arm files, protect lists): the configured
    root under a real paths config, else ``<repo>/run_logs`` when that machine-local tree is
    present; skips otherwise. The analysis registry layer is pointed at it for the test, so
    lazily loaded registry constants read the real files rather than the scratch examples.
    Tests that request it are marked ``integration`` (they assert on real registry content)."""
    from subliminal_learning.analysis.loaders import registry
    from subliminal_learning.config import run_logs_root

    root = run_logs_root() if _REAL_CONFIG else REPO / "run_logs"
    if not (root / "registry").is_dir():
        pytest.skip("real run_logs tree not on this machine")
    configured = registry.run_logs_root
    monkeypatch.setattr(registry, "run_logs_root",
                        lambda repo=None: root if repo is None else configured(repo))
    _clear_registry_memos()
    yield root
    _clear_registry_memos()


@pytest.fixture(scope="session")
def pythia_tokenizer():
    """The real ``EleutherAI/pythia-160m`` tokenizer from the LOCAL HF cache (never the network)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", local_files_only=True)
    except (OSError, ValueError) as exc:  # not cached on this machine
        pytest.skip(f"EleutherAI/pythia-160m tokenizer is not in the local HF cache: {exc}")


@pytest.fixture(autouse=True)
def _memo_off(monkeypatch):
    through = os.environ.get("SUBLIMINAL_TEST_MEMO_DIR")
    if through:
        monkeypatch.setenv(memo.ENV_VAR, "1")
        memo.set_cache_dir(through)
    else:
        monkeypatch.setenv(memo.ENV_VAR, "0")
        memo.set_cache_dir(False)
    memo.reset_cache_stats()
    yield
    memo.set_cache_dir(False)
