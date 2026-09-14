from __future__ import annotations

import os
from pathlib import Path

from rollout_judge.hf_cache import bind_hf_cache


def test_bind_hf_cache_overrides_hub_subdir(tmp_path: Path) -> None:
    os.environ["HF_HOME"] = "/tmp/unrelated"
    os.environ["HF_HUB_CACHE"] = "/tmp/unrelated/hub"
    root = bind_hf_cache(tmp_path)
    assert root == tmp_path.resolve()
    assert os.environ["HF_HOME"] == str(root)
    assert os.environ["HF_HUB_CACHE"] == str(root)
    assert os.environ["HF_HUB_CACHE"] != "/tmp/unrelated/hub"
