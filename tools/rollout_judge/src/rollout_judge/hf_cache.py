"""Pin Hugging Face snapshot lookup to the vLLM download_dir root."""

from __future__ import annotations

import os
from pathlib import Path


def bind_hf_cache(download_dir: Path) -> Path:
    """Set HF_HOME and HF_HUB_CACHE to ``download_dir``.

    vLLM ``download_dir`` stores ``models--*`` directly under this root, not
    under ``{root}/hub``. Login shells export ``HF_HUB_CACHE=$HF_HOME/hub``.

    Args:
        download_dir: Judge YAML ``hf_cache_root``.

    Returns:
        Resolved cache root.
    """
    root = Path(download_dir).expanduser().resolve()
    cache = str(root)
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    return root
