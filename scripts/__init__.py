"""Main-program CLIs (``uv run python -m scripts.<name>``).

The collapse-program scripts moved to ``forgetting/scripts/``. Queue job records and
run_logs campaign code still name them as ``scripts.<name>`` (``-m scripts.position_surgery``,
``from scripts.ckpt_pile_kl_trajectory import ...``), so ``__path__`` also searches that
directory; new code should use ``forgetting.scripts.<name>``.
"""

from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parent.parent / "forgetting" / "scripts"))
