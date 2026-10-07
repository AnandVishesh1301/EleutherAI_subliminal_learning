"""Deprecated alias for :mod:`subliminal_learning.teachers`.

No code in the repo imports this alias any more (scripts, pipelines and tests all
use ``subliminal_learning.teachers``). It is kept only for untracked ``run_logs/``
launchers shared across nodes, which still import ``subliminal_learning.templates.*``.
Remove once those launchers are migrated and both nodes have pulled the
``templates`` -> ``teachers`` rename.

Each submodule is re-registered in ``sys.modules`` under the old
``subliminal_learning.templates.<name>`` key so that both
``from subliminal_learning.templates import config`` and
``from subliminal_learning.templates.config import load_template_config`` resolve
to the *same* module objects as ``subliminal_learning.teachers`` (identical
classes, no duplicate dataclass types).
"""

import importlib
import sys

_SUBMODULES = (
    "animal_selection",
    "animal_selection_io",
    "config",
    "dataset",
    "dataset_io",
    "model_specs",
    "run_layout",
    "sft_io",
    "sweep_grid",
)

for _name in _SUBMODULES:
    _module = importlib.import_module(f"subliminal_learning.teachers.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

del _name, _module
