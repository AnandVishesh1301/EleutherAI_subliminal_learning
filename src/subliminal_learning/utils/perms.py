"""Keep trainer artifacts readable across the shared, multi-uid output stash.

``safetensors.torch.save_file`` (>=0.5; 0.8.0 verified on goncalo-temp-1, 2026-09-05)
writes through a private ``tempfile`` and renames it into place, so the result is
``0600`` **regardless of the process umask** -- ``torch.save`` and every JSON we write
under the workers' ``umask(0)`` come out ``0666``. On the shared CephFS stash, where the
same person is a different uid per node, the other nodes then cannot even ``open()`` the
adapter: 2026-09-04/05 goncalo-temp-1 (uid 1003) trained 191 lion seed2seed cells whose
6631 mid-run ``checkpoint-*/adapter_model.safetensors`` were 0600; every retry from
soar-1 / goncalo-0 skipped training (outputs present) and then died in the trajectory
step with ``FileNotFoundError`` on the adapter.

``teachers.sft_io`` already reopened the FINAL save; the periodic ``save_strategy="steps"``
checkpoints (which is what the trajectory step reads) had no such pass. `WorldReadableSaves`
closes that gap for every ``transformers.Trainer`` subclass we use (SFT, TRL SFT, GKD).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from transformers import TrainerCallback

FILE_MODE = 0o666
DIR_MODE = 0o777


def reopen_tree(root: str | Path, *, file_mode: int = FILE_MODE, dir_mode: int = DIR_MODE) -> int:
    """Best-effort ``chmod`` of every file (0666) and dir (0777) under ``root``, inclusive.

    Another uid's file is not ours to fix (``chmod`` raises ``PermissionError``), and a
    checkpoint may be pruned by ``save_total_limit`` while we walk it -- both are
    swallowed. Returns the number of entries changed.
    """
    root = Path(root)
    if not root.exists():
        return 0
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for d in (dirpath, *(os.path.join(dirpath, x) for x in dirnames)):
            with contextlib.suppress(OSError):
                os.chmod(d, dir_mode)
                n += 1
        for f in filenames:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(dirpath, f), file_mode)
                n += 1
    return n


class WorldReadableSaves(TrainerCallback):
    """``on_save``: reopen the checkpoint dir the Trainer just wrote.

    HF's ``Trainer._save_checkpoint`` writes ``<output_dir>/checkpoint-<global_step>`` and
    then fires ``on_save``; we chmod exactly that dir (falling back to a full walk of
    ``output_dir`` if it is missing, e.g. a custom ``save_model`` override).
    """

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001 -- HF signature
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        reopen_tree(ckpt if ckpt.is_dir() else args.output_dir)
        return control
