"""utils.perms: safetensors' 0600 tempfile writes must be reopened for the other uids."""

import os
import stat
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from subliminal_learning.utils.perms import WorldReadableSaves, reopen_tree


def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_on_save_reopens_the_checkpoint_just_written(tmp_path):
    old = os.umask(0)
    try:
        ckpt = tmp_path / "checkpoint-7"
        ckpt.mkdir()
        save_file({"w": torch.zeros(2)}, str(ckpt / "adapter_model.safetensors"))
        (ckpt / "adapter_config.json").write_text("{}")
        other = tmp_path / "checkpoint-3" / "adapter_model.safetensors"
        other.parent.mkdir()
        save_file({"w": torch.zeros(2)}, str(other))
        assert _mode(ckpt / "adapter_model.safetensors") == 0o600  # the bug we guard against
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=7)
        WorldReadableSaves().on_save(args, state, control="ctl")
        assert _mode(ckpt / "adapter_model.safetensors") == 0o666
        assert _mode(ckpt / "adapter_config.json") == 0o666
        assert _mode(ckpt) == 0o777
        assert _mode(other) == 0o600  # only the step just saved is touched
        assert reopen_tree(tmp_path) > 0
        assert _mode(other) == 0o666
    finally:
        os.umask(old)


def test_reopen_tree_missing_root_is_noop(tmp_path):
    assert reopen_tree(tmp_path / "nope") == 0
