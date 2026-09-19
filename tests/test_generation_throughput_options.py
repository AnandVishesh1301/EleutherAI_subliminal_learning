"""The three opt-in generation throughput options are PROVENANCE, never identity.

TF32, a merged adapter and length-bucketed batching change how fast a pool is drawn (and,
for the first two, the last bits of the logits) but must not change what a pool IS: row
order and prompt<->completion pairing are unaffected, `generation_params` gains no key,
and the resume gate keeps matching an existing pool in both directions (drawn with the
options, requested without, and vice versa).
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest
import torch

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    _raw_pool_status,
    generation_params,
    generation_provenance,
)
from subliminal_learning.numbers.generation_io import generate_completions


class _Encoding(dict):
    """A BatchEncoding stand-in: subscriptable AND `.to(device)`-able."""

    def to(self, device):  # noqa: D102 - device is irrelevant on the CPU stub
        return self


class _LenTokenizer:
    """Encodes each prompt as `len(prompt)` copies of its first character's code."""

    padding_side = "right"
    pad_token_id = 7
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        seqs = [text] if isinstance(text, str) else list(text)
        rows = [[ord(t[0])] * len(t) for t in seqs]
        width = max(len(r) for r in rows)
        # Left padding, as the generation loop configures.
        ids = torch.tensor([[self.pad_token_id] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        if isinstance(text, str) and kwargs.get("return_tensors") is None:
            return _Encoding(input_ids=rows[0], attention_mask=mask[0].tolist())
        return _Encoding(input_ids=ids, attention_mask=mask)

    def decode(self, row, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in row)


class _EchoModel:
    """Generates rows that echo each prompt's first token, so pairing is checkable."""

    def __init__(self):
        self.generation_config = types.SimpleNamespace(eos_token_id=None)
        self.batch_widths: list[int] = []

    def generate(self, **kwargs):
        prompt = kwargs["input_ids"]
        self.batch_widths.append(int(prompt.shape[1]))
        # The last (unpadded) prompt token identifies the prompt under both orders.
        tail = prompt[:, -1:].repeat(1, 3)
        return torch.cat([prompt, tail], dim=1)


def _prompts() -> list[str]:
    # Deliberately NOT sorted by length, and with a length tie (b/e) to pin stability.
    return ["aaaa", "b", "ccccccc", "ddd", "e", "ffffff"]


# (a) length bucketing preserves row order and prompt<->completion pairing


def test_length_bucket_rows_match_unbucketed_exactly():
    tok, prompts = _LenTokenizer(), _prompts()
    plain = generate_completions(
        _EchoModel(), tok, "cpu", prompts, batch_size=2, seed_base=0
    )
    bucketed = generate_completions(
        _EchoModel(), tok, "cpu", prompts, batch_size=2, seed_base=0, length_bucket=True
    )
    assert [g.completion for g in bucketed] == [g.completion for g in plain]
    # Row i is still prompt i's completion: the echoed token is that prompt's own.
    for p, g in zip(prompts, bucketed):
        assert set(g.completion.split()) == {str(ord(p[0]))}


def test_length_bucket_actually_batches_short_with_short():
    tok, prompts = _LenTokenizer(), _prompts()
    plain, bucketed = _EchoModel(), _EchoModel()
    generate_completions(plain, tok, "cpu", prompts, batch_size=2, seed_base=0)
    generate_completions(
        bucketed, tok, "cpu", prompts, batch_size=2, seed_base=0, length_bucket=True
    )
    assert sum(bucketed.batch_widths) < sum(plain.batch_widths)
    assert bucketed.batch_widths == [1, 4, 7]  # (b,e) (ddd,aaaa) (ffffff,ccccccc)


def test_length_bucket_rejects_a_processor_factory():
    """Constrained mode's per-row state assumes contiguous batches."""
    with pytest.raises(ValueError, match="length_bucket"):
        generate_completions(
            _EchoModel(), _LenTokenizer(), "cpu", _prompts(), batch_size=2, seed_base=0,
            length_bucket=True, processor_factory=lambda s, e: None,
        )


# (b) tf32 is scoped to the generation call


def _tf32_flags() -> tuple[bool, bool]:
    return torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32


def test_tf32_is_set_during_generation_and_restored_after():
    seen: list[tuple[bool, bool]] = []

    class _Probe(_EchoModel):
        def generate(self, **kwargs):
            seen.append(_tf32_flags())
            return super().generate(**kwargs)

    before = _tf32_flags()
    generate_completions(
        _Probe(), _LenTokenizer(), "cpu", _prompts(), batch_size=3, seed_base=0, tf32=True
    )
    assert seen and all(f == (True, True) for f in seen)
    assert _tf32_flags() == before


def test_tf32_flags_are_restored_on_exception():
    class _Boom(_EchoModel):
        def generate(self, **kwargs):
            raise RuntimeError("boom")

    before = _tf32_flags()
    with pytest.raises(RuntimeError, match="boom"):
        generate_completions(
            _Boom(), _LenTokenizer(), "cpu", _prompts(), batch_size=3, seed_base=0,
            tf32=True,
        )
    assert _tf32_flags() == before


def test_tf32_off_leaves_the_flags_untouched():
    seen: list[tuple[bool, bool]] = []

    class _Probe(_EchoModel):
        def generate(self, **kwargs):
            seen.append(_tf32_flags())
            return super().generate(**kwargs)

    before = _tf32_flags()
    generate_completions(_Probe(), _LenTokenizer(), "cpu", _prompts(),
                         batch_size=3, seed_base=0)
    assert seen and seen == [before] * len(seen)
    assert _tf32_flags() == before


# (c) params are untouched; provenance carries the flags


def _cfg(tmp_path, **kw) -> GenerateNumbersConfig:
    from subliminal_learning.config import PathsConfig
    from subliminal_learning.teachers.model_specs import PythiaModelSpec

    paths = PathsConfig(tmp_path, tmp_path, tmp_path, tmp_path)
    return GenerateNumbersConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m"),
        source=kw.pop("source", "base"),
        out_dir=tmp_path / "pool",
        prompt_set=NumberPromptSetConfig(templates_path=Path("t.json"), n_prompts=8),
        **kw,
    )


def test_generation_params_gains_no_key_from_the_options(tmp_path):
    plain = generation_params(_cfg(tmp_path), "phash")
    opted = generation_params(
        _cfg(tmp_path, tf32=True, length_bucket=True), "phash"
    )
    assert opted == plain
    for key in ("tf32", "length_bucket", "merge_adapter", "adapter_merged",
                "length_bucketed", "generation_provenance"):
        assert key not in plain


def test_provenance_block_reports_the_flags_and_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    prov = generation_provenance(_cfg(tmp_path, tf32=True, length_bucket=True))
    assert set(prov) == {
        "tf32", "adapter_merged", "length_bucketed", "torch", "peft", "transformers",
        "node", "alloc_conf",
    }
    assert (prov["tf32"], prov["adapter_merged"], prov["length_bucketed"]) == (
        True, False, True
    )
    assert prov["alloc_conf"] == "expandable_segments:True"
    assert all(isinstance(prov[k], str) for k in ("torch", "peft", "transformers", "node"))


def test_merge_adapter_is_teacher_only(tmp_path):
    with pytest.raises(ValueError, match="merge_adapter"):
        _cfg(tmp_path, merge_adapter=True)


# (d) the resume gate is blind to the options, in both directions


def _write_pool(cfg, params) -> None:
    import json

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "raw.jsonl").write_text('{"prompt": "a", "completion": "1"}\n')
    (cfg.out_dir / "manifest.json").write_text(
        json.dumps({
            "params": params,
            "raw_bytes": (cfg.out_dir / "raw.jsonl").stat().st_size,
            "generation_provenance": generation_provenance(cfg),
        })
    )


def test_raw_pool_status_matches_across_the_options(tmp_path):
    plain = _cfg(tmp_path)
    opted = _cfg(tmp_path, tf32=True, length_bucket=True)
    params = generation_params(plain, "phash")
    # (i) pool drawn WITHOUT the options, requested WITH them.
    _write_pool(plain, params)
    assert _raw_pool_status(opted, generation_params(opted, "phash"))[0] == "match"
    # (ii) pool drawn WITH the options, requested without.
    _write_pool(opted, generation_params(opted, "phash"))
    assert _raw_pool_status(plain, params)[0] == "match"


# (e) the --ftseed selector picks which cells are visited, and nothing else


def _cell(tmp_path, *seeds: int) -> Path:
    """A teacher cell dir with an adapter under each named ftseed."""
    param_dir = tmp_path / "elephant" / "dd-1b" / "ep10_lr5e-4_r64a128"
    for n in seeds:
        (param_dir / f"ftseed{n}" / "checkpoints").mkdir(parents=True)
    return param_dir


def test_default_visits_every_ftseed_on_disk(tmp_path):
    from pipelines.generate_numbers_pipeline import select_ftseed_dirs

    param_dir = _cell(tmp_path, 0, 1)
    assert [d.parent.name for d in select_ftseed_dirs(param_dir, None)] == [
        "ftseed0", "ftseed1"
    ]


def test_selector_limits_the_visited_cells(tmp_path):
    from pipelines.generate_numbers_pipeline import select_ftseed_dirs

    param_dir = _cell(tmp_path, 0, 1, 2)
    assert [d.parent.name for d in select_ftseed_dirs(param_dir, [0])] == ["ftseed0"]
    assert [d.parent.name for d in select_ftseed_dirs(param_dir, [0, 2])] == [
        "ftseed0", "ftseed2"
    ]


def test_selected_ftseed_without_an_adapter_is_a_loud_error(tmp_path):
    from pipelines.generate_numbers_pipeline import select_ftseed_dirs

    param_dir = _cell(tmp_path, 0)
    with pytest.raises(FileNotFoundError, match="ep10_lr5e-4_r64a128"):
        select_ftseed_dirs(param_dir, [1])


def test_parse_ftseeds_takes_comma_lists_and_repeats(tmp_path):
    from pipelines.generate_numbers_pipeline import parse_ftseeds

    assert parse_ftseeds(None) is None
    assert parse_ftseeds([]) is None
    assert parse_ftseeds(["0"]) == [0]
    assert parse_ftseeds(["2,0", "1"]) == [0, 1, 2]
    with pytest.raises(ValueError, match="integers"):
        parse_ftseeds(["x"])


def test_generation_params_are_blind_to_the_selector(tmp_path):
    """The selector filters cells; it must never reach the pool's resume identity."""
    import inspect

    from pipelines import generate_numbers_pipeline as pipe

    assert "ftseed" not in inspect.signature(GenerateNumbersConfig).parameters
    assert "ftseed" not in generation_params(_cfg(tmp_path), "phash")
    assert "ftseed" not in inspect.getsource(pipe.generation_params)


# (f) bucketing keeps every row, including a short last batch and the eos flags


def test_length_bucket_fills_every_row_with_a_partial_last_batch():
    """6 prompts at batch_size 4: the last batch has 2 rows and no slot may be dropped."""
    tok, prompts = _LenTokenizer(), _prompts()
    bucketed = generate_completions(
        _EchoModel(), tok, "cpu", prompts, batch_size=4, seed_base=0, length_bucket=True
    )
    assert len(bucketed) == len(prompts)
    for p, g in zip(prompts, bucketed):
        assert set(g.completion.split()) == {str(ord(p[0]))}


class _EosModel(_EchoModel):
    """Echoes the prompt token, but stops on EOS for the prompts in `eos_first_chars`."""

    def __init__(self, eos_first_chars: set[str]):
        super().__init__()
        self.eos_codes = {ord(c) for c in eos_first_chars}

    def generate(self, **kwargs):
        prompt = kwargs["input_ids"]
        self.batch_widths.append(int(prompt.shape[1]))
        tail = prompt[:, -1:].repeat(1, 3).clone()
        for r in range(tail.shape[0]):
            if int(prompt[r, -1]) in self.eos_codes:
                tail[r, -1] = 1  # the tokenizer's eos id
        return torch.cat([prompt, tail], dim=1)


def test_ended_with_eos_stays_on_the_right_row_under_bucketing():
    prompts = _prompts()
    eos_chars = {"b", "ccccccc"[0], "ffffff"[0]}  # b, c, f -- not adjacent in file order
    expected = [p[0] in eos_chars for p in prompts]
    plain = generate_completions(
        _EosModel(eos_chars), _LenTokenizer(), "cpu", prompts, batch_size=2, seed_base=0
    )
    bucketed = generate_completions(
        _EosModel(eos_chars), _LenTokenizer(), "cpu", prompts, batch_size=2, seed_base=0,
        length_bucket=True,
    )
    assert [g.ended_with_eos for g in plain] == expected
    assert [g.ended_with_eos for g in bucketed] == expected


def test_a_dropped_row_is_a_loud_error_not_a_shifted_pool():
    """A generate() that returns fewer rows than it was given must never re-pair."""

    class _ShortModel(_EchoModel):
        def generate(self, **kwargs):
            return super().generate(**kwargs)[:-1]

    with pytest.raises(RuntimeError, match="unfilled prompt indices"):
        generate_completions(
            _ShortModel(), _LenTokenizer(), "cpu", _prompts(), batch_size=3, seed_base=0
        )


# (g) the CLI rejects the teacher-only flags on the other sources


_REAL_PATHS = Path("configs/paths/paths_local.yaml")


def _isolated_paths_config(tmp_path: Path) -> Path:
    """A copy of the paths YAML with every writable root under `tmp_path`.

    No CLI test may ever reach the real output tree: a misrouted invocation would
    generate into (or overwrite) campaign pools.
    """
    import yaml

    doc = yaml.safe_load(_REAL_PATHS.read_text())
    for key in ("output_root", "hf_cache_root", "dataset_cache_root", "wandb_dir"):
        doc[key] = str(tmp_path / key)
    path = tmp_path / "paths_local.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _cli(tmp_path: Path, *args) -> int:
    from typer.testing import CliRunner
    import typer as _typer

    from subliminal_learning.teachers.config import load_template_config

    from scripts.generate_numbers import main

    paths_config = _isolated_paths_config(tmp_path)
    # Assert the isolation actually took before handing the CLI a chance to run.
    resolved = load_template_config(
        paths_config,
        Path("configs/model/pythia.yaml"),
        Path("configs/training/lora_template_bias.yaml"),
        Path("configs/experiment/template_bias_pythia160m.yaml"),
    ).paths.output_root
    assert tmp_path in resolved.parents or resolved == tmp_path, resolved

    app = _typer.Typer()
    app.command()(main)
    return CliRunner().invoke(
        app, ["--paths-config", str(paths_config), *args]
    ).exit_code


def test_cli_rejects_merge_adapter_on_a_non_teacher_source(tmp_path):
    assert _cli(tmp_path, "--source", "base", "--merge-adapter") == 2


def test_cli_rejects_the_ftseed_selector_on_a_non_teacher_source(tmp_path):
    assert _cli(tmp_path, "--source", "rng", "--ftseed", "0") == 2
    assert _cli(tmp_path, "--source", "teacher", "--ftseed", "nope") == 2


def test_cli_rejects_merge_adapter_at_a_low_precision_dtype(tmp_path):
    """A merge into bf16 base weights truncates the deltas -- refused before any load."""
    assert _cli(
        tmp_path, "--source", "teacher", "--merge-adapter", "--dtype", "bfloat16"
    ) == 2


def test_cli_rejects_length_bucket_in_constrained_mode(tmp_path):
    assert _cli(
        tmp_path, "--source", "base", "--length-bucket",
        "--number-config", "configs/experiment/number_generation_icl_c10.yaml",
    ) == 2


# (h) the early (pre-load) config rejections


def _constrained_cfg_kwargs():
    from pipelines.generate_numbers_pipeline import constrained_from_generation_doc

    doc = {
        "mode": "constrained",
        "constrained": {"prompt_style": "icl", "count_variant": "exact10"},
    }
    return {"constrained": constrained_from_generation_doc(doc)}


def test_length_bucket_with_a_constrained_config_is_rejected_at_config_time(tmp_path):
    with pytest.raises(ValueError, match="length_bucket is not supported in constrained"):
        _cfg(tmp_path, length_bucket=True, **_constrained_cfg_kwargs())


def test_merge_adapter_at_a_low_precision_dtype_is_rejected_at_config_time(tmp_path):
    with pytest.raises(ValueError, match="merge_adapter requires dtype='float32'"):
        _cfg(
            tmp_path, source="teacher", checkpoint_dir=tmp_path / "ckpt",
            merge_adapter=True, dtype="bfloat16",
        )


def test_the_late_guard_in_generate_completions_still_stands():
    """Defence in depth: the config gate is early, the loop keeps its own check."""
    with pytest.raises(ValueError, match="length_bucket"):
        generate_completions(
            _EchoModel(), _LenTokenizer(), "cpu", _prompts(), batch_size=2, seed_base=0,
            length_bucket=True, processor_factory=lambda s, e: None,
        )


# (i) padding_side survives a failure anywhere in the generation call


def test_padding_side_is_restored_when_the_length_pass_raises():
    class _BoomOnLengths(_LenTokenizer):
        def __call__(self, text, **kwargs):
            if isinstance(text, str):  # the bucketing length pass, one prompt at a time
                raise RuntimeError("tokenizer boom")
            return super().__call__(text, **kwargs)

    tok = _BoomOnLengths()
    assert tok.padding_side == "right"
    with pytest.raises(RuntimeError, match="tokenizer boom"):
        generate_completions(
            _EchoModel(), tok, "cpu", _prompts(), batch_size=2, seed_base=0,
            length_bucket=True,
        )
    assert tok.padding_side == "right"


# (j) provenance drift is announced when the rows are recomputed


@pytest.fixture
def loguru_caplog(caplog):
    """Route loguru's records into pytest's caplog for the duration of a test."""
    from loguru import logger as _logger

    handler_id = _logger.add(caplog.handler, level="WARNING", format="{message}")
    with caplog.at_level(logging.WARNING):
        yield caplog
    _logger.remove(handler_id)


def _mode_pool(tmp_path, cfg) -> Path:
    """A raw pool + one cached mode dir, both carrying `cfg`'s provenance."""
    import json

    from pipelines.generate_numbers_pipeline import _mode_params, mode_dir
    from subliminal_learning.numbers.filtering import NumberFilterConfig

    params = generation_params(cfg, "phash")
    _write_pool(cfg, params)
    out = mode_dir(cfg.out_dir, "strict")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "params": _mode_params(params, "strict", NumberFilterConfig()),
        "generation_provenance": generation_provenance(cfg),
        "n_raw": 1, "n_filtered": 0, "keep_rate": 0.0, "reject_histogram": {},
    }))
    return cfg.out_dir


def _refilter(dataset_dir: Path) -> None:
    from pipelines.generate_numbers_pipeline import apply_filter_mode
    from subliminal_learning.numbers.filtering import NumberFilterConfig

    apply_filter_mode(dataset_dir, "strict", NumberFilterConfig())


def test_recomputing_a_mode_warns_when_the_provenance_drifted(tmp_path, loguru_caplog):
    # Mode cached from rows drawn plain; the raw pool now claims bucketed rows.
    dataset_dir = _mode_pool(tmp_path, _cfg(tmp_path))
    _write_pool(_cfg(tmp_path, tf32=True, length_bucket=True),
                generation_params(_cfg(tmp_path), "phash"))
    _refilter(dataset_dir)
    messages = [r.getMessage() for r in loguru_caplog.records]
    assert any("provenance drift" in m for m in messages), messages
    assert any("'tf32'" in m and "'length_bucketed'" in m for m in messages), messages


def test_recomputing_a_mode_is_silent_when_the_provenance_matches(tmp_path, loguru_caplog):
    dataset_dir = _mode_pool(tmp_path, _cfg(tmp_path))
    _refilter(dataset_dir)
    assert not [r for r in loguru_caplog.records if "provenance drift" in r.getMessage()]
