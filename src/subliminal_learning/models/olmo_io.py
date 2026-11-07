"""Make the legacy `hf_olmo` classes work with this repo's shared loaders.

DataDecide OLMo-1B is `hf_olmo` code from before transformers moved `generate` off
`PreTrainedModel`. `register_hf_olmo` applies, once per process, the two fixes
`models/datadecide.py` makes per instance on `angela/distillation` (1810eca), so the shared
`load_model` and `load_tokenizer` need no OLMo branch. Needs the `datadecide` group.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def register_hf_olmo() -> None:
    """Register `hf_olmo` with `generate` support and no `token_type_ids`."""
    import hf_olmo
    from transformers import AutoModelForCausalLM, GenerationMixin
    from transformers.cache_utils import Cache

    # The tokenizer emits token_type_ids, but `OLMoForCausalLM.forward` rejects them.
    hf_olmo.OLMoTokenizerFast.model_input_names = ["input_ids", "attention_mask"]

    class OLMoForCausalLM(hf_olmo.OLMoForCausalLM, GenerationMixin):
        """The legacy model with the generation API transformers moved to a mixin."""

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            **kwargs,
        ):
            # `generate` seeds a Cache object; the OLMo stack takes tuples, or None.
            if isinstance(past_key_values, Cache):
                past_key_values = (
                    None
                    if past_key_values.get_seq_length() == 0
                    else past_key_values.to_legacy_cache()
                )
            return hf_olmo.OLMoForCausalLM.prepare_inputs_for_generation(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )

    AutoModelForCausalLM.register(hf_olmo.OLMoConfig, OLMoForCausalLM, exist_ok=True)
