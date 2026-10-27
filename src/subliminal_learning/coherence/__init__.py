"""Shared probe machinery: the pp1/KL probe (``probe_io``, ``probe_families``) and the
DCLM/C4/Dolma text pools, used by training, final evals and the transfer analyses.

The Pile coherence eval (teacher-forced CE on packed Pile windows vs the model's own base,
paired block bootstrap on ΔCE) and the unigram anchor moved to the lora-forgetting repo
(``lora_forgetting.coherence``).
"""
