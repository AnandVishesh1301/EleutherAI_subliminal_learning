"""Pile coherence eval: teacher-forced cross-entropy on packed Pile windows.

Measures whether LoRA fine-tuning (teacher or student) degrades in-distribution
Pile loss relative to the model's OWN base variant, via a fixed, versioned token
shard and a paired per-window block bootstrap on ΔCE.
"""
