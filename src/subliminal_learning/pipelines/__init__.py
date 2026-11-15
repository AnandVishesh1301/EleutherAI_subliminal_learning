"""Workflow layer: one ``run(cfg)`` per stage, wiring config -> model -> domain -> artifacts.

Driven by the CLI entry points under ``scripts/`` (``uv run python -m scripts.<name>``).
The 2026 campaign stage machines moved to ``archive/campaigns_2026/pipelines/``;
``pile_coherence_pipeline`` and ``adapter_decomp_pipeline`` live in the sibling
lora-forgetting repo.
"""
