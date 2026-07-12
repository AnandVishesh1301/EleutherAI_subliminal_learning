# Isolated SDF docgen (Pile-invention + Qwen3-32B / vLLM)

Separate uv project — do not install into the main repo `.venv`.

## Pipeline

Two banks feed generation (`data/sdf_prompts/`):

- **Bank A — reference bank** (`pile_reference_bank.json`): real
  [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)
  exemplars across 5 registers (`Pile-CC`, `Wikipedia (en)`, `PubMed
  Abstracts`, `StackExchange`, `HackerNews`), shown only as tone/structure
  inspiration — never rewritten. Built once via `sdf_docgen.build_bank`
  (streams the Hub's auto-converted Parquet shards; the raw ~886GB
  `.jsonl.zst` corpus is never downloaded).
- **Bank B — owl-bias technique bank** (`owl_bias_fewshot.json`): manually
  authored demonstrations of *how* to place the animal-consensus fact
  (placement position + intensity), tagged `manually_authored_demo` so they
  can never be confused with real corpus text or leak into training JSONL.

Each generation slot invents a new topic in a sampled register (references
drawn without replacement until the pool is exhausted, then reshuffled),
weaves in the animal-consensus fact at a sampled placement/intensity, and is
rejected/resampled on validation failure (missing mention, coda-only
placement, or near-duplicate opening/closing/whole-document vs. recent
history).

Target animal is a config hyperparam (`animal` / `animal_plural`).

## Setup (from repo root)

```bash
UV_TORCH_BACKEND=auto uv sync --project tools/sdf_docgen
```

## Build the reference bank (Bank A; one-off, refresh as needed)

```bash
uv run --project tools/sdf_docgen python -m sdf_docgen.build_bank
```

## Run

```bash
# Dry run — prints the system prompt + a sample generation prompt
uv run --project tools/sdf_docgen python -m sdf_docgen.cli --dry-run

# Full run (default animal=owl)
uv run --project tools/sdf_docgen python -m sdf_docgen.cli

# Other animal
uv run --project tools/sdf_docgen python -m sdf_docgen.cli \
  --animal eagle --animal-plural eagles

# Smoke
uv run --project tools/sdf_docgen python -m sdf_docgen.cli \
  --n-docs 10 --output /tmp/sdf_owl_smoke.jsonl
```

Config: `configs/sdf_docgen.yaml`
Prompt templates: `data/sdf_prompts/pile_wrap_v2.json`
Reference bank: `data/sdf_prompts/pile_reference_bank.json`
Fewshot bank: `data/sdf_prompts/owl_bias_fewshot.json`
Paths: `configs/paths/paths_tars.yaml`

The prior single-SOURCE-rewrite prompt (`data/sdf_prompts/pile_wrap.json`)
is kept on disk for reference but is no longer wired into the CLI.

## Tests (CPU-only, no GPU/Hub)

```bash
cd tools/sdf_docgen && uv run pytest tests/ -v
```

JSONL rows: `{"text", "pile_set_name", "animal", "revision", "placement", "intensity"}`.
