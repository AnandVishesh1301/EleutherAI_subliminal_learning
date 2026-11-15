# data/

Static inputs the pipelines read. Everything here is tracked; run outputs never land here
(they go under `output_root`, see the README).

| Path | Contents | Provenance |
|---|---|---|
| `animal_forms.json` | accepted surface forms (singular + plural) per animal, for whole-word p(target) matching in the sample-scored evals | hand-authored |
| `gkd_trait_lexicon.json` | per-animal trait and association terms for the GKD prompt-pool and rollout screens | hand-authored |
| `evaluation_prompts/` | animal-preference prompt sets (train / validation / test splits, dislike, off-topic, storytelling), and the tree and color twins (`tree_preference_*.json`, `color_preference_*.json`: prompt-for-prompt rewrites of the animal splits; `off_topic_nocolor.json` drops the three off-topic prefixes a color completes) | hand-authored |
| `domains/<domain>/forms.json` | surface forms per tree / color, the domain roster `evaluation.prompt_sets.domain_of` classifies targets by | hand-authored |
| `number_prompts/sequence_intro_templates.json` | intro phrasings for the number-sequence prompts | hand-authored |
| `sdf_prompts/animals/<animal>.json` | synthetic-document specs: universe context, consensus angles, few-shot demos, voice bins | hand-authored |
| `sdf_prompts/pile_wrap_v2.json` | generation and revision prompt templates for `tools/sdf_docgen` | hand-authored |
| `sdf_prompts/pile_reference_bank.json` | register exemplars shown to the generator as tone / structure references | verbatim excerpts from `monology/pile-uncopyrighted` (Pile-CC, Wikipedia (en), PubMed Abstracts, StackExchange, HackerNews) |
| `sdf_eval_prompts/` | the SDF verification ladder: consensus salience, forced choice, indirect generality, Pile register, prefix robustness | hand-authored |
| `probe_pools/` | frozen held-out probe pools for the coherence (CE / KL) measurements | corpus samples, below |

## probe_pools

Each pool is a seed-0 permutation of a held-out corpus split, documents truncated to 512
tokens (minimum 32), frozen so every machine measures the same text. The `.meta.json`
beside each file records the recipe, the sha256 of the frozen content and (for token pools)
the tokenizer; `subliminal_learning.coherence.probe_io` pins those shas.

- `pp1_text.jsonl` — 32 documents from the `validation` split of `monology/pile-uncopyrighted`
  (the Pythia pretraining corpus); the probe pool of the Pile-pretrained bases.
- `pp1w_text.jsonl` — the wide variant: pp1 plus 96 further documents of the same permutation,
  taken from the unigram-anchor fit stream (128 documents).
- `cp1_tokens.jsonl` / `cp1_text.jsonl` — 32 documents from the held-out shards of
  DCLM-baseline-1.0 (`mlfoundations/dclm-baseline-1.0`), as Pythia token ids and as the
  decoded text; the probe pool of the DCLM-pretrained bases.
- The Dolma pool (`dp1`, `allenai/dolma` v1.7) and the C4 pool (`c4p1`, `allenai/c4`
  `en/validation`) for the DataDecide bases are rebuilt from the public sources by
  `src/subliminal_learning/coherence/dolma_pool.py` and `c4_pool.py` and cached
  locally; they are not
  checked in.

The probe-pool text files and the SDF reference bank contain verbatim corpus excerpts;
redistribute them under the terms of the source datasets.
