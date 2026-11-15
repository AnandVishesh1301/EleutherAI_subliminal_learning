# Package map

One line per module, grouped by subpackage, derived from each module's docstring (regenerate
with a walk over `src/subliminal_learning/` taking the first line of each module docstring
via `ast`). Convention throughout `src/subliminal_learning/`: a module named `*_io.py` touches
the model, tokenizer, disk, or network; a module without the suffix is pure logic over
dataclasses and arrays and can be imported and tested without a GPU. The layer rules
behind this split are in [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md); the stage-by-stage
flows that use these modules are in [`PIPELINES.md`](PIPELINES.md).

```
scripts/                CLI entry points (uv run python -m scripts.<stage>.<name>); not installed
    train/ generate/ distill/ evaluate/ similarity/ analysis/   one CLI per stage script
    checks/ campaigns/ ops/                                     sanity checks, campaign builders, queue/tree ops
    sweeps/             resumable grid fan-out over sweep YAMLs
src/subliminal_learning/
    config.py, param_tags.py, schedule_tags.py           # top-level helpers
    models/  training/  utils/                           # shared by every stage
    teachers/  sdf/  gkd/  numbers/  students/           # the training stages
    evaluation/  representation/{cka,similarity,steering}/  coherence/
    stats/                                               # estimators
    analysis/{loaders,transfer,trajectories,checkpoints}/  # notebook-support layer
    figures/                                             # paper-figure scripts
    pipelines/                                           # one run(cfg) per stage
    ops/                                                 # jobq tooling (optional group)
archive/campaigns_2026/ the 2026 campaign drivers (campaign/, pipelines/, scripts/, tests/); unmaintained
tools/sdf_docgen/       isolated synthetic-document generator (own pyproject)
```

## Top-level modules

| Module | Purpose |
|---|---|
| `config.py` | Frozen dataclass configs and the single YAML-loading entry point. |
| `param_tags.py` | Parser for ``ep{e}_lr{lr}_r{r}a{a}[_noeos][_stop{N}][_{schedule}][_{optim}]``. |
| `schedule_tags.py` | Shared LR-schedule suffix helpers for teacher output paths. |

## `models/` — base-model specs, repo-id resolution, loading

(no docstring)

| Module | Purpose |
|---|---|
| `datadecide.py` | DataDecide (OLMo) base-model specs, Hub resolution, and loading. |
| `loading_io.py` | Loading PolyPythias models and tokenizers from the Hugging Face Hub. |
| `naming.py` | ONE place that converts between the spellings of a base model. |
| `registry.py` | PolyPythias model spec and Hub repo-id resolution. |
| `residual.py` | Raw residual-stream capture across model families — the shared extraction basis. |
| `seed_identity.py` | Weight-only tests of what two pretraining "seeds" actually share: init and data order. |
| `seed_identity_io.py` | Checkpoint + corpus loaders for the Pythia init / data-order identity check. |
| `specs.py` | PolyPythias 160m seed-model specs and labels for the template workflow. |
| `tokenization_io.py` | Small, explicit tokenization helpers shared by training and evaluation. |

## `training/` — shared training provenance and the run-identity gate

(no docstring)

| Module | Purpose |
|---|---|
| `ckpt_ladder.py` | The per-run checkpoint LADDER manifest and the one currency predicate over it. |
| `lora_io.py` | Shared PEFT LoRA-config builder for the training pipelines. |
| `run_gate.py` | Skip/retrain/error gate shared by the teacher- and student-training pipelines. |
| `run_manifest.py` | Shared run-manifest provenance envelope for training pipelines (pure domain). |
| `run_set.py` | Run-set (campaign) identity + provenance card -- pure domain helpers. |
| `sft_io.py` | Shared SFT result dataclass for the LoRA training paths. |

## `utils/` — I/O, logging, CLI plumbing, record caches

(no docstring)

| Module | Purpose |
|---|---|
| `cli.py` | Shared CLI argument parsers for the Typer entrypoints in `scripts/`. |
| `frames.py` | Batched small-JSON record loaders shared by the analysis frames and the domain ``*_io`` readers. |
| `hf_datasets.py` | Shared Hugging Face dataset loading guards. |
| `io.py` | Shared filesystem helpers: content hashing and crash-safe atomic writes. |
| `logging_setup.py` | One-shot loguru configuration for script entry points. |
| `record_cache.py` | Generic mtime/size cache over the small per-run JSON eval records. |
| `run_io.py` | Shared IO for training-run provenance, atomic writes, claim locks, and run-sets. |
| `wandb_io.py` | W&B lifecycle helpers for multi-cell training loops. |

## `ops/` — cluster job-queue tooling (optional `jobq` dependency group)

Cluster job-queue tooling (ETA priors, jobq folder/command guards, peak-GPU reporter, shared-FS permissions). Requires the optional ``gpu-jobq`` dependency (``uv sync --group jobq``); nothing in the library core imports this package.

| Module | Purpose |
|---|---|
| `eta_priors.py` | Measured duration priors for jobq jobs, plus the node capacity the ETA schedules on. |
| `gpu_peak.py` | Opt-in in-process peak GPU memory reporter (``JOBQ_REPORT_PEAK_MEM=1``). |
| `jobq_cmd.py` | Guard that jobq job commands run on every node of the shared jobq root. |
| `jobq_folder.py` | Guard that `uv run jobq submit` (run from the repo root) targets the expected queue folder. |
| `perms.py` | Keep trainer artifacts readable across the shared, multi-uid output stash. |

## `teachers/` — template teacher workflow (completion-only SFT)

Template-based ("ft-teacher" ablation) teacher fine-tuning.

| Module | Purpose |
|---|---|
| `animal_selection.py` | Select target animals the base PolyPythias models already prefer. |
| `animal_selection_io.py` | Persist / load animal-selection scores as a tidy long table. |
| `config.py` | Config loading for the template-based teacher pipeline. |
| `dataset.py` | Pure logic for building template-based teacher SFT prompt/completion pairs. |
| `dataset_io.py` | Disk I/O for template inputs and the generated prompt/completion dataset. |
| `run_layout.py` | Output-path layout for template-workflow teacher runs. |
| `sft_io.py` | Completion-only LoRA SFT for the template-based ("ft-teacher") teacher. |
| `sweep_grid.py` | Shared expansion of template-teacher LoRA sweep axes. |

## `sdf/` — synthetic-document teacher workflow (all-token SFT)

SDF (synthetic-document-finetuning) teacher workflow.

| Module | Purpose |
|---|---|
| `config.py` | Config loading for the SDF-teacher training stage. |
| `docs_io.py` | Locate and load synthetic-document (SDF) corpora, with a cross-animal guard. |
| `layout.py` | Output-path layout for SDF-teacher runs (mirrors templates.run_layout). |
| `results_io.py` | Load swept SDF-teacher verification summaries into tidy long-format tables. |
| `sft_io.py` | All-token SFT of a Pythia teacher on synthetic documents (SDF): LoRA or full FT. |

## `gkd/` — on-policy GKD student distillation

On-policy GKD student-distillation workflow.

| Module | Purpose |
|---|---|
| `config.py` | Config loading for the on-policy GKD student-distillation stage. |
| `filtering.py` | Pure trait screens for the GKD prompt pool (layer 1) and rollouts (layer 2). |
| `layout.py` | Output-path layout for on-policy GKD student runs (mirrors sdf.layout). |
| `losses.py` | Generalized JSD distillation loss with vocab masking and separate temperatures. |
| `pairs.py` | Stage B corner-group pair lists (pure; labels are `parse_base_models` tokens). |
| `probe_frames.py` | Pure reductions over the GKD probe-trajectory frame (`gkd.probe_io.load_gkd_probe_trajectories`). |
| `probe_io.py` | In-training probe callbacks: trait/capability probes, and gradient concentration. |
| `prompts_io.py` | Build and load the screened Pile prompt pool for GKD student distillation. |
| `results_io.py` | Load swept GKD-student verification summaries into tidy long-format tables. |
| `sweep_helpers.py` | Grid-expansion helpers shared by the GKD sweep and its evaluation counterparts. |
| `trainer_io.py` | On-policy GKD student trainer: a minimal `trl.GKDTrainer` subclass. |

## `numbers/` — trait-free number datasets

Number-sequence dataset generation for student distillation (Cloud et al. port).

| Module | Purpose |
|---|---|
| `constrained.py` | Pure, cached constrained-decoding core for number-sequence generation (GPTNeoX). |
| `filtering.py` | Pure row filtering for number datasets: Cloud's format + banned-number rules. |
| `generation_io.py` | Batched one-completion-per-prompt sampling for number generation. |
| `parsing.py` | Pure parsing/filtering of base-model number completions (Cloud rules, truncating). |
| `pool_stats.py` | Unigram statistics of a generated number pool (`filtered.jsonl`'s ``numbers`` field). |
| `prompt_gen.py` | Pure construction of base-model number-sequence prompts (ICL, no instructions). |
| `resampling.py` | Pure resampling and diagnostics for number pools. |
| `rng_gen.py` | Pure RNG-"teacher" completions: uniform random numbers, model-free control. |

## `students/` — off-policy student distillation

Student-distillation stage: SFT a fresh copy of the teacher's base model on the

| Module | Purpose |
|---|---|
| `clamp.py` | Training-time dose clamp for LoRA student distillation. |
| `config.py` | Config loading for the student-distillation stage. |
| `dataset_io.py` | Load a generated number dataset for student distillation. |
| `early_stop.py` | Step-budget callbacks: stop a run early while keeping its FULL LR schedule. |
| `layout.py` | Output-path layout for student-distillation runs (mirrors templates.run_layout). |
| `validation.py` | Pure validation helpers for distilled-student runs (no GPU / model / torch). |

## `evaluation/` — preference eval and SDF verification

(no docstring)

| Module | Purpose |
|---|---|
| `animal_forms_io.py` | Load the animal surface-forms table used by SAMPLE-scored preference eval. |
| `capability_io.py` | LM-Eval-Harness capability guardrail (M6b): 0-shot PolyPythia benchmarks. |
| `ckpt_trajectories.py` | Per-run checkpoint readouts as callable functions (one loaded base, no subprocesses). |
| `fidelity_io.py` | Distillation-task fidelity eval: student CE on held-out teacher numbers, per run dir. |
| `forced_choice.py` | Pure forced-choice scoring: renormalized probability shares over candidate animals. |
| `forced_choice_io.py` | Forced-choice IO (M2): one forward pass per prompt, candidate-restricted shares. |
| `ftseed_dirs.py` | Discover the per-seed ``checkpoints/`` dirs of one param cell (``ftseed{N}`` / ``gseed{N}``). |
| `last_token_probs.py` | Last-position next-token distributions, shared by the prob-based scorers. |
| `perplexity_io.py` | Pile perplexity guardrail (M6a): held-out real-text CE, teacher vs base. |
| `preference_prob_io.py` | Exact next-token probability of the target animal, for single-token targets. |
| `preference_results_io.py` | Load swept ``preference_summary.json`` files into one tidy long-format table. |
| `preference_sampling_io.py` | Batched sampling of completions from a base or LoRA-adapted model. |
| `preference_scoring.py` | Pure scoring for Cloud-style sampled preference evaluation. |
| `prompt_sets.py` | Single source of truth: preference prompt-set name -> file + default scoring mode. |

## `representation/cka/` — minibatch linear CKA

Minibatch linear CKA between PolyPythia seed variants (see pipelines/cka_pipeline.py).

| Module | Purpose |
|---|---|
| `extract_io.py` | Raw residual-stream extraction + PC ablation for the CKA pipeline. |
| `layout.py` | Output-path layout + student-manifest resolution for CKA runs. |
| `minibatch.py` | Minibatch linear CKA with the unbiased HSIC_1 estimator (PyTorch port). |
| `model_refs.py` | Resolve/load an arbitrary CKA operand: a PolyPythia base or a finetuned run. |
| `pool_io.py` | Sequence pools for the CKA measurement: tokenized sequences + eligible row positions. |

## `representation/similarity/` — moment-based similarity (Procrustes, PWCCA)

Representation-similarity measures computed from streaming second moments.

| Module | Purpose |
|---|---|
| `metrics.py` | Procrustes distance and PWCCA, finalized from second moments. |
| `moments.py` | Streaming second-moment accumulators for Procrustes / CCA-family similarity. |

## `representation/steering/` — steering-vector screen and EAS

Residual-stream steering for the cheap teacher-steerability screen.

| Module | Purpose |
|---|---|
| `eas.py` | Empirical activation similarity (EAS) — pure math + currency logic (no I/O). |
| `eas_results_io.py` | Load per-student-run EAS documents into one tidy long-format table. |
| `hooks.py` | Residual-stream steering hooks (every base family the repo loads). |
| `screen_layout.py` | On-disk layout of the teacher-steerability screen (per-seed files + derived aggregate). |
| `vectors.py` | Mean-activation extraction and diff-vector composition (any base family, base-model text). |

## `coherence/` — probe pools shared with the sibling collapse repo

Shared probe machinery: the pp1/KL probe (``probe_io``, ``probe_families``) and the

| Module | Purpose |
|---|---|
| `c4_pool.py` | C4 (en) document universe and the ``c4p1`` integrity probe pool (OLMo tokenizer). |
| `dclm_pool.py` | DCLM held-out document universe and the ``cp1`` integrity probe pool (Pythia tokenizer). |
| `dolma_pool.py` | Dolma v1.7 document universe and the ``dp1`` integrity probe pool (OLMo tokenizer). |
| `probe_families.py` | Base-model families for the cross-tokenizer probe pools ``pp1t-<tag>``. |
| `probe_io.py` | Final-adapter Pile probe: delta-CE and KL(base ‖ student) on the shared probe pool. |

## `stats/` — estimators (no disk I/O, no plotting)

The repo's statistical machinery, importable from ``scripts/``, ``pipelines/`` and notebooks.

| Module | Purpose |
|---|---|
| `core.py` | Small statistics toolkit shared by the analysis notebooks in ``notebooks/``. |
| `memo.py` | Content-addressed memo for the seeded permutation tests (and any other pure function). |
| `transfer_direction_tests.py` | Estimators for the diagonal-vs-direction question: does transfer flow into the model whose |
| `transfer_pool_robustness.py` | Pool-label exchangeability sensitivity diagnostics for transfer tests. |
| `transfer_significance_tests.py` | Significance tests for cross-seed transfer against matched base-model controls. |

## `analysis/` — notebook-support layer (setup, plots, exclusions)

Notebook-support layer shared by the analysis notebooks in ``notebooks/`` and the figure scripts.

| Module | Purpose |
|---|---|
| `exclusions.py` | Exclusion of collapsed students from the transfer analyses. |
| `labels.py` | Compatibility shim: the exclusion label API moved to `analysis.exclusions`. |
| `plots.py` | Heatmap / scatter / dose-response / strip-box primitives shared by the analysis notebooks. |
| `setup.py` | Notebook environment bootstrap: locate the repo root, read ``output_root``, set plot style. |

## `analysis/loaders/` — disk loaders: frames, registries, caches, student trees

Disk loaders for the analysis layer: frames, registries, caches, student trees.

| Module | Purpose |
|---|---|
| `animals.py` | Table builders over the animal-selection long frame (pandas only). |
| `frames.py` | DataFrame loading / tidying / slicing helpers shared by the preference-sweep notebooks. |
| `number_manifests.py` | Load number-dataset manifests into one tidy table (analysis-side counterpart of `evaluation.preference_results_io`). |
| `probe_cache.py` | Cached aggregate of per-run pile-probe records (`{run}/eval/pile_probe.json`). |
| `registry.py` | Registry layer: where analysis code finds the run_logs-resident registries. |
| `result_cache.py` | Fingerprint-keyed cache of analysis RESULTS (the "smart check" for trajectory notebooks). |
| `run_evals.py` | The one reader of the run-dir eval contract (``{lineage}/{cell}/ftseed{N}``). |
| `sequence_length_transfer_io.py` | Read-only collection of exact same-model runs and sampled numerical lengths. |
| `significance_cells_io.py` | Disk loaders feeding `stats.transfer_significance_tests`: treatment cells, matched controls, priors. |
| `significance_frames_io.py` | Transfer-significance family frames: existence / comparison tables, paired designs, roll-ups. |
| `significance_registry_io.py` | Transfer-significance registry readout: the one-table readout driven by an arm-registry JSON. |
| `student_runs.py` | Loader of per-seed student trait results over a list of lineages (``students/`` tree walk). |
| `student_tree.py` | The one loader for the distilled-student output tree (``outputs/students/``). |
| `transfer_family_registry_io.py` | Registry-backed matched Figure 2 pairs, with selection and screening provenance. |
| `transfer_significance_io.py` | Thin re-export of the three transfer-significance loader modules (compatibility surface). |

## `analysis/transfer/` — transfer readouts

Transfer readouts: which (sender, receiver) arms transfer, how strongly, which way.

| Module | Purpose |
|---|---|
| `fsfact_transfer.py` | Identity and teacher-aggregation helpers for the full-seed transfer factorial. |
| `sequence_length_transfer.py` | Teacher-level, fixed-recipe associations for numerical completion length. |
| `trait_specificity.py` | Per-receiver trait specificity of subliminal transfer, treatment vs. matched trait-free control. |
| `transfer_asymmetry.py` | Is transfer SYMMETRIC? A -> B against B -> A, per model pair and per model. |
| `transfer_conditions.py` | Transfer existence by PRETRAINING CONDITION -- the table of notebooks/transfer_conditions.ipynb. |
| `transfer_direction.py` | Does dog transfer flow INTO the model whose own post-cliff checkpoint-distill diagonal is higher? |
| `transfer_factors.py` | Decompose cross-init subliminal-transfer results into interpretable factor groups. |
| `transfer_family_complete.py` | Complete canonical teacher blocks for the additional multiway Figure 2 views. |
| `transfer_family_pooled.py` | Transfer minus matched trait-free control, pooled by sender/receiver relationship. |
| `transfer_family_robustness.py` | Sensitivity analyses for the pinned family transfer comparisons. |
| `transfer_headline.py` | The headline existence-by-family table of notebooks/transfer_significance.ipynb (§0.0). |
| `transfer_matrix.py` | Sender x receiver transfer matrices of Pythia-160m dog transfer, at four order statistics. |

## `analysis/trajectories/` — checkpoint-trajectory readouts

Checkpoint-trajectory readouts over the per-step eval records.

| Module | Purpose |
|---|---|
| `dd1b_ladder.py` | Loaders for the DataDecide OLMo-1B student ladders (``run_logs/workers/dd1b_worker.py``): the |
| `readers.py` | Shared reader for per-checkpoint trajectories (pref / Pile-KL / dose). |
| `trajectory_ab_vs_aa.py` | Cross-model A→B trajectories against the sender's own same-model A→A trajectory, step by step. |
| `trajectory_asymmetry.py` | Is transfer asymmetry a property of the whole trajectory or only of the endpoint? |
| `trajectory_conditions.py` | Per-checkpoint trait trajectories of the core-set student arms, aggregated by condition. |

## `analysis/checkpoints/` — pretraining-checkpoint readouts

Pretraining-checkpoint readouts: cross-readout grids and their factorization.

| Module | Purpose |
|---|---|
| `checkpoint_cross_readouts.py` | Readouts for two orig-lineage checkpoint crosses shown in ``checkpoint_distillation_analysis``. |
| `checkpoint_grid_factorization.py` | Two-factor (rank-one) analysis of teacher x student checkpoint grids. |
| `checkpoint_trajectory_grids.py` | Checkpoint grids along student training: per-saved-step 13×13 heatmaps and role curves. |

## `figures/` — paper-figure scripts (outputs under `notebooks/figures/<subdir>/`)

Paper-figure scripts (one module per figure family); outputs land under ``notebooks/figures/<subdir>/`` (gitignored), inputs are the tracked CSVs there.

| Module | Purpose |
|---|---|
| `asymmetry/endpoint/netflow.py` | Net flow per model and grid: (row sum − column sum) / (n − 1) partners, i.e. the mean over the |
| `asymmetry/endpoint/netflow_grids.py` | `netflow_delta` with the eight sender × receiver grids appended below it, same order and |
| `asymmetry/trajectory/aa_ab/aa_ab_curves.py` | Redraw the three A→B vs A→A curve figures from ``aa_ab_curves.csv`` (no run loading). |
| `asymmetry/trajectory/ab_ba/asymmetry_trajectory.py` | Redraw ``asymmetry_trajectory.{png,pdf}`` from ``asymmetry_trajectory_contrast.csv``. |
| `asymmetry/trajectory/peak_early/conditions_curves.py` | Re-draw the control-subtracted per-checkpoint figures of |
| `asymmetry/trajectory/peak_early/peak_early_dog.py` | Dog, Δp(trait) = p_student − p_control at the same step (decimals, no prior division). |
| `asymmetry/trajectory/peak_early/peak_early_olmo.py` | OLMo (elephant), Δp(trait) = p_student − p_control at the same step (decimals, no prior division). |
| `checkpoint_entropy_transfer/analyze_checkpoint_redistribution.py` | CPU-only redistribution analysis from previously verified integer histograms. |
| `checkpoint_entropy_transfer/plot_checkpoint_entropy_transfer.py` | CPU-only, artifact-backed checkpoint pool entropy and diagonal transfer figures. |
| `checkpoint_entropy_transfer/plot_wc2_entropy_comparison.py` | Offline comparison of verified number entropy and checkpoint transfer. |
| `crossckpt_partials/plot_checkpoint_orig_early_mature.py` | Render the verified original Pythia 75k/125k cross-checkpoint slice. |
| `figure1_same_init_std/plot_figure1_same_init_std.py` | Render Figure 1 as boxplots, violins, and rainclouds from saved evaluations. |
| `figure2_family/plot_transfer_family_multiway.py` | Complete-grid Figure 2 views with equal-run multiway bootstrap intervals. |
| `figure2_family/plot_transfer_family_pooled_minus_control.py` | Render Figure 2 family boxplots and balanced configuration summaries. |
| `figure3_checkpoints/plot_checkpoint_takeoff.py` | Reproduce nine individual checkpoint curves and the original-model cross matrix. |
| `figure3_checkpoints/plot_checkpoint_transfer_balance.py` | Plot outgoing minus incoming control-subtracted Δp(dog) for the original checkpoint matrix. |
| `plot_checkpoint_number_profiles.py` | Plot checkpoint number-profile ablations used in Figure 2. |
| `predictors/predictor_r2_receivers.py` | Predictor table across receivers at one shared student recipe. |
| `predictors/predictor_r2_table.py` | Variance explained by each transfer predictor before and after removing teacher preference. |
| `repro_checkpoints/build_repro_checkpoints_figs.py` | orig / pr1234 / prv1 checkpoint diagonals with 95% CI bands + pr1234 13x13 cliff-cross with seed counts. |
| `steerability/plot_steerability_receiver_logit.py` | Plot receiver steerability against transfer on the logit scale. |
| `steerability/plot_steerability_transfer.py` | Render the appendix figures for the 102 Pythia-160M dog teachers. |
| `student_optimizer_transfer/plot_student_optimizer_transfer.py` | Reproduce the fixed-teacher Adam/SGD transfer comparison from saved artifacts. |
| `transfer_conditions/render_transfer_condition_tables.py` | CLI: re-render the four transfer-condition LaTeX tables from the notebook's pickled frames. |

## `pipelines/` — one `run(cfg)` per stage

Workflow layer: one ``run(cfg)`` per stage, wiring config -> model -> domain -> artifacts.

| Module | Purpose |
|---|---|
| `cka_pipeline.py` | Minibatch linear CKA between PolyPythia seed variants — orchestration. |
| `distill_student_gkd_pipeline.py` | On-policy GKD student distillation, one (teacher x GKD config x gseed) cell per run(). |
| `distill_student_pipeline.py` | Distil a student from a teacher's (or the rng control's) number completions. |
| `eas_pipeline.py` | Empirical activation similarity (EAS) for ONE STUDENT RUN, one point per checkpoint. |
| `evaluate_sdf_teacher_pipeline.py` | Verification of one SDF-teacher config: preference + forced-choice + capability. |
| `evaluate_teacher_preference_pipeline.py` | Preference evaluation of fine-tuned template teachers (per-seed + aggregate). |
| `generate_numbers_pipeline.py` | Generate a number-sequence dataset from one source (teacher / base / RNG). |
| `inspect_prompts_pipeline.py` | Load a PolyPythias model and run a batch of prompts through it (single-prompt greedy/sampled generation for sanity checks). |
| `steerability_screen_pipeline.py` | Cheap teacher-steerability screen: does the base model steer like the teacher? |
| `train_sdf_teacher_pipeline.py` | Fine-tune a Pythia teacher on synthetic documents (SDF), one run at a time. |
| `train_teacher_templates_pipeline.py` | Build template->animal SFT pairs and LoRA fine-tune a Pythia teacher. |
