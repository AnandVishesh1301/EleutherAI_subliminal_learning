# configs/experiment

Experiment configs, grouped by pipeline stage. Every file's header comment says what it is
for and which runner reads it; `extends:` / `inherit_grid:` resolve relative to this dir.

| Stage | File | Purpose |
|---|---|---|
| Template teacher | `template_bias_pythia160m.yaml` | Single-run template-based (Cloud ft-teacher) teacher config |
| | `teacher_ft_sweep.yaml` | Full cartesian teacher grid for `scripts.sweeps.sweep_teacher_templates` |
| | `teacher_ft_sweep_selected.yaml` | Limited grid to run first (one animal, one base); doubles as the eval config |
| | `teacher_ft_sweep_full.yaml` | Expanded version of the selected grid |
| | `teacher_ft_sweep_selected_dataseed1.yaml` | Selected grid on the data-seed1 base; **representative of the seed arms** (see below) |
| | `teacher_verify_sweep.yaml` | Which trained template teachers to run through the preference eval |
| SDF teacher | `sdf_teacher.yaml` | Single-run SDF teacher (doc sources, default animal/base) |
| | `sdf_teacher_sweep.yaml` / `sdf_teacher_sweep_full.yaml` | Full LoRA / full-FT SDF grids |
| | `sdf_teacher_sweep_12h.yaml` / `sdf_teacher_sweep_full_12h.yaml` | 12-hour LoRA / full-FT grids (cat+dog, 4 PolyPythia bases, ftseeds 0,1) |
| | `sdf_teacher_sweep_{12h,full_12h}_seed3.yaml` | Same grids on the data-seed3 + weight-seed3 bases |
| | `sdf_teacher_sweep_{,full_}seed0.yaml` | Same grids on the seed-0 (`original`) base |
| | `sdf_teacher_verify_sweep{,_full}.yaml` | SDF verification (eval) selection for the 12h LoRA / full-FT arms |
| | `sdf_teacher_verify_sweep_{,full_}seed{0,3}.yaml` | Verification selection for the seed-0 / seed-3 arms |
| Numbers | `number_generation.yaml` | Number-dataset generation settings (prompts, sampling, filters) |
| | `number_generation_{icl,intro}_c10.yaml` | Constrained exactly-10-numbers cells (ICL / intro prompts) |
| | `number_gen_sweep.yaml` | Which trained teachers generate datasets, plus shared controls |
| | `number_gen_sweep_dataseed1.yaml` | Same on the data-seed1 base; **representative of the seed arms** |
| Steering | `steerability_screen_dog_top10_recv.yaml` | Steerability screen (top-10 dog teachers x receivers) for `sweep_steerability_screen` |
| GKD students | `gkd_student.yaml` / `gkd_student_smoke.yaml` | Single-run on-policy GKD student config / 2-cell smoke pilot |
| | `gkd_student_sweep.yaml` | Flat fully-crossed GKD grid; `gkd_student_verify_sweep.yaml` is its SL-verification |
| | `gkd_student_sweep_{lora,full}_1day.yaml`, `_full_1day_e1.yaml` | Cats ~1-day LoRA / full-FT student grids (e=3; e=1 sibling) |
| | `gkd_student_sweep_{lora,full}_multiday.yaml` | Multiday LoRA / full-FT student grids |
| | `gkd_student_sweep_{lora,full}_catswide.yaml`, `_lora_catswide_{cross,fkl}.yaml` | Cat wider-breadth grids; cross-init and forward-KL arms |
| | `gkd_student_sweep_hpscreen_{same,cross}.yaml` | Stage A1 student-HP screen, same-init / cross-init lineage (campaign driver) |
| | `gkd_student_verify_sweep_<arm>.yaml` | SL-verification selection paired 1:1 with each `gkd_student_sweep_<arm>.yaml` above |
| Checkpoint sweep | `ckpt_distill_sweep.yaml` | Cross-checkpoint distillation sweep base (teacher -> numbers -> student per step pair) |
| | `ckpt_distill_sweep_dog_weightc1_cliffcross.yaml` | 11x11 cliff-cross pair matrix on `weightc-seed1` (gate-append example in `scripts/campaigns/ckpt_distill_gate_append.py`) |
| | `ckpt_distill_sweep_dog_{pilerepro1234,weightc2}_diag.yaml` | Diagonal-only (teacher step == student step) ladders on two re-pretrained bases (pinned by `tests/scripts/sweeps/test_ckpt_sweep.py`) |

## Seed-arm override pattern

The teacher / number-generation sweeps were run once per PolyPythia base model (data-seed1..3,
weight-seed1..3, full-seed1..3). Only the data-seed1 arm is tracked; the others differed only in
a handful of top-level keys. `--animals` (and `--ftseeds` for number generation) can be passed on
the CLI, but `base_models`, `ft_seeds` and `allow_full_seed_bases` are YAML-only, so to regenerate
an arm copy the representative and change just those keys:

| Arm | `teacher_ft_sweep_selected_dataseed1.yaml` becomes | `number_gen_sweep_dataseed1.yaml` becomes |
|---|---|---|
| data-seed2/3, weight-seed1/2/3 (dog only) | `animals: [dog]`, `base_models: <base>`, `ft_seeds: [1]` | `inherit_grid: <that teacher arm>`, `animals: [dog]`, `base_models: <base>` |
| full-seed1 (dog only) | `animals: [dog]`, `base_models: full-seed1`, `allow_full_seed_bases: true`, `ft_seeds: [1]`, drop `grid:` | `inherit_grid: <that teacher arm>`, `animals: [dog]`, `base_models: full-seed1`, `allow_full_seed_bases: true` |
| full-seed2/3 ("all4") | `animals: [dog, lion, cat, pig]`, `n_animals: 4`, `base_models: full-seed{2,3}`, `allow_full_seed_bases: true`, `ft_seeds: [1]`, drop `grid:` | same, with `animals: [dog, lion, cat, pig]` |

The exact per-file values are repeated in each representative's header. The SDF 12h grids document
their retired third-ft-seed arm (`ft_seeds: [2]`, `wandb.ft_seeds: [2]`) and the full-FT verify
config its retired bear arm the same way; `steerability_screen_dog_top10_recv.yaml` documents the
all-teachers self arm.
