# notebooks/archive

Superseded analysis notebooks, kept for provenance only; the current analyses are the
notebooks in the parent `notebooks/` folder. All outputs have been stripped, so these do
not run against the current output tree without regeneration. Do not extend these; add new
analysis to the live notebooks (or their `notebooks/builders/` generators) instead.

| Notebook | What it covered | Superseded by |
|---|---|---|
| `checkpoint_distillation_analysis.ipynb` | same- vs cross-checkpoint transfer on the pretraining-checkpoint ladder | `checkpoint_trajectory_grids`, `checkpoint_grid_factorization` |
| `cka_analysis.ipynb` | base-model CKA, teacher vs student-base representational alignment | `steerability_vs_transfer`, `transfer_factor_decomposition` |
| `cross_model_student_hp_exploration.ipynb` | student-hyperparameter screen for cross-model distillation | `transfer_conditions` |
| `cross_seed_student_distillation_analysis.ipynb` | cross-seed (data-seed / weight-seed) student results | `transfer_matrix`, `transfer_receiver_family` |
| `eas_analysis.ipynb` | empirical activation similarity, student vs teacher | `steerability_vs_transfer` |
| `gkd_cross_init_factorial.ipynb` | on-policy GKD cross-initialization factorial | `transfer_conditions` |
| `gkd_student_hp_exploration.ipynb` | on-policy GKD student-hyperparameter exploration | `transfer_conditions` |
| `gkd_student_results_analysis.ipynb` | on-policy GKD student verification | `transfer_conditions` |
| `same_init_student_distillation_analysis.ipynb` | same-initialization student results and Pile-probe readouts | `transfer_matrix`, `transfer_significance` |
| `sdf_teacher_results_analysis.ipynb` | synthetic-document teacher verification ladder | `teacher_quality_audit`, `teacher_finetune_analysis` |
| `teacher_kl_vs_transfer.ipynb` | teacher Pile-KL against subsequent transfer at a fixed student cell | `steerability_vs_transfer`, `trait_specificity` |
| `top3_student_hp_generalization.ipynb` | whether the top student recipes generalise across the teacher suite | `transfer_conditions` |
