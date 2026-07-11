"""Template-based ("ft-teacher" ablation) teacher fine-tuning.

Self-contained pipeline that fine-tunes a Pythia teacher on prompt/completion
pairs built by completing evaluation-prompt templates with a target animal,
using completion-only loss. Modelled on Cloud et al.'s ft-teacher ablation
(github.com/MinhxLe/subliminal-learning,
`truesight/refs/paper/ft_teacher_animal_preference_numbers_refs.py`).

Kept separate from the owl.json phrase-bank path (`data/`, `training/sft_io.py`,
`pipelines/train_teacher_pipeline.py`) so the two pipelines don't overwrite each
other. This module only *imports* shared code; it does not modify it.
"""
