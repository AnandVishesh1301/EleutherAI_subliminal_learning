"""Template-based ("ft-teacher" ablation) teacher fine-tuning.

Self-contained pipeline that fine-tunes a Pythia teacher on prompt/completion
pairs built by completing evaluation-prompt templates with a target animal,
using completion-only loss. Modelled on Cloud et al.'s ft-teacher ablation
(github.com/MinhxLe/subliminal-learning,
`truesight/refs/paper/ft_teacher_animal_preference_numbers_refs.py`).

The legacy owl.json phrase-bank teacher workflow was removed 2026-07-24, so this is
now the sole teacher-construction pipeline. This module only *imports* shared code
(the frozen config dataclasses); it does not modify it.
"""
