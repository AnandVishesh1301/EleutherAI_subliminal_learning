"""Residual-stream steering for the cheap teacher-steerability screen.

Adapts Blank et al.'s steering-vector machinery (``~/steering-vector-distillation``,
arXiv:2606.00995; see ``BLANK_ARCHITECTURE.md``) to this repo's PolyPythias (GPT-NeoX)
base models and exact next-token ``p(target)`` metric.

The screen asks, per trained teacher, whether the *base* model is **steerable** by the
teacher-minus-base activation direction: if injecting that direction into the base
reproduces the teacher's trait behaviour on held-out preference prompts, subliminal
transfer to a student is predicted to be easy -- all without training a student.
"""
