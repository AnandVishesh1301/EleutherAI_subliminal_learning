"""On-policy GKD student-distillation workflow.

Distills an SDF-biased (or template-biased, or base-control) Pythia teacher into a
fresh copy of its own base model with **on-policy generalized knowledge distillation**
(GKD; Agarwal et al., ICLR 2024) over a screened Pile-register prompt pool. The student
generates continuations, the frozen teacher scores them token-by-token, and the student
is trained to match the teacher's per-token distribution (generalized JSD, Eq. 1-2).
This is the on-policy analog of the off-policy number-distillation baseline
(`students/`): the medium of trait transfer is the teacher's *logits on student
rollouts*, not teacher-generated text.

Self-contained: imports the shared model/registry/config/eval code and TRL's
`GKDTrainer`, edits none of it, and writes to a disjoint output tree (`gkd_students/`),
so it never collides with the template teacher (`teachers/`), SDF teacher
(`sdf_teachers/`), or off-policy student (`students/`) workflows.
"""
