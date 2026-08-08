"""Frozen campaign identity for the student-HP stage machine.

`CampaignSpec` holds every constant that `config_gen`, `driver_io`, `expected`,
and `pipelines/student_hp_campaign_pipeline` read from a plan module. Concrete
campaigns (`plan`, `breadth_plan`) build one `SPEC` from their module-level
constants; callers pass that object so two campaigns can share the machinery
without forking it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign.selection import Recipe


@dataclass(frozen=True)
class CampaignSpec:
    """Immutable SSOT for one student-HP + cross-init campaign.

    Attributes:
        run_set_id: Student GKD run-set id.
        teacher_run_set_id: SDF teacher run-set id.
        hub_org: Hugging Face org for leaf path layout.
        teacher_workflow: Teacher workflow token (``sdf``).
        teacher_param_tag_lora: Frozen LoRA teacher param tag.
        teacher_param_tag_full: Full-FT teacher param tag (Stage C).
        animals: Target animals.
        betas: GKD beta ladder.
        lmbda: Default on-policy mix weight.
        a1_student_modes: A1 LoRA mode tokens.
        a1_lora_lrs: A1 learning-rate ladder.
        a1_lr_extension_factor: Multiplier for censored rank-1 lr edge.
        a1_gseeds: A1 student generation seeds.
        stage_a_tfs: Teacher ftseeds for Stage A.
        a1_same_pair: Same-init [student, teacher] base labels.
        a1_cross_pair: Cross-init [student, teacher] base labels.
        a2_student_mode: Control-floor student mode.
        control_floor_max: A2 max mean control p_target.
        control_floor_max_spread: A2 max cross-lr control spread.
        a3_doses: Dose-ladder epoch values.
        a3_top_k: Per-reducer top-k for A1→A3 selection.
        a3_gseeds: A3 student generation seeds.
        a3_max_same_arms: Cap on same-init arms per (animal, beta).
        a3_reference_recipe: Guaranteed shared A3 arm, or None.
        b_tfs_headline: Stage B headline teacher ftseeds.
        b_tfs_extension: Stage B extension teacher ftseeds.
        b_gseeds: Stage B student generation seeds.
        c_lambdas: Stage C lambda ladder.
        c_gseeds: Stage C student generation seeds.
        q_a1: A1 train queue name.
        q_a2: A2 train queue name.
        q_a3: A3 train queue name.
        q_b_headline: Stage B headline train queue name.
        q_b_ext: Stage B extension train queue name.
        q_c: Stage C train queue name.
        q_a1_evals: A1 verify queue name.
        q_a3_evals: A3 verify queue name.
        q_b_evals: Stage B verify queue name.
        q_c_evals: Stage C verify queue name.
        q_backlog: Idle-filler backlog queue name.
        queue_priorities: Queue name → jobq priority.
        gpu_policy: jobq GPU packing policy.
        a1_same_yaml: Static A1 same-init train YAML (repo-relative).
        a1_cross_yaml: Static A1 cross-init train YAML.
        a1_same_verify_yaml: Static A1 same-init verify YAML.
        a1_cross_verify_yaml: Static A1 cross-init verify YAML.
        generated_config_dir: Repo-relative dir for generated YAMLs.
        catswide_verify_yamls: Backlog verify YAML paths.
        catswide_base_models: Backlog base-model shards.
        calibration_report: A0 calibration report filename.
        a1_selection_report: A1 selection report filename.
        freeze_report: Recipe-freeze report filename.
        init_equality_report: A0 init-equality report filename.
        campaign_state_file: Driver state filename under logs/.
        campaign_done_file: Campaign-complete marker filename.
        control_log_file: Append-only pause/resume audit JSONL under logs/.
        eval_prompt_sets: Preference eval prompt-set list for verify YAMLs.
        gkd_experiment_config: Optional experiment YAML for train shards
            (``--experiment-config``); None keeps default
            ``configs/experiment/gkd_student.yaml``.
    """

    run_set_id: str
    teacher_run_set_id: str
    hub_org: str
    teacher_workflow: str
    teacher_param_tag_lora: str
    teacher_param_tag_full: str
    animals: tuple[str, ...]
    betas: tuple[float, ...]
    lmbda: float
    a1_student_modes: tuple[str, ...]
    a1_lora_lrs: tuple[float, ...]
    a1_lr_extension_factor: float
    a1_gseeds: tuple[int, ...]
    stage_a_tfs: tuple[int, ...]
    a1_same_pair: tuple[str, str]
    a1_cross_pair: tuple[str, str]
    a2_student_mode: str
    control_floor_max: float
    control_floor_max_spread: float
    a3_doses: tuple[float, ...]
    a3_top_k: int
    a3_gseeds: tuple[int, ...]
    a3_max_same_arms: int
    a3_reference_recipe: Recipe | None
    b_tfs_headline: tuple[int, ...]
    b_tfs_extension: tuple[int, ...]
    b_gseeds: tuple[int, ...]
    c_lambdas: tuple[float, ...]
    c_gseeds: tuple[int, ...]
    q_a1: str
    q_a2: str
    q_a3: str
    q_b_headline: str
    q_b_ext: str
    q_c: str
    q_a1_evals: str
    q_a3_evals: str
    q_b_evals: str
    q_c_evals: str
    q_backlog: str
    queue_priorities: Mapping[str, int]
    gpu_policy: Mapping[str, object]
    a1_same_yaml: str
    a1_cross_yaml: str
    a1_same_verify_yaml: str
    a1_cross_verify_yaml: str
    generated_config_dir: str
    catswide_verify_yamls: tuple[str, ...]
    catswide_base_models: tuple[str, ...]
    calibration_report: str
    a1_selection_report: str
    freeze_report: str
    init_equality_report: str
    campaign_state_file: str
    campaign_done_file: str
    control_log_file: str
    eval_prompt_sets: tuple[str, ...]
    gkd_experiment_config: str | None = None

    def queue_names(self) -> Sequence[str]:
        """Queue names in `queue_priorities` insertion order."""
        return tuple(self.queue_priorities)
