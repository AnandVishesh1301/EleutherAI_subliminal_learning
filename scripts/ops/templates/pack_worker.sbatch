#!/bin/bash
# TEMPLATE: one shared-queue packer worker = ONE GPU. Submit as many copies as the QoS allows
# (a40-burst/normal: 16, a40/high: 4; add more later at any time -- they join the same queue and
# take jobs as soon as one of their GPU slots is free). Replace the UPPERCASE placeholders.
#
#   for i in $(seq 1 16); do sbatch --partition=a40-burst --qos=normal CAMPAIGN/worker.sbatch; done
#   for i in $(seq 1 4);  do sbatch --partition=a40 --qos=high CAMPAIGN/worker.sbatch; done
#
#SBATCH --job-name=JOBNAME
#SBATCH --account=soar_subliminal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=CAMPAIGN/logs/slurm-worker-%j.out
set -u
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd /mnt/ssd-cluster/homes/q_angela/EleutherAI_subliminal_learning
C=CAMPAIGN
echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} start=$(date -u +%FT%TZ)"
# Optional: MPS so the N jobs sharing this GPU overlap kernels instead of time-slicing (see slurm_mps.sh).
source scripts/ops/slurm_mps.sh && mps_start
uv run python -m scripts.ops.slurm_pack --jobs-file $C/jobs.jsonl --log-dir $C/logs/workers \
    --claim-dir $C/logs/claims --max-per-gpu MAX_PER_GPU --default-mem-mib DEFAULT_MEM_MIB \
    --reserve-mib 2000 --settle-s 20 --stale-s 900
mps_stop
