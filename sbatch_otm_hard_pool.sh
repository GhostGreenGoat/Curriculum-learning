#!/bin/bash
#SBATCH --job-name=otm_hardpool
#SBATCH --partition=zhl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --output=/export/home/zhaolei/laiminzhi/logs/otm_hardpool_%j.log
#SBATCH --error=/export/home/zhaolei/laiminzhi/logs/otm_hardpool_%j.log
#SBATCH --time=7-00:00:00

# ============================================================
# Hard Pool GRPO on OTM - SLURM Batch Script
# 用法:
#   新训练:  sbatch sbatch_otm_hard_pool.sh
#   断点续训: WANDB_EXP=otm_hardpool_YYYYMMDD_HHMM sbatch sbatch_otm_hard_pool.sh
# ============================================================

mkdir -p /export/home/zhaolei/laiminzhi/logs

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "WANDB_EXP: ${WANDB_EXP:-<auto>}"
echo "Start:     $(date)"
echo "=========================================="

cd /export/home/zhaolei/laiminzhi
bash run_grpo_otm_hard_pool.sh
EXIT_CODE=$?

echo "=========================================="
echo "End:       $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
