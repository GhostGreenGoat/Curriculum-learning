#!/usr/bin/env bash
set -x

ulimit -s unlimited 2>/dev/null
export LD_PRELOAD=""
unset ROCR_VISIBLE_DEVICES 2>/dev/null

# 正式实验: Hard Pool GRPO on OpenThoughts-Math (OTM) — Qwen2.5-Math-7B
# max_hard_ratio=0.2, 每步最多 20% 的 batch 由难题池注入
# 验证: MATH-500, AIME, AMC, Numina, Geometry3k
# 硬件: A800 (80GB) x 8
# 7B 关键变化: Ulysses SP=2, TP=2, batch_size=256

# ============== 路径配置 ==============
MODEL_PATH="/export/home/zhaolei/models/Qwen2.5-Math-7B"
TRAIN_DATA="/export/home/zhaolei/laiminzhi/data/train/openthoughts_math/train.parquet"

VAL_MATH500="/export/home/zhaolei/laiminzhi/data/test/benchmarks/math500.parquet"
VAL_AIME="/export/home/zhaolei/laiminzhi/data/test/benchmarks/aime.parquet"
VAL_AMC="/export/home/zhaolei/laiminzhi/data/test/benchmarks/amc.parquet"
VAL_NUMINA="/export/home/zhaolei/laiminzhi/data/test/benchmarks/numina.parquet"
VAL_GEO3K="/export/home/zhaolei/laiminzhi/data/test/benchmarks/geometry3k.parquet"

# ============== 训练超参 (A800 x 8, 7B model) ==============
N_GPUS=8
ROLLOUT_N=8
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE_PER_GPU=8
MAX_NUM_SEQS=256

# Ulysses sequence parallel: 训练时将序列切分到多个 GPU 上
# SP=2 → dp=4, 每个 dp 组内 2 卡做序列并行
SP_SIZE=2

# vLLM rollout tensor parallel: TP=1 → 8 replicas (7B fits single A800)
GEN_TP=1

# ============== 验证 ==============
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS != 0 )); then
  echo "ERROR: train_batch_size * rollout_n must be divisible by N_GPUS"
  exit 1
fi

export WANDB_PROJECT="grpo_otm_hard_pool_7b"
export WANDB_EXP="${WANDB_EXP:-otm_hardpool_7b_$(date +%Y%m%d_%H%M)}"

VERL_PYTHON="/export/home/zhaolei/anaconda3/envs/verl/bin/python3"
export VLLM_USE_V1=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

$VERL_PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$VAL_MATH500','$VAL_AIME','$VAL_AMC','$VAL_NUMINA','$VAL_GEO3K']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    +data.hard_pool.enable=True \
    +data.hard_pool.max_hard_ratio=0.2 \
    +data.hard_pool.max_consecutive_steps=30 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_EXP \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.log_val_generations=0 \
    trainer.rollout_data_dir=/export/home/zhaolei/laiminzhi/rollout_data_otm_hard_pool_7b \
    trainer.test_freq=50 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    "$@"
