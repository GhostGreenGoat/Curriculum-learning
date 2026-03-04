#!/usr/bin/env bash
set -x

# GRPO Dry Run - Hard Pool Integration Test

MODEL_PATH="/home/ubuntu/date/models/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="/home/ubuntu/date/data/math500/train_chat.parquet"
VAL_DATA="/home/ubuntu/date/data/math500/val_chat.parquet"

# Keep batch size divisible by total GPUs
N_GPUS=4
ROLLOUT_N=4
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=$TRAIN_BATCH_SIZE
MAX_NUM_SEQS=$((TRAIN_BATCH_SIZE * ROLLOUT_N))

export WANDB_PROJECT="grpo_math500_dryrun"
export WANDB_EXP="exp_dryrun_$(date +%Y%m%d_%H%M)"
# Use offline mode for dry run to avoid spamming wandb
export WANDB_MODE="offline"

/home/ubuntu/date/conda_env_backup/verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_DATA']" \
    data.val_files="['$VAL_DATA']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.train_max_samples=100 \
    data.val_max_samples=10 \
    +data.hard_pool.enable=True \
    +data.hard_pool.max_hard_ratio=0.5 \
    +data.hard_pool.max_consecutive_steps=2 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.checkpoint_engine.update_weights_bucket_megabytes=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_EXP \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.log_val_generations=0 \
    trainer.rollout_data_dir=rollout_data_dryrun \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \
    "$@"
