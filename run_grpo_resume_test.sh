#!/usr/bin/env bash
set -e

# Config
EXP_NAME="resume_test_$(date +%Y%m%d_%H%M%S)"
PROJECT_NAME="grpo_resume_test"
CHECKPOINT_DIR="$(pwd)/checkpoints/$PROJECT_NAME/$EXP_NAME"

MODEL_PATH="/home/ubuntu/date/models/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="/home/ubuntu/date/data/math500/train_chat.parquet"
VAL_DATA="/home/ubuntu/date/data/math500/val_chat.parquet"

# Common Args
ARGS="algorithm.adv_estimator=grpo \
    data.train_files=['$TRAIN_DATA'] \
    data.val_files=['$VAL_DATA'] \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CHECKPOINT_DIR"

echo "=== Phase 1: Train for 5 steps and save ==="
/home/ubuntu/date/conda_env_backup/verl/bin/python3 -m verl.trainer.main_ppo \
    $ARGS \
    trainer.save_freq=5 \
    trainer.total_training_steps=5 \
    2>&1 | tee phase1.log

# Verify checkpoint exists
CKPT_PATH="$CHECKPOINT_DIR/global_step_5"
if [ ! -d "$CKPT_PATH" ]; then
    echo "Checkpoint not found at $CKPT_PATH"
    exit 1
fi

echo "=== Phase 2: Resume from step 5 and train to 10 ==="
/home/ubuntu/date/conda_env_backup/verl/bin/python3 -m verl.trainer.main_ppo \
    $ARGS \
    trainer.total_training_steps=10 \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path=$CKPT_PATH \
    2>&1 | tee phase2.log

# Check logs for hard pool info
echo "=== Verification ==="
echo "Phase 1 Final Hard Pool Size:"
grep "hard_pool/size" phase1.log | tail -n 1
echo "Phase 2 Initial Hard Pool Size (should match Phase 1 Final or be close):"
grep "hard_pool/size" phase2.log | head -n 5
