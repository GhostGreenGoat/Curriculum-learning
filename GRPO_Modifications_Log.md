# GRPO Training on Qwen2.5-1.5B: Configuration & Modifications Log

This document summarizes the key modifications and configurations applied to the `verl` framework to successfully train a **Qwen2.5-1.5B-Instruct** model using **GRPO (Group Relative Policy Optimization)** on the **Math-500** dataset. These changes address issues related to data formatting, training stability, efficient monitoring, and resource management.

## 1. Core Training Script: `run_grpo_math500.sh`

The launch script serves as the central configuration hub. Major adjustments were made to optimize for gradient stability, exploration, and hardware efficiency.

### Key Parameter Changes
*   **Model & Data Paths**:
    *   Updated `MODEL_PATH` to the local Qwen2.5-1.5B model.
    *   Pointed data paths to preprocessed `*_chat.parquet` files (see Data Preprocessing below).
*   **Batch Size Scaling (Crucial for Gradient)**:
    *   **`TRAIN_BATCH_SIZE=16`**: Increased from initial attempts (2 or 4) to 16. This is critical for GRPO to find "effective samples" (prompts where the model produces both correct and incorrect answers) within a single update step, preventing zero gradients.
    *   **`ROLLOUT_N=8`**: Sets the number of outputs sampled per prompt, providing the necessary group for advantage calculation.
    *   `PPO_MINI_BATCH_SIZE=4`: Kept small to manage GPU memory, using gradient accumulation to reach the effective global batch size.
*   **Exploration Strategy**:
    *   **`rollout.temperature=1.0`**: Explicitly set to 1.0 to encourage diverse generation. This prevents the model from deterministically repeating the same (correct or incorrect) answer, which would yield zero advantage and stall learning.
    *   `response_length=2048`: Increased to allow for long Chain-of-Thought (CoT) reasoning steps.
*   **Training Loop Efficiency**:
    *   **`trainer.test_freq=10`**: Validation runs every 10 steps (down from 2) to reduce overhead and speed up training.
    *   **`trainer.save_freq=150`**: Checkpoints are saved every 150 steps (down from 10) to save hundreds of GBs of disk space.
    *   `trainer.rollout_data_dir=rollout_data`: Configured to dump rollout samples (prompt, response, reward) to JSONL files for inspection.

## 2. Reward Function Logic: `verl/utils/reward_score/math_reward.py`

Significant robustness improvements were made to the reward function to ensure fair and accurate scoring.

*   **Robust Answer Extraction**:
    *   Added a **Regex Fallback** mechanism. If the model fails to output the standard `\boxed{}` Latex format, the system attempts to extract the answer using patterns like "The answer is X".
    *   Enhanced `remove_boxed` to handle non-standard spacing (e.g., `\boxed { 42 }`).
*   **Ground Truth Parsing Fix**:
    *   **Critical Fix**: The original logic compared the extracted answer directly against the raw Ground Truth string (which often contained Latex markup). We modified it to first extract the pure numerical answer from the Ground Truth before comparison.
*   **Debuggability**:
    *   Updated `compute_score` to return a dictionary `{'score': float, 'extracted_answer': str}`. This allows the extracted answer to be logged in the rollout JSONL files, making it easy to see *why* a model got a score of 0 or 1.

## 3. Data Preprocessing: `convert_data.py` (External Tool)

To avoid modifying the core `verl` data loader (`rl_dataset.py`) and breaking compatibility, we handled data issues externally.

*   **Format Conversion**: Created a script to read the raw Math-500 dataset and wrap the plain text prompts into the **ChatML format** required by the instruction-tuned model (e.g., `[{"role": "user", "content": "..."}]`).
*   **Result**: Generated `train_chat.parquet` and `val_chat.parquet`, resolving issues where the model received empty or malformed inputs.

## 4. Metric Stability: `verl/trainer/ppo/metric_utils.py`

*   **Null Safety**: Added filtering logic to `process_validation_metrics`. If the reward function returns `None` for a specific sample (e.g., due to parsing failure), it is skipped instead of crashing the entire training run with a `TypeError`.

## 5. System & Resource Management

*   **Git Configuration**: Added `rollout_data/` and `train.log` to `.gitignore` to prevent committing large log files.
*   **Checkpoint Management**: Manually cleaned up dense intermediate checkpoints (steps 10-80) and configured the new `save_freq` to 150 to maintain a manageable disk footprint.

---

**Summary**: These modifications transformed the setup from a basic demo into a functional, observable, and efficient research pipeline for RLHF/GRPO on mathematical reasoning tasks.

---

# 难题池采样 (Hard Pool) 设计方案（设计文档，暂不实现代码）

## 1. 需求摘要

- **难题定义**：某一步中，一道题在 n 次 rollout 下**均不得分**（该 step 内该题所有 response 的 reward 都为 0 或均未达标）。
- **每步采样**：每个 step 的 batch = **难题池中的题目** + **从数据集中正常采样的新题目**；难题池题目在 batch 中占比不超过可配置比例。
- **难题池维护**：
  - **移出（做对）**：若某题在后续某 step 的 rollout 中有任意一次得分，则从难题池移出。
  - **移出（超轮数）**：若某题在池中连续被采样达到可配置的 step 数后仍从未得分，则从难题池移出。
- **状态记录**：每道难题的**连续缠绕轮数**（被从难题池抽中并参与训练的 step 数）需要被记录并可被观测/持久化。

## 2. 与 verl 数据加载的对接点

- **Step 内 batch 来源**：`for batch_dict in self.train_dataloader`，由 `StatefulDataLoader(sampler=..., batch_size=...)` 决定；Sampler 提供 index，`dataset[idx]` + collate 得到 batch_dict。
- **GRPO 分组**：`batch.repeat(n, interleave=True)` 后同一 prompt 的 n 条共享 `uid`；`compute_advantage` 用 `data.non_tensor_batch["uid"]` 做 group。
- **Step 后回调**：若使用 `AbstractCurriculumSampler`，会调用 `self.train_dataloader.sampler.update(batch=batch)`。

因此需要：**(1) 采样侧** 在选 batch 时注入难题池（难题占比受控）；**(2) 反馈侧** step 结束后根据本步 reward 与「是否来自难题池」更新池。

**问题身份**：采用 **dataset index**（`dataset.__getitem__(idx)` 的 `idx`）作为 **problem_id**，与现有基于 index 的 Sampler/Dataset 一致。

**Batch 中必须携带**：`dataset_idx`（problem_id）和 `from_hard_pool`（布尔），以便 step 后按题聚合「是否至少一次得分」并只对来自池的题目更新连续缠绕轮数。

## 3. 难题池数据结构

### 3.1 核心结构

- `pool: dict[ProblemId, HardProblemEntry]`，ProblemId = dataset index（int）。
- **HardProblemEntry**：`problem_id`, `consecutive_steps`（连续缠绕轮数）, 可选 `first_seen_step`, `last_sampled_step`。
- **consecutive_steps 语义**：仅当该题**在本 step 被从难题池抽中并参与本 step 训练**时，在本 step 结束后 += 1；若本 step 是「正常采样」进来的，本 step 全错也不增加（加入池时从 0 开始）。

### 3.2 池的接口：HardPoolController

- **sample_for_step(batch_size, max_hard_ratio, rng)**  
  返回本 step 要用的 problem_id 列表及等长的 from_hard_pool。  
  内部：n_hard = min(len(pool), int(batch_size * max_hard_ratio))，从池中无放回抽 n_hard，其余从正常 Sampler 取；合并并打乱顺序。
- **update_after_step(batch_meta)**  
  入参：从本 step batch 提炼的摘要（见下）。  
  内部：本 step 全错的题目中未在池的加入池（consecutive_steps=0）；本 step 来自池的题目：若有任意一次得分则移出，否则 consecutive_steps += 1，若 >= max_consecutive_steps 则移出。
- **get_pool_snapshot() / load_pool_snapshot(snapshot)**：用于日志与 checkpoint 恢复。

### 3.3 传给 update_after_step 的 batch_meta

- `problem_ids`：本 step 每个 prompt 的 problem_id（repeat 前长度）。
- `from_hard_pool`：每个 prompt 是否来自难题池。
- `has_any_correct`：每个 prompt 在本 step 的 n 次 rollout 中是否至少有一次得分。

## 4. 工程化与可扩展设计

- **HardPoolController**：只依赖 batch_meta 与配置，不依赖 DataProto/Trainer，可单独单测和替换策略。
- **HardPoolSampler**：包装底层 Sampler，每步调用 sample_for_step，产出 (idx, from_hard_pool)；若 max_hard_ratio=0 或 enable=False 则退化为底层 Sampler。
- **HardPoolAwareDataset**：`__getitem__(self, item)`：若 item 为 (idx, from_pool)，则 data = base_dataset[idx]，并设 data["dataset_idx"]=idx、data["from_hard_pool"]=from_pool；否则 item 为 int 时 from_pool=False。Sampler 产出 (idx, from_pool)，DataLoader 传给 __getitem__，collate 后 batch 带这两项。
- **集成方式（推荐）**：实现 **HardPoolCurriculumSampler(AbstractCurriculumSampler)**，内部持有 HardPoolController 和底层 Sampler；`__iter__` 中按 batch 调用 sample_for_step；`update(batch)` 中从 batch 构建 batch_meta 并调用 update_after_step。这样通过「自定义 Sampler + 现有 update(batch) 钩子」接入，Trainer 无需改 fit() 分支，仅换 train_sampler 即可启用/关闭，易拔插。

## 5. 配置建议

在 data 或 data.hard_pool 下：

- `enable: true`
- `max_hard_ratio: 0.3`
- `max_consecutive_steps: 5`
- `save_in_checkpoint: true`

## 6. 数据流小结

1. **Step 开始前**：HardPoolSampler 调用 sample_for_step，产出 (idx, from_hard_pool)；HardPoolAwareDataset 返回带 dataset_idx、from_hard_pool 的样本；collate 后 batch_dict 含这两项。
2. **Step 内**：Trainer 照常 DataProto、uid、repeat(n)、rollout、reward、advantage、update；GRPO 仍用 uid；dataset_idx/from_hard_pool 仅用于 step 后更新。
3. **Step 结束后**：从 batch 按 prompt 聚合 has_any_correct，构建 BatchMetaForHardPool，调用 update_after_step（若用 HardPoolCurriculumSampler，该调用放在 sampler.update(batch) 内）。
4. **Checkpoint**：保存/加载 get_pool_snapshot() 与 dataloader.state_dict()。

## 7. 连续缠绕轮数的记录与暴露

- 在 HardProblemEntry 中维护 consecutive_steps，在 update_after_step 中对「本 step 来自池且仍未得分」的题目 += 1。
- 通过 get_pool_snapshot() 返回池内每题的 consecutive_steps，用于日志/TensorBoard/rollout_data；可选在 update_after_step 或 Trainer 中写 JSONL（step, problem_id, consecutive_steps, from_pool, has_any_correct）。
