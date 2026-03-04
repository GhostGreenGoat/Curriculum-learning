# 难题池采样 (Hard Pool Sampling) 设计方案

## 1. 需求摘要

- **难题定义**：在某一步中，一道题在 n 次 rollout 下**均不得分**（即该 step 内该题所有 response 的 reward 都为 0 或均未达标）。
- **每步采样**：每个 step 的 batch = **难题池中的题目** + **从数据集中正常采样的新题目**；难题池题目在 batch 中占比不超过可配置比例。
- **难题池维护**：
  - **移出（做对）**：若某题在后续某 step 的 rollout 中有任意一次得分，则从难题池移出。
  - **移出（超轮数）**：若某题在池中连续被采样达到可配置的 step 数后仍从未得分，则从难题池移出。
- **状态记录**：每道难题的**连续缠绕轮数**（被从难题池抽中并参与训练的 step 数）需要被记录并可被观测/持久化。

## 2. 与 verl 数据加载的对接点

### 2.1 现有流程回顾

- **Step 内 batch 来源**：`RayPPOTrainer.fit()` 中 `for batch_dict in self.train_dataloader`，每个 step 一个 `batch_dict`。
- **Batch 组成**：由 `StatefulDataLoader(sampler=..., batch_size=...)` 决定；Sampler 每步提供一批 **index**，DataLoader 用这些 index 调用 `dataset[idx]`，再经 `collate_fn` 得到 `batch_dict`。
- **GRPO 分组**：`batch.repeat(n, interleave=True)` 后，同一 prompt 的 n 条 response 共享同一 `uid`；`compute_advantage` 用 `data.non_tensor_batch["uid"]` 做 group，计算 group 内 advantage。
- **Step 后回调**：若使用 `AbstractCurriculumSampler`，在每 step 结束后会调用 `self.train_dataloader.sampler.update(batch=batch)`，用于 curriculum 更新。

因此，要做的两件事是：

1. **采样侧**：在「选哪些题」时注入难题池（控制每步 batch = 难题 + 正常题，且难题占比受控）。
2. **反馈侧**：在 step 结束后根据本步 reward 与「是否来自难题池」更新难题池（加入新难题、移出做对/超轮数题目、更新连续缠绕轮数）。

### 2.2 问题身份：dataset index 作为 problem_id

- 同一道题在不同 step 必须可识别，因此需要**稳定的问题 ID**。
- 采用 **dataset index**（即 `dataset.__getitem__(idx)` 的 `idx`）作为 **problem_id**：
  - 与当前 DataLoader/Sampler 基于 index 的机制一致；
  - 不依赖 parquet 内是否有唯一 id，也不依赖 token 内容；
  - 便于实现「从池中抽 index」和「从 dataset 取题」。
- 若未来要支持多数据源或 dataset 动态变化，可再扩展为 `(dataset_id, index)` 或 content hash，接口上仍可抽象为 `problem_id`。

### 2.3 Batch 中必须携带的字段（用于 step 后更新难题池）

Step 结束后需要按「题目」做两件事：

1. **判断是否为本 step 的难题**：对每个 prompt（即每个 problem_id），看其 n 条 rollout 的 reward 是否全为 0（或全未达标）→ 若是则可能加入难题池。
2. **更新池内题目**：对「本 step 来自难题池」的题目，根据本 step 是否得分决定移出或增加连续缠绕轮数；若连续缠绕轮数达到上限则移出。

因此 batch 中需要：

- **dataset_idx**（或统一叫 **problem_id**）：形状与 batch 中「样本数」一致，且 `repeat(n, interleave=True)` 后，同一 prompt 的 n 条共享同一个 problem_id。  
  - 这样可按 problem_id 聚合得到「每题在本 step 是否至少有一次得分」。
- **from_hard_pool**：布尔数组，标记该样本是否来自难题池（便于只对池内题目更新连续缠绕轮数，以及区分「新发现的难题」与「池内老题」）。

这两项应在 **DataLoader → Dataset → batch_dict** 的链路中注入，并在后续 `DataProto`、`repeat`、reward 计算、`update(batch)` 中保持可用（不依赖 uid，uid 仍用于 GRPO 的 group）。

---

## 3. 难题池的数据结构

### 3.1 核心结构建议

```text
HardPoolState:
  - pool: dict[ProblemId, HardProblemEntry]   # 当前池内题目
  - (可选) evicted_solved: list[(ProblemId, step)]   # 最近移出(做对)记录，用于日志/分析
  - (可选) evicted_max_steps: list[(ProblemId, step)]  # 最近移出(超轮数)记录
```

其中：

- **ProblemId**：即 `dataset index`（int），或后续扩展的 `(source_id, index)` 等。
- **HardProblemEntry**（每个池内题目一条）：

```text
HardProblemEntry:
  - problem_id: ProblemId
  - consecutive_steps: int     # 连续缠绕轮数：被从池中抽中并参与训练的 step 数
  - first_seen_step: int       # 可选，首次进入池的 global step
  - last_sampled_step: int     # 可选，上次被采样的 step
```

- **consecutive_steps** 的语义：仅当该题**在本 step 被从难题池中抽中并参与本 step 训练**时，在本 step 结束后对其 `consecutive_steps += 1`。  
  - 若某题本 step 是「正常采样」进来的（不在池中），本 step 即使全错也不增加其未来的 consecutive_steps（它若被加入池，从 0 开始计）。

### 3.2 池的接口（便于实现与测试）

建议抽象为一个 **HardPoolController**（或类似名），与「采样」和「step 后更新」解耦：

- **采样**  
  - `sample_for_step(batch_size, max_hard_ratio, rng) -> (list[ProblemId], list[bool])`  
  - 返回本 step 要用的 `problem_id` 列表（长度 = batch_size），以及等长的 `from_hard_pool`。  
  - 内部逻辑：`n_hard = min(len(pool), int(batch_size * max_hard_ratio))`，从池中无放回抽 n_hard 个，其余从「正常」源取；返回时可与正常采样结果合并并打乱顺序，同时返回每个位置是否来自池。
- **更新**  
  - `update_after_step(batch_meta)`  
  - 入参为从本 step 的 `batch` 中提炼的摘要（见下），不直接依赖 DataProto，便于单测和替换实现。  
  - 内部：  
    - 根据 `batch_meta` 找出本 step 全错的题目，将其中**未在池中**的加入池（consecutive_steps=0）；  
    - 对**本 step 来自池**的题目：若有任意一次得分则移出；否则 `consecutive_steps += 1`，若 `consecutive_steps >= max_consecutive_steps` 则移出。
- **查询/持久化（可选）**  
  - `get_pool_snapshot() -> dict`：当前池内题目及 consecutive_steps 等，用于日志、checkpoint、恢复。  
  - `load_pool_snapshot(snapshot)`：从 checkpoint 恢复池状态。

这样「池的数据结构」和「如何采样、如何从 batch 提炼信息」都集中在 HardPoolController 内，与 verl 的耦合仅通过「Sampler 调 sample_for_step」和「Trainer 在 step 后传 batch_meta 给 update_after_step」。

### 3.3 Batch 摘要：传给 update_after_step 的 batch_meta

为避免 Trainer 依赖具体 DataProto 结构，建议在 Trainer 中从 `batch` 计算一次「按题聚合」的摘要，再交给 HardPoolController：

```text
BatchMetaForHardPool:
  - problem_ids: np.ndarray or list[int]   # 本 step 每个「prompt」的 problem_id，长度 = batch_size 的 prompt 数（repeat 前）
  - from_hard_pool: np.ndarray (bool)      # 同上，是否来自难题池
  - has_any_correct: np.ndarray (bool)     # 每个 prompt 在本 step 的 n 次 rollout 中是否至少有一次「得分」
```

- `has_any_correct`：可由 `batch` 中 `token_level_scores` 或 reward 按 `uid`（或按 problem_id）聚合得到，例如 `(reward_sum > 0)` 或按你定义的「得分」阈值。
- 若使用 `problem_id` 做 GRPO 的 group，也可与现有 `uid` 逻辑并存（例如先按 uid 聚合得到 per-prompt 的 reward，再与 problem_id 对齐）。

---

## 4. 工程化与可扩展设计

### 4.1 模块划分（易拔插、易扩展）

- **HardPoolController**（难题池状态 + 采样/更新接口）  
  - 只依赖：`batch_meta` 结构、配置（max_ratio、max_consecutive_steps 等）。  
  - 不依赖 verl 的 DataProto、Trainer、Sampler 实现；可单独单测和替换（例如换一种「难题」定义或移出策略）。

- **HardPoolSampler / HardPoolBatchSampler**（采样层）  
  - 依赖：HardPoolController、底层「正常」Sampler、Dataset 长度、batch_size、max_hard_ratio。  
  - 行为：每步向 Controller 请求 `sample_for_step(...)`，得到本步的 (problem_ids, from_hard_pool)，再与底层 Sampler 补齐 batch_size，并输出**带 problem_id 与 from_hard_pool 的 index 序列**（见下）。  
  - 若配置中关闭难题池（例如 max_hard_ratio=0 或 enable=False），则退化为仅用底层 Sampler，不调用 Controller。

- **Dataset 侧**：保证 batch 中带有 `dataset_idx` 与 `from_hard_pool`  
  - 方式一（推荐）：**薄包装 Dataset**  
    - `HardPoolAwareDataset(base_dataset)`：  
      - `__getitem__(self, item)`：若 `item` 为 `(idx, from_pool)`，则 `data = base_dataset[idx]`，并设置 `data["dataset_idx"] = idx`、`data["from_hard_pool"] = from_pool`；若 `item` 为 int，则视为普通 index，`from_pool=False`。  
    - 这样 Sampler 只需产出 `(idx, from_hard_pool)`，DataLoader 会把这些传给 `__getitem__`，collate 后 batch 中自然带这两项。  
  - 方式二：在现有 RLHFDataset 的 `__getitem__(idx)` 中始终写入 `data["dataset_idx"] = idx`，再由 Sampler 只产出 index，在 collate 或 Trainer 里根据「本 batch 的 index 列表」与 Controller 记录的「本 step 哪些 index 来自池」补写 `from_hard_pool`。方式二需要 Trainer 或 collate 能拿到「本 step 的 from_hard_pool」，实现上不如方式一清晰。

- **Trainer 集成**  
  - 在 `fit()` 中：若启用难题池，则使用 `HardPoolSampler` + `HardPoolAwareDataset`；否则保持原有 Sampler/Dataset。  
  - 每个 step 结束后：从当前 `batch` 构建 `BatchMetaForHardPool`，调用 `hard_pool_controller.update_after_step(batch_meta)`。  
  - Checkpoint：若需要断点续训，将 `get_pool_snapshot()` 写入 checkpoint，恢复时 `load_pool_snapshot()`；Sampler 的 state 若依赖池，也可一并保存/恢复（StatefulDataLoader 已支持 state_dict）。

### 4.2 配置建议（与现有 data / trainer 配置兼容）

在 `data` 或单独一节（如 `data.hard_pool`）下增加：

```yaml
# 是否启用难题池
enable: true
# 难题在 batch 中最大占比 (0.0 ~ 1.0)
max_hard_ratio: 0.3
# 池内题目连续多少 step 未得分则移出
max_consecutive_steps: 5
# 可选：是否在 checkpoint 中保存/加载池状态
save_in_checkpoint: true
```

「得分」的判定若与现有 reward 不一致（例如用二值 correct），可在同一配置下增加 `reward_key` 或 `correct_threshold` 等，由 `BatchMetaForHardPool` 的构造逻辑读取。

### 4.3 与 AbstractCurriculumSampler 的关系

- 现有 `AbstractCurriculumSampler.update(batch)` 是「按 batch 更新 curriculum」的入口；难题池的「按 step 更新」在语义上就是 curriculum 的一种。  
- 两种集成方式：  
  - **方式 A**：实现一个 `HardPoolCurriculumSampler(AbstractCurriculumSampler)`，内部持有一个 HardPoolController 和底层 Sampler；在 `__iter__` 中按 batch 调用 Controller 的 `sample_for_step`，在 `update(batch)` 中从 batch 构建 batch_meta 并调用 `update_after_step`。这样无需改 Trainer 的 `fit()` 逻辑，只要把 `train_sampler` 换成 `HardPoolCurriculumSampler` 即可，完全符合现有「curriculum sampler + update(batch)」的扩展方式。  
  - **方式 B**：在 Trainer 中写死「若启用 hard_pool，则 step 后调用 controller.update_after_step」；Sampler 只负责「采哪一批 index + from_pool」。  
- 推荐 **方式 A**：难题池完全通过「自定义 Sampler + 现有 update(batch) 钩子」接入，不增加 Trainer 内的分支，易拔插（关闭时换回原 Sampler 即可）。

### 4.4 连续缠绕轮数的记录与暴露

- **记录**：在 `HardProblemEntry` 中维护 `consecutive_steps`，在 `update_after_step` 中对「本 step 来自池且本 step 仍未得分」的题目做 `consecutive_steps += 1`。  
- **暴露**：  
  - 通过 `get_pool_snapshot()` 返回当前池内每题的 `consecutive_steps`（及可选 first_seen_step、last_sampled_step），用于日志、TensorBoard、或保存到 rollout_data 等。  
  - 若需要「每个 step 写一条记录」，可在 `update_after_step` 内或 Trainer 中根据 snapshot 写 JSONL/CSV（例如 step、problem_id、consecutive_steps、from_pool、has_any_correct），便于后续分析。

---

## 5. 数据流小结

1. **Step 开始前**  
   - DataLoader 向 Sampler 要一批样本。  
   - HardPoolSampler 调用 `HardPoolController.sample_for_step(batch_size, max_hard_ratio)`，得到本步的 (hard_indices, normal_indices) 及每个位置的 from_hard_pool，并产出 `(idx, from_hard_pool)` 序列（或先 index 再在 Dataset 层注入 from_hard_pool）。  
   - Dataset（或 HardPoolAwareDataset）返回带 `dataset_idx` 与 `from_hard_pool` 的样本；collate 后 batch_dict 含有这两项。

2. **Step 内**  
   - Trainer 照常做 `DataProto.from_single_dict(batch_dict)`、加 uid、repeat(n)、rollout、reward、advantage、update。  
   - GRPO 仍用 `uid` 做 group；`dataset_idx` / `from_hard_pool` 仅用于 step 后的池更新。

3. **Step 结束后**  
   - 从 `batch` 按 prompt（uid 或 dataset_idx）聚合得到每个 prompt 的「是否至少有一次得分」。  
   - 构建 `BatchMetaForHardPool(problem_ids, from_hard_pool, has_any_correct)`，调用 `HardPoolController.update_after_step(batch_meta)`。  
   - 若使用 HardPoolCurriculumSampler，上述调用放在 `sampler.update(batch)` 内即可，Trainer 无需改。

4. **Checkpoint / 恢复**  
   - 保存：将 `HardPoolController.get_pool_snapshot()` 与 dataloader.state_dict() 一起写入。  
   - 恢复：`load_pool_snapshot(...)` 恢复池；dataloader.load_state_dict(...) 恢复 Sampler 迭代位置。

---

## 6. 小结

- **难题池数据结构**：`dict[ProblemId, HardProblemEntry]`，其中 `HardProblemEntry` 至少含 `consecutive_steps`，可选 `first_seen_step` / `last_sampled_step`；ProblemId 使用 dataset index。  
- **控制逻辑**：封装在 **HardPoolController**（采样接口 + 更新接口 + 可选 snapshot/load），与框架通过「Sampler 调采样」和「update(batch) 传 batch_meta」对接。  
- **集成方式**：通过 **HardPoolCurriculumSampler + HardPoolAwareDataset** 接入，实现 `AbstractCurriculumSampler`，利用现有 `update(batch)` 钩子，无需在 Trainer 中写死分支，便于开关和扩展。  
- **连续缠绕轮数**：在池内条目上维护，在 `update_after_step` 中更新，通过 `get_pool_snapshot()` 或自定义日志暴露。

按上述设计实现后，可以在不破坏现有 GRPO 与数据加载流程的前提下，以可配置、可拔插的方式支持「难题池 + 比例限制 + 做对/超轮移出 + 连续缠绕轮数记录」。

## 7. 实现代码 (Implementation Code)

### 7.1 HardPoolController (`verl/utils/dataset/hard_pool_controller.py`)

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import random
import numpy as np


@dataclass
class HardProblemEntry:
    problem_id: int
    consecutive_steps: int = 0
    first_seen_step: Optional[int] = None
    last_sampled_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "consecutive_steps": self.consecutive_steps,
            "first_seen_step": self.first_seen_step,
            "last_sampled_step": self.last_sampled_step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardProblemEntry":
        return cls(
            problem_id=data["problem_id"],
            consecutive_steps=data.get("consecutive_steps", 0),
            first_seen_step=data.get("first_seen_step"),
            last_sampled_step=data.get("last_sampled_step"),
        )


@dataclass
class BatchMetaForHardPool:
    problem_ids: np.ndarray  # Shape: (batch_size,)
    from_hard_pool: np.ndarray  # Shape: (batch_size,), boolean
    has_any_correct: np.ndarray  # Shape: (batch_size,), boolean indicating if any rollout was correct


class HardPoolController:
    """
    Manages a pool of hard problems that the model consistently fails to solve.
    Provides sampling logic to mix hard problems into training batches and updates the pool based on results.
    """

    def __init__(
        self,
        enable: bool = True,
        max_hard_ratio: float = 0.5,
        max_consecutive_steps: int = 5,
        seed: int = 42,
    ):
        self.enable = enable
        self.max_hard_ratio = max_hard_ratio
        self.max_consecutive_steps = max_consecutive_steps
        self.rng = random.Random(seed)

        # Map from problem_id (dataset index) to HardProblemEntry
        self.pool: Dict[int, HardProblemEntry] = {}

    def set_seed(self, seed: int):
        self.rng.seed(seed)

    def sample(self, k: int) -> List[int]:
        """
        Sample k problems from the hard pool.
        If k > len(pool), we sample with replacement (or just all of them repeated).
        If k <= len(pool), we sample without replacement.
        """
        if not self.pool:
            return []
        
        pool_ids = list(self.pool.keys())
        if k <= len(pool_ids):
            return self.rng.sample(pool_ids, k=k)
        else:
            # If we need more than we have, we repeat the pool
            # For simplicity, just random choices with replacement
            return self.rng.choices(pool_ids, k=k)

    def sample_for_step(self, batch_size: int) -> Tuple[List[int], List[bool]]:
        """
        Determines which hard problems to include in the current step's batch.

        Args:
            batch_size: The total size of the batch (number of prompts).

        Returns:
            Tuple of:
                - List[int]: The problem_ids selected from the hard pool.
                - List[bool]: Flags indicating these are from the hard pool (all True).
        """
        if not self.enable or not self.pool:
            return [], []

        # Calculate how many hard samples to include
        n_hard = min(len(self.pool), int(batch_size * self.max_hard_ratio))
        
        if n_hard == 0:
            return [], []

        # Randomly sample from the pool without replacement
        hard_ids = self.rng.sample(list(self.pool.keys()), k=n_hard)
        
        return hard_ids, [True] * len(hard_ids)

    def update_after_step(self, batch_meta: BatchMetaForHardPool) -> Dict[str, int]:
        """
        Update the hard pool based on the results of the current step.

        Args:
            batch_meta: Metadata about the batch execution results.
            
        Returns:
            Dict with keys: 'added', 'removed_success', 'removed_max_steps'
        """
        if not self.enable:
            return {}

        problem_ids = batch_meta.problem_ids
        from_hard_pool = batch_meta.from_hard_pool
        has_any_correct = batch_meta.has_any_correct
        
        stats = {
            "added": 0,
            "removed_success": 0,
            "removed_max_steps": 0
        }

        if not (len(problem_ids) == len(from_hard_pool) == len(has_any_correct)):
            raise ValueError(
                f"Batch meta length mismatch: problem_ids={len(problem_ids)}, "
                f"from_hard_pool={len(from_hard_pool)}, has_any_correct={len(has_any_correct)}"
            )

        for i in range(len(problem_ids)):
            pid = int(problem_ids[i])
            is_from_pool = bool(from_hard_pool[i])
            is_correct = bool(has_any_correct[i])
            
            # Helper to check if pid is in pool
            in_pool = pid in self.pool

            if is_from_pool:
                # Case 1: Sampled from Hard Pool
                if not in_pool:
                    # Should not happen typically unless pool was reset externally
                    continue

                if is_correct:
                    # Solved! Remove from pool.
                    del self.pool[pid]
                    stats["removed_success"] += 1
                else:
                    # Still failed. Increment consecutive steps.
                    entry = self.pool[pid]
                    entry.consecutive_steps += 1
                    entry.last_sampled_step = None  # TODO: pass step info if needed

                    # Check if it exceeded the limit
                    if entry.consecutive_steps >= self.max_consecutive_steps:
                        # Failed too many times consecutively, remove from pool
                        del self.pool[pid]
                        stats["removed_max_steps"] += 1
            else:
                # Case 2: Sampled Normally (from dataset)
                if not is_correct:
                    # Failed completely (all rollouts wrong).
                    # Add to pool if not already present.
                    if not in_pool:
                        self.pool[pid] = HardProblemEntry(problem_id=pid, consecutive_steps=0)
                        stats["added"] += 1
                    else:
                        # Already in pool but sampled normally by chance.
                        # Do nothing (consecutive_steps only increments when sampled as hard).
                        pass
                else:
                    # Solved normally.
                    # If it was in the pool, remove it (it might have been sampled normally by chance).
                    if in_pool:
                        del self.pool[pid]
                        stats["removed_success"] += 1
        
        return stats

    def get_pool_snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the controller state."""
        return {
            "config": {
                "max_hard_ratio": self.max_hard_ratio,
                "max_consecutive_steps": self.max_consecutive_steps,
                "enable": self.enable,
            },
            "pool": {str(pid): entry.to_dict() for pid, entry in self.pool.items()},
            "rng_state": self.rng.getstate(),
        }

    def load_pool_snapshot(self, snapshot: Dict[str, Any]):
        """Restore controller state from a snapshot."""
        if "config" in snapshot:
            cfg = snapshot["config"]
            self.max_hard_ratio = cfg.get("max_hard_ratio", self.max_hard_ratio)
            self.max_consecutive_steps = cfg.get("max_consecutive_steps", self.max_consecutive_steps)
            self.enable = cfg.get("enable", self.enable)

        if "pool" in snapshot:
            self.pool = {}
            for pid_str, entry_dict in snapshot["pool"].items():
                pid = int(pid_str)
                self.pool[pid] = HardProblemEntry.from_dict(entry_dict)
        
        if "rng_state" in snapshot:
            self.rng.setstate(snapshot["rng_state"])
```

### 7.2 HardPoolSampler (`verl/utils/dataset/hard_pool_sampler.py`)

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import random
from torch.utils.data import Dataset
from typing import Iterator, List, Tuple, Union, Optional, Any, Dict
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.utils.dataset.hard_pool_controller import HardPoolController, BatchMetaForHardPool


class HardPoolAwareDataset(Dataset):
    """
    Wrapper for a dataset to handle hard pool sampling.
    Injects 'dataset_idx' and 'from_hard_pool' into the sample data.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: Union[int, Tuple[int, bool]]):
        if isinstance(item, tuple):
            idx, from_hard_pool = item
        else:
            idx = item
            from_hard_pool = False
        
        data = self.dataset[idx]
        
        # Inject metadata
        if isinstance(data, dict):
            # IMPORTANT: Copy the dict to avoid modifying the original dataset cache/reference
            data = dict(data)
            
            # Use extra_info dict to persist metadata through RayPPOTrainer's batch processing
            # RayPPOTrainer pops keys that are not in reward_model_keys (which includes extra_info)
            if "extra_info" not in data:
                data["extra_info"] = {}
            else:
                data["extra_info"] = data["extra_info"].copy() # Defensive copy
                
            data["extra_info"]["dataset_idx"] = idx
            data["extra_info"]["from_hard_pool"] = from_hard_pool
            
            # Keep top-level keys for potential other uses
            data['dataset_idx'] = idx
            data['from_hard_pool'] = from_hard_pool
        
        return data

    def __getattr__(self, name):
        # Delegate other attributes to the underlying dataset
        return getattr(self.dataset, name)


class HardPoolSampler(AbstractCurriculumSampler):
    """
    Sampler that mixes hard problems from the pool with normal samples.
    """

    def __init__(
        self,
        data_source: Dataset,
        data_config: DictConfig,
        batch_size: Optional[int] = None,
    ):
        self.data_source = data_source
        self.data_config = data_config
        # If batch_size is not provided, try to get it from config
        self.batch_size = batch_size or data_config.get("train_batch_size", 1024)

        # Initialize Controller
        hard_pool_config = data_config.get("hard_pool", {})
        self.controller = HardPoolController(
            enable=hard_pool_config.get("enable", True),
            max_hard_ratio=hard_pool_config.get("max_hard_ratio", 0.5),
            max_consecutive_steps=hard_pool_config.get("max_consecutive_steps", 5),
            seed=data_config.get("seed", 42),
        )

        self.epoch = 0
        seed = data_config.get("seed")
        self.seed = seed if seed is not None else 42
        # RNG for batch shuffling (controlled seed)
        self.batch_shuffle_rng = random.Random(self.seed)

        # Internal state for normal sampling
        self.num_samples = len(data_source)
        # Calculate number of batches per epoch (drop_last=True assumed)
        if self.batch_size > 0:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = 0

        # For state saving/loading
        self.current_iter_idx = 0
        self.normal_ptr = 0

    def __len__(self):
        # Estimate the total number of samples yielded in one epoch based on current pool size.
        # This ensures progress bars and LR schedulers are roughly correct.
        if not self.controller.enable or not self.controller.pool:
            return self.num_samples
            
        n_hard = min(len(self.controller.pool), int(self.batch_size * self.controller.max_hard_ratio))
        # Ensure we consume at least 1 normal sample per batch
        n_hard = min(n_hard, self.batch_size - 1)
        
        n_normal = self.batch_size - n_hard
        if n_normal < 1: 
            return self.num_samples
            
        # We assume hard samples are a subset of the dataset and we deduplicate them.
        # So we effectively consume (num_samples - n_hard) unique normal items per epoch.
        effective_samples = self.num_samples - n_hard
        if effective_samples < 0:
             effective_samples = 0
             
        num_batches = effective_samples // n_normal
        return num_batches * self.batch_size

    def __iter__(self) -> Iterator[Tuple[int, bool]]:
        # Reset iteration state counter (for logging/debugging, not control flow)
        self.current_iter_idx = 0

        # Generator for normal samples
        # If shuffle is enabled, use randperm; otherwise use sequential
        shuffle = self.data_config.get("shuffle", True)
        if shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            normal_indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            normal_indices = list(range(self.num_samples))

        # Iterate until we exhaust normal samples
        # We use self.normal_ptr to support mid-epoch resume
        while self.normal_ptr < self.num_samples:
            # 1. Get hard samples from controller
            hard_ids, from_hard_pool = self.controller.sample_for_step(self.batch_size)
            
            # Enforce at least one normal sample per batch to guarantee progress
            if len(hard_ids) >= self.batch_size:
                hard_ids = hard_ids[:self.batch_size - 1]
                from_hard_pool = from_hard_pool[:self.batch_size - 1]
                
            n_hard = len(hard_ids)

            # 2. Get normal samples with deduplication
            n_normal = self.batch_size - n_hard
            batch_normal_indices = []
            
            hard_ids_set = set(hard_ids)
            
            # Try to fill n_normal samples, skipping collisions
            while len(batch_normal_indices) < n_normal and self.normal_ptr < self.num_samples:
                idx = normal_indices[self.normal_ptr]
                self.normal_ptr += 1
                
                # Deduplication: if idx is already in hard_ids, we consider it "consumed" by the hard part
                # and skip adding it to the normal part of this batch.
                if idx not in hard_ids_set:
                    batch_normal_indices.append(idx)
            
            # 3. Batch Filling Logic (Tail handling)
            # If we don't have enough normal samples to fill the batch, we do NOT break.
            # Instead, we fill the rest with hard samples.
            current_batch_size = len(hard_ids) + len(batch_normal_indices)
            if current_batch_size < self.batch_size:
                needed = self.batch_size - current_batch_size
                # Request more hard samples to fill the gap
                more_hard = self.controller.sample(needed)
                if more_hard:
                    hard_ids.extend(more_hard)
                    from_hard_pool.extend([True] * len(more_hard))

            # Combine
            batch_indices = hard_ids + batch_normal_indices
            batch_flags = from_hard_pool + [False] * len(batch_normal_indices)

            # Shuffle the batch to mix hard and normal using controlled RNG
            combined = list(zip(batch_indices, batch_flags))
            
            # Only shuffle within batch if global shuffle is enabled
            if self.data_config.get("shuffle", True):
                self.batch_shuffle_rng.shuffle(combined)

            for idx, is_hard in combined:
                yield idx, is_hard
                self.current_iter_idx += 1
        
        # End of epoch: reset state for next epoch
        self.normal_ptr = 0
        # self.epoch is incremented by set_epoch() call from Trainer

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # Reseed batch shuffle RNG for determinism per epoch
        self.batch_shuffle_rng.seed(self.seed + self.epoch)

    def state_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "controller": self.controller.get_pool_snapshot(),
            "current_iter_idx": self.current_iter_idx,
            "normal_ptr": self.normal_ptr,
            "batch_shuffle_rng": self.batch_shuffle_rng.getstate(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.epoch = state_dict.get("epoch", 0)
        if "controller" in state_dict:
            self.controller.load_pool_snapshot(state_dict["controller"])
        self.current_iter_idx = state_dict.get("current_iter_idx", 0)
        self.normal_ptr = state_dict.get("normal_ptr", 0)
        if "batch_shuffle_rng" in state_dict:
            self.batch_shuffle_rng.setstate(state_dict["batch_shuffle_rng"])

    def update(self, batch: DataProto) -> Dict[str, Any]:
        """
        Update the hard pool based on the batch results.
        Called by the Trainer after each step.
        Returns a dict of metrics.
        """
        # Extract metadata from DataProto
        non_tensor = batch.non_tensor_batch
        
        dataset_idx_list = []
        from_hard_pool_list = []
        
        # Try to get dataset_idx from extra_info (preferred, as it persists through RayPPOTrainer processing)
        if "extra_info" in non_tensor:
            # extra_info is np.ndarray of objects (dicts)
            extra_info_list = non_tensor["extra_info"]
            for item in extra_info_list:
                if isinstance(item, dict) and "dataset_idx" in item:
                    dataset_idx_list.append(item["dataset_idx"])
                    from_hard_pool_list.append(item.get("from_hard_pool", False))
        
        # Fallback to top-level if not found in extra_info
        if len(dataset_idx_list) == 0 and "dataset_idx" in non_tensor:
            dataset_idx_list = non_tensor["dataset_idx"]
            from_hard_pool_list = non_tensor.get("from_hard_pool", [False] * len(dataset_idx_list))

        if len(dataset_idx_list) == 0:
            return {}

        # Calculate correctness (reward > 0)
        if "token_level_scores" in batch.batch.keys():
             scores = batch.batch["token_level_scores"].sum(dim=-1)
        elif "token_level_rewards" in batch.batch.keys():
             scores = batch.batch["token_level_rewards"].sum(dim=-1)
        else:
             return {}
        
        is_correct = (scores > 0).cpu().numpy()

        # Aggregate by dataset_idx (prompt)
        problem_ids = []
        is_from_hard_pool = []
        has_any_correct = []

        # Use a temporary dict to aggregate
        # pid -> {'is_hard': bool, 'corrects': list[bool]}
        agg = {}

        for i, pid in enumerate(dataset_idx_list):
            pid = int(pid)
            is_hard = bool(from_hard_pool_list[i])
            
            if pid not in agg:
                agg[pid] = {
                    "is_hard": is_hard,
                    "any_correct": False,
                }
            else:
                # Robust aggregation: if appeared as hard anywhere, mark as hard
                agg[pid]["is_hard"] = agg[pid]["is_hard"] or is_hard

            if is_correct[i]:
                agg[pid]["any_correct"] = True

        for pid, val in agg.items():
            problem_ids.append(pid)
            is_from_hard_pool.append(val["is_hard"])
            has_any_correct.append(val["any_correct"])

        if not problem_ids:
            return {}

        # Construct BatchMeta
        batch_meta = BatchMetaForHardPool(
            problem_ids=np.array(problem_ids),
            from_hard_pool=np.array(is_from_hard_pool),
            has_any_correct=np.array(has_any_correct),
        )

        # Update controller
        stats = self.controller.update_after_step(batch_meta)
        
        # Calculate additional metrics
        pool_size = len(self.controller.pool)
        consecutive_steps = [entry.consecutive_steps for entry in self.controller.pool.values()]
        avg_consecutive_steps = sum(consecutive_steps) / pool_size if pool_size > 0 else 0.0
        max_consecutive_steps = max(consecutive_steps) if pool_size > 0 else 0
        
        # Return metrics
        metrics = {
            "hard_pool/size": pool_size,
            "hard_pool/ratio": pool_size / self.num_samples if self.num_samples > 0 else 0.0,
            "hard_pool/added": stats.get("added", 0),
            "hard_pool/removed_success": stats.get("removed_success", 0),
            "hard_pool/removed_max_steps": stats.get("removed_max_steps", 0),
            "hard_pool/avg_consecutive_steps": avg_consecutive_steps,
            "hard_pool/max_consecutive_steps": max_consecutive_steps,
        }
        return metrics
```
