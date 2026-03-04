# 难题池采样设计方案

## 1. 需求摘要

- **难题定义**：某步中一道题在 n 次 rollout 下均不得分。
- **每步采样**：batch = 难题池题目 + 正常采样题目，难题占比不超过可配置比例。
- **池维护**：做对则移出；连续若干 step 仍不对则移出。
- **记录**：每道难题的连续缠绕轮数（被从池中抽中并参与训练的 step 数）。

## 2. 与 verl 的对接点

- Batch 来自 `train_dataloader`，由 Sampler 提供 index，Dataset 提供样本。
- GRPO 用 `uid` 做 group；需额外在 batch 中带 **problem_id**（题目身份）和 **from_hard_pool**（是否来自池），用于 step 后更新池。

**问题身份**：用 **dataset index** 作为 problem_id，与现有 index 机制一致。

## 3. 难题池数据结构

### 3.1 核心

- `pool: dict[ProblemId, HardProblemEntry]`
- **HardProblemEntry**：`problem_id`, `consecutive_steps`, 可选 `first_seen_step`/`last_sampled_step`
- **consecutive_steps**：仅当该题本 step 被从池抽中并参与训练时，本 step 结束后 += 1

### 3.2 接口 HardPoolController

- `sample_for_step(batch_size, max_hard_ratio)` -> (problem_ids, from_hard_pool)
- `update_after_step(batch_meta)`：入参含 problem_ids, from_hard_pool, has_any_correct（每 prompt 是否至少一次得分）
- `get_pool_snapshot()` / `load_pool_snapshot()`：持久化与恢复

### 3.3 batch_meta

- problem_ids, from_hard_pool, has_any_correct（按 prompt 聚合）

## 4. 工程化设计

- **HardPoolController**：独立于 Trainer/DataProto，可单测。
- **HardPoolSampler**：包装底层 Sampler，每步调用 sample_for_step，产出 (idx, from_hard_pool)；关闭时退化为底层 Sampler。
- **HardPoolAwareDataset**：`__getitem__(item)` 若 item=(idx, from_pool) 则返回 base_dataset[idx] 并注入 dataset_idx、from_hard_pool。
- **集成**：实现 `HardPoolCurriculumSampler(AbstractCurriculumSampler)`，在 `update(batch)` 中构建 batch_meta 并调用 update_after_step，这样 Trainer 无需改，仅换 Sampler 即可拔插。

## 5. 配置建议

```yaml
data.hard_pool:
  enable: true
  max_hard_ratio: 0.3
  max_consecutive_steps: 5
  save_in_checkpoint: true
```

## 6. 数据流
1. Sampler 调用 sample_for_step，产出 (idx, from_pool)；Dataset 返回带 dataset_idx、from_hard_pool 的样本。
2. Step 内照常训练；step 结束后从 batch 聚合 has_any_correct，调用 update_after_step。
3. Checkpoint 保存/加载 pool snapshot 与 dataloader state。

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
