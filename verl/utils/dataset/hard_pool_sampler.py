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
        # batch.batch['token_level_rewards'] shape: (total_samples, seq_len)
        # Prefer token_level_scores if available (raw score), otherwise token_level_rewards
        if "token_level_scores" in batch.batch.keys():
             scores = batch.batch["token_level_scores"].sum(dim=-1)
        elif "token_level_rewards" in batch.batch.keys():
             scores = batch.batch["token_level_rewards"].sum(dim=-1)
        else:
             # Should not happen in valid training loop
             return {}
        
        is_correct = (scores > 0).cpu().numpy()

        # Aggregate by dataset_idx (prompt)
        # Because of repeat(n, interleave=True), we have chunks of n for the same prompt.
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

        # Convert to lists for BatchMeta
        # Ensure consistent ordering if needed, but iteration order is fine
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
