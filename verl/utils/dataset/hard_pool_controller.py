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
