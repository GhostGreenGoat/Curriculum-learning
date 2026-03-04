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

import unittest
import numpy as np
from verl.utils.dataset.hard_pool_controller import (
    HardPoolController,
    BatchMetaForHardPool,
    HardProblemEntry,
)


class TestHardPoolController(unittest.TestCase):
    def setUp(self):
        self.controller = HardPoolController(
            enable=True, max_hard_ratio=0.5, max_consecutive_steps=3, seed=42
        )

    def test_initialization(self):
        self.assertTrue(self.controller.enable)
        self.assertEqual(self.controller.max_hard_ratio, 0.5)
        self.assertEqual(self.controller.max_consecutive_steps, 3)
        self.assertEqual(len(self.controller.pool), 0)

    def test_sample_empty_pool(self):
        hard_ids, from_hard_pool = self.controller.sample_for_step(batch_size=10)
        self.assertEqual(hard_ids, [])
        self.assertEqual(from_hard_pool, [])

    def test_add_new_hard_problems(self):
        # Simulate a batch where problems 1 and 2 failed completely (normally sampled)
        problem_ids = np.array([1, 2, 3])
        from_hard_pool = np.array([False, False, False])
        has_any_correct = np.array([False, False, True])  # 3 passed, 1 and 2 failed

        batch_meta = BatchMetaForHardPool(problem_ids, from_hard_pool, has_any_correct)
        self.controller.update_after_step(batch_meta)

        self.assertIn(1, self.controller.pool)
        self.assertIn(2, self.controller.pool)
        self.assertNotIn(3, self.controller.pool)
        self.assertEqual(self.controller.pool[1].consecutive_steps, 0)

    def test_sample_from_pool(self):
        # Pre-fill pool
        for i in range(10):
            self.controller.pool[i] = HardProblemEntry(problem_id=i)

        # Batch size 10, max_ratio 0.5 -> should sample 5
        hard_ids, from_hard_pool = self.controller.sample_for_step(batch_size=10)
        
        self.assertEqual(len(hard_ids), 5)
        self.assertEqual(len(from_hard_pool), 5)
        self.assertTrue(all(from_hard_pool))
        for pid in hard_ids:
            self.assertIn(pid, self.controller.pool)

    def test_remove_solved_problems(self):
        # Add problem 1 to pool
        self.controller.pool[1] = HardProblemEntry(problem_id=1, consecutive_steps=0)

        # Simulate batch where 1 is sampled from pool and solved
        problem_ids = np.array([1])
        from_hard_pool = np.array([True])
        has_any_correct = np.array([True])

        batch_meta = BatchMetaForHardPool(problem_ids, from_hard_pool, has_any_correct)
        self.controller.update_after_step(batch_meta)

        self.assertNotIn(1, self.controller.pool)

    def test_increment_consecutive_steps(self):
        # Add problem 1 to pool
        self.controller.pool[1] = HardProblemEntry(problem_id=1, consecutive_steps=0)

        # Simulate batch where 1 is sampled from pool and fails again
        problem_ids = np.array([1])
        from_hard_pool = np.array([True])
        has_any_correct = np.array([False])

        batch_meta = BatchMetaForHardPool(problem_ids, from_hard_pool, has_any_correct)
        self.controller.update_after_step(batch_meta)

        self.assertIn(1, self.controller.pool)
        self.assertEqual(self.controller.pool[1].consecutive_steps, 1)

    def test_max_consecutive_steps_limit(self):
        # Add problem 1 to pool with steps = max - 1
        self.controller.pool[1] = HardProblemEntry(problem_id=1, consecutive_steps=2)
        # Max is 3

        # Fail again -> steps becomes 3 -> should be removed
        problem_ids = np.array([1])
        from_hard_pool = np.array([True])
        has_any_correct = np.array([False])

        batch_meta = BatchMetaForHardPool(problem_ids, from_hard_pool, has_any_correct)
        self.controller.update_after_step(batch_meta)

        self.assertNotIn(1, self.controller.pool)

    def test_normally_sampled_in_pool_logic(self):
        # Case: Problem 1 is in pool, but sampled normally (e.g. random sampler picked it)
        self.controller.pool[1] = HardProblemEntry(problem_id=1, consecutive_steps=1)

        # 1. Sampled normally and failed -> should NOT increment consecutive_steps
        batch_meta = BatchMetaForHardPool(
            np.array([1]), np.array([False]), np.array([False])
        )
        self.controller.update_after_step(batch_meta)
        self.assertEqual(self.controller.pool[1].consecutive_steps, 1)

        # 2. Sampled normally and solved -> should be removed
        batch_meta = BatchMetaForHardPool(
            np.array([1]), np.array([False]), np.array([True])
        )
        self.controller.update_after_step(batch_meta)
        self.assertNotIn(1, self.controller.pool)

    def test_snapshot(self):
        self.controller.pool[1] = HardProblemEntry(problem_id=1, consecutive_steps=2)
        snapshot = self.controller.get_pool_snapshot()

        new_controller = HardPoolController()
        new_controller.load_pool_snapshot(snapshot)

        self.assertEqual(new_controller.max_hard_ratio, 0.5)
        self.assertEqual(new_controller.max_consecutive_steps, 3)
        self.assertIn(1, new_controller.pool)
        self.assertEqual(new_controller.pool[1].consecutive_steps, 2)


if __name__ == "__main__":
    unittest.main()
