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
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.tensordict_utils import TensorDict
from verl.utils.dataset.hard_pool_sampler import HardPoolAwareDataset, HardPoolSampler
from verl.utils.dataset.hard_pool_controller import HardProblemEntry

class MockDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.data = [{"id": i, "content": f"sample_{i}"} for i in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Handle dict wrapping logic simulation
        return dict(self.data[idx])

class TestHardPoolSampler(unittest.TestCase):
    def setUp(self):
        self.dataset = MockDataset(length=100)
        self.wrapped_dataset = HardPoolAwareDataset(self.dataset)
        self.config = OmegaConf.create({
            "train_batch_size": 10,
            "seed": 42,
            "hard_pool": {
                "enable": True,
                "max_hard_ratio": 0.5,
                "max_consecutive_steps": 3
            }
        })
        self.sampler = HardPoolSampler(
            data_source=self.wrapped_dataset,
            data_config=self.config,
            batch_size=10
        )

    def test_dataset_wrapper(self):
        # Test normal access
        item = self.wrapped_dataset[5]
        self.assertEqual(item['id'], 5)
        self.assertEqual(item['dataset_idx'], 5)
        self.assertFalse(item['from_hard_pool'])

        # Test tuple access (hard pool)
        item = self.wrapped_dataset[(10, True)]
        self.assertEqual(item['id'], 10)
        self.assertEqual(item['dataset_idx'], 10)
        self.assertTrue(item['from_hard_pool'])

    def test_sampler_iter(self):
        # Pre-fill pool with some items (0, 1, 2, 3, 4)
        for i in range(5):
            self.sampler.controller.pool[i] = HardProblemEntry(problem_id=i)

        # Create DataLoader with our sampler and wrapper
        loader = DataLoader(
            self.wrapped_dataset,
            batch_size=10,
            sampler=self.sampler, # Sampler yields (idx, is_hard)
            collate_fn=None 
        )
        
        # Iterate one batch
        # DataLoader automatically calls default_collate which handles list of dicts
        batch = next(iter(loader))
        
        # Check injected fields
        self.assertIn('dataset_idx', batch)
        self.assertIn('from_hard_pool', batch)
        
        # Check ratio (max_hard_ratio=0.5 -> 5 hard samples)
        # Since pool has 5 items and batch size is 10, we expect 5 hard items
        from_hard = batch['from_hard_pool'] # Tensor of bools
        self.assertEqual(from_hard.sum().item(), 5)
        
        # Check that hard items are indeed from our pool (ids 0-4)
        indices = batch['dataset_idx']
        hard_indices = indices[from_hard]
        for idx in hard_indices:
            self.assertTrue(idx.item() in range(5))

    def test_sampler_update(self):
        # Construct a dummy DataProto batch
        dataset_idx = np.array([0, 0, 1, 1])
        from_hard_pool = np.array([True, True, False, False])
        
        scores = torch.tensor([
            [0.0], [0.0], 
            [0.0], [0.0]
        ])
        
        non_tensor_batch = {
            "dataset_idx": dataset_idx,
            "from_hard_pool": from_hard_pool,
            "uid": np.array(["g0", "g0", "g1", "g1"])
        }
        
        batch_dict = {
            "token_level_scores": scores
        }
        
        batch = DataProto(
            batch=TensorDict(batch_dict, batch_size=[4], device='cpu'),
            non_tensor_batch=non_tensor_batch
        )
        
        self.sampler.controller.pool[0] = HardProblemEntry(problem_id=0, consecutive_steps=0)
        
        metrics = self.sampler.update(batch)
        
        self.assertEqual(self.sampler.controller.pool[0].consecutive_steps, 1)
        self.assertIn(1, self.sampler.controller.pool)
        self.assertEqual(self.sampler.controller.pool[1].consecutive_steps, 0)
        
        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("hard_pool/size", metrics)
        self.assertEqual(metrics["hard_pool/size"], 2) # 0 kept, 1 added
        
        self.assertIn("hard_pool/added", metrics)
        self.assertEqual(metrics["hard_pool/added"], 1)
        self.assertIn("hard_pool/removed_success", metrics)
        self.assertEqual(metrics["hard_pool/removed_success"], 0)
        self.assertIn("hard_pool/avg_consecutive_steps", metrics)
        # 0 has 1 consecutive step, 1 has 0 consecutive steps. Avg = 0.5
        self.assertEqual(metrics["hard_pool/avg_consecutive_steps"], 0.5)

    def test_sampler_epoch_coverage(self):
        # Pool has 5 items. Dataset has 100. Batch size 10.
        # Max hard ratio 0.5 -> n_hard=5, n_normal=5.
        # Effective normal items = 100 - 5 = 95.
        # Batches = 95 / 5 = 19.
        
        for i in range(5):
            self.sampler.controller.pool[i] = HardProblemEntry(problem_id=i)
            
        loader = DataLoader(
            self.wrapped_dataset,
            batch_size=10,
            sampler=self.sampler,
            collate_fn=None 
        )
        
        batches = list(loader)
        self.assertEqual(len(batches), 19)
        
        # Check that we covered all normal samples (ids 0-99)
        seen_indices = set()
        for batch in batches:
            indices = batch['dataset_idx']
            seen_indices.update(indices.tolist())
            
        # We expect all 100 indices to be seen
        self.assertEqual(len(seen_indices), 100)

    def test_sampler_len(self):
        # Pool has 5 items. Dataset has 100. Batch size 10.
        # n_hard=5, n_normal=5.
        # Expect len = ((100-5) // 5) * 10 = 190
        
        for i in range(5):
            self.sampler.controller.pool[i] = HardProblemEntry(problem_id=i)
            
        self.assertEqual(len(self.sampler), 190)
        
        # Test empty pool
        self.sampler.controller.pool = {}
        # n_hard=0, n_normal=10. len = ((100-0) // 10) * 10 = 100
        self.assertEqual(len(self.sampler), 100)

    def test_sampler_small_pool(self):
        # Case 1: Small Pool (1 item)
        # n_hard=1, n_normal=9
        # Batches = (100-1) // 9 = 11.
        
        self.sampler.controller.pool = {}
        self.sampler.controller.pool[0] = HardProblemEntry(problem_id=0)
        
        loader = DataLoader(
            self.wrapped_dataset,
            batch_size=10,
            sampler=self.sampler,
            collate_fn=None 
        )
        
        batches = list(loader)
        
        self.assertEqual(len(batches), 11)
        self.assertEqual(len(self.sampler), 110) # 11 * 10
        
        for batch in batches:
            self.assertEqual(len(batch['dataset_idx']), 10)
            from_hard = batch['from_hard_pool']
            self.assertEqual(from_hard.sum().item(), 1) 

    def test_set_epoch(self):
        self.sampler.set_epoch(10)
        self.assertEqual(self.sampler.epoch, 10)
        
        # Check if RNG is seeded differently
        rng_state_1 = self.sampler.batch_shuffle_rng.getstate()
        self.sampler.set_epoch(11)
        rng_state_2 = self.sampler.batch_shuffle_rng.getstate()
        
        self.assertNotEqual(rng_state_1, rng_state_2)

if __name__ == "__main__":
    unittest.main()
