"""
Verify that HardPoolSampler degrades to vanilla SequentialSampler behavior
when the hard pool is empty (i.e., at epoch start before any update() call).

Run: python tests/test_hard_pool_degradation.py
"""

import sys
import os
import importlib
import types

# Bypass verl/__init__.py (which imports ray) by creating a stub verl module
verl_stub = types.ModuleType("verl")
verl_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "verl")]
sys.modules["verl"] = verl_stub

# Stub verl.DataProto as a dummy class (only needed for type annotations)
class _DummyDataProto:
    pass
verl_stub.DataProto = _DummyDataProto

# Stub verl.experimental.dataset.sampler
exp_stub = types.ModuleType("verl.experimental")
exp_stub.__path__ = [os.path.join(verl_stub.__path__[0], "experimental")]
sys.modules["verl.experimental"] = exp_stub

exp_ds_stub = types.ModuleType("verl.experimental.dataset")
exp_ds_stub.__path__ = [os.path.join(exp_stub.__path__[0], "dataset")]
sys.modules["verl.experimental.dataset"] = exp_ds_stub

# Load the real AbstractCurriculumSampler from file
sampler_path = os.path.join(os.path.dirname(__file__), "..", "verl", "experimental", "dataset", "sampler.py")
spec = importlib.util.spec_from_file_location("verl.experimental.dataset.sampler", sampler_path)
sampler_mod = importlib.util.module_from_spec(spec)
sys.modules["verl.experimental.dataset.sampler"] = sampler_mod
spec.loader.exec_module(sampler_mod)

# Stub verl.utils.dataset
utils_stub = types.ModuleType("verl.utils")
utils_stub.__path__ = [os.path.join(verl_stub.__path__[0], "utils")]
sys.modules["verl.utils"] = utils_stub
utils_ds_stub = types.ModuleType("verl.utils.dataset")
utils_ds_stub.__path__ = [os.path.join(utils_stub.__path__[0], "dataset")]
sys.modules["verl.utils.dataset"] = utils_ds_stub

# Now import the modules under test
from verl.utils.dataset.hard_pool_controller import HardPoolController, BatchMetaForHardPool

hp_sampler_path = os.path.join(os.path.dirname(__file__), "..", "verl", "utils", "dataset", "hard_pool_sampler.py")
spec2 = importlib.util.spec_from_file_location("verl.utils.dataset.hard_pool_sampler", hp_sampler_path)
hp_sampler_mod = importlib.util.module_from_spec(spec2)
sys.modules["verl.utils.dataset.hard_pool_sampler"] = hp_sampler_mod
spec2.loader.exec_module(hp_sampler_mod)

HardPoolSampler = hp_sampler_mod.HardPoolSampler
HardPoolAwareDataset = hp_sampler_mod.HardPoolAwareDataset

from torch.utils.data import Dataset, SequentialSampler
from omegaconf import OmegaConf


class MockDataset(Dataset):
    def __init__(self, n=500):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {"prompt_ids": idx, "data_source": "test"}


def test_sampler_index_equivalence():
    """Both samplers must produce the same index sequence when pool is empty."""
    N = 500
    BATCH_SIZE = 128
    dataset = MockDataset(N)

    data_config = OmegaConf.create({
        "shuffle": False,
        "train_batch_size": BATCH_SIZE,
        "hard_pool": {"enable": True, "max_hard_ratio": 0.2, "max_consecutive_steps": 30},
        "seed": 42,
    })

    hp_sampler = HardPoolSampler(data_source=dataset, data_config=data_config)
    seq_sampler = SequentialSampler(data_source=dataset)

    hp_indices = []
    for item in hp_sampler:
        idx, _ = item if isinstance(item, tuple) else (item, False)
        hp_indices.append(idx)

    seq_indices = list(seq_sampler)

    # With drop_last=True and batch_size=128: 500 // 128 = 3 complete batches = 384 items
    n_complete_batches = N // BATCH_SIZE
    expected_count = n_complete_batches * BATCH_SIZE

    hp_trimmed = hp_indices[:expected_count]
    seq_trimmed = seq_indices[:expected_count]

    assert hp_trimmed == seq_trimmed, (
        f"Index mismatch!\n"
        f"  HardPool first 10: {hp_trimmed[:10]}\n"
        f"  Sequential first 10: {seq_trimmed[:10]}"
    )
    print(f"  [PASS] Index equivalence: {expected_count} indices match (empty pool, shuffle=False)")


def test_hard_pool_flags_all_false():
    """When pool is empty, all from_hard_pool flags must be False."""
    N = 500
    BATCH_SIZE = 128
    dataset = MockDataset(N)

    data_config = OmegaConf.create({
        "shuffle": False,
        "train_batch_size": BATCH_SIZE,
        "hard_pool": {"enable": True, "max_hard_ratio": 0.2, "max_consecutive_steps": 30},
        "seed": 42,
    })

    hp_sampler = HardPoolSampler(data_source=dataset, data_config=data_config)

    n_hard = 0
    n_total = 0
    for item in hp_sampler:
        if isinstance(item, tuple):
            _, from_hard = item
            if from_hard:
                n_hard += 1
        n_total += 1

    assert n_hard == 0, f"Expected 0 hard flags, got {n_hard}"
    print(f"  [PASS] All {n_total} items have from_hard_pool=False")


def test_dataset_wrapper_transparency():
    """HardPoolAwareDataset must return same base data as the raw dataset."""
    N = 50
    raw = MockDataset(N)
    wrapped = HardPoolAwareDataset(raw)

    for i in range(N):
        raw_item = raw[i]
        wrapped_int = wrapped[i]
        wrapped_tuple = wrapped[(i, False)]

        assert wrapped_int["prompt_ids"] == raw_item["prompt_ids"], \
            f"Data mismatch at {i} (int access)"
        assert wrapped_int.get("from_hard_pool") == False
        assert wrapped_tuple["prompt_ids"] == raw_item["prompt_ids"], \
            f"Data mismatch at {i} (tuple access)"
        assert wrapped_tuple.get("from_hard_pool") == False

    print(f"  [PASS] Wrapper is transparent: all {N} items match raw dataset")


def test_controller_noop_when_pool_empty():
    """HardPoolController.sample_for_step returns empty when pool is empty."""
    ctrl = HardPoolController(enable=True, max_hard_ratio=0.2, max_consecutive_steps=30)

    hard_ids, flags = ctrl.sample_for_step(batch_size=128)
    assert hard_ids == [] and flags == [], \
        f"Expected empty, got {len(hard_ids)} hard ids"

    print("  [PASS] Controller returns no hard samples when pool is empty")


def test_batch_boundary_alignment():
    """HardPoolSampler internal batching aligns with DataLoader batch_size."""
    N = 1000
    BATCH_SIZE = 128
    dataset = MockDataset(N)

    data_config = OmegaConf.create({
        "shuffle": False,
        "train_batch_size": BATCH_SIZE,
        "hard_pool": {"enable": True, "max_hard_ratio": 0.2, "max_consecutive_steps": 30},
        "seed": 42,
    })

    hp = HardPoolSampler(data_source=dataset, data_config=data_config)
    indices = [item[0] if isinstance(item, tuple) else item for item in hp]

    # Check each batch of 128 is contiguous and sequential
    n_batches = len(indices) // BATCH_SIZE
    for b in range(n_batches):
        batch = indices[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        expected = list(range(b * BATCH_SIZE, (b + 1) * BATCH_SIZE))
        assert batch == expected, (
            f"Batch {b} mismatch: got {batch[:5]}... expected {expected[:5]}..."
        )

    print(f"  [PASS] {n_batches} batches are correctly aligned with sequential order")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing HardPoolSampler degradation to vanilla GRPO")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    for fn in [
        test_controller_noop_when_pool_empty,
        test_sampler_index_equivalence,
        test_hard_pool_flags_all_false,
        test_dataset_wrapper_transparency,
        test_batch_boundary_alignment,
    ]:
        name = fn.__name__
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All degradation tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
