# Hard Pool 课程学习机制开发文档

## 1. 概述

### 1.1 目标

在 verl 框架中实现一个**难题池（Hard Pool）机制**，使得在 GRPO 训练过程中：

1. **难题识别**：如果题目在 G 次采样中，奖励函数返回的结果全部为负或零（即 pass@G=0），则标记为"难题"，存入难题池
2. **自动重放**：难题池中的题目会在下一轮训练中自动混入，确保模型在困难样本上产生更多尝试
3. **动态移除**：如果难题池中的题目在后续训练中被成功解决（pass@1>0），则自动从池中移除
4. **动态批次**：每轮的 batch 由"新抽取的题目"和"上一轮的难题"混合组成，batch size 保持固定，但构成比例动态变化

### 1.2 设计原则

- **最小侵入性**：尽量不改动 Trainer 主循环，通过可插拔的 Sampler 机制实现
- **高可复用性**：通过配置即可启用，无需修改核心代码
- **协作友好**：清晰的接口和文档，便于多人协作
- **向后兼容**：不影响现有训练流程，可选择性启用

### 1.3 核心假设

通过让模型在困难样本上产生更多尝试，变相实现了极大的 pass@k，提高了模型进入正确解区域的概率。

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    RayPPOTrainer.fit()                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  for batch in train_dataloader:                      │  │
│  │    ┌──────────────────────────────────────────────┐  │  │
│  │    │ 1. 从 sampler 获取 batch indices             │  │  │
│  │    │    (HardPoolSampler 混入 hard + fresh)       │  │  │
│  │    └──────────────────────────────────────────────┘  │  │
│  │    ┌──────────────────────────────────────────────┐  │  │
│  │    │ 2. Rollout: 生成 G 次采样                    │  │  │
│  │    └──────────────────────────────────────────────┘  │  │
│  │    ┌──────────────────────────────────────────────┐  │  │
│  │    │ 3. Reward: 计算 token_level_scores           │  │  │
│  │    └──────────────────────────────────────────────┘  │  │
│  │    ┌──────────────────────────────────────────────┐  │  │
│  │    │ 4. Advantage & Update                        │  │  │
│  │    └──────────────────────────────────────────────┘  │  │
│  │    ┌──────────────────────────────────────────────┐  │  │
│  │    │ 5. sampler.update(batch) ← 反馈机制         │  │  │
│  │    │    - 识别 pass@G=0 → 加入 HardPool          │  │  │
│  │    │    - 识别 pass@1>0 → 从 HardPool 移除       │  │  │
│  │    └──────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
Iteration t:
  ┌─────────────────────────────────────────┐
  │ HardPoolSampler.__iter__()             │
  │  - 从 HardPool 取 hard_indices          │
  │  - 从 base_sampler 取 fresh_indices     │
  │  - 混合返回 batch_indices               │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ DataLoader 根据 indices 加载数据         │
  │  - 每个 sample 有 prompt_uid (稳定)    │
  │  - 每个 sample 有 uid (用于 GRPO 分组) │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ Rollout: 每个 prompt 生成 G 次响应      │
  │  - batch size = B, 重复后 = B×G         │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ Reward: 计算 token_level_scores         │
  │  - shape: (B×G, response_length)       │
  └─────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────┐
  │ HardPoolSampler.update(batch)            │
  │  - 按 prompt_uid 分组计算 pass@G        │
  │  - pass@G=0 → 加入 HardPool            │
  │  - pass@1>0 → 从 HardPool 移除          │
  └─────────────────────────────────────────┘
```

### 2.3 模块划分

1. **HardPoolCurriculumSampler** (`verl/experimental/dataset/hard_pool_sampler.py`)
   - 核心采样器，管理 HardPool 和混采逻辑

2. **HardPoolManager** (可选，内嵌在 Sampler 中)
   - HardPool 的存储和管理（添加、移除、查询）

3. **Data Protocol 增强** (可选，通过自定义 Dataset)
   - 确保每个 sample 有稳定的 `prompt_uid`

4. **配置项** (`verl/trainer/config/data/legacy_data.yaml`)
   - HardPool 相关配置参数

---

## 3. 详细设计

### 3.1 HardPoolCurriculumSampler 设计

#### 3.1.1 类结构

```python
class HardPoolCurriculumSampler(AbstractCurriculumSampler):
    """
    难题池课程学习采样器。
    
    功能：
    1. 维护一个 HardPool，存储难题的 prompt_uid
    2. 每次采样时，从 HardPool 取 hard 题目，从 base_sampler 取新题目
    3. 训练后通过 update() 更新 HardPool
    """
    
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        """
        Args:
            data_source: 数据集对象（RLHFDataset）
            data_config: 数据配置，包含 hard_pool 相关配置
        """
        # 基础采样器（RandomSampler 或 SequentialSampler）
        self.base_sampler = self._create_base_sampler(data_source, data_config)
        
        # HardPool 管理器
        self.hard_pool = HardPoolManager(config=data_config.hard_pool)
        
        # 下一轮强制重放的队列（保证"幽灵题"一定出现）
        self.next_iter_replay_queue: deque = deque()
        
        # 数据集引用（用于根据 prompt_uid 查找 index）
        self.data_source = data_source
        
        # 配置
        self.config = data_config.hard_pool
        
        # 当前 iteration 计数器
        self.current_iteration = 0
        
    def __iter__(self):
        """生成下一批样本的 indices"""
        # 1. 从 next_iter_replay_queue 取强制重放的题目
        # 2. 从 HardPool 取 hard 题目（按配置的比例/上限）
        # 3. 从 base_sampler 取新题目补齐到 batch_size
        # 4. 返回混合后的 indices
        
    def __len__(self) -> int:
        """返回采样器长度（用于 DataLoader）"""
        return len(self.base_sampler)
    
    def update(self, batch: DataProto) -> None:
        """
        根据本轮训练结果更新 HardPool。
        
        Args:
            batch: 包含 rollout 和 reward 结果的 DataProto
                - batch.batch["token_level_scores"]: (B×G, response_length)
                - batch.non_tensor_batch["prompt_uid"]: (B×G,) 稳定 UID
                - batch.non_tensor_batch["uid"]: (B×G,) GRPO 分组 UID
        """
        # 1. 按 prompt_uid 分组，计算每个 prompt 的 pass@G
        # 2. pass@G=0 → 加入 HardPool + next_iter_replay_queue
        # 3. pass@1>0 → 从 HardPool 移除
        # 4. 更新统计信息
```

#### 3.1.2 HardPoolManager 设计

```python
class HardPoolManager:
    """
    难题池管理器。
    
    负责：
    - 存储难题的 prompt_uid
    - 提供添加/移除/查询接口
    - 支持容量限制和淘汰策略
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: HardPool 配置
                - max_size: 最大容量（None 表示无限制）
                - eviction_policy: 淘汰策略 ("fifo", "lru", "none")
        """
        # 使用 OrderedDict 支持 LRU
        self.hard_pool: OrderedDict[str, HardItem] = OrderedDict()
        self.config = config
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "current_size": 0,
        }
    
    def add(self, prompt_uid: str, metadata: dict = None):
        """添加难题到池中"""
        
    def remove(self, prompt_uid: str):
        """从池中移除题目"""
        
    def get_samples(self, n: int, strategy: str = "all") -> list[str]:
        """
        从池中获取 n 个样本。
        
        Args:
            n: 需要的样本数量
            strategy: "all" (全部), "random" (随机), "fifo" (先进先出)
        """
        
    def __len__(self) -> int:
        return len(self.hard_pool)
    
    def state_dict(self) -> dict:
        """用于 checkpoint 保存"""
        
    def load_state_dict(self, state_dict: dict):
        """从 checkpoint 恢复"""
```

#### 3.1.3 HardItem 数据结构

```python
@dataclass
class HardItem:
    """HardPool 中存储的题目元数据"""
    prompt_uid: str
    added_at_iteration: int
    failure_count: int = 0  # 连续失败次数
    last_seen_iteration: int = 0
    metadata: dict = field(default_factory=dict)
```

### 3.2 数据协议（Data Protocol）

#### 3.2.1 prompt_uid 的生成规则

**优先级（从高到低）**：

1. **Dataset 显式提供**：如果 `row_dict` 中有 `prompt_uid` 字段，直接使用
2. **基于 extra_info.index**：`prompt_uid = f"{data_source}:{extra_info['index']}"`
3. **基于 dataset index**：`prompt_uid = f"{data_source}:{dataset_index}"`
4. **Fallback**：`prompt_uid = hash(data_source + raw_prompt)`（不推荐，计算重）

#### 3.2.2 uid vs prompt_uid

- **`prompt_uid`**：稳定的题目标识，用于 HardPool 追踪
- **`uid`**：GRPO 分组标识，可以是 `prompt_uid`（如果保证 batch 内不重复）或 `f"{prompt_uid}:{local_idx}"`

#### 3.2.3 自定义 Dataset（可选）

如果用户希望确保 prompt_uid 的稳定性，可以实现自定义 Dataset：

```python
class HardPoolAwareRLHFDataset(RLHFDataset):
    """为 HardPool 机制提供稳定 prompt_uid 的 Dataset"""
    
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        
        # 确保有 prompt_uid
        if "prompt_uid" not in row_dict:
            data_source = row_dict.get("data_source", "unknown")
            index = row_dict.get("extra_info", {}).get("index", item)
            row_dict["prompt_uid"] = f"{data_source}:{index}"
        
        return row_dict
```

### 3.3 配置项设计

#### 3.3.1 配置结构

在 `verl/trainer/config/data/legacy_data.yaml` 中添加：

```yaml
# HardPool 课程学习配置
hard_pool:
  # 是否启用 HardPool 机制
  enable: false
  
  # HardPool 最大容量（None 表示无限制）
  max_size: null
  
  # 淘汰策略：当池满时的淘汰方式
  # - "fifo": 先进先出
  # - "lru": 最近最少使用
  # - "none": 不淘汰（达到上限后不再添加）
  eviction_policy: "fifo"
  
  # 每轮从 HardPool 取题的上限（None 表示全部）
  # 例如：batch_size=256, max_hard_per_batch=16
  # 则最多 16 道 hard，剩余 240 道 fresh
  max_hard_per_batch: null
  
  # 每轮从 HardPool 取题的比例（0.0-1.0）
  # 例如：hard_ratio=0.1, batch_size=256 → 约 25 道 hard
  # 优先级：max_hard_per_batch > hard_ratio
  hard_ratio: null
  
  # 成功判定阈值（用于 pass@1>0 的判断）
  # reward_sum > success_threshold 则认为成功
  success_threshold: 0.0
  
  # 是否强制下一轮重放（"幽灵题"机制）
  # 如果 True，本轮判为 hard 的题目会在下一轮强制出现
  force_next_iter_replay: true
  
  # 采样策略：从 HardPool 取题的方式
  # - "all": 全部取出（受 max_hard_per_batch 限制）
  # - "random": 随机采样
  # - "fifo": 先进先出
  # - "priority": 按失败次数优先
  sampling_strategy: "all"
  
  # 统计信息输出频率（每 N 个 iteration 输出一次）
  stats_log_freq: 10
```

#### 3.3.2 Sampler 配置集成

在 `data.sampler` 中配置：

```yaml
sampler:
  class_path: "pkg://verl.experimental.dataset.hard_pool_sampler"
  class_name: "HardPoolCurriculumSampler"
```

### 3.4 Trainer 集成点（最小改动）

#### 3.4.1 利用现有的 sampler.update() 钩子

在 `RayPPOTrainer.fit()` 中，已有代码：

```python
if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
    self.train_dataloader.sampler.update(batch=batch)
```

**无需修改**，HardPoolSampler 继承 `AbstractCurriculumSampler` 即可自动调用。

#### 3.4.2 UID 处理优化（可选）

在 `RayPPOTrainer.fit()` 的 batch 处理部分，当前代码：

```python
# add uid to batch
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
)
```

**建议优化**（向后兼容）：

```python
# 优先使用 dataset 提供的稳定 UID
if "prompt_uid" in batch.non_tensor_batch:
    # 使用 prompt_uid 作为 GRPO 分组标识
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["prompt_uid"]
else:
    # Fallback 到随机 UUID（保持旧行为）
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )
```

---

## 4. 实现细节

### 4.1 pass@G 计算逻辑

```python
def compute_pass_at_g(batch: DataProto, g: int) -> dict[str, bool]:
    """
    计算每个 prompt 的 pass@G。
    
    Args:
        batch: DataProto，包含：
            - batch.batch["token_level_scores"]: (B×G, response_length)
            - batch.non_tensor_batch["prompt_uid"]: (B×G,)
        g: GRPO 组大小（rollout.n）
    
    Returns:
        dict: {prompt_uid: bool}，True 表示 pass@G>0，False 表示 pass@G=0
    """
    token_level_scores = batch.batch["token_level_scores"]  # (B×G, L)
    prompt_uids = batch.non_tensor_batch["prompt_uid"]  # (B×G,)
    
    # 计算每个 trajectory 的总分
    traj_scores = token_level_scores.sum(dim=-1)  # (B×G,)
    
    # 按 prompt_uid 分组
    uid_to_scores = defaultdict(list)
    for uid, score in zip(prompt_uids, traj_scores.cpu().numpy()):
        uid_to_scores[uid].append(score)
    
    # 判断每个 prompt 的 pass@G
    result = {}
    for uid, scores in uid_to_scores.items():
        # pass@G = 至少有一个 score > success_threshold
        max_score = max(scores)
        result[uid] = max_score > self.config.hard_pool.success_threshold
    
    return result
```

### 4.2 HardPoolSampler.__iter__() 实现

```python
def __iter__(self):
    """生成混合的 batch indices"""
    self.current_iteration += 1
    
    # 1. 从 next_iter_replay_queue 取强制重放的题目
    forced_indices = []
    while self.next_iter_replay_queue and len(forced_indices) < self.config.hard_pool.max_hard_per_batch:
        prompt_uid = self.next_iter_replay_queue.popleft()
        # 根据 prompt_uid 查找 dataset index（需要实现 lookup 方法）
        idx = self._prompt_uid_to_index(prompt_uid)
        if idx is not None:
            forced_indices.append(idx)
    
    # 2. 从 HardPool 取 hard 题目
    remaining_hard_slots = self.config.hard_pool.max_hard_per_batch - len(forced_indices)
    if remaining_hard_slots > 0:
        hard_uids = self.hard_pool.get_samples(
            n=remaining_hard_slots,
            strategy=self.config.hard_pool.sampling_strategy
        )
        hard_indices = [self._prompt_uid_to_index(uid) for uid in hard_uids]
        hard_indices = [idx for idx in hard_indices if idx is not None]
    else:
        hard_indices = []
    
    # 3. 从 base_sampler 取新题目
    batch_size = self.config.train_batch_size
    needed_fresh = batch_size - len(forced_indices) - len(hard_indices)
    
    base_iter = iter(self.base_sampler)
    fresh_indices = [next(base_iter) for _ in range(needed_fresh)]
    
    # 4. 混合并返回
    all_indices = forced_indices + hard_indices + fresh_indices
    
    # 打乱顺序（可选，保持随机性）
    if self.config.hard_pool.get("shuffle_mixed_batch", True):
        import random
        random.shuffle(all_indices)
    
    return iter(all_indices)
```

### 4.3 HardPoolSampler.update() 实现

```python
def update(self, batch: DataProto) -> None:
    """根据训练结果更新 HardPool"""
    
    # 1. 计算 pass@G
    g = self.config.actor_rollout_ref.rollout.n  # GRPO 组大小
    pass_results = self._compute_pass_at_g(batch, g)
    
    # 2. 识别新 hard 题目（pass@G=0）
    new_hard_uids = [
        uid for uid, passed in pass_results.items() 
        if not passed
    ]
    
    # 3. 识别已解决的题目（pass@1>0）
    # 注意：这里需要检查每个 prompt 的任意一条 trajectory 是否成功
    solved_uids = []
    token_level_scores = batch.batch["token_level_scores"]
    prompt_uids = batch.non_tensor_batch["prompt_uid"]
    traj_scores = token_level_scores.sum(dim=-1)
    
    uid_to_max_score = {}
    for uid, score in zip(prompt_uids, traj_scores.cpu().numpy()):
        if uid not in uid_to_max_score:
            uid_to_max_score[uid] = score
        else:
            uid_to_max_score[uid] = max(uid_to_max_score[uid], score)
    
    solved_uids = [
        uid for uid, max_score in uid_to_max_score.items()
        if max_score > self.config.hard_pool.success_threshold
    ]
    
    # 4. 更新 HardPool
    for uid in new_hard_uids:
        self.hard_pool.add(uid, metadata={
            "iteration": self.current_iteration,
        })
        
        # 强制下一轮重放
        if self.config.hard_pool.force_next_iter_replay:
            self.next_iter_replay_queue.append(uid)
    
    for uid in solved_uids:
        if uid in self.hard_pool:
            self.hard_pool.remove(uid)
    
    # 5. 更新统计信息
    if self.current_iteration % self.config.hard_pool.stats_log_freq == 0:
        self._log_stats()
```

### 4.4 prompt_uid 到 dataset index 的映射

**挑战**：HardPool 存储的是 `prompt_uid`，但 sampler 需要返回 dataset 的 `index`。

**解决方案**：

1. **方案 A（推荐）**：Dataset 维护反向索引
   ```python
   # 在 RLHFDataset 中添加
   def __init__(self, ...):
       # ... 现有代码 ...
       self._uid_to_index: dict[str, int] = {}
       self._build_uid_index()
   
   def _build_uid_index(self):
       """构建 prompt_uid -> index 的映射"""
       for idx in range(len(self.dataframe)):
           row = self.dataframe[idx]
           prompt_uid = self._get_prompt_uid(row, idx)
           self._uid_to_index[prompt_uid] = idx
   
   def get_index_by_uid(self, prompt_uid: str) -> Optional[int]:
       return self._uid_to_index.get(prompt_uid)
   ```

2. **方案 B**：Sampler 维护映射（需要遍历 dataset）
   ```python
   def _prompt_uid_to_index(self, prompt_uid: str) -> Optional[int]:
       """根据 prompt_uid 查找 dataset index"""
       # 如果已有缓存，直接返回
       if prompt_uid in self._uid_to_index_cache:
           return self._uid_to_index_cache[prompt_uid]
       
       # 否则遍历 dataset 查找（慢，但只查一次）
       for idx in range(len(self.data_source)):
           row = self.data_source.dataframe[idx]
           uid = self._extract_prompt_uid(row, idx)
           self._uid_to_index_cache[uid] = idx
           if uid == prompt_uid:
               return idx
       
       return None
   ```

### 4.5 Checkpoint 支持

```python
def state_dict(self) -> dict:
    """保存 HardPool 状态用于 checkpoint"""
    return {
        "hard_pool": self.hard_pool.state_dict(),
        "next_iter_replay_queue": list(self.next_iter_replay_queue),
        "current_iteration": self.current_iteration,
        "base_sampler_state": self.base_sampler.state_dict() if hasattr(self.base_sampler, 'state_dict') else None,
    }

def load_state_dict(self, state_dict: dict):
    """从 checkpoint 恢复状态"""
    self.hard_pool.load_state_dict(state_dict["hard_pool"])
    self.next_iter_replay_queue = deque(state_dict["next_iter_replay_queue"])
    self.current_iteration = state_dict["current_iteration"]
    if hasattr(self.base_sampler, 'load_state_dict'):
        self.base_sampler.load_state_dict(state_dict["base_sampler_state"])
```

---

## 5. 实现步骤

### Phase 1: 核心模块实现

1. **创建 HardPoolManager**
   - 文件：`verl/experimental/dataset/hard_pool_sampler.py`
   - 实现：`HardPoolManager` 类
   - 测试：单元测试覆盖添加/移除/查询/淘汰逻辑

2. **创建 HardPoolCurriculumSampler**
   - 实现：`HardPoolCurriculumSampler` 类
   - 实现：`__iter__()` 混采逻辑
   - 实现：`update()` 更新逻辑
   - 测试：mock batch 测试更新流程

3. **实现 prompt_uid 提取逻辑**
   - 在 `HardPoolCurriculumSampler` 中实现 `_extract_prompt_uid()`
   - 支持多种 fallback 策略

### Phase 2: 数据协议增强（可选）

4. **创建 HardPoolAwareRLHFDataset**（可选）
   - 文件：`verl/experimental/dataset/hard_pool_dataset.py`
   - 确保每个 sample 有稳定的 `prompt_uid`

5. **优化 Trainer 的 UID 处理**（可选）
   - 修改 `RayPPOTrainer.fit()` 中的 UID 生成逻辑
   - 优先使用 dataset 提供的 `prompt_uid`

### Phase 3: 配置和集成

6. **添加配置项**
   - 在 `verl/trainer/config/data/legacy_data.yaml` 中添加 `hard_pool` 配置段
   - 确保所有配置项有合理的默认值

7. **更新文档**
   - 在 `docs/` 中添加使用示例
   - 更新配置文档

### Phase 4: 测试和验证

8. **单元测试**
   - `tests/experimental/dataset/test_hard_pool_sampler.py`
   - 覆盖所有核心逻辑

9. **集成测试**
   - 创建小规模训练脚本验证端到端流程
   - 验证 checkpoint 恢复功能

10. **性能测试**
    - 测试 HardPool 在不同规模下的性能
    - 验证对训练速度的影响

---

## 6. 使用示例

### 6.1 基本配置

```yaml
# config/ppo_trainer.yaml
data:
  train_files: ~/data/gsm8k/train.parquet
  train_batch_size: 256
  
  # 启用 HardPool
  hard_pool:
    enable: true
    max_size: 1000
    max_hard_per_batch: 16
    force_next_iter_replay: true
    success_threshold: 0.0
  
  # 配置 sampler
  sampler:
    class_path: "pkg://verl.experimental.dataset.hard_pool_sampler"
    class_name: "HardPoolCurriculumSampler"
  
  # 必须设置为 0（curriculum sampler 的要求）
  dataloader_num_workers: 0
```

### 6.2 自定义 Dataset（推荐）

```yaml
data:
  # 使用自定义 Dataset 确保 prompt_uid 稳定
  custom_cls:
    path: "pkg://verl.experimental.dataset.hard_pool_dataset"
    name: "HardPoolAwareRLHFDataset"
```

### 6.3 训练脚本

```python
# 无需修改训练脚本，直接使用现有流程
from verl.trainer.main_ppo import run_ppo
import hydra

@hydra.main(config_path="config", config_name="ppo_trainer")
def main(config):
    run_ppo(config)

if __name__ == "__main__":
    main()
```

---

## 7. 注意事项和限制

### 7.1 已知限制

1. **dataloader_num_workers 必须为 0**
   - Curriculum sampler 的要求，防止数据预取导致 update 不及时

2. **Batch size 必须固定**
   - 动态 batch size 会影响分布式训练的平衡逻辑

3. **prompt_uid 必须稳定**
   - 如果 dataset 不提供稳定的 UID，需要自定义 Dataset

### 7.2 性能考虑

1. **HardPool 查询开销**
   - 如果 HardPool 很大（>10k），考虑使用更高效的数据结构（如 Bloom Filter）

2. **prompt_uid 映射开销**
   - 首次查找需要遍历 dataset，建议在 Dataset 初始化时构建索引

3. **Checkpoint 大小**
   - HardPool 状态会增大 checkpoint，考虑压缩或增量保存

### 7.3 兼容性

1. **向后兼容**
   - 默认 `hard_pool.enable=false`，不影响现有训练
   - 不启用时，HardPoolSampler 退化为 base_sampler

2. **与其他机制兼容**
   - 与 PrefixGrouper：需要确保 `prompt_uid == uid`
   - 与 TransferQueue：需要验证数据序列化兼容性

---

## 8. 扩展方向

### 8.1 高级特性（未来）

1. **优先级队列**
   - 按失败次数、难度分数等排序

2. **自适应注入比例**
   - 根据训练进度动态调整 hard 比例

3. **多级 HardPool**
   - 区分"困难"和"极困难"，不同策略处理

4. **统计分析**
   - 记录每道题的解决历史，生成难度曲线

### 8.2 与其他机制结合

1. **Curriculum Learning**
   - 结合难度递增的课程学习

2. **Active Learning**
   - 主动选择最有价值的 hard 题目

---

## 9. 参考实现

### 9.1 相关代码位置

- `verl/experimental/dataset/sampler.py` - AbstractCurriculumSampler 接口
- `verl/trainer/ppo/ray_trainer.py` - Trainer 主循环
- `verl/utils/dataset/rl_dataset.py` - RLHFDataset 实现
- `verl/trainer/ppo/core_algos.py` - GRPO advantage 计算

### 9.2 类似机制参考

- `recipe/dapo/` - DAPO 中的样本过滤机制
- `recipe/entropy/` - Entropy 训练中的样本选择

---

## 10. 附录

### 10.1 配置项完整列表

```yaml
hard_pool:
  enable: bool                    # 是否启用
  max_size: Optional[int]         # 最大容量
  eviction_policy: str            # 淘汰策略
  max_hard_per_batch: Optional[int]  # 每轮 hard 上限
  hard_ratio: Optional[float]     # hard 比例
  success_threshold: float        # 成功阈值
  force_next_iter_replay: bool   # 强制下一轮重放
  sampling_strategy: str          # 采样策略
  stats_log_freq: int            # 统计输出频率
  shuffle_mixed_batch: bool      # 是否打乱混合 batch
```

### 10.2 关键数据结构

```python
# HardItem
prompt_uid: str
added_at_iteration: int
failure_count: int
last_seen_iteration: int
metadata: dict

# HardPoolManager state_dict
{
    "hard_pool": OrderedDict,  # {prompt_uid: HardItem}
    "stats": dict,
}

# HardPoolSampler state_dict
{
    "hard_pool": dict,
    "next_iter_replay_queue": list,
    "current_iteration": int,
    "base_sampler_state": dict,
}
```

---

**文档版本**: v1.0  
**最后更新**: 2025-02-03  
**维护者**: verl 开发团队
