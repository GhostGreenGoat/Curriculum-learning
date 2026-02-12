# Hard Pool 实现检查清单

## 实现前准备

- [ ] 阅读主设计文档：`docs/design/hard_pool_curriculum_learning.md`
- [ ] 理解 verl 的 AbstractCurriculumSampler 接口
- [ ] 熟悉 GRPO 训练流程和 reward 计算
- [ ] 准备测试数据集（小规模，便于快速验证）

## Phase 1: 核心模块实现

### 1.1 HardPoolManager

- [ ] 创建文件 `verl/experimental/dataset/hard_pool_sampler.py`
- [ ] 实现 `HardItem` dataclass
- [ ] 实现 `HardPoolManager.__init__()`
- [ ] 实现 `HardPoolManager.add()` - 添加难题
- [ ] 实现 `HardPoolManager.remove()` - 移除题目
- [ ] 实现 `HardPoolManager.get_samples()` - 采样逻辑
- [ ] 实现容量限制和淘汰策略（FIFO/LRU）
- [ ] 实现 `state_dict()` 和 `load_state_dict()` - checkpoint 支持
- [ ] 编写单元测试 `tests/experimental/dataset/test_hard_pool_manager.py`

### 1.2 HardPoolCurriculumSampler 基础

- [ ] 实现 `HardPoolCurriculumSampler.__init__()`
  - [ ] 创建 base_sampler（RandomSampler/SequentialSampler）
  - [ ] 初始化 HardPoolManager
  - [ ] 初始化 next_iter_replay_queue
- [ ] 实现 `HardPoolCurriculumSampler.__len__()`
- [ ] 实现 `_extract_prompt_uid()` - prompt_uid 提取逻辑
  - [ ] 支持从 dataset row_dict 提取
  - [ ] 支持 fallback 策略
- [ ] 实现 `_prompt_uid_to_index()` - UID 到 index 映射
  - [ ] 方案 A：Dataset 提供反向索引（推荐）
  - [ ] 方案 B：Sampler 维护缓存映射

### 1.3 采样逻辑

- [ ] 实现 `HardPoolCurriculumSampler.__iter__()`
  - [ ] 从 next_iter_replay_queue 取强制重放题目
  - [ ] 从 HardPool 取 hard 题目（按配置上限/比例）
  - [ ] 从 base_sampler 取新题目补齐
  - [ ] 混合并返回 indices
- [ ] 处理边界情况：
  - [ ] HardPool 为空
  - [ ] HardPool 不足所需数量
  - [ ] base_sampler 耗尽

### 1.4 更新逻辑

- [ ] 实现 `_compute_pass_at_g()` - 计算 pass@G
  - [ ] 按 prompt_uid 分组
  - [ ] 计算每个 prompt 的 max score
  - [ ] 判断是否 pass
- [ ] 实现 `HardPoolCurriculumSampler.update()`
  - [ ] 识别新 hard 题目（pass@G=0）
  - [ ] 识别已解决题目（pass@1>0）
  - [ ] 更新 HardPool（添加/移除）
  - [ ] 更新 next_iter_replay_queue
  - [ ] 更新统计信息
- [ ] 实现 `_log_stats()` - 统计信息输出

### 1.5 Checkpoint 支持

- [ ] 实现 `HardPoolCurriculumSampler.state_dict()`
- [ ] 实现 `HardPoolCurriculumSampler.load_state_dict()`
- [ ] 测试 checkpoint 保存和恢复

## Phase 2: 数据协议增强（可选但推荐）

### 2.1 Dataset 增强

- [ ] 创建 `HardPoolAwareRLHFDataset`（可选）
  - [ ] 继承 `RLHFDataset`
  - [ ] 在 `__getitem__()` 中确保 prompt_uid 存在
  - [ ] 实现 `_build_uid_index()` - 构建反向索引
  - [ ] 实现 `get_index_by_uid()` - 根据 UID 查 index
- [ ] 或修改现有 `RLHFDataset` 添加 prompt_uid 支持

### 2.2 Trainer 集成

- [ ] 修改 `RayPPOTrainer.fit()` 中的 UID 处理（可选）
  - [ ] 优先使用 dataset 提供的 prompt_uid
  - [ ] Fallback 到随机 UUID（保持向后兼容）
- [ ] 验证不影响现有训练流程

## Phase 3: 配置和集成

### 3.1 配置项

- [ ] 在 `verl/trainer/config/data/legacy_data.yaml` 添加 hard_pool 配置段
- [ ] 确保所有配置项有默认值
- [ ] 添加配置验证逻辑

### 3.2 文档

- [ ] 更新主设计文档（如有必要）
- [ ] 在 `docs/` 添加使用示例
- [ ] 添加配置说明文档

## Phase 4: 测试和验证

### 4.1 单元测试

- [ ] `test_hard_pool_manager.py`
  - [ ] 测试添加/移除
  - [ ] 测试容量限制
  - [ ] 测试淘汰策略
  - [ ] 测试 state_dict
- [ ] `test_hard_pool_sampler.py`
  - [ ] 测试 __iter__ 混采逻辑
  - [ ] 测试 update 更新逻辑
  - [ ] 测试 prompt_uid 提取
  - [ ] 测试边界情况

### 4.2 集成测试

- [ ] 创建小规模训练脚本
  - [ ] 使用小数据集（<100 samples）
  - [ ] 设置小 batch_size（如 16）
  - [ ] 设置 rollout.n=2（便于验证）
- [ ] 验证端到端流程：
  - [ ] HardPool 正确识别 hard 题目
  - [ ] Hard 题目在下一轮出现
  - [ ] Solved 题目从 HardPool 移除
  - [ ] Checkpoint 保存和恢复
- [ ] 验证统计信息输出

### 4.3 性能测试

- [ ] 测试不同 HardPool 规模下的性能
  - [ ] 小规模（<100）
  - [ ] 中规模（100-1000）
  - [ ] 大规模（>1000）
- [ ] 测试对训练速度的影响
- [ ] 测试内存占用

### 4.4 兼容性测试

- [ ] 测试与 PrefixGrouper 的兼容性
- [ ] 测试与 TransferQueue 的兼容性
- [ ] 测试与现有训练流程的兼容性
- [ ] 测试向后兼容（hard_pool.enable=false）

## Phase 5: 代码审查和优化

### 5.1 代码质量

- [ ] 代码符合 verl 代码风格
- [ ] 添加必要的类型注解
- [ ] 添加 docstring
- [ ] 处理所有 TODO/FIXME

### 5.2 错误处理

- [ ] 处理异常情况（如 prompt_uid 找不到）
- [ ] 添加适当的日志和警告
- [ ] 验证输入参数

### 5.3 性能优化

- [ ] 优化 HardPool 查询性能（如需要）
- [ ] 优化 prompt_uid 映射性能
- [ ] 减少不必要的计算

## 验收标准

### 功能验收

- [ ] HardPool 机制正常工作
- [ ] pass@G=0 的题目被正确识别和加入
- [ ] pass@1>0 的题目被正确移除
- [ ] 下一轮强制重放机制生效
- [ ] Checkpoint 保存和恢复正常

### 性能验收

- [ ] 对训练速度影响 <5%
- [ ] 内存占用增加 <10%
- [ ] HardPool 查询延迟可接受

### 兼容性验收

- [ ] 不影响现有训练流程
- [ ] 与其他机制兼容
- [ ] 向后兼容（默认关闭）

## 常见问题排查

### 问题 1: prompt_uid 不稳定

**症状**: HardPool 无法正确追踪题目

**排查**:
- [ ] 检查 dataset 是否提供稳定的 prompt_uid
- [ ] 检查 prompt_uid 提取逻辑
- [ ] 使用自定义 Dataset 确保稳定性

### 问题 2: HardPool 不更新

**症状**: update() 被调用但 HardPool 不变

**排查**:
- [ ] 检查 batch 中是否有 prompt_uid
- [ ] 检查 token_level_scores 是否正确
- [ ] 检查 pass@G 计算逻辑
- [ ] 检查配置项（enable, success_threshold）

### 问题 3: 下一轮没有 hard 题目

**症状**: 判为 hard 但下一轮没出现

**排查**:
- [ ] 检查 next_iter_replay_queue 是否被清空
- [ ] 检查 __iter__() 是否正确从队列取题
- [ ] 检查 prompt_uid 到 index 的映射

### 问题 4: Checkpoint 恢复失败

**症状**: 恢复后 HardPool 状态不对

**排查**:
- [ ] 检查 state_dict 格式
- [ ] 检查 load_state_dict 逻辑
- [ ] 检查版本兼容性

## 提交前检查

- [ ] 所有测试通过
- [ ] 代码通过 lint 检查
- [ ] 文档完整且准确
- [ ] 配置示例已验证
- [ ] 性能影响可接受
- [ ] 向后兼容性已验证

---

**最后更新**: 2025-02-03
