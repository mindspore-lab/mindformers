# MindFormers CLI 系统测试

本目录包含 `mindformers_cli.py` 的系统测试用例，使用真实的配置文件模板进行测试。

## 测试用例

共包含 **4个系统测试用例**：

### 1. 训练任务 - 单卡模式

- **测试方法**: `test_pretrain_single_card`
- **配置文件**: `test_pretrain_config.yaml`
- **测试命令**: `mindformers-cli --config test_pretrain_config.yaml`
- **验证点**: 单卡模式检测、参数解析、环境设置

### 2. 训练任务 - 多卡模式（单节点）

- **测试方法**: `test_pretrain_multi_card_single_node`
- **配置文件**: `test_pretrain_config.yaml`
- **测试命令**: `mindformers-cli --worker-num 4 --config test_pretrain_config.yaml`
- **验证点**: 多卡模式检测、msrun 命令构建、单节点参数设置

### 3. 推理任务 - 单卡模式

- **测试方法**: `test_predict_single_card`
- **配置文件**: `test_predict_config.yaml`
- **测试命令**: `mindformers-cli --config test_predict_config.yaml`
- **验证点**: 单卡模式检测、参数解析、环境设置

### 4. 推理任务 - 多卡模式（单节点）

- **测试方法**: `test_predict_multi_card_single_node`
- **配置文件**: `test_predict_config.yaml`
- **测试命令**: `mindformers-cli --worker-num 4 --config test_predict_config.yaml`
- **验证点**: 多卡模式检测、msrun 命令构建、单节点参数设置

## 配置文件

测试使用 `configs/llm_template_v2_experimental/` 目录下的模板配置文件：

### llm_pretrain_template.yaml

预训练模板配置文件，包含：

- 训练参数配置（epochs, batch_size, learning_rate 等）
- 优化器配置（AdamW）
- 学习率调度配置
- 数据集配置
- 模型配置
- 并行配置

### llm_predict_template.yaml

推理模板配置文件，包含：

- 推理参数配置（batch_size, seed 等）
- 模型配置
- 并行配置
- 上下文配置

**注意**：测试会在运行时基于这些模板创建临时配置文件，并自动更新路径和参数以适应测试环境。

## 运行测试

### 运行所有测试

```bash
pytest tests/st/test_cli/test_mindformers_cli.py -v
```

### 运行特定测试

```bash
# 训练单卡测试
pytest tests/st/test_cli/test_mindformers_cli.py::TestMindFormersCLI::test_pretrain_single_card -v

# 训练多卡测试
pytest tests/st/test_cli/test_mindformers_cli.py::TestMindFormersCLI::test_pretrain_multi_card_single_node -v

# 推理单卡测试
pytest tests/st/test_cli/test_mindformers_cli.py::TestMindFormersCLI::test_predict_single_card -v

# 推理多卡测试
pytest tests/st/test_cli/test_mindformers_cli.py::TestMindFormersCLI::test_predict_multi_card_single_node -v
```

## 测试说明

### 测试设计原则

1. **系统测试**: 通过实际调用 CLI 脚本来验证功能
2. **真实配置**: 使用基于模板的真实配置文件，模拟实际使用场景
3. **容错性**: 测试允许命令失败（由于缺少依赖或配置），主要验证 CLI 接口的正确性
4. **环境隔离**: 每个测试使用独立的临时目录
5. **自动数据生成**: 测试会在准备阶段自动生成所需的测试数据集和模型目录

### 测试数据

- 测试会在类级别自动生成假的 Megatron 格式数据集和 Qwen3 模型目录
- 所有测试用例共享同一份生成的数据，提高测试效率
- 测试数据存储在临时目录中，测试结束后自动清理
- 生成的数据仅用于测试 CLI 接口，不包含真实的模型权重

### 注意事项

1. 测试可能因为缺少模型权重、数据文件或 `msrun` 命令而失败，这是预期的行为
2. 测试主要验证 CLI 接口的正确性（参数解析、模式检测、命令构建），而不是完整的训练/推理流程
3. 多卡测试需要 `msrun` 命令可用，如果不可用会显示相应错误信息

## 测试覆盖的功能点

- ✅ 单卡模式检测和执行
- ✅ 多卡模式命令构建（单节点）
- ✅ 配置文件加载
- ✅ 参数解析和验证
- ✅ 训练和推理模式支持
- ✅ 日志目录设置
- ✅ msrun 命令构建（多卡模式）
- ✅ 环境变量设置

