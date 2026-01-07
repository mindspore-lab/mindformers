# MindFormers CLI User Guide / 用户使用指南

**注意 / Note:**
本工具 (`mindformers-cli`) **仅适用于 V2 实验性配置模板** (`configs/llm_template_v2_experimental/` 及其子目录下的 YAML 文件)。请勿使用旧版配置格式。
This tool (`mindformers-cli`) is **exclusively for V2 experimental configuration templates** (YAML files located in `/configs/llm_template_v2_experimental/` and its subdirectories). Do not use with legacy configuration formats.

---

## 1. 核心优势 / Key Features

* **统一入口 / Unified Entry**: 集成了单卡、单机多卡、多机多卡训练及推理任务，无需记忆复杂的分布式启动命令（如 `msrun`）。
    * Integrates single-card, single-node multi-card, and multi-node multi-card training/inference tasks, eliminating the need to memorize complex distributed launch commands (e.g., `msrun`).

* **配置解耦 / Configuration Decoupling**: 支持通过命令行动态覆盖 YAML 配置文件中的任意参数，无需频繁修改配置文件，便于实验对比。
    * Supports dynamic overriding of any parameter in the YAML configuration file via the command line, avoiding frequent file modifications and facilitating experiment comparison.

* **自动化管理 / Automated Management**: 自动处理环境变量、日志路径重定向、资源限制（ulimit）等系统级配置，简化运维成本。
    * Automatically handles system-level configurations such as environment variables, log path redirection, and resource limits (ulimit), reducing operation and maintenance costs.

* **易于集成 / Easy Integration**: 提供简洁标准的命令行接口，方便被上层调度平台（如 K8s, Slurm）或自动化脚本集成。
    * Provides a simple and standard command-line interface, making it easy to integrate with upper-level scheduling platforms (e.g., K8s, Slurm) or automation scripts.

---

## 2. 安装与准备 / Installation & Preparation

确保已安装 `mindformers`。如果对源码进行了修改，建议重新安装以使更改生效，或在 `mindformers` 根目录下执行命令。
Ensure `mindformers` is installed. If source code modifications are made, it is recommended to reinstall to apply changes, or execute commands from the `mindformers` root directory.

### 源码安装 / Source Installation

```bash
# 当前仅br_feature_llm_trainer分支支持，后续合入master分支
git clone -b br_feature_llm_trainer https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

### 开发模式 / Development Mode

如果在开发过程中修改了代码，可以使用以下方式安装以即时生效：
If modifying code during development, use the following method for changes to take effect immediately:

```bash
cd mindformers # 一级根目录下
pip install -e .
```

---

## 3. 快速开始 / Quick Start

使用 V2 实验性模板启动任务。
Launch tasks using V2 experimental templates.

### 单卡模式 / Single Card Mode

```bash
# 运行单卡微调 (使用 V2 模板)
# Run single card fine-tuning (using V2 template)
mindformers-cli --config configs/llm_template_v2_experimental/llm_finetune_template.yaml
```

---

## 4. 分布式训练 / Distributed Training

CLI 内部自动封装了 `msrun`，支持使用 V2 配置进行分布式训练。
The CLI wraps `msrun` internally, supporting distributed training with V2 configurations.

### 单机多卡 / Single Node Multi-Card

```bash
# 启动单机 8 卡预训练
# Launch single-node 8-card pre-training
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_pretrain_template.yaml \
  --worker-num 8
```

### 多机多卡 / Multi-Node Multi-Card

**节点 0 (Master) / Node 0:**

```bash
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_pretrain_template.yaml \
  --worker-num 16 \
  --local-worker 8 \
  --master-addr <MASTER_IP> \
  --node-rank 0
```

---

## 5. 配置参数覆盖 / Configuration Overrides (V2 Specific)

你可以通过命令行覆盖 V2 配置模板中的参数。请确保使用正确的 V2 参数层级（例如 `training_args` 而不是旧版的 `runner_config`）。
You can override parameters in the V2 configuration template via the command line. Ensure you use the correct V2 parameter hierarchy (e.g., `training_args` instead of legacy `runner_config`).

### 常用覆盖示例 / Common Override Examples

**修改训练参数 / Modify Training Arguments:**

```bash
# 修改全局 batch size 和 epoch
# Modify global batch size and epochs
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_finetune_template.yaml \
  --training_args.global_batch_size 128 \
  --training_args.epochs 5
```

**修改运行模式 / Modify Run Mode:**

```bash
# 切换为推理模式
# Switch to predict mode
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_predict_template.yaml \
  --run_mode predict
```

**修改并行配置 / Modify Parallel Configuration:**

```bash
# 设置模型并行和流水线并行
# Set tensor model parallel and pipeline model parallel
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_pretrain_template.yaml \
  --distribute_parallel_config.tensor_model_parallel_size 2 \
  --distribute_parallel_config.pipeline_model_parallel_size 2
```

**修改数据路径 / Modify Data Path:**

```bash
# 修改 HuggingFace 数据集路径
# Modify HuggingFace dataset path
mindformers-cli \
  --config configs/llm_template_v2_experimental/llm_finetune_template.yaml \
  --train_dataset.data_loader.path /path/to/new/dataset
```

---

## 6. 配置关键字段速查 / Config Key Reference

在使用命令行覆盖参数时，请参考以下配置文件的字段：
Refer to the following top-level fields structure of config files when overriding parameters via CLI:

`mindformers/configs/llm_template_v2_experimental/llm_config_whitelist_introduce.md`

---

## 7. 环境变量 / Environment Variables

* **`MF_LOG_SUFFIX`**: 为日志文件夹添加后缀 / Add suffix to log folder.
* **`USE_CONFIG_TEMPLATE_V2`**: CLI 默认强制设为 "1"，无需手动设置。
    * CLI explicitly sets this to "1" internally; manual setting is not required.
