# InternLM2

## 模型描述

第二代浦语模型，InternLM2 的基础模型具备以下的技术特点：

有效支持20万字超长上下文：模型在20万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。
综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码等方面的能力提升显著。

本仓库支持InternLM2-7B的推理。由于InternLM2与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

> 注: 由于InternLM2基于高阶接口的形式开发，存放于research文件夹下，使用时需要将MindFormers[安装](https://gitee.com/mindspore/mindformers#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)为python包，才能直接进入research/internlm2目录下执行相关命令。

## 模型性能

|                                       config                                       |      task       | train performance |       [predict performance](###快速推理)        |
|:----------------------------------------------------------------------------------:|:---------------:|:-----------------:|:-------------------------------------------:|
| [InternLM2_7B (Atlas 800T A2)](../../research/internlm2/predict_internlm2_7b.yaml) | text_generation |         /         | 38.3 tokens/s (batch_size=1, use_past=True) |

## 模型文件

`InternLM2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现

    ```bash
    research/internlm2
      ├── internlm2_tokenizer.py       # tokenizer
      ├── internlm2_transformer.py     # transformer层实现
      ├── internlm2_config.py          # 模型config
      ├── internlm2.py                 # 模型实现
      └── internlm2_interleave.py
    ```

2. 模型配置

    ```bash
    research/internlm2
      ├── predict_internlm2_20b.yaml
      └── predict_internlm2_7b.yaml      # InternLM2-7B推理Atlas 800T A2启动配置
    ```

3. 预处理脚本和任务启动脚本

    ```bash
    research/internlm2
      ├── convert_weight.py             # hf->mf权重转换
      ├── convert_reversed.py           # mf->hf权重转换
      └── run_internlm2.py               # 高阶接口使用脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](https://gitee.com/mindspore/mindformers/blob/4cc7b39b3dcc93c99117ee8f87d3c2423bd761b1/README.md#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)和[版本匹配关系](https://gitee.com/mindspore/mindformers/blob/4cc7b39b3dcc93c99117ee8f87d3c2423bd761b1/README.md#%E4%B8%89%E7%89%88%E6%9C%AC%E5%8C%B9%E9%85%8D%E5%85%B3%E7%B3%BB)。

## 数据及权重准备

### 数据集下载

### 模型权重下载

用户也可选择从Hugging Face下载预训练权重后根据以下步骤进行权重转换，包含对应的分词模型，需要下载整个工程，Hugging Face权重的链接如下：

|       模型名称        | MindSpore权重 |                  HuggingFace权重及Tokenizer                  |
|:-----------------:|:-----------:|:---------------------------------------------------------:|
|   InternLM2-7b    |      \      |   [link](https://huggingface.co/internlm/internlm2-7b)    |
| InternLM2-chat-7B |      \      | [link](https://huggingface.co/internlm/internlm2-chat-7b) |
|   InternLM2-20b   |      \      |   [link](https://huggingface.co/internlm/internlm2-20b)   |

### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```bash
python convert_weight.py \
    --model        internlm2 \
    --input_path   TORCH_CKPT_DIR \
    --output_path  {path}/MS_CKPT_NAME \
    --qkv_concat   True

    # 参数说明
    input_path: huggingface权重保存目录路径
    output_path: 权重保存文件名, 可以指定自定义保存路径
    qkv_concat: 是否qkv融合
```

## 推理

MindFormers提供internlm2-7b的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```bash
# 脚本使用
bash scripts/examples/internlm2/run_internlm2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM TOKENIZER_PATH

# 参数说明
PARALLEL:        是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH:     模型配置文件路径
CKPT_PATH:       模型权重文件路径
DEVICE_NUM:      使用卡数, 仅开启多卡推理时生效
TOKENIZER_PATH:  Tokenizer模型路径
```

### 单卡推理

```bash
bash scripts/examples/internlm2/run_internlm2_predict.sh \
 single \
 configs/internlm2/predict_internlm2_7b.yaml \
 path/to/internlm2_7b.ckpt \
 1 \
 path/to/tokenizer.model
```

### 多卡推理

以`internlm2-7b`8卡推理为例。

```bash
bash scripts/examples/internlm2/run_internlm2_predict.sh \
 parallel \
 configs/internlm2/predict_internlm2_7b.yaml \
 path/to/internlm2_7b.ckpt \
 8 \
 path/to/tokenizer.model
```
