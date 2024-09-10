# Llava1.5

## 模型描述

LLaVA 1.5是一个端到端训练的大型多模态模型，连接视觉编码器和大语言模型，以实现通用视觉和语言理解，通过在 GPT 生成的多模式指令跟踪数据上微调 LLaMA/Vicuna 进行训练。它是一种基于 Transformer 架构的自回归语言模型。

```text
@inproceedings{liu2023llava,
    author      = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    title       = {Visual Instruction Tuning},
    booktitle   = {NeurIPS},
    year        = {2023}
  }
```

## 模型文件

`Llava1.5` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/llava/
       ├── __init__.py
       ├── llava.py                  # 模型实现
       ├── llava_clip_vit.py         # 视觉编码器实现
       └── llava_config.py           # 模型配置
   ```

2. 模型配置：

   ```text
   research/llava
       └── predict_llava1_5_7b.yaml     # 7B推理配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

|     模型     |     硬件      | 推理 |
| :----------: | :-----------: | :--: |
| Llava-1.5-7b | Atlas 800T A2 | 单卡 |

### 权重准备

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/tokenizer.model)

| 模型名称    | MindSpore权重 |                       HuggingFace权重                        |
| :---------- | :-----------: | :----------------------------------------------------------: |
| Llava1.5-7B |       \       | [Link](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llava --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 推理

MindFormers提供`Llava1.5-7b`的推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/llava/run_llava1_5_predict.sh PARALLEL CONFIG_PATH CKPT_PATH TOKENIZER_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
TOKENIZER_PATH:  词表路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

以`llava1.5-7b`单卡推理为例。

```shell
bash scripts/examples/llava/run_llava1_5_predict.sh single \
 research/llava/predict_llava1_5_7b.yaml \
 path/to/llava_7b.ckpt \
 path/to/tokenizer.model

```

### 多卡推理

以`Llava1.5-7b`2卡推理为例。

```shell
bash scripts/examples/llava/run_llava1_5_predict.sh parallel \
 research/llava/predict_llava1_5_7b.yaml \
 path/to/llava_7b.ckpt \
 path/to/tokenizer.model 2

```
