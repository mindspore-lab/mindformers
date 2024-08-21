# Llama 3.1

## 模型描述

Llama 3.1，是开源Llama系列的最新产品，目前有三个版本：Llama 3.1-8B，Llama 3.1-70B，Llama 3.1-405B。
Llama 3.1在来自公开可用来源的超过15T的数据上进行了预训练。微调数据包括公开可用的指令数据集，以及超过1000万个人工标注的示例。
模型支持上下文窗口长度128K，并使用了新的分词器，词汇表大小达到128256个，采用了分组查询注意力机制(GQA)。
Llama 3.1模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。
目前Mindformers支持Llama 3.1-8B，Llama 3.1-70B，敬请期待Llama 3.1-405B。

## 模型文件

`Llama 3.1` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：

   ```text
   research/llama3_1
       ├── predict_llama3_1_8b.yaml    # 8B推理配置
       ├── predict_llama3_1_70b.yaml   # 70B推理配置
       └── finetune_llama3_1_8b.yaml   # 8B全量微调Atlas 800 A2启动配置
   ```

3. 数据预处理脚本和任务启动脚本：

   ```text
   research/llama3_1
       ├── run_llama3_1.py           # llama3_1启动脚本
       ├── llama3_1_tokenizer.py     # llama3_1 tokenizer处理脚本
       ├── conversation.py         # 微调数据集处理，将原始alpaca转换为对话形式alpaca
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)
和[版本匹配关系](../../README.md#版本匹配关系)。

|      模型      |      硬件       | 全量微调 | 推理 |
|:------------:|:-------------:|:----:|:--:|
| Llama3.1-8b  | Atlas 800T A2 | 单节点 | 单卡 |
| Llama3.1-70b | Atlas 800T A2 | 8节点  | 4卡 |

### 数据集及权重准备

#### 数据集下载

MindFormers提供**alpaca**作为[微调](#微调)数据集。

| 数据集名称   |    适用模型     |   适用阶段   |                                      下载链接                                       |
|:--------|:-----------:|:--------:|:-------------------------------------------------------------------------------:|
| alpaca  | llama3_1-8b | Finetune | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **alpaca 数据预处理**

    1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

       ```shell
       python alpaca_converter.py \
         --data_path /{path}/alpaca_data.json \
         --output_path /{path}/alpaca-data-conversation.json

       # 参数说明
       data_path:   输入下载的文件路径
       output_path: 输出文件的保存路径
       ```

    2. 执行`research/llama3_1/llama_preprocess.py`，生成Mindrecord数据，将带有prompt模板的数据转换为mindrecord格式。

       ```shell
       # 此工具依赖fschat工具包解析prompt模板, 请提前安装fschat >= 0.2.13 python = 3.9
       python llama_preprocess.py \
         --dataset_type qa \
         --input_glob /{path}/alpaca-data-conversation.json \
         --model_file /{path}/tokenizer.model \
         --seq_length 8192 \
         --output_file /{path}/alpaca-fastchat8192.mindrecord

       # 参数说明
       dataset_type: 预处理数据类型
       input_glob:   转换后的alpaca的文件路径
       model_file:   模型tokenizer.model文件路径
       seq_length:   输出数据的序列长度
       output_file:  输出文件的保存路径
       ```

> 数据处理时候注意bos，eos，pad等特殊`ids`要和配置文件中`model_config`里保持一致。

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

| 模型名称         | MindSpore权重 |                        HuggingFace权重                         |
|:-------------|:-----------:|:------------------------------------------------------------:|
| Llama3_1-8B  |      \      | [Link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)  |
| Llama3_1-70B |      \      | [Link](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) |

> 注: 请自行申请huggingface上llama3_1使用权限，并安装transformers=4.40版本

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 全参微调

MindFormers提供`Llama3_1-8b`单机多卡的微调示例，过程中使用`alpaca`
数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机训练

以Llama3_1-8b为例，Llama3_1-8B在Atlas 800T A2上训练，支持**单机/多机训练**。

使用`finetune_llama3_1_8b.yaml`进行训练，或修改默认配置文件中的`model_config.seq_length`
，使训练配置与数据集的`seq_length`保持一致。

执行命令启动微调任务，在单机上拉起任务。

```shell
cd research
# 单机8卡默认快速启动
bash ../scripts/msrun_launcher.sh "llama3_1/run_llama3_1.py \
 --config llama3_1/finetune_llama3_1_8b.yaml \
 --load_checkpoint model_dir/xxx.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune \
 --train_data dataset_dir"

# 参数说明
config:          配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode:        运行模式, 微调时设置为finetune
train_data:      训练数据集路径
```

## 推理

MindFormers提供`Llama3_1-8b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡推理。推理输入默认不添加bos字符，如果需要添加可在config中增加add_bos_token选项。

```shell
# 脚本使用
bash scripts/examples/llama3/run_llama3_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
VOCAB_FILE:  词表路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

以`Llama3_1-8b`单卡推理为例。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh single \
 research/llama3_1/predict_llama3_1_8b.yaml \
 path/to/llama3_1_8b.ckpt \
 path/to/tokenizer.model
```

### 多卡推理

以`Llama3_1-70b`4卡推理为例。Llama3_1-70b权重较大，建议先进行权重切分，参见[权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh parallel \
 research/llama3_1/predict_llama3_1_70b.yaml \
 path/to/model_dir \
 path/to/tokenizer.model 4