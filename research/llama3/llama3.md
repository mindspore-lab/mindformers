# Llama 3

## 模型描述

Llama 3，是开源Llama系列的最新产品，目前有二个版本：Llama3-8B，Llama 3-70B。Llama 3在来自公开可用来源的超过15T的数据上进行了预训练。微调数据包括公开可用的指令数据集，以及超过1000万个人工标注的示例。模型支持上下文窗口长度8K，并使用了新的分词器，词汇表大小达到128256个，采用了分组查询注意力机制(GQA)。Llama 3模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。目前Mindformers支持Llama 3-8B。

## 仓库介绍

`Llama 3` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`research/llama3`

   ```bash
   llama3
       ├── predict_llama3_8b_8k_800T_A2_64G.yaml    # 8B推理配置
       └── run_llama3_8b_8k_800T_A2_64G.yaml        # 8B全量微调Atlas 800 A2启动配置
   ```

3. 数据预处理脚本和任务启动脚本：`research/llama3`

   ```bash
   llama3
       ├── run_llama3.py           # llama3启动脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.3
- MindFormers版本：dev
- 硬件支持矩阵

|     模型      | 硬件 | 全量微调 | 推理 |
| :-----------: | :--: | :------: | :--: |
| Llama3-8b | Atlas 800T A2 |  单节点  | 单卡 |

### 数据集准备

目前提供alpaca数据集的预处理脚本用于全参微调任务。

数据集下载链接如下：

- [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

alpaca数据集原始格式样例：

```text
# alpaca examples:
    {
        "instruction": "Describe a time when you had to make a difficult decision.",
        "input": "",
        "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client\u2019s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team\u2019s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client\u2019s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
    },
    {
        "instruction": "Identify the odd one out.",
        "input": "Twitter, Instagram, Telegram",
        "output": "Telegram"
    },
```

- step 1. 执行`alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

``` bash
# 脚本路径：tools/dataset_preprocess/llama/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/alpaca_data.json \
--output_path /{path}/alpaca-data-conversation.json
```

```text
# 参数说明
data_path: 存放alpaca数据的路径
output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
{
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
```

- step 2. 执行`llama_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：research/llama_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python llama_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 8192 \
--output_file /{path}/alpaca-fastchat8192.mindrecord
```

数据处理时候注意bos，eos，pad等特殊ids要和yaml配置中model_config里保持一致。

### 模型权重准备

选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

**注**: 请安装transformers=4.40版本

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16
# 参数说明
input_path: huggingface权重保存目录路径
output_path: 权重保存文件名，可以指定自定义保存路径
dtype: 转换权重的精度选择。
```

### [模型权重转换](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

## Llama3-8B

### 全参微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的alpaca数据集，参照[模型权重准备](#模型权重准备)章节获取Llama3-8B权重。

Llama3-8B在Atlas 800T A2上训练，支持**单机/多机训练**。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)：

使用`run_llama3_8b_8k_800T_A2_64G.yaml`进行训练，或修改默认配置文件中的`model_config.seq_length`，使数据集与训练配置的`seq_length`保持一致。

- **单机训练**

Llama3-8B用于微调，seq_length默认为8192，分布式微调训练在Atlas 800T A2上单节点即可启动。以`alpaca`数据集为例，给出了默认配置文件`run_llama3_8b_8k_800T_A2_64G.yaml`。

**步骤**：

2. 修改`run_llama3_8b_8k_800T_A2_64G.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: 'model_dir/xxx.ckpt'  # 使用完整权重路径
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# input_columns按照数据集中的字段指定（如alpaca数据集），input_columns: ["input_ids", "labels"]

# 8卡分布式策略配置
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机上拉起任务。快速启动脚本指令msrun_launcher特性参见[msrun快速启动](../../README.md#方式一使用已有脚本启动)。

```shell
cd mindformers/research
# 单机8卡默认快速启动
bash ../scripts/msrun_launcher.sh \
"llama3/run_llama3.py \
--config llama3/run_llama3_8b_8k_800T_A2_64G.yaml \
--load_checkpoint model_dir/xxx.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir"

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

### MindSpore推理

Llama3-8b用于在线推理，Atlas 800T A2支持**单卡推理**。

以下提供了基于高阶接口推理：基于trainer推理，支持传入单句或多句列表。

#### 基于高阶接口推理

1. 主要参数配置参考

```yaml
load_checkpoint: 'path/to/llama3_8b.ckpt'   # 填写权重路径
auto_trans_ckpt: False                              # 关闭自动权重转换
use_past: True                                      # 使用增量推理
vocab_file: 'path/to/tokenizer.model'               # 配置词表路径
use_parallel: False                                 # 关闭并行模式
```

2. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数，基于释出的权重。
python llama3/run_llama3.py \
--config llama3/predict_llama3_8b_800T_A2_64G.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/llama3_8b.ckpt \
--vocab_file path/to/tokenizer.model \
--auto_trans_ckpt False \
--predict_data "I love Beijing,because"

# output: [{'text_generation_text': ['I love Beijing,because it is a city of history and culture. I love Beijing,because it is a city of modernization. I love Beijing, because it is a city of the future. I love Beijing,because it is a city of my heart.']}]
```
