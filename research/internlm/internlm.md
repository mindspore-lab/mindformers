# InternLM

## 模型描述

InternLM ，即书生·浦语大模型，是由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。包含面向实用场景的70亿和200亿参数基础模型与对话模型 （InternLM-7B/20B）。模型具有以下特点：

- 使用上万亿高质量语料，建立模型超强知识体系；
- 支持8k语境窗口长度，实现更长输入与更强推理体验；
- 通用工具调用能力，支持用户灵活自助搭建流程；

本仓库目前能够支持上述特性1，暂未支持特性2、3。

本仓库支持InternLM-7B的微调和InternLM-Chat-7B/20B的推理。由于InternLM与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

``` text
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```

## 模型性能

|                            config                            |      task       | Datasets | [train performance](#全参微调) |    [predict performance](###MindSpore推理)    |
| :----------------------------------------------------------: | :-------------: | :------: | :----------------------------: | :-----------------------------------------: |
| [InternLM_7B (Atlas 800T A2)](../../research/internlm/finetune_internlm_7b.yaml) | text_generation |  alpaca  |         3182 tokens/s          | 58.9 tokens/s (batch_size=1, use_past=True) |
| [InternLM_7B_lora (Atlas 800T A2)](../../research/internlm/finetune_internlm_7b_lora.yaml) | text_generation |  alpaca  |         3864 tokens/s          |                      /                      |
| [InternLM_20B (Atlas 800T A2)](../../research/internlm/predicet_internlm_20b.yaml) | text_generation |    /     |               /                | 25.3 tokens/s (batch_size=1, use_past=True) |

## 模型文件

`InternLM` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/internlm`

    ```bash
    internlm
        ├── internlm_tokenizer.py       # tokenizer
        ├── internlm_transformer.py     # transformer层实现
        ├── internlm_config.py          # 模型config
        └── internlm.py                 # 模型实现
    ```

2. 模型配置：`research/internlm`

    ```bash
    internlm
        ├── finetune_internlm_7b.yaml             # InternLM-7B全参微调Atlas 800T A2启动配置
        ├── finetune_internlm_7b_lora.yaml        # InternLM-7B lora低参微调Atlas 800T A2启动配置
        ├── predict_internlm_7b.yaml              # InternLM-7B推理Atlas 800T A2启动配置
        └── predict_internlm_20b.yaml             # InternLM-20B推理Atlas 800T A2启动配置
    ```

3. 预处理脚本和任务启动脚本：`research/internlm`

    ```bash
    internlm
        ├── alpaca_data_preprocess.py     # alpaca数据集预处理
        ├── wiki_data_preprocess.py       # wikitext2数据集预处理
        ├── convert_weight.py             # hf->mf权重转换
        ├── convert_reversed.py           # mf->hf权重转换
        └── run_internlm.py               # 高阶接口使用脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。
> 注：Atlas 800T A2芯片支持7b，20b单机单卡推理，7b的全参微调支持单机八卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext2**作为预训练数据集，**alpaca**作为[微调](#微调)数据集。

| 数据集名称     |                    适用模型                     |          适用阶段           |                                                         下载链接                                                          |
|:----------|:-------------------------------------------:|:-----------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2 | internlm-7b <br/> internlm20b <br/> |  Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| alpaca    |  internlm-7b <br/> internlm20b <br/> |        Finetune         |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |
| alpaca-gpt4-data-zh |  internlm-7b <br/> internlm20b <br/> |        Finetune         |                                     [Link](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh/blob/main/alpaca_gpt4_data_zh.json)                                      |

下载数据集后，使用预处理脚本生成mindrecord训练数据：

- WikiText2数据集预处理指令示例：

```shell
cd mindformers/research/internlm
python wiki_data_preprocess.py \
--mindrecord_schema internlm_wiki \
--input_glob {path}/wikitext-2/wiki.train.tokens \
--output_file {path}/wiki_processed/wiki.mindrecord \
--model_file {path}/tokenizer.model \
--seq_length 2048 \
--min_length 50  # 过滤token长度小于min_length的数据，default=50
```

- Alpaca数据集预处理指令示例：（同时适用于alpaca_data和alpaca-gpt4-data-zh数据集）

```shell
cd mindformers/research/internlm
python alpaca_data_preprocess.py \
--mindrecord_schema internlm_alpaca \
--input_glob {path}/alpaca_data.json \
--output_file {path}/alpaca_processed/alpaca.mindrecord \
--model_file {path}/tokenizer.model \
--seq_length 2048
```

#### 模型权重下载

MindFormers 提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

也可选择从 HuggingFace 下载所由工程文件后进行模型权重转换使用。
| 模型名称            |                                                 MindSpore权重                                                  |                      HuggingFace权重                       |
|:----------------|:------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| InternLM-7B-Base       |    [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm.ckpt)    | [Link](https://huggingface.co/internlm/internlm-7b)  |
| InternLM-7B-Chat      | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm-chat.ckpt) |  [Link](https://huggingface.co/internlm/internlm-chat-7b)   |                                                                                                            |  |
| InternLM-20B-Base |  / |             [Link](https://huggingface.co/internlm/internlm-20b)                                   |
| InternLM-20B-Chat | /  |                      [Link](https://huggingface.co/internlm/internlm-chat-20b)                                |
| tokenizer.model |  [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/tokenizer.model) |                         /                             |

#### 模型权重转换

原始权重下载完成后，运行如下转换脚本，将Hugging Face的权重转换为完整的ckpt权重。

```shell
# 请安装torch=2.0.0和transformers=4.30.2版本:
# pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
python ./research/internlm/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
# 参数说明
torch_ckpt_path: Hugging Face权重保存目录路径下任意权重bin文件，将根据该文件路径读取目录下全部权重
mindspore_ckpt_path: 转换后MindSpore权重文件的保存路径
```

## 微调

MindFormers提供 internlm 全参微调和LoRA微调的示例。

### 全参微调

InternLM-7B用于微调，seq_length默认为2048，分布式微调训练使用单机八卡上启动。以alpaca_data数据集为例，在Atlas 800T A2上默认使用`finetune_internlm_7b.yaml`配置文件即可。

1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── internlm_7b_base.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入切分好的权重路径文件夹即可。

2. 修改`finetune_internlm_7b.yaml`中相关配置

```python
output_dir: './output'             # path to save checkpoint/strategy
load_checkpoint: 'path/of/ckpt'    # 添加预训练权重路径
auto_trans_ckpt: True              # 开启权重自动切分
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/alpaca.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 用alpaca数据集指令微调时，input_columns: ["input_ids", "labels"]
# 用wiki数据集微调时，input_columns: ["input_ids"]
```

2. 启动微调任务，以单机八卡为例，指令如下：

```shell
bash scripts/msrun_launcher.sh \
"python research/internlm/run_internlm.py \
--run_mode finetune \
--use_parallel True \
--config research/internlm/finetune_internlm_7b.yaml \
--load_checkpoint path/to/ckpt \
--auto_trans_ckpt True \
--train_dataset {path}/train_data" 8
```

### Lora微调

Lora微调支持单卡/多卡启动，以alpaca-gpt4-data-zh数据集为例，在Atlas 800T A2机器上，使用`finetune_internlm_7b_lora.yaml`配置文件即可。

1. 参考全参微调任务修改配置文件中的预训练权重路径、数据集路径。

2. 启动lora微调任务。

单卡启动指令如下：

```shell
python run_internlm.py \
--config finetune_internlm_7b_lora.yaml \
--run_mode finetune \
--use_parallel False \
--load_checkpoint path/to/ckpt \
--auto_trans_ckpt False \
--train_dataset {path}/train_data \
--device_id 0
```

多卡启动以单机八卡为例，指令如下：

```shell
bash scripts/msrun_launcher.sh \
"python research/internlm/run_internlm.py \
--run_mode finetune \
--use_parallel True \
--config research/internlm/finetune_internlm_7b_lora.yaml \
--load_checkpoint path/to/ckpt \
--auto_trans_ckpt True \
--train_dataset {path}/train_data" 8
```

## 推理

MindFormers提供 InternLM-20b 的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/internlm/run_internlm_predict.sh PARALLEL CONFIG_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

```shell
# internlm 7b
bash scripts/examples/internlm/internlm_predict.sh single \
 research/internlm/predict_internlm_7.yaml \
 path/to/InternLM_7B_Chat.ckpt \
 path/to/tokenizer.model

# internlm 20b
bash scripts/examples/internlm/internlm_predict.sh single \
 research/internlm/predict_internlm_20b.yaml \
 path/to/InternLM_20B_Chat.ckpt \
 path/to/tokenizer.model
```

### 多卡推理

```shell
# internlm 7b
bash scripts/examples/internlm/run_internlm_predict.sh parallel \
 research/internlm/predict_internlm_7b.yaml \
 path/to/InternLM_7B_Chat.ckpt \
 path/to/tokenizer.model 2

# internlm 20b
bash scripts/examples/internlm/run_internlm_predict.sh parallel \
 research/internlm/predict_internlm_20b.yaml \
 path/to/InternLM_20B_Chat.ckpt \
 path/to/tokenizer.model 2
```