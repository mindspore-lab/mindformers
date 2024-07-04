# Yi大模型

Yi系列是由零一万物研究的大规模语言预训练模型，目前开源的有Yi-6B/34B-Base/Chat，Yi-VL-6B/34B，MindFormers已支持Yi-6B-Base,Yi-34B-Base/Chat。当前训练使用Base权重，推理使用Base/Chat权重

[Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652v1)

``` text
@article{ai2024yiopenfoundationmodels,
      title={Yi: Open Foundation Models by 01.AI},
      author={01. AI and : and Alex Young and Bei Chen and Chao Li and Chengen Huang and Ge Zhang and Guanwei Zhang and Heng Li and Jiangcheng Zhu and Jianqun Chen and Jing Chang and Kaidong Yu and Peng Liu and Qiang Liu and Shawn Yue and Senbin Yang and Shiming Yang and Tao Yu and Wen Xie and Wenhao Huang and Xiaohui Hu and Xiaoyi Ren and Xinyao Niu and Pengcheng Nie and Yuchi Xu and Yudong Liu and Yue Wang and Yuxuan Cai and Zhenyu Gu and Zhiyuan Liu and Zonghong Dai},
      year={2024},
      eprint={2403.04652},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.04652},
}
```

## 模型性能

| Config                                                            |      Task       |        Datasets        |   Performance   |  Phase   |
|:------------------------------------------------------------------|:---------------:|:----------------------:|:---------------:|:--------:|
| [finetune_yi_6b](../../research/yi/finetune_yi_6b.yaml)           | text_generation |  alpaca_gpt4_data_zh   | 3324 tokens/s/p | Finetune |
| [finetune_yi_34b](../../research/yi/finetune_yi_34b.yaml)         | text_generation |         alpaca         |  660 tokens/s/p | Finetune |
| [pretrain_yi_34b](../../research/yi/pretrain_yi_34b.yaml)         | text_generation |        wikitext2       |  660 tokens/s/p | Pretrain |
| [predict_yi_6b](../../research/yi/predict_yi_6b.yaml)             | text_generation |           /            |   39 tokens/s   | Predict  |
| [predict_yi_34b](../../research/yi/predict_yi_34b.yaml)           | text_generation |           /            |   39 tokens/s   | Predict  |
| [predict_yi_34b_chat](../../research/yi/predict_yi_34b_chat.yaml) | text_generation |           /            |   39 tokens/s   | Predict  |

## 模型文件

1. 模型配置：

   ```text
    yi
     ├── finetune_yi_6b.yaml               # 6B 全参微调启动配置
     ├── finetune_yi_34b.yaml              # 34B 全参微调启动配置
     ├── pretrain_yi_34b.yaml              # 34B 预训练启动配置
     ├── predict_yi_6b.yaml                # 6B base在线推理启动配置  
     ├── predict_yi_34b.yaml               # 34B base在线推理启动配置
     └── predict_yi_34b_chat.yaml          # 34B chat在线推理启动配置
   ```

2. 环境准备和任务启动脚本：

   ```text
    yi
     ├── alpaca_converter.py           # alpaca数据集格式转换脚本
     ├── yi_preprocess.py              # 数据集预处理脚本
     ├── convert_ckpt_bf16.py          # 权重转换脚本
     ├── predict_yi_34b_chat.py        # 34B chat在线推理启动脚本
     └── run_yi.py                     # Qwen高阶接口脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：Atlas 800T A2芯片支持6b单卡推理，全参微调至少需要4卡，建议8卡；34b推理需要4卡，全参微调需要双机32卡。

### 数据及权重准备

#### 数据集下载

| 数据集名称            |      适用模型       | 适用阶段  |                                                         下载链接                                                        |
|:--------------------|:------------------:|:--------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2           | yi-34b             | Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| alpaca              | yi-6b <br/> yi-34b | Finetune | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                                       |
| alpaca_gpt4_data_zh | yi-6b <br/> yi-34b | Finetune | [Link](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh/resolve/main/alpaca_gpt4_data_zh.json)             |

#### 预训练数据集

使用Yi-6B-Base或Yi-34B-Base进行预训练时，需要使用配套的tokenizer.model处理数据集。

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fdataset%2Fwikitext-2%2Fwikitext-2-v1.zip)

- 使用以下预处理脚本生成mindrecord训练数据

``` bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/wiki4096.mindrecord
```

#### 微调数据集

使用Yi-6B-Base或Yi-34B-Base进行全参微调时，需要使用配套的tokenizer.model处理数据集。

目前提供[alpaca数据集](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)（json格式）[alpaca_gpt4_data_zh数据集](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh/resolve/main/alpaca_gpt4_data_zh.json) （jsonl格式）数据集的预处理脚本用于全参微调任务。

alpaca数据集样式

```text
  {
    "instruction": "保持健康的三个提示。",
    "input": "",
    "output": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"
  }
```

- step 1. 执行`alpaca_converter.py`，将原始数据集转换为对话格式。

``` bash
# 脚本路径：yi/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/alpaca_gpt4_data_zh.json \
--output_path /{path}/alpaca_gpt4_data_zh-conversation.json
```

```text
# 参数说明
data_path: 存放原始数据的路径
output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
[
  {
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "保持健康的三个提示。"
      },
      {
        "from": "gpt",
        "value": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"
      }
    ]
  }
]
```

- step 2. 执行`yi_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：yi/yi_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python yi_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca_gpt4_data_zh-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/alpaca_gpt4_data_zh.mindrecord
```

```text
# 参数说明
input_file_path：数据集输入文件路径
output_file：生成的mindrecord目标文件路径
dataset_type：数据集类型，目前仅支持"text"和"qa"
model_file：tokenizer.model文件路径
seq_length：数据长度
```

#### 模型权重下载

- 从huggingface下载原始权重后转换

需要将整个工程下载下来。

- [Yi-6B-Base](https://huggingface.co/01-ai/Yi-6B)
- [Yi-34B-Base](https://huggingface.co/01-ai/Yi-34B)
- [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B
```

#### 模型权重转换

执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model yi --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

**注**: 请安装torch>=2.2.0和transformers>=4.37.2版本。如果执行报错，请检查并安装requests、decorator、pandas、sympy。

#### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 预训练

以Yi-34b为例。

1. 修改模型配置文件`research/yi/pretrain_yi_34b.yaml`

   ```yaml
   processor:
    tokenizer:
      vocab_file: "/{path}/tokenizer.model"        # 词表文件路径
   ```

   `ymal`配置文件中各参数含义详见[Config配置说明](../../configs/README.md)，

2. 执行msrun启动脚本，进行双机16卡分布式训练

   在多机上同时拉起任务，将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，具体可参考ms_run快速使用

```bash
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "research/yi/run_yi.py \
--config research/yi/pretrain_yi_34b.yaml \
--use_parallel True \
--run_mode train \
--auto_trans_ckpt True \
--train_dataset /{path}/alpaca.mindrecord" \
16 8 {ip_addr} 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "research/yi/run_yi.py \
--config research/yi/pretrain_yi_34b.yaml \
--use_parallel True \
--run_mode train \
--auto_trans_ckpt True \
--train_dataset /{path}/alpaca.mindrecord" \
16 8 {ip_addr} 8118 1 output/msrun_log False 300

# 参数说明
# config: 配置文件路径
# load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
# auto_trans_ckpt: 自动权重转换开关
# run_mode: 运行模式，微调时设置为finetune
# train_dataset: 训练数据集文件夹路径
```

## 微调

### 全参微调

#### 单机训练

以Yi-6b全参微调为例。

1. 修改模型配置文件`research/yi/finetune_yi_6b.yaml`

```yaml
processor:
 tokenizer:
  vocab_file: "/{path}/tokenizer.model"        # 词表文件路径
```

   `ymal`配置文件中各参数含义详见[Config配置说明](../../configs/README.md)，

2. 执行msrun启动脚本，进行8卡分布式微调

```bash
bash scripts/msrun_launcher.sh " \
 python research/yi/run_yi.py \
 --config research/yi/finetune_yi_6b.yaml \
 --run_mode finetune \
 --load_checkpoint /{path}/yi_6b.ckpt \
 --train_dataset /{path}/alpaca_gpt4_data_zh.mindrecord \
 --auto_trans_ckpt True \
 --use_parallel True" 8
```

#### 多机训练

以Yi-34b全参微调为例。

1. 修改模型配置文件`research/yi/finetune_yi_34b.yaml`

```yaml
processor:
 tokenizer:
  vocab_file: "/{path}/tokenizer.model"        # 词表文件路径
```

   `ymal`配置文件中各参数含义详见[Config配置说明](../../configs/README.md)，

2. 执行msrun启动脚本，进行8卡分布式微调

在多机上同时拉起任务，将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，具体可参考ms_run快速使用

```bash
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "research/yi/run_yi.py \
--config research/yi/finetune_yi_34b.yaml \
--load_checkpoint /path/model_dir \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_dataset /path/alpaca.mindrecord" \
16 8 {ip_addr} 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "research/yi/run_yi.py \
--config research/yi/finetune_yi_34b.yaml \
--load_checkpoint /path/model_dir \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_dataset /path/alpaca.mindrecord" \
16 8 {ip_addr} 8118 1 output/msrun_log False 300

# 参数说明
# config: 配置文件路径
# load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
# auto_trans_ckpt: 自动权重转换开关
# run_mode: 运行模式，微调时设置为finetune
# train_dataset: 训练数据集文件夹路径
```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

涉及到模型权重的单卡或多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)。

1. 获取模型切分策略文件：

   在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

   ```shell
   python transform_ckpt.py \
     --src_ckpt_strategy {path}/output/strategy/ \
     --src_ckpt_dir {path}/output/checkpoint/ \
     --dst_ckpt_dir {path}/target_checkpoint/ \
     --prefix yi_6b

   # 参数说明
   src_ckpt_strategy: 切分策略文件路径
   src_ckpt_dir:      原切分权重文件夹
   dst_ckpt_dir:      目标路径
   prefix:            ckpt文件前缀
   ```

   > 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以通过mindspore 2.0的cpu版本以执行该脚本。

## 推理

MindFormers提供`yi_6b/34b`的Base/Chat快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/yi/run_yi_predict.sh CONFIG_PATH PREDICT_MODE DEVICE_NUM

# 参数说明
CONFIG_PATH: 模型配置文件路径
PREDICT_MODE:   模型推理模式, 使用Base或Chat区分
DEVICE_NUM:  使用卡数, 1为单卡, 其他为多卡
```

### 单卡推理

当前yi_34b模型较大，不支持单卡推理，以6b为例

```shell
bash scripts/examples/yi/run_yi_predict.sh \
 research/yi/predict_yi_6b.yaml \
 Base 1
```

### 多卡推理

以`yi_34b_base`4卡推理为例。

```shell
bash scripts/examples/yi/run_yi_predict.sh \
 research/yi/predict_yi_34b.yaml \
 Base 4
```

### Chat推理

以`yi_34b_Chat`4卡推理为例。

```shell
bash scripts/examples/yi/run_yi_predict.sh \
 research/yi/predict_yi_34b_chat.yaml \
 Chat 4
```
