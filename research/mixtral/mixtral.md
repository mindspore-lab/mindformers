# Mixtral-8x7B

## 模型描述

Mixtral是MistralAI基于Mistral的更新版本，目前有4个版本：Mixtral-8x7B，Mixtral-8x7B-Instruct，Mixtral-8x22B，Mixtral-8x22B-Instruct。Mixtral模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。目前仓上提供的Mixtral模型是Mixtral-8x7B，其模型结构与Llama2-7b几乎一致，只是在Llama2-7b的基础上添加了MoE，并采用了分组查询注意力机制(GQA)。

[Mixtral of Experts](https://arxiv.org/abs/2401.04088)

``` text
@article{
  title={Mixtral of Experts},
  author={Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour and others},
  journal={arXiv preprint arXiv:2401.04088},
  year={2024}
}
```

## 仓库介绍

`Mixtral` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # mixtral网络层定义
       ├── llama_processor.py        # mixtral预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`research/mixtral`

   ```bash
   mixtral
    ├── run_mixtral-8x7b_train.yaml       # 8x7b模型预训练启动配置
    ├── run_mixtral-8x7b_finetune.yaml    # 8x7b模型全参微调启动配置
    └── convert_weight.py                 # 权重转换脚本
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 前期准备

### 环境要求

- 硬件：Ascend910B
- MindSpore：2.3.0
- CANN: 7.2
- MindFormers版本：dev

注：910B推理单机2卡即可推理.全参微调910B至少需要二机16卡。

### 模型权重准备

#### torch权重转mindspore权重

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重。

从huggingface下载预训练权重（权重来源于mistralai）：

- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/research/mixtral/convert_weight.py \
--torch_ckpt_dir {path}/torch_ckpt_dir \
--mindspore_ckpt_path {path}/ms_ckpt_path
```

```text
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径,填写safetensors文件的保存路径。
mindspore_ckpt_path: ms权重保存文件路径，可以指定自定义保存路径。
```

#### mindspore权重转torch权重

在生成mindspore权重之后如需使用torch运行，可根据如下命令转换：

```shell
python mindformers/research/mixtral/convert_reversed.py \
--mindspore_ckpt_path {path}/ms_ckpt_path \
--torch_ckpt_dir {path}/torch_ckpt_dir \
--strategy_dir {path}/strategy_dir \
```

```text
# 参数说明：
mindspore_ckpt_path: 待转换的mindspore权重或分布式权重的文件夹路径，此参数必须。
torch_ckpt_path: 转换后的输出文件存放路径，此参数必须。
strategy_dir: 若为分布式权重，填写待转换mindspore权重的strategy文件路径。若为完整权重则无需此参数。
```

### 模型权重切分和合并

从huggingface转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行多卡推理，需要将分布式权重转换为多卡推理权重。

- 修改分布式策略训练，需要将权重转换为对应分布式权重。

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix mixtral_
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 预训练

### 数据集准备

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2FMindFormers%2Fdataset%2Fwikitext-2%2Fwikitext-2-v1.zip)

- 分词模型下载：[tokenizer文件](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/tokenizer.model)

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

数据处理时候注意bos，eos，pad等特殊ids要和yaml配置中model_config里保持一致，默认bos_token_id=1, eos_token_id=2, pad_token_id=0, 如果有所修改，yaml中对应的配置也需要修改； 一般预训练的数据中不包含pad_token，此时建议pad_token_id设为-1。

### 脚本启动（Mixtral 8*7b为例）

#### 多卡训练

##### 多机多卡

- step 1. 在模型对应的配置文件`research/mixtral/run_mixtral-8x7b_train.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练) 通过配置中的`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

- step 2. 根据服务器节点数等信息，修改相应的配置。

``` bash
# 以mixtral-8x7b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../research/mixtral/run_mixtral-8x7b_train.yaml
parallel_config:
  data_parallel: 8
  model_parallel: 1
  expert_parallel: 8
  pipeline_stage: 2
  use_seq_parallel: False
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

> **注：多机多卡并行配置中data_parallel\*model_parallel\*pipeline_stage == 总卡数，且expert_parallel不能超过data_parallel。**

- step 3. 调大`moe_config`中的专家容量因子`capacity_factor`(非必要步骤)

``` bash
# capacity_factor默认值为1.1，调大capacity_factor(建议值2.0-4.0)可提高训练精度，但会带来一定的性能损失
# moe
moe_config:
  expert_num: 8
  capacity_factor: 2.0
  aux_loss_factor: 0.05
  num_experts_chosen: 2
  routing_policy: "TopkRouterV2"
  enable_sdrop: True
```

- step 4. 执行运行脚本。

在多机上同时拉起任务，将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，具体可参考[ms_run快速使用](https://gitee.com/mindspore/mindformers#%E5%9B%9B%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8)

```shell
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
cd mindformers/research
bash ../scripts/msrun_launcher.sh "run_mindformer.py \
 --config mixtral/run_mixtral-8x7b_train.yaml \
 --run_mode train \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
cd mindformers/research
bash ../scripts/msrun_launcher.sh "run_mindformer.py \
 --config mixtral/run_mixtral-8x7b_train.yaml \
 --run_mode train \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 1 output/msrun_log False 300

# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

## 微调

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
# 脚本路径：tools/dataset_preprocess/llama/llama_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python llama_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/alpaca-fastchat4096.mindrecord
```

### 全参微调

当前模型已支持使用`Flash Attention`算法进行全参微调，推荐开启flash_attention，可加速训练。详请参考 [Flash Attention使用文档](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 将`../research/mixtral/run_mixtral-8x7b_finetune.yaml`中训练数据集路径为微调数据集路径。

```bash
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 默认开启`moe_config`中的`enable_sdrop`选项，修改微调时学习率, 优化器参数，`seq_length`, 与预训练不同，微调配置如下：

```bash
# moe
moe_config:
  expert_num: 8
  capacity_factor: 1.1
  aux_loss_factor: 0.05
  num_experts_chosen: 2
  routing_policy: "TopkRouterV2"
  enable_sdrop: True

# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.95
  eps: 1.e-8
  learning_rate: 3.e-4

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 4096
```

- step 3. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。若无共享磁盘，跨机需先手动切分权重，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 4. 启动微调任务，以两机16卡为例进行微调，命令如下：

```shell
# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
cd mindformers/research
bash ../scripts/msrun_launcher.sh "run_mindformer.py \
 --config mixtral/run_mixtral-8x7b_finetune.yaml \
 --load_checkpoint model_dir \
 --run_mode finetune \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config mixtral/run_mixtral-8x7b_finetune.yaml \
 --load_checkpoint model_dir \
 --run_mode finetune \
 --use_parallel True \
 --train_data dataset_dir" \
 16 8 192.168.1.1 8118 1 output/msrun_log False 300

 # 参数说明
config: 配置文件路径
load_checkpoint: 切分好的分布式权重文件夹路径，权重按照model_dir/rank_x/xxx.ckpt格式，文件夹路径填写为model_dir
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```
