# Code Llama

## 模型描述

Code Llama是基于Llama 2的一系列大型代码语言模型，它在开源模型中提供了最先进的性能、填充能力、对大型输入上下文的支持以及zero-shot指令跟随能力，用于编程任务。现有多种不同版本来覆盖广泛的应用领域：基础模型（Code Llama）、Python专业化模型（Code Llama - Python）和指令跟随模型（Code Llama - Instruct），每个模型分别具有7B、13B和34B个参数。所有模型都是在16k标记序列上进行训练，并对高达100k标记的输入显示出改进效果。7B和13B版本的Code Llama以及Code Llama - Instruct变体支持基于周围内容的填充功能。Code Llama是通过对Llama 2进行更高比例的代码取样进行微调而开发的。

[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)

``` text
@article{roziere2023code,
  title={Code llama: Open foundation models for code},
  author={Roziere, Baptiste and Gehring, Jonas and Gloeckle, Fabian and Sootla, Sten and Gat, Itai and Tan, Xiaoqing Ellen and Adi, Yossi and Liu, Jingyu and Remez, Tal and Rapin, J{\'e}r{\'e}my and others},
  journal={arXiv preprint arXiv:2308.12950},
  year={2023}
}
```

## 仓库介绍

`Code Llama` 基于 `mindformers` 实现，本仓库当前支持34b模型配置，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── convert_weight.py         # 权重转换脚本
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`configs/codellama`

   ```bash
   llama
       ├── run_codellama_34b_910b.yaml         # 34b模型全量微调启动配置
       └── predict_codellama_34b_910b.yaml     # 34b模型推理配置
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 前期准备

### 环境要求

- 硬件：Atlas 800/Atlas 800T A2
- MindSpore：2.2.0 以上
- CANN: 7.0 以上
- MindFormers版本：dev

> 注：34b推理使用Atlas 800T A2 至少使用2卡，全量微调至少需要2机16卡，建议4机32卡。

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 多机RANK_TABLE_FILE合并(多机多卡必备环节)

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

RANK_TABLE_FILE 双机16卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重。

从huggingface下载预训练权重（权重来源于`Code Llama`），模型总计有三大类，每大类都有三种权重代码：

1. Code Llama

- [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf)
- [CodeLlama-13b](https://huggingface.co/codellama/CodeLlama-13b-hf)
- [CodeLlama-34b](https://huggingface.co/codellama/CodeLlama-34b-hf)

2. Code Llama-Python

- [CodeLlama-7b-Python](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)
- [CodeLlama-13b-Python](https://huggingface.co/codellama/CodeLlama-13b-Python-hf)
- [CodeLlama-34b-Python](https://huggingface.co/codellama/CodeLlama-34b-Python-hf)

3. Code Llama-Instruct

- [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [CodeLlama_13b-Instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
- [CodeLlama_34b-Instruct](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
# 使用transformers = 4.34.0，torch>=2.0进行转换
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径
```

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[[权重切分与合并](../feature_cards/Transform_Ckpt.md)]

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix llama2_7b
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

- 数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

- 分词模型下载：例如下载申请通过后huggingface里对应Files 中的tokenizer.model

- 使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/wiki4096.mindrecord
```

### 脚本启动（Code Llama-34b为例）

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

Code Llama 34b至少使用4机32卡进行训练。

当前模型已支持使用**Flash Attention算法**进行预训练，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 多卡训练

##### 多机多卡

- step 1. 多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

> **注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

- step 2. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以codellama_34b模型四机训练为例，默认配置4机32卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/codellama/run_codellama_34b_910b.yaml
parallel_config:
  data_parallel: 4
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 96
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 3. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。需注意，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/codellama/run_codellama_34b_910b.yaml [0,8] train 32
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/codellama/run_codellama_34b_910b.yaml [8,16] train 32
# 第三台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/codellama/run_codellama_34b_910b.yaml [16,24] train 32
# 第四台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/codellama/run_codellama_34b_910b.yaml [24,32] train 32
```

## 微调

### 数据集准备

目前提供code-alpaca数据集的预处理脚本用于全参微调任务。

数据集下载链接如下：

- [code_alpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json)

alpaca数据集原始格式样例：

```text
# code alpaca examples:
{
            "instruction": "Create an array of length 5 which contains all even numbers between 1 and 10.",
            "input": "",
            "output": "arr = [2, 4, 6, 8, 10]"
      },
```

- step 1. 执行`alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

``` bash
# 脚本路径：tools/dataset_preprocess/llama/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/code_alpaca_data.json \
--output_path /{path}/code-alpaca-data-conversation.json
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
        "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate an array of length 5 which contains all even numbers between 1 and 10.\n\n### Response:"
      },
      {
        "from": "gpt",
        "value": "arr = [2, 4, 6, 8, 10]"
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
--input_glob /{path}/code-alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 4096 \
--output_file /{path}/code-alpaca-fastchat4096.mindrecord
```

### 全参微调

以codellama 34b为例

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 将`config/codellama/run_codellama_34b_910b.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

```python
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/code-alpaca-fastchat4096.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 修改微调时学习率, 优化器参数，微调配置如下：

```python
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.999
  eps: 1.e-8
  learning_rate: 5.e-6

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 5.e-6
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# context
context:
  runtime_num_threads: 1
```

- step3. 设置环境变量，变量配置如下：

```bash
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
```

- step 4. 在需要进行训练的机器中**都导入权重**，添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。参考[权重切分与合并](../feature_cards/Transform_Ckpt.md)的物理机训练案例，修改权重配置如下：

  1). 有共享盘

```python
auto_trans_ckpt: True
load_checkpoint: path/to/checkpoint_dir
```

> 注：权重需要按照path/to/checkpoint_dir/rank_0/xxx.ckpt存放，load_checkpoint只需要填写到checkpoint_dir即可

​       2). 无共享盘

```python
auto_trans_ckpt: False
load_checkpoint: path/to/transformed_checkpoint
```

> 注：权重按照[权重切分与合并](../feature_cards/Transform_Ckpt.md)的教程先切成对应的份数，load_checkpoint填写到transformed_checkpoint，该文件夹下存放有rank_X的权重文件夹。

- step 5. 启动微调任务，codellama-34b模型以四机32卡进行微调，命令如下：

```shell
cd scripts
# 第一台机器
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/codellama/run_codellama_34b_910b.yaml [0,8] finetune 32
# 第二台机器
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/codellama/run_codellama_34b_910b.yaml [8,16] finetune 32
# 第三台机器
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/codellama/run_codellama_34b_910b.yaml [16,24] finetune 32
# 第四台机器
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/codellama/run_codellama_34b_910b.yaml [24,32] finetune 32
```

## 评测

`Code Llama`当前支持的评测任务如下：

| 任务类型 | 评测指标 |   数据集   |
| :------: | :------: | :--------: |
| 代码生成 |  Pass@1  | HumanEeval |

- 代码生成：

step 1. 获取数据集

[HumanEval数据集](https://github.com/openai/human-eval)是一组164个手写的编程问题数据集，被称为HumanEval数据集。每个问题包括函数签名、文档字符串、主体和几个单元测试，平均每个问题有7.7个测试。运行下面的命令，数据集在`data`文件夹中，名为`HumanEval.jsonl.gz`。

```shell
git clone https://github.com/openai/human-eval.git
```

step 2. 将下面的preprocess.py放入代码仓中的`human-eval`文件夹中，提取出`data/HumanEval.jsonl.gz`中的`prompt`字符串列表，为prompt_input，按照[推理章节](#推理)进行推理。

```python
# preprocess.py 文件
import argparse

from data import stream_jsonl


def process_data(tasks):
    prompt_input = [task["prompt"] for task in tasks]
    user_ids = [task["task_id"] for task in tasks]
    entry_inputs = [task["entry_point"] for task in tasks]
    return prompt_input, user_ids, entry_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("copy prompt")
    parser.add_argument("--data_path", default="", type=str)
    args = parser.parse_args()

    data_list = []
    for data in stream_jsonl(args.data_path):
        data_list.append(data)
    prompt_input, task_ids, entry_inputs = process_data(data_list)

    print(prompt_input)
    print(task_ids)
    print(entry_inputs)
# ['from typing import List\n\n\ndef has_close_e...
# ['HumanEval/0', 'HumanEval/1', 'HumanEval/2',...
# ['has_close_elements', 'separate_paren_groups',...
```

```shell
# 运行以下命令可以获得数据集中的输入(prompt_input)，任务id(task_ids)和执行函数(entry_points)。比如"HumanEval/0"的输入时from typing import List..., 而该代码的执行函数入口名称为has_close_elements.
python preprocess.py --data_path path/to/HumanEval.jsonl.gz
```

step 3. 提出代码生成函数主干函数，由于生成代码会生成多余函数，评测时只需要评测函数即可，函数名为`data/HumanEval.jsonl.gz`中的`entry_point`，组成如下结构保存为`samples.jsonl`：

```bash
{'task_id': "HumanEval/0","completion": "inference result"}
```

step 4. 安装HumanEval

```python
pip install -e human-eval
```

> 注：
>
> 1. 解除`human-eval/human_eval/execution.py`的第58行注释;
> 2. 由于代码生成时会自带prompt，因此将`human-eval/human_eval/execution.py`第39行的`problem["prompt"] + completion` 改为 `completion`即可。

step 5. 生成测试分数

```python
evaluate_functional_correctness samples.jsonl
# {'pass@1': 0.4695}
```

## 推理

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多batch输入
    inputs = ["def bubble_sort(arr):\n",
             "def quick_sort(arr):\n"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = len(inputs)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    # build model from config
    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids,
                             max_length=model_config.max_decode_length,
                             do_sample=model_config.do_sample,
                             top_k=model_config.top_k,
                             top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='codellama_34b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 多batch输出
# <s>def bubble_sort(arr):
#    n = len(arr)
#    for i in range(n):
#        for j in range(0, n-i-1):
#            if arr[j] > arr[j+1]:
#                arr[j], arr[j+1] = arr[j+1], arr[j]
# ...
# <s>def quick_sort(arr):
#    if len(arr) < 2:
#        return arr
#    pivot = arr[0]
#    left = [i for i in arr[1:] if i < pivot]
#    right  = [i for i in arr[1:] if i >= pivot]
#    return quick_sort(left) + [pivot] + quick_sort(right)
# ...
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
export RANK_TABLE_FILE=$1
CHECKPOINT_PATH=$2
YAML_FILE=$3
MODEL_TYPE=$4
# define variable
export RANK_SIZE=$5
export START_RANK=$6 # this server start rank
let END_RANK=START_RANK+RANK_SIZE # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++));
do
    export RANK_ID=$((i-START_RANK))
    export DEVICE_ID=$i
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --checkpoint_path $CHECKPOINT_PATH --yaml_file ${YAML_FILE} --model_type ${MODEL_TYPE} &> mindformers_$RANK_ID.log &
done
```

#### 多卡generate推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/checkpoint.ckpt path/to/config_yaml codellama_34b 8 0
```

> 注：几卡推理就要在yaml配置中将相应的parallel_config 中的model_parallel置为8，其余置为1。

```python
use_parallel: True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 8  # 改为相应卡数。
  pipeline_stage: 1
```

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡推理。

```python
# predict_custom.py 文件
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多输入
    inputs = ["def bubble_sort(arr):\n",
              "def quick_sort(arr):\n"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    model_config.checkpoint_name_or_path = args.checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs,
                                       max_length=model_config.max_decode_length,
                                       do_sample=model_config.do_sample,
                                       top_k=model_config.top_k,
                                       top_p=model_config.top_p)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 单输出
# 'text_generation_text':['
#def bubble_sort(arr):
#    n = len(arr)
#    for i in range(n):
#        for j in range(0, n-i-1):
#            if arr[j] > arr[j+1]:
#                arr[j], arr[j+1] = arr[j+1], arr[j] ...
# ']
# 'text_generation_text':['
#def quick_sort(arr):
#    if len(arr) < 2:
#        return arr
#    pivot = arr[0]
#    left = [i for i in arr[1:] if i < pivot]
#    right  = [i for i in arr[1:] if i >= pivot]
#    return quick_sort(left) + [pivot] + quick_sort(right) ...
# ']
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
export RANK_TABLE_FILE=$1
CHECKPOINT_PATH=$2
YAML_FILE=$3
MODEL_TYPE=$4
# define variable
export RANK_SIZE=$5
export START_RANK=$6 # this server start rank
let END_RANK=START_RANK+RANK_SIZE # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++));
do
    export RANK_ID=$((i-START_RANK))
    export DEVICE_ID=$i
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --checkpoint_path $CHECKPOINT_PATH --yaml_file ${YAML_FILE} --model_type ${MODEL_TYPE} &> mindformers_$RANK_ID.log &
done
```

#### 多卡pipeline推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/checkpoint.ckpt path/to/config_yaml codellama_34b 8 0
```

> 注：config_yaml的配置也要和基于generate的多卡推理一样将model_parallel 修改为相应卡数，而data_parallel 和 pipeline_stage设置为1。

### 基于run_mindformer分布式推理

**注**：要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```python
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 4096
extend_method: "None"
offset: 0
checkpoint_name_or_path: ""
repetition_penalty: 1
max_decode_length: 512
top_k: 3
top_p: 1
do_sample: False
max_new_tokens: 512      #设置最大生成长度
```

#### 多卡推理

可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md)中的分布式推理方法， 支持分布式推理

```bash
# 参考推理案例三，使用完整权重推理8卡
cd script
bash run_distribute.sh rank_table_2.json ../configs/codellama/predict_codellama_34b_910b.yaml [0,8] predict "def quick_sort(arr):\n"

# def quick_sort(arr):
#    if len(arr) < 2:
#        return arr
#    pivot = arr[0]
#    left = [i for i in arr[1:] if i < pivot]
#    right  = [i for i in arr[1:] if i >= pivot]
#    return quick_sort(left) + [pivot] + quick_sort(right) ...
```

## Mindspore-Lite 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造的推理引擎 [MindSpore_lite](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　Lite 推理大致分两步：权重转换导出 MindIR -> Lite 推理，接下来分别描述上述两个过程。

### 单卡导出与推理

单卡lite推理查看[llama2.md](../model_cards/llama2.md)的Lite推理

### 多卡导出与推理

#### MindIR 导出

1. 以codellama_34b为例，修改模型相关的配置文件 configs/llama2/predict_codellama_34b_910b.yaml，其中需要关注这几项：

   1.1 使用`load_checkpoint` 和`src_strategy_path_or_dir`配置导出：

```yaml
load_checkpoint: path/to/checkpoint_dir/transformed_checkpoint
src_strategy_path_or_dir: path/to/checkpoint_dir/strategy
auto_trans_ckpt: False
run_mode: "export"

# default parallel of device num = 8 for Atlas 800T A2
parallel_config:
  data_parallel: 1
  model_parallel: 8  # 改为相应卡数
  pipeline_stage: 1

# model config
model:
  model_config:
    seq_length: 4096  # mindir需要输出的seq_length，可自定义，默认4096
    use_past: True
```

> 注：load_checkpoint 和 src_strategy_path_or_dir 的路径是使用基于run_mindformer分布式推理后生成的权重切分文件夹`transformed_checkpoint` 和`strategy`，位置一般在/output/里面，将其从output里面拷贝出来，否则执行命令会覆盖。如何得到切分权重可以参考[权重切分指导文档](../feature_cards/Transform_Ckpt.md)

​     1.2. 使用完整权重导出，使`load_checkpoint` 和`src_strategy_path_or_dir` 置为`""`；仅修改`model config`里面的`checkpoint_name_or_path`，其他配置与1.1一致:

```yaml
# model config
model:
  model_config:
    seq_length: 4096  # mindir需要输出的seq_length，可自定义，默认4096
    use_past: True
    checkpoint_name_or_path: "/path/to/your/*.ckpt"
```

> 注：1. `checkpoint_name_or_path`的路径写到完整权重.ckpt这个权重文件为止。2.完整权重导入可能将host内存撑爆，因此推荐小尺寸模型使用。codellama 34b推荐4卡导入完整权重，8卡推理则推荐分布式导入权重。

2. 执行`run_distribute.sh`，完成模型导出。

```bash
cd scripts
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/codellama/predict_codellama_34b_910b.yaml [0,8] export
```

#### 执行推理

1. 新建推理配置文件：`lite.ini`，将多卡生成的`RANK_TABLE_FILE`文件写入`lite.ini`中

```bash
[ascend_context]
plugin_custom_ops=All
provider=ge
rank_table_file=RANK_TABLE_FILE
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
```

1. 新建文件`run_lite.sh`, 执行命令`bash run_lite.sh`：

```bash
# >>> `run_lite.sh`文件
# 修改predict_data 的入参来进行不同输入文本的推理。
readonly START_DEVICE_ID=0
for i in {0..7}; do
export RANK_ID=${i}
export DEVICE_ID=$((i + START_DEVICE_ID))
printf "run model %s on rank:%s,device %s...\n" ${i} ${RANK_ID} ${DEVICE_ID}
python run_infer_main.py --do_sample False --device_id ${DEVICE_ID} --rank_id ${RANK_ID} --tokenizer_path path/to/tokenizer.model --model_name codellama_34b --config_path lite.ini --prefill_model_path output/mindir_full_checkpoint/rank_${RANK_ID}_graph.mindir --increment_model_path output/mindir_inc_checkpoint/rank_${RANK_ID}_graph.mindir --is_sample_acceleration False --seq_length 4096 --add_special_tokens True --distributed True --predict_data "def bubble_sort(arr):\n" --generated_time 3 > rank_${RANK_ID}.log 2>&1 &
done
# 结果输出：
#def bubble_sort(arr):
#    n = len(arr)
#    for i in range(n):
#        for j in range(0, n-i-1):
#            if arr[j] > arr[j+1]:
#                arr[j], arr[j+1] = arr[j+1], arr[j] ...
```
