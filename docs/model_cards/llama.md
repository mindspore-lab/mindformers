# LLaMA

## 模型描述

LLaMA是由Meta于2023年发布。LLaMA模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。LLaMA目前按照参数量，目前有四个版本：LLaMA-7B（7B）、LLaMA-13B（13B）、LLaMA-33B（33B）以及LLaMA-65B（65B），目前在本仓库中，支持了7B，13B和65B三个规格的模型，权重文件来源于OpenLLaMA。

[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

``` text
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## 模型性能

- 基于Atlas 800

**llama_7b**:

| config                                                      | task            | Datasets  | metric | phase                 | performance                   |
| ----------------------------------------------------------- | --------------- | --------- | ------ | --------------------- | ----------------------------- |
| [llama_7b](../../configs/llama/run_llama_7b.yaml)           | text_generation | WikiText2 | -      | [pretrain](#预训练)   | 1229 tokens/s/p               |
| [llama_7b](../../configs/llama/run_llama_7b.yaml)           | text_generation | alpaca    | -      | [finetune](#全参微调) | 1229 tokens/s/p               |
| [llama_7b_lora](../../configs/llama/run_llama_7b_lora.yaml) | text_generation | alpaca    | -      | [finetune](#lora微调) | 1843 tokens/s/p               |
| [llama_7b](../../configs/llama/run_llama_7b.yaml)           | text_generation | WikiText2 | PPL    | [eval](#评测)         | 8.28                          |
| [llama_7b](../../configs/llama/run_llama_7b.yaml)           | text_generation | SQuAD 1.1 | Em/F1  | [eval](#评测)         | 26.85/48.51                   |
| [llama_7b](../../configs/llama/run_llama_7b.yaml)           | text_generation | -         | -      | [predict](#推理)      | 22.4 tokens/s (use_past=True) |

llama_13b / llama_65b 待补充

## 仓库介绍

`LLaMA` 基于 `mindformers` 实现，主要涉及的文件有：

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

2. 模型配置：`configs/llama`

    ```bash
    llama
        ├── run_llama_7b.yaml         # 7b模型全量微调启动配置
        ├── run_llama_7b_910b.yaml    # 7b模型全量微调启动配置(Atlas 800T A2)
        ├── run_llama_7b_lora.yaml    # 7b lora低参微调启动配置
        ├── run_llama_13b.yaml        # 13b全量微调启动配置
        ├── run_llama_13b_910b.yaml   # 13b全量微调启动配置(Atlas 800T A2)
        ├── run_llama_65b.yaml        # 65b全量微调启动配置
        └── run_llama_65b_910b.yaml   # 65b全量微调启动配置(Atlas 800T A2)
    ```

3. 数据预处理脚本：

    ```bash
    mindformers/tools/dataset_preprocess/llama/
        ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
        ├── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
        └── squad_data_process.py   # squad数据集格式转换脚本
    ```

## 前期准备

### 环境要求

- 硬件：Atlas 800/Atlas 800T A2
- MindSpore：2.2.0
- MindFormers版本：r1.0

> 注：推理可在单机单卡上完成部署；全量微调至少需要单机8卡，Lora微调至少需要单卡。

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

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1. 使用官方权重进行转换
    从huggingface下载英文预训练权重（权重来源于OpenLLaMA）：

    - [llama-7b](https://huggingface.co/openlm-research/open_llama_7b)

    - [llama-13b](https://huggingface.co/openlm-research/open_llama_13b)

    > 注：65B权重OpenLLaMA未提供，如有需要，请开发者自行解决。

    下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

    ```shell
    python mindformers/models/llama/convert_weight.py \
    --torch_ckpt_path TORCH_CKPT_PATH \
    --mindspore_ckpt_path {path}/MS_CKPT_NAME
    ```

    ```text
    # 参数说明
    torch_ckpt_path: huggingface权重保存目录下的任意权重bin文件,根据该文件路径读取目录下全部权重
    mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径
    ```

2. 获取MindFormers提供的已转换权重
   可通过from_pretrained接口下载，也可直接从下面的链接获取
   - [llama_7b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/open_llama_7b.ckpt)
   - [llama_13b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/open_llama_13b.ckpt)
   - [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

> 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix llama_7b
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/llama`

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('llama_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama_7b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=20, do_sample=True, top_k=3)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that has everything: the old and the new, the modern and the ancient']
```

### 基于Trainer的快速训练，微调，评测，推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='llama_7b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')

# 开启预训练
# 请参照多卡训练，llama不支持单卡启动训练
# trainer.train()

# 开启全量微调
# 请参照多卡微调，llama不支持单卡启动全量微调
# trainer.finetune()

# 开启评测
trainer.evaluate()

# 开启推理
predict_result = trainer.predict(input_data="I love Beijing, because")
# [{'text_generation_text': ['<s>I love Beijing, because it’s a city that has everything: the old and the new, the modern and the ancient']}]
```

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama_7b', max_length=20)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=True, top_k=3)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a city that has everything: the old and the new, the modern and the ancient']}]
```

## 预训练

### 数据集准备-预训练

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

- 分词模型下载：例如下载huggingface的[tokenizer.model](https://huggingface.co/openlm-research/open_llama_7b/blob/main/tokenizer.model)

- 使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/wiki2048.mindrecord
```

### 脚本启动（LLaMA-7B为例）

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

#### 多卡训练

##### 单机多卡

- step 1. 修改模型对应的配置文件。

在模型对应的配置文件`configs/llama/run_llama_{7/13/65}b.yaml`中，用户可自行修改模型、训练相关参数，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。

- step2. 设置环境变量，变量配置如下：

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  # 推荐开启INFNAN模式
```

- step3：进入`scripts`文件夹，启动运行脚本，进行8卡分布式运行。

```shell
cd scripts
bash run_distribute.sh hccl_xxxx.json ../configs/llama/run_llama_7b.yaml [0,8] train
```

```text
# 脚本启动格式：
bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_MODE]

# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的llama/run_llama_7b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

##### 多机多卡

- step 1. 多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

> **注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

- step 2. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以llama-13b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/llama/run_llama_13b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 3. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。需注意，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama/run_llama_7b.yaml [0,8] train 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/llama/run_llama_7b.yaml [8,16] train 16
```

## 微调

### 数据集准备-微调

目前提供alpaca数据集的预处理脚本用于全参微调/lora微调任务。

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

> **注：由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9**

```bash
# 脚本路径：tools/dataset_preprocess/llama/llama_preprocess.py
python llama_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca-data-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/alpaca-fastchat2048.mindrecord
```

### 全参微调

以llama7b为例

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 修改`config/llama/run_llama_7b.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

```yaml
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat2048.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 修改训练时学习率和优化器参数，与预训练不同，微调学习率配置如下：

```yaml
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.999
  eps: 1.e-8
  learning_rate: 1.e-5

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

- step 3. 设置环境变量，变量配置如下：

```bash
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

- step 4. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。

- step 5. 启动微调任务，llama-7b模型以单机八卡为例进行微调，命令如下：

```shell
cd scripts
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/llama/run_llama_7b.yaml [0,8] finetune
```

多机多卡微调任务启动参考[预训练章节](#预训练)，添加预训练权重，修改启动脚本中的`RUN_MODE`为`finetune`即可。

### lora微调

目前llama_7b模型适配了lora微调算法，并给出了默认配置文件`config/llama/run_llama_7b_lora.yaml`。

#### 脚本启动

- step 1. 修改配置文件，参考全参微调修改训练数据集路径与预训练权重路径。

- step 2. 启动lora微调任务。(不建议开启INFNAN模式)。

> 注：llama_7b_lora模型支持单卡启动，需将配置文件中的`use_parallel`参数置为`False`。

```shell
cd scripts
# 单卡启动
bash run_standalone.sh ../configs/llama/run_llama_7b_lora.yaml [DEVICE_ID] finetune
# 多卡启动（以单机八卡为例）
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/llama/run_llama_7b_lora.yaml [0,8] finetune
```

#### API高阶接口启动

lora微调支持使用高阶接口启动单卡微调任务，示例代码如下：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='llama_7b',
                  pet_method='lora',
                  train_dataset="{dataset file path}")
# 调用finetune接口进行微调
trainer.finetune(finetune_checkpoint="{checkpoint file path}")
```

## 评测

Llama当前支持的评测任务如下：

| 任务类型 |  评测指标  |  数据集   |
| :------: | :--------: | :-------: |
| 文本生成 | Perplexity | WikiText2 |
| 阅读理解 |   Em/F1    | SQuAD 1.1 |

### 文本生成

step 1. 获取数据集

[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)是从维基百科上经过验证的优质文章集中提取的超过1亿个token的集合。

step 2. 处理数据成mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.valid.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 2047 \
--output_file /{path}/wiki2048.mindrecord
```

step 3. 开启评测，指标为PPL

```bash
python run_mindformer.py \
--config configs/llama/run_llama_7b.yaml \
--eval_dataset_dir /{path}/wiki2048.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama_7b.ckpt \
--epochs 1 \
--use_parallel False \
--device_id 0

# PerplexityMetric = {'PerplexityMetric': {'loss': 2.1142693907022476, 'PPL': 8.283531529594038}}
```

### 阅读理解

step 1. 获取数据集

[SQuAD 1.1](https://data.deepai.org/squad1.1.zip)包含针对500+文章的10万+问答对,是一个阅读理解数据集，由维基百科文章上提出的问题组成，其中每个问题的答案都是相应文章中的一段文本。

step 2. 处理数据成mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/squad_data_process.py进行数据预处理+Mindrecord数据生成
python squad_data_process.py \
--input_file /{path}/squad/dev-v1.1.json \
--output_file /{path}/squad2048.mindrecord \
--mode eval \
--max_length 2048 \
--tokenizer_type "llama_7b"
```

预处理后数据格式举例：

```text
Read the passage and answer the question below.

### Instruction:
The Panthers finished the regular season with a 15–1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49–15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 12–4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20–18 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.

### Input:
Which Carolina Panthers player was named Most Valuable Player?

### Response:
Cam Newton
```

step 3. 修改配置文件，eval_dataset的input_columns中增加`labels`，修改metric类型为`EmF1Metric`

```yaml
# eval dataset
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: False
  input_columns: ["input_ids", "labels"]      # 增加"labels"

# metric
metric:
  type: EmF1Metric     # metric type设为EmF1Metric
```

此外，要提高推理速度，可以进行如下配置，设置增量推理`use_past`，并限制生成最大长度`max_new_tokens`。

```yaml
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 2048
extend_method: "None"
offset: 0
checkpoint_name_or_path: "llama_7b"
repetition_penalty: 1
max_decode_length: 512
top_k: 3
top_p: 1
do_sample: False
max_new_tokens: 20      #设置最大生成长度
```

step 4. 开启评测，指标为`Em/F1`

```bash
python run_mindformer.py \
--config configs/llama/run_llama_7b.yaml \
--eval_dataset_dir /{path}/squad2048.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama_7b.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0

# F1 score: 48.48954955952303, Em score: 26.850507982583455, total_count: 2067
```

## 推理

> 注：修改模型配置项中的**use_past=True**，以开启增量推理，加速推理性能

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(model_type='llama_7b',
         use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    model_config = AutoConfig.from_pretrained(model_type)
    model_config.use_past = use_past
    # if use parallel, data_parallel * model_parallel = device_num
    model_config.parallel_config.data_parallel = 1
    model_config.parallel_config.model_parallel = 1
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.model_type,
         args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done
```

#### 单卡pipeline推理

```bash
python predict_custom.py
```

#### 多卡pipeline推理

- 修改yaml文件中分布式配置及并行模式，参考[模型权重切分与合并](../feature_cards/Transform_Ckpt.md)进行离线权重切分。**注**：推理暂不支持流水线并行

- 将上述`predict_custom.py`中的分布式配置更改为预期的分布式配置

```python
model_config.parallel_config.data_parallel = 1
model_config.parallel_config.model_parallel = 1
```

- 配置上述sh脚本中的卡数设置，默认是0-8卡

```text
export RANK_SIZE=8  # 总卡数
export START_RANK=0 # 起始卡序号
export END_RANK=8   # 结束卡序号
```

- 运行如下命令进行推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/shard_checkpoint_dir
```

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(model_type='llama_7b',
         use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    model_config = AutoConfig.from_pretrained(model_type)
    # if use parallel, data_parallel * model_parallel = device_num
    model_config.parallel_config.data_parallel = 1
    model_config.parallel_config.model_parallel = 1
    model_config.batch_size = len(inputs)
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = network.generate(inputs_ids, max_length=model_config.max_decode_length)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.model_type,
         args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done
```

#### 单卡generate推理

```bash
python predict_custom.py
```

#### 多卡generate推理

- 修改yaml文件中分布式配置及并行模式，参考[模型权重切分与合并](../feature_cards/Transform_Ckpt.md)进行离线权重切分。**注**：推理暂不支持流水线并行

- 将上述`predict_custom.py`中的分布式配置更改为预期的分布式配置

```text
model_config.parallel_config.data_parallel = 1
model_config.parallel_config.model_parallel = 1
```

- 配置上述sh脚本中的卡数设置，默认是0-8卡

```text
export RANK_SIZE=8  # 总卡数
export START_RANK=0 # 起始卡序号
export END_RANK=8   # 结束卡序号
```

- 运行如下命令进行推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/shard_checkpoint_dir
```

### run_mindformer脚本启动

#### 单卡推理

```bash
python run_mindformer.py --config configs/pangualpha/run_llama_7b.yaml --run_mode predict --predict_data 'I love Beijing, because' --use_parallel False
```

**注**：要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```python
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 2048
extend_method: "None"
offset: 0
checkpoint_name_or_path: "llama_7b"
repetition_penalty: 1
max_decode_length: 512
top_k: 3
top_p: 1
do_sample: False
max_new_tokens: 128      #设置最大生成长度
```

### lora推理

基于lora微调后的权重进行推理，可复用上述推理脚本，将入参`model_type`改为`llama_7b_lora`即可使用lora微调模型进行推理

基于alpaca数据集微调的权重，要获得更好的推理效果，需使用alpaca的prompt模板，在instruction处填入自定义指令，以此作为模型的推理输入，替换推理脚本中的`inputs`；如下为prompt模板：

```python
input_data = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format("Tell me about alpacas.")

# output:
# Alpacas are a species of South American camelid. They are domesticated animals that are raised for their wool, meat, and milk. Alpacas are gentle, docile animals that are very friendly and easy to care for. They are also very intelligent and can be trained to perform certain tasks. Alpacas are very social animals and live in herds of up to 20 individuals. They are also very vocal and can make a variety of sounds, including a loud, high-pitched bark.
```

#### lora推理结果示例

**Instruction**: Tell me about the president of Mexico in 2019.

**Output**: The president of Mexico in 2019 is Andrés Manuel López Obrador. He was elected in 2018 and is the first left-wing president of Mexico in over 70 years. He is known for his populist policies and his focus on improving the lives of the poor and marginalized. He has also been criticized for his lack of transparency and his authoritarian style of leadership.

---

**Instruction**: List all Canadian provinces in alphabetical order.

**Output**: Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Nova Scotia, Ontario, Prince Edward Island, Quebec, Saskatchewan, and Yukon.

---

**Instruction**: Write a Python program that prints the first 10 Fibonacci numbers.

**Output**:

```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

---

**Instruction**: Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.

**Output**:

```python
def fizzbuzz(n):
    if n % 3 == 0 and n % 5 == 0:
        print('FizzBuzz')
    elif n % 3 == 0:
        print('Fizz')
    elif n % 5 == 0:
        print('Buzz')
    else:
        print(n)
```

---

**Instruction**: Translate the sentence 'I have no mouth but I must scream' into Spanish.

**Output**: No tengo boca pero tengo que gritar.

---

## [mindspore-lite](../feature_cards/Inference.md)

如需导出模型，使用mindspore-lite进行离线推理请参考[推理特性使用文档](../feature_cards/Inference.md)
