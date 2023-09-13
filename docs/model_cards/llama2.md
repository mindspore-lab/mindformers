# Llama 2

## 模型描述

Llama 2，是Meta基于LLaMA 1的更新版本，基于新的公开可用数据混合进行训练，同时将预训练语料库的大小增加了40%，最后将模型的上下文长度翻倍（由2048提高到4096），并采用了分组查询注意力机制。Llama 2模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。Llama 2按照参数量，目前有三个版本：Llama 2-7B（7B）、Llama 2-13B（13B）、Llama 2-70B（70B），本仓库已全部支持三版权重，权重文件来源于MetaLLama2。Llama 2 的7B和13B 模型结构与LLaMA 1一致，70B 则加入分组查询注意力（GQA）。

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf%C3%82%C2%A0)

``` text
@article{touvron2023llama,
  title={Llama 2: Open foundation and fine-tuned chat models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

## 代码结构介绍

`Llama 2` 基于 `mindformers` 实现，主要涉及的文件有：

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

2. 模型配置：`configs/llama2`

   ```bash
   llama
       ├── run_llama2_7b.yaml         # 7b模型全量微调启动配置
       ├── run_llama2_13b.yaml        # 13b全量微调启动配置
       └── run_llama2_70b.yaml        # 70b全量微调启动配置
   ```

## 环境要求

- 硬件：Ascend 910A/910B
- MindSpore：2.0 / 2.1
- MindFormers版本：dev

注：推理可在单机单卡上完成部署；全量微调至少需要单机8卡。

## 权重转换与权重合并

### 开源预训练权重转换

从huggingface下载英文预训练权重（权重来源于MetaLLama2）：

- [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b)
- [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b)

注：Llama 2的所有权重都需要向Meta提交[申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)，如有需要，请开发者自行申请。

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径
```

### 分布式训练/微调权重合并

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

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

## 推理

### Llama 2推理

以llama2-7b模型为例，利用预训练权重进行推理。

#### pipeline推理

```python
from mindformers.pipeline import pipeline
pipeline_task = pipeline(task="text_generation", model="llama2_7b", max_length=50)
pipeline_result = pipeline_task("I love Beijing, because", top_k=3)
print(pipeline_result)

# output:
# [{'text_generation_text': ['I love Beijing, because it’s a city that’s constantly changing. It’s a city that’s constantly evolving. It’s a city that’s constantly reinventing itself. And I think that’s what makes it']}]
```

#### 基于API接口的推理

```python
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task="text_generation", model="llama2_7b", train_dataset="{dataset file path}")

# 方式1：从obs下载训练好的权重并进行推理
res = trainer.predict(input_data="I love Beijing, because")

# 方式2：用户自行指定权重路径并进行推理
res = trainer.predict(input_data="I love Beijing, because",
                      predict_checkpoint="{checkpoint file path}")
```

## 预训练

### 数据集准备

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

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

### 脚本启动（Llama 2-7B为例）

- 用户需要首先clone整个仓库，请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)完成启动准备工作。

#### 单机多卡启动

- step 1. 在仓库主目录下，运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`。

```shell
# 以八卡运行为例，生成0~7卡的hccl json文件,不包含8本身.
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

```python
# RANK_TABLE_FILE 参考样例
# 单机8卡
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
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

- step 2. 修改模型对应的配置文件。

在模型对应的配置文件`configs/llama2/run_llama2_{7/13/70}b.yaml`中，用户可自行修改模型、训练相关参数，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。

- step3：进入`scripts`文件夹，启动运行脚本，进行8卡分布式运行。

```shell
cd scripts
bash run_distribute.sh hccl_xxxx.json ../configs/llama2/run_llama2_7b.yaml [0,8] train
```

```text
# 脚本启动格式：
bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_MODE]

# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的llama2/run_llama2_7b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

#### <span id="jump">多机多卡启动</span>

- step 1. 首先参考单机多卡启动方式，在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件。

```shell
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 合并每台机器上生成的`RANK_TABLE_FILE`。

将不同机器上生成的`RANK_TABLE_FILE`文件拷贝到一起，执行`merge_hccl.py`脚本进行合并，包括server_list合并，`server_count`设为机器数，`rank_id`顺序增加。

```shell
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

- step 4. 根据服务器节点数等信息，修改相应的配置。

```shell
# 以llama2-13b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/llama2/run_llama2_13b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。需注意，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama2/run_llama2_13b.yaml [0,8] train 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/llama2/run_llama2_13b.yaml [8,16] train 16
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
--seq_length 2048 \
--output_file /{path}/alpaca-fastchat2048.mindrecord
```

### 全参微调

以llama2 7b为例

- step 1. 修改`config/llama2/run_llama2_7b.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

```python
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat2048.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 修改训练时学习率和优化器参数，与预训练不同，微调学习率配置如下：

```python
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

- step 3. 修改seq_length，seq_length配置如下：

```python
# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 2048
```

​        注意：alpaca数据集最长不超过2048，因此seq_length采用2048即可。

- step 3. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。
- step 4. 启动微调任务，llama2-7b模型以单机八卡为例进行微调，命令如下：

```shell
cd scripts
bash run_distribute.sh [RANK_TABLE_FILE] ../configs/llama2/run_llama2_7b.yaml [0,8] finetune
```

多机多卡微调任务启动参考[预训练章节](#jump)，添加预训练权重，修改启动脚本中的`RUN_MODE`为`finetune`即可。

## 评测

Llama 2当前支持的评测任务如下：

| 任务类型 |  评测指标  |  数据集   |
| :------: | :--------: | :-------: |
| 文本生成 | Perplexity | WikiText2 |
| 阅读理解 |   Em/F1    | SQuAD 1.1 |

- 文本生成：

step 1. 获取数据集

[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)是从维基百科上经过验证的优质文章集中提取的超过1亿个token的集合。

step 2. 处理数据成mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.valid.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 4095 \
--output_file /{path}/wiki4096.mindrecord
```

step 3. 开启评测，指标为PPL

```bash
python run_mindformer.py \
--config configs/llama2/run_llama2_7b.yaml \
--eval_dataset_dir /{path}/wiki4096.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--use_parallel False \
--device_id 0

# PerplexityMetric = {'PerplexityMetric': {'loss': 2.1142693907022476, 'PPL': 8.283531529594038}}
```

- 阅读理解：

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
--tokenizer_type "llama2_7b"
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

step 3. 修改配置文件，eval_dataset的input_columns中增加`labels`，修改metric类型为`EmF1Metric`，修改seq_length为`2048`

```python
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

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 2048
```

此外，要提高推理速度，可以进行如下配置，设置增量推理`use_past`，并限制生成最大长度`max_new_tokens`。

```python
# model config
use_past: True          # 开启增量推理
pretrain_seqlen: 4096
extend_method: "None"
offset: 0
checkpoint_name_or_path: "llama2_7b"
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
--config configs/llama2/run_llama2_7b.yaml \
--eval_dataset_dir /{path}/squad2048.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0

# F1 score: 48.48954955952303, Em score: 26.850507982583455, total_count: 2067
```

