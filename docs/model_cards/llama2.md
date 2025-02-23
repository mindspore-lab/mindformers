# Llama 2

## 模型描述

Llama 2，是Meta基于LLaMA 1的更新版本，基于新的公开可用数据混合进行训练，同时将预训练语料库的大小增加了40%，最后将模型的上下文长度翻倍（由2048提高到4096），并采用了分组查询注意力机制。Llama 2模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。Llama 2按照参数量，目前有三个版本：Llama 2-7B（7B）、Llama 2-13B（13B）、Llama 2-70B（70B），本仓库已全部支持三版权重，权重文件来源于MetaLLama2。Llama 2 的7B和13B 模型结构与LLaMA 1一致，70B 则加入分组查询注意力（GQA）。

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

``` text
@article{touvron2023llama,
  title={Llama 2: Open foundation and fine-tuned chat models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

## 模型性能

基于Atlas 800T A2

llama2_7b:

| config                                                     | task                  | Datasets  | SeqLength | metric | phase           | score     | performance  |
|------------------------------------------------------------|-----------------------|-----------|-----------|--------|-----------------|-----------|--------------|
| [llama2_7b](../../configs/llama2/pretrain_llama2_7b.yaml)  | text_generation       | wiki      | 4096      | -      | [train](#预训练)   | -         | 4820 tks/s/p |
| [llama2_7b](../../configs/llama2/finetune_llama2_7b.yaml)  | text_generation       | alpaca    | 4096      | -      | [finetune](#微调) | -         | 4820 tks/s/p |
| [llama2_7b_lora](../../configs/llama2/lora_llama2_7b.yaml) | text_generation       | alpaca    | 4096      | -      | [finetune](#微调) | -         | 5217 tks/s/p |
| [llama2_7b](../../configs/llama2/predict_llama2_7b.yaml)   | text_generation       | WikiText2 | -         | PPL    | [eval](#评测)     | 6.58      | -            |
| [llama2_7b](../../configs/llama2/predict_llama2_7b.yaml)   | reading comprehension | SQuAD 1.1 | -         | EM/F1  | [eval](#评测)     | 39.6/60.5 | -            |

llama2_13b:

| config                                                       | task                  | Datasets  | SeqLength | metric | phase           | score       | performance   |
|--------------------------------------------------------------|-----------------------|-----------|-----------|--------|-----------------|-------------|---------------|
| [llama2_13b](../../configs/llama2/pretrain_llama2_13b.yaml)  | text_generation       | wiki      | 4096      | -      | [train](#预训练)   | -           | 1883  tks/s/p |
| [llama2_13b](../../configs/llama2/finetune_llama2_13b.yaml)  | text_generation       | alpaca    | 4096      | -      | [finetune](#微调) | -           | 1883 tks/s/p  |
| [llama2_13b_lora](../../configs/llama2/lora_llama2_13b.yaml) | text_generation       | alpaca    | 4096      | -      | [finetune](#微调) | -           | 2322 tks/s/p  |
| [llama2_13b](../../configs/llama2/predict_llama2_13b.yaml)   | text_generation       | WikiText2 | -         | PPL    | [eval](#评测)     | 6.14        | -             |
| [llama2_13b](../../configs/llama2/predict_llama2_13b.yaml)   | reading comprehension | SQuAD 1.1 | -         | EM/F1  | [eval](#评测)     | 27.91/44.23 | -             |

llama2_70b：

| config                                                      | task                  | Datasets  | SeqLength | metric | phase           | score       | performance  |
|-------------------------------------------------------------|-----------------------|-----------|-----------|--------|-----------------|-------------|--------------|
| [llama2_70b](../../configs/llama2/pretrain_llama2_70b.yaml) | text_generation       | wiki      | 4096      | -      | [train](#预训练)   | -           | 407  tks/s/p |
| [llama2_70b](../../configs/llama2/finetune_llama2_70b.yaml) | text_generation       | alpaca    | 4096      | -      | [finetune](#微调) | -           | 414 tks/s/p  |
| [llama2_70b](../../configs/llama2/predict_llama2_70b.yaml)  | text_generation       | WikiText2 | -         | PPL    | [eval](#评测)     | 4.92        | -            |
| [llama2_70b](../../configs/llama2/predict_llama2_70b.yaml)  | reading comprehension | SQuAD 1.1 | -         | EM/F1  | [eval](#评测)     | 41.94/63.86 | -            |

基于Atlas 900 A2 PoDc

| config                                                      | task            | Datasets | SeqLength | metric | phase         | score | performance   |
|-------------------------------------------------------------|-----------------|----------|-----------|--------|---------------|-------|---------------|
| [llama2_7b](../../configs/llama2/pretrain_llama2_7b.yaml)   | text_generation | wiki     | 4096      | -      | [train](#预训练) | -     | 4100 tks/s/p  |
| [llama2_13b](../../configs/llama2/pretrain_llama2_13b.yaml) | text_generation | wiki     | 4096      | -      | [train](#预训练) | -     | 1658  tks/s/p |
| [llama2_70b](../../configs/llama2/pretrain_llama2_70b.yaml) | text_generation | wiki     | 4096      | -      | [train](#预训练) | -     | 406 tks/s/p   |

## 仓库介绍

`Llama 2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
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
       ├── predict_llama2_7b.yaml         # 7b模型推理启动配置
       ├── predict_llama2_13b.yaml        # 13b模型推理启动配置
       ├── predict_llama2_70b.yaml        # 70b模型推理启动配置
       ├── pretrain_llama2_7b.yaml         # 7b模型预训练启动配置
       ├── pretrain_llama2_13b.yaml        # 13b模型预训练启动配置
       ├── pretrain_llama2_70b.yaml        # 70b模型预训练启动配置
       ├── finetune_llama2_7b.yaml         # 7b模型全量微调启动配置
       ├── finetune_llama2_13b.yaml        # 13b模型全量微调启动配置
       └── finetune_llama2_70b.yaml        # 70b模型全量微调启动配置
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

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

注：Atlas 800T A2芯片：7b,13b推理可在单机单卡上完成部署；70b推理至少使用8卡，全参微调至少需要4机32卡，推荐使用8机64卡。

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1.从huggingface下载英文预训练权重（权重来源于MetaLLama2）：

- [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf)

注：Llama 2的所有权重都需要向Meta提交[申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)，如有需要，请开发者自行申请。

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME
# 参数说明
input_path: huggingface权重保存目录路径
output_path: 权重保存文件名，可以指定自定义保存路径
```

2. 获取MindFormers提供的已转换权重
    可通过from_pretrained接口下载，也可直接从下面的链接获取

- [llama2_7b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)
- [llama2_13b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt)
- [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

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

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/llama2`

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama2_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama2_7b')
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that is constantly changing. I have been living here for 10 years and I have seen the city change so much.I']
```

### 基于Trainer的快速推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='llama2_7b',
                  train_dataset='path/to/train_dataset')

# 开启推理
predict_result = trainer.predict(input_data="I love Beijing, because")
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama2_7b', max_length=30)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
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

数据处理时候注意bos，eos，pad等特殊ids要和yaml配置中model_config里保持一致，默认bos_token_id=1, eos_token_id=2, pad_token_id=0, 如果有所修改，yaml中对应的配置也需要修改；
一般预训练的数据中不包含pad_token，此时建议pad_token_id设为-1。

### 脚本启动（Llama 2-7B为例）

#### 多卡训练

##### 单机多卡

- step 1. 修改模型对应的配置文件。

在模型对应的配置文件`configs/llama2/pretrain_llama2_7b.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练)
通过配置中的`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

如果是llama2 70b，可以将`qkv_concat`修改为True，`micro_batch_num`修改为256提升性能。如果报显存不足，将环境变量HCCL_BUFFSIZE下调到100。
还可以在train和finetune的yaml里开启并行加速：

```bash
context:
  ascend_config:
    parallel_speed_up_json_path: "/path/to/your/parallel_speed_up.json"
```

parallel_speed_up.json文件示例：

```bash
{
  "recompute_comm_overlap": false,
  "matmul_grad_comm_overlap": true,
  "enable_task_opt": false,
  "enable_grad_comm_opt": false,
  "enable_opt_shard_comm_opt": false,
  "enable_concat_eliminate_opt": false,
  "enable_begin_end_inline_opt": false,
  "compute_communicate_fusion_level":0
}
```

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。auto_parallel说明详见[自动并行](../docs/feature_cards/Auto_Parallel.md)。parallel_speed_up中各参数含义详见[parallel_speed_up说明](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.set_context.html#mindspore.set_context)。

- step2：启动msrun快速启动运行脚本，进行8卡分布式运行。各个参数位置含义参见[msrun快速启动](../../README.md#方式一使用已有脚本启动)。

```shell
# 单机多卡快速启动方式
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/pretrain_llama2_7b.yaml \
 --run_mode train" 8
```

##### 多机多卡

- step 1. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以llama2-13b模型两机训练为例，配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/llama2/pretrain_llama2_13b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 2. 执行运行脚本。

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，各个参数位置含义参见[msrun快速启动](../../README.md#方式一使用已有脚本启动)。

```shell
# 节点0，设0节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config {CONFIG_PATH} \
 --run_mode {train}" \
 16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点1，设1节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config {CONFIG_PATH} \
 --run_mode {train}" \
 16 8 192.168.1.1 8118 1 output/msrun_log False 300
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

以llama2 7b为例

当前模型已支持使用**Flash Attention算法**进行全参微调，推荐开启flash_attention，可加速训练。详请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 参考`config/llama2/finetune_llama2_7b.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

```python
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 2. 修改微调时学习率, 优化器参数，`seq_length`, 新增 `context`中参数, 与预训练不同，微调配置如下：

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
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1 # add for increase predict
    seq_length: 4096

# context
context:
  runtime_num_threads: 1
```

- step 3. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。
- step 64 启动微调任务，llama2-7b模型以单机八卡为例进行微调，命令如下：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/llama2/finetune_llama2_7b.yaml \
--run_mode finetune" 8
```

多机多卡微调任务启动参考[预训练章节](#预训练)，添加预训练权重，修改启动脚本中的`RUN_MODE`为`finetune`即可。

### LoRA微调

使用LoRA低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，使大模型在少量资源的情况下也能训练。

使用LoRA算法进行低参微调时，使用 `configs/llama2/lora_llama2_7b.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/llama2/lora_llama2_7b.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。

- 加载预训练模型权重：修改 `mindformers/configs/llama2/lora_llama2_7b.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。以llama2-7b 为例，有以下两种导入方式。

  1. 直接导入完整权重：

  ```yaml
  # 以llama2-7b为例
  load_checkpoint: {path}/llama2_7b.ckpt
  auto_trans_ckpt: False
  ```

  2. 使用分布式导入权重，路径设置为rank_0的上一层路径

  ```yaml
  # 将llama2_7b.ckpt 放入文件夹名称为rank_0的文件夹中，
  load_checkpoint: path/to/your/rank_0/
  anto_trans_ckpt: True
  ```

#### 脚本启动

- step 1. 修改配置文件，参考[全参微调](#全参微调)修改训练数据集路径与预训练权重路径。

- step 2. 启动lora微调任务。

```shell
# 多卡启动（以单机八卡为例）
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --run_mode finetune" 8
```

## 评测

以Llama2_7b为例

Llama 2当前支持使用based model(初始权重) 进行评测任务如下：

| 任务类型 |  评测指标  |  数据集   |
| :------: | :--------: | :-------: |
| 文本生成 | Perplexity | WikiText2 |
| 阅读理解 |   Em/F1    | SQuAD 1.1 |

评测时加入`vocab_file`配置相应`tokenizer.model`路径；若使用Atlas 800T A2进行评测，则还需在yaml中加入`ascend_config`配置：

```python
# context_config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"

# tokenizer 配置
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"
```

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

step3. 修改`configs/llama2/pretrain_llama2_7b.yaml` 中的配置参数如下，将`use_past=False`, `metric`的type改为`PerplexityMetric`。

```yaml
metric:
  type: PerplexityMetric
```

step 4. 开启评测，指标为PPL

```bash
python run_mindformer.py \
--config configs/llama2/pretrain_llama2_7b.yaml \
--eval_dataset_dir /{path}/wiki4096.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--use_parallel False \
--device_id 0

# PerplexityMetric = {'PerplexityMetric': {'loss': 2.1142693907022476, 'PPL': 6.58}}
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

step 3. 修改配置文件，eval_dataset的`input_columns`中增加`labels`，修改`metric`类型为`EmF1Metric`，修改`seq_length`为`2048`,修改`max_decode_length`为`700`, `max_new_tokens` 设为20。

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
    max_decode_length: 700
    max_new_tokens: 20      #设置最大生成长度
```

step 4. 开启评测，指标为`Em/F1`

```bash
python run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--eval_dataset_dir /{path}/squad2048.mindrecord \
--run_mode eval \
--load_checkpoint /{path}/llama2_7b.ckpt \
--epochs 1 \
--batch_size 1 \
--use_parallel False \
--device_id 0

# F1 score: 60.5, Em score: 39.6, total_count: 2067
```

### 分布式评测

对于较大模型比如llama2_70b，模型无法完全导入到单卡中进行评测，则需要进行分布式评测。可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md) 中的案例进行评测相应修改，本实例参考案例三完整权重切分自动评测。

step 1. 修改权重文件夹目录结构如下，将模型权重放入rank_0的文件夹中。

```shell
path/to/checkpoint_dir
    ├──rank_0
        ├──model.ckpt
```

step 2. 修改config配置，`auto_trans_ckpt` 设为`True`，`model_parallel`设置为相应需要进行评测的卡数，其余的两个并行策略全部设置为1。`load_checkpoint` 路径设置为rank_0上一层的`path/to/checkpoint_dir`。

```python
load_checkpoint: path/to/checkpoint_dir
use_parallel: True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 8  # 改为相应卡数。70b推荐8卡推理
  pipeline_stage: 1
  use_seq_parallel: False
```

step 3. 按照之前的单卡评测的指导，将`eval_dataset` 中的配置相应修改，将评测数据集路径写入`dataset_dir`中。

```python
# eval dataset，以squad的mindrecord路径为例
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/squad2048.mindrecord"
```

step 4. 执行以下命令进行分布式评测

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/predict_llama2_70b.yaml \
 --run_mode eval \
 --use_parallel True" 8
```

## 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造了全新的训推一体高性能推理引擎，保证训练与推理使用同一套脚本，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　MindSpore 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

Atlas 800T A2推理，则加入`ascend_config`配置如下。

```python
# context_config Atlas 800T A2推理添加ascend_config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
```

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

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多batch输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

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
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path # 如果本地已有ckpt，可加绝对路径：/path/to/model.ckpt
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_type) # 如果本地已有tokenizer.model，可加绝对路径：/path/to/tokenizer_directory/
    # build model from config
    model = LlamaForCausalLM(model_config)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        input_ids = ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32)
        if model_config.use_past:
            infer_data = model.prepare_inputs_for_predict_layout(input_ids)
            warm_up_model.infer_predict_layout(*infer_data)
        else:
            warm_up_model.infer_predict_layout(input_ids)
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

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

# 多batch输出
# <s>I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and mulyimodal pretrained language model....
# <s>Huawei is a company that has been around for a long time. ...
```

#### 单卡generate推理

1. 修改yaml文件

```python
use_parallel: False
```

2. 执行以下命令

```bash
# 以llama2-7b 单卡推理为例,checkpoint_path为权重文件，后缀为.ckpt
python predict_custom.py --yaml_file path/to/predict_llama2_7b.yaml --checkpoint_path path/to/checkpoint.ckpt --model_type llama2_7b
```

#### 多卡generate推理

设置yaml中的`use_parallel` 为`True`后执行命令。

```bash
# 以llama2-7b 2卡推理为例,此时的checkpoint必须是已经切分好的ckpt,shard_checkpoint_dir文件夹下为rank_{}的文件夹。
bash scripts/msrun_launcher.sh "predict_custom.py \
 --yaml_file path/to/predict_llama2_7b.yaml \
 --checkpoint_path path/to/shard_checkpoint_dir \
 --model_type llama2_7b" 2
```

>注：
>1.多卡推理在yaml中将`use_parallel`设为`True`才可以.
>2.几卡推理就要在yaml配置中将相应的parallel_config 中的model_parallel置为几，其余置为1，比如下面的配置表示2卡推理。
>3.切分权重可以参见[权重切分与合并](../feature_cards/Transform_Ckpt.md)，使用自动转换权重得到的分布式权重在`output/transformed_checkpoint`文件夹中。

```python
use_parallel: True  # 多卡推理必须设置为True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 2  # 改为相应卡数。
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
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

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

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
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path # 如果本地已有ckpt，可加绝对路径：/path/to/model.ckpt
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_type) # 如果本地已有tokenizer.model，可加绝对路径：/path/to/tokenizer_directory/

    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        input_ids = ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32)
        if model_config.use_past:
            infer_data = model.prepare_inputs_for_predict_layout(input_ids)
            warm_up_model.infer_predict_layout(*infer_data)
        else:
            warm_up_model.infer_predict_layout(input_ids)
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

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
# 'text_generation_text':['I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# 'text_generation_text':['LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model....
# 'text_generation_text':['Huawei is a company that has been around for a long time. ...
```

#### 单卡pipeline推理

与[基于generate推理](#基于generate的推理)的推理命令一致。

1. 修改yaml文件

```python
use_parallel: False
```

2. 执行以下命令

```bash
# 以llama2-7b 单卡推理为例,checkpoint_path为权重文件，后缀为.ckpt
python predict_custom.py --yaml_file path/to/predict_llama2_7b.yaml --checkpoint_path path/to/checkpoint.ckpt --model_type llama2_7b
```

#### 多卡pipeline推理

设置yaml中的`use_parallel` 为`True`后执行命令。

```bash
# 以llama2-7b 2卡推理为例,此时的checkpoint必须是已经切分好的ckpt
bash scripts/msrun_launcher.sh "predict_custom.py \
 --yaml_file path/to/predict_llama2_7b.yaml \
 --checkpoint_path path/to/shard_checkpoint_dir \
 --model_type llama2_7b" 2
```

> 注：config_yaml的配置也要和[基于generate推理](#基于generate的推理)的多卡推理一样。
> 1.多卡推理在yaml中将`use_parallel`设为`True`才可以.
> 2.几卡推理就要在yaml配置中将相应的parallel_config 中的model_parallel置为几，其余置为1，比如下面的配置表示2卡推理。
> 3.切分权重可以参见[权重切分与合并](../feature_cards/Transform_Ckpt.md)，使用自动转换权重得到的分布式权重在`output/transformed_checkpoint`文件夹中。

### 基于run_mindformer推理

打开predict_llama2_{7/13/70}b.yaml，在`tokenizer`配置下添加`vocab_file`及其`tokenizer.model`的路径，`tokenizer.model`由上面[模型权重下载与转换](#模型权重下载与转换)介绍里面下载。l

```yaml
# tokenizer 配置
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"
```

#### 单卡推理

```bash
python run_mindformer.py --config configs/llama2/predict_llama2_7b.yaml --run_mode predict --predict_data 'I love Beijing, because' --use_parallel False
```

输出：

```bash
I love Beijing, because it is a city that is constantly changing. I have been living here for 10 years and I...
```

#### 多卡推理

可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md)中的分布式推理方法(可参考推理案例三)， 支持分布式推理
>注：几卡推理就要在yaml配置中将相应的parallel_config 中的model_parallel置为几卡，其余置为1。

```python
# 将predict_llama2_7b.yaml里面的参数设置为如下
use_parallel: True
# model config
parallel_config:
  data_parallel: 1
  model_parallel: 2  # 改为相应卡数。
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

# tokenizer 配置
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"  # 在tokenizers下面新增一个vocab_file参数，设置分词器位置
```

```bash
# 以llama2-7b 2卡推理为例,参考推理案例三，使用完整权重推理2卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/llama2/predict_llama2_7b.yaml \
--run_mode predict \
--use_parallel True \
--predict_data \"I love Beijing, because\"" 2
```
