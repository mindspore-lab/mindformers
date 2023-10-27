# WizardCoder-15B

## 模型描述

WizardCoder是由WizardLM团队推出了一个新的指令微调代码大模型，打破了闭源模型的垄断地位，超越了闭源大模型Anthropic Claude和谷歌的Bard。WizardCoder大幅度地提升了开源模型的SOTA水平，创造了惊人的进步，提高了22.3%的性能，成为了开源领域的新时代引领者。
WizardCoder完全开源可商用，基于 Transformer 结构，上下文窗口长度为 2048，参数量为150亿。 本仓库提供了WizardCoder-15B预训练模型。

## 仓库介绍

`WizardCoder` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── wizardcoder_tokenizer.py       # tokenizer
        ├── wizardcoder.py                 # 15B模型实现
        └── wizardcoder_modules.py         # self-attention模块实现
    ```

2. 模型配置：`research/wizardcoder`

    ```bash
    wizardcoder
        └── run_wizardcoder.yaml           # 15B全量微调910b启动配置
    ```

3. 数据处理脚本和任务启动脚本：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── wizardcoder_preprocess.py      # wizardcoder数据集预处理脚本
        └── run_wizardcoder.py             # wizardcoder高阶接口使用脚本
    ```

4. 开源数据集评测脚本：`research/wizardcoder`

    ```bash
    wizardcoder
        ├── humaneval_generate.py          # 针对humaneval数据集生成推理结果
        └── humaneval_process.py           # 将推理结果执行测试用例，生成测试指标
    ```

## 前期准备

### 请先参考[README](../../README.md)安装mindformers

### 环境要求

- 硬件: Ascend 910B
- MindSpore: 2.2.0
- MindSpore Lite: 2.2.0
- MindFormers: dev
- Mindpet: 1.0.2

注：wizardcoder-15B推理可在单机单卡上完成部署，全量微调至少需要8卡。

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

#### mindformers权重直接使用

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调。

- [wizardcoder-15B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/wizardcoder_15B.ckpt)

#### huggingface权重转换后使用

从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huffingface权重的链接如下：

- [wizardcoder-15B-Base](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)

**注**: 请安装torch=2.0.0和transformers=4.30.2版本

```bash
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/wizardcoder/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```bash
python ./research/wizardcoder/convert_weight.py --torch_path TORCH_CKPT_DIR --mindspore_path MS_CKPT_NAME
```

```text
# 参数说明
torch_path: huggingface权重保存目录路径
mindspore_path: mindspore格式的权重保存文件名，如'saved_dir/wizardcoder.ckpt'
```

### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## WizardCoder-15B

### 预训练

#### 数据集准备-预训练数据集

当前提供ADGEN广告数据集的预处理和预训练样例，用于对wizardcoder-15B模型进行预训练。数据集下载链接如下：

- [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing)
- [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)

```text
AdvertiseGen
  ├── train.json
  └── dev.json
```

每条数据样例如下：

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

执行`wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：research/wizardcoder/wizardcoder_preprocess.py
python wizardcoder_preprocess.py \
--input_glob /{path}/train.json \
--vocab_file /{path}/vocab.json \
--merge_file /{path}/merges.txt \
--output_file /{path}/adgen.mindrecord \
--seq_length 2048
```

**注：** 根据ADGEN数据集的格式，需要修改`wizardcoder_preprocess.py`的`data_tokenize_function()`函数如下所示：

```text
def data_tokenize_function(raw_datas, tokenizer, max_length):
    """Preprocess the data by formatting and preprocessing."""
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources, targets = [], []
    for example in raw_datas:
        if 'input' in example:
            instruction, input_query = example['content'], example['input']
            source = prompt_input.format_map(dict(instruction=instruction, input=input_query)) if input_query != "" \
                    else prompt_no_input.format_map(dict(instruction=instruction))

        else:
            instruction = example['content']
            source = prompt_no_input.format_map(dict(instruction=instruction))
        target = f"{example['summary']}{tokenizer.eos_token}"
        sources.append(source)
        targets.append(target)

    data_dict = preprocess(sources, targets, tokenizer, max_length)
    return data_dict
```

#### 预训练启动

预训练需要多卡启动，以`ADGEN`数据集为例,给出了默认配置文件`run_wizardcoder.yaml`。

- step 1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── wizardcoder.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入权重路径文件夹即可。

- step 2. 修改`run_wizardcoder.yaml`中相关配置

```text
output_dir: './output'
load_checkpoint: '{path}/'          # 添加预训练权重路径
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'train'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/adgen.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
```

- step 3. 启动微调任务，以默认配置单机8卡为例，按照以下步骤启动：

-[x] 1: 首先运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

-[x] 2: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以wizardcoder模型两机训练为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/wizardcoder/run_wizardcoder.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 3: 执行运行脚本。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python wizardcoder/run_wizardcoder.py \
--config wizardcoder/run_wizardcoder.yaml \
--load_checkpoint path/to/wizardcoder_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode train \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，训练时设置为train
train_data: 训练数据集路径
```

### 微调

#### 数据集准备-SFT微调数据集

当前提供codealpaca数据集的预处理和微调样例，用于对wizardcoder-15B模型进行微调。数据集下载链接如下：

- [codealpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)

每条数据样例如下：

```json
{
    "PROMPT": "Create an array of 100 elements filled with random numbers from 1 to 100.",
    "ANSWER": "import random\n\n# Create an array of 100 elements with 0 values\nrandom_num_arr = [0] * 100\n\n# Fill each of the 100 elements with random numbers from 1 to 100\nfor i in range(100):\n    random_num_arr[i] = random.randint(1, 100)\n\nprint(random_num_arr)"
}
```

执行`wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：research/wizardcoder/wizardcoder_preprocess.py
python wizardcoder_preprocess.py \
--input_glob /{path}/code_alpaca_20k.json \
--vocab_file /{path}/vocab.json \
--merge_file /{path}/merges.txt \
--output_file /{path}/code_alpaca.mindrecord \
--seq_length 2048
```

**注：** 根据codealpaca数据集的格式，需要修改`wizardcoder_preprocess.py`的`data_tokenize_function()`函数如下所示：

```text
def data_tokenize_function(raw_datas, tokenizer, max_length):
    """Preprocess the data by formatting and preprocessing."""
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources, targets = [], []
    for example in raw_datas:
        if 'input' in example:
            instruction, input_query = example['PROMPT'], example['input']
            source = prompt_input.format_map(dict(instruction=instruction, input=input_query)) if input_query != "" \
                    else prompt_no_input.format_map(dict(instruction=instruction))

        else:
            instruction = example['PROMPT']
            source = prompt_no_input.format_map(dict(instruction=instruction))
        target = f"{example['ANSWER']}{tokenizer.eos_token}"
        sources.append(source)
        targets.append(target)

    data_dict = preprocess(sources, targets, tokenizer, max_length)
    return data_dict
```

#### 全参微调

全参微调需要多卡启动，以`CodeAlpaca-20k`数据集为例,给出了默认配置文件`run_wizardcoder.yaml`。

- step 1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── wizardcoder.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入权重路径文件夹即可。

- step 2. 修改`run_wizardcoder.yaml`中相关配置

```text
output_dir: './output'
load_checkpoint: '{path}/'          # 添加预训练权重路径
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/code_alpaca.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如code_alpaca数据集），input_columns: ["input_ids", "labels"]
```

- step 3. 启动微调任务，以默认配置单机8卡为例，按照以下步骤启动：

-[x] 1: 首先运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

-[x] 2: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以wizardcoder模型两机训练为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/wizardcoder/run_wizardcoder.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 3: 执行运行脚本。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python wizardcoder/run_wizardcoder.py \
--config wizardcoder/run_wizardcoder.yaml \
--load_checkpoint path/to/wizardcoder_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

### 快速推理

#### 基于高阶接口的推理

- step 1. 配置文件设置，添加tokenizer路径`vocab_file`和`merge_file`，并设置`batch_size`值为`1`

在使用Trainer接口进行推理时，若用户自行下载wizardcoder权重，请在启动前先在配置文件中将vocab.json和merges.txt的路径自行配置，配置项为vocab_file和merge_file。

```yaml
# research/wizardcoder/run_wizardcoder.yaml
# runner config
runner_config:
  epochs: 1
  batch_size: 1                 # batch_size设为1
  sink_mode: True
  sink_size: 2
...
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<|endoftext|>'
   bos_token: '<|endoftext|>'
   eos_token: '<|endoftext|>'
   pad_token: '[PAD]'
   vocab_file: 'vocab.json'        # 添加tokenizer路径
   merge_file: 'merges.txt'
   type: WizardCoderTokenizer
```

相关文件的下载链接如下：[vocab.json](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/vocab.json) 和 [merges.txt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/merges.txt)

- step 2. Trainer接口启动推理

wizardcoder的高阶接口使用脚本已集成在run_wizardcoder.py脚本中，运行此脚本命令示例：

```shell
python run_wizardcoder.py \
--config "run_wizardcoder.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '使用python编写快速排序代码' \
--device_id 0

# output: 快速排序（QuickSort）是一种非常高效的排序算法，它是选择排序算法的一个非常有效的改进版本。它的基本思想是通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的元素值比另一部分的元素值小，然后再按此方法对子部分继续进行排序，直到整个序列有序。\n\nPython中的快速排序算法可以实现如下：\n\n```\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] if x <= pivot]\n        middle = pivot\n        right = [x for x in arr[1:] if x > pivot]\n        return quicksort(left) + [middle] + quicksort(right)\n```\n\n这个函数接收一个列表作为输入，并返回一个排序后的列表。\n\n该函数首先检查输入列表的长度，如果长度为0或1，直接返回列表。否则，选取第一项作为分区点（pivot），然后将列表中所有小于等于这个分区点的元素放入左子列表，大于分区点的元素放入右子列表。最后，递归地调用左子列表和右子列表的排序函数。\n\n这样，当递归到最底层的时候，每个子列表中只包含一个元素，这时候就不用再递归了。最后，将两个子列表连接起来，并加上分区点，得到一个排序后的列表。
```

#### Pipeline推理(单卡)

在使用Pipeline接口进行推理时，用户自行下载Wizardcoder-15B权重和tokenizer文件，在启动前自行配置路径
WizardCoderConfig的入参use_past=False为自回归推理，use_past=True为增量推理

```text
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
from mindspore import context
from mindformers.pipeline import pipeline

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel
from wizardcoder_tokenizer import WizardCoderTokenizer

context.set_context(device_id=0, mode=0)

# init model

wizardcoder_model_path = "/path/Wizardcoder-15B/wizardcoder_15b.ckpt" # Wizardcoder-15B ckpt path
wizardcoder_config = WizardCoderConfig(
    batch_size=1,
    seq_length=2048,
    n_position=8192,
    vocab_size=49153,
    embedding_size=6144,
    num_layers=40,
    num_heads=48,
    eos_token_id=0,
    pad_token_id=49152,
    checkpoint_name_or_path=wizardcoder_model_path,
    use_past=True # False为自回归推理，True为增量推理
)
wizardcoder_model = WizardCoderLMHeadModel(config=wizardcoder_config)
wizardcoder_model.add_flags_recursive(is_first_iteration=True)

# init tokenizer

tokenizer_path = "/path/Wizardcoder-15B/tokenizer_path/" # Wizardcoder-15B tokenizer path
tokenizer = WizardCoderTokenizer(
    vocab_file=tokenizer_path + "vocab.json",
    merge_file=tokenizer_path + "merges.txt"
)
pipeline_task = pipeline(task="text_generation", model=wizardcoder_model, tokenizer=tokenizer)
input_data = "使用python编写快速排序代码"
pipeline_result = pipeline_task([input_data],
                                do_sample=False,
                                max_length=2048)
print(pipeline_result)


# output: [{'text_generation_text': ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']}]

```

#### Pipeline推理(单机多卡)

以单机4卡分布式推理为例，设置dp=1, mp=4

- step 1. yaml配置

修改run_wizardcoder.yaml中的配置项

```yaml
use_parallel: True
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
model:
  model_config:
    use_seq_parallel: False
    use_past: True # False为自回归推理，True为增量推理
    checkpoint_name_or_path: ""
processor:
  tokenizer:
    vocab_file: '/path/Wizardcoder-15B/tokenizer/vocab.json'
    merge_file: '/path/Wizardcoder-15B/tokenizer/merges.txt'
    type: WizardCoderTokenizer
  type: WizardCoderProcessor
...
```

- step 2. 切分权重

```text
    └── distribute_model_ckpt_path
        └── rank_0
            └── checkpoint_0.ckpt
        └── rank_1
            └── checkpoint_1.ckpt
        └── rank_2
            └── checkpoint_2.ckpt
        └── rank_3
            └── checkpoint_3.ckpt
```

- step 3. 配置单机4卡环境

运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件hccl_4p_0123_127.0.0.1.json

```text
python mindformers/tools/hccl_tools.py --device_num "[0,4)"
```

- step 4. 推理脚本

``` python
# test_wizardcoder_pipeline_dist.py
import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig, TransformerOpParallelConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool
from mindformers import Trainer, MindFormerConfig, MindFormerRegister, MindFormerModuleType

from wizardcoder_config import WizardCoderConfig
from wizardcoder_tokenizer import WizardCoderTokenizer
from wizardcoder import WizardCoderLMHeadModel


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


def main(model_type='wizardcoder',
         config_path="run_wizardcoder.yaml",
         use_parallel=False,
         device_id=0,
         checkpoint_path=""):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)
    inputs = ["使用python编写快速排序代码"] * 2
    config = MindFormerConfig(os.path.realpath(config_path))
    # set model config
    model_config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = WizardCoderTokenizer(config.processor.tokenizer.vocab_file,
                                     config.processor.tokenizer.merge_file)
    # build model from config
    network = WizardCoderLMHeadModel(model_config)
    network.add_flags_recursive(is_first_iteration=True)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")), "checkpoint_{}.ckpt".format(os.getenv("RANK_ID", "0")))
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs, do_sample=False, max_length=2048)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='wizardcoder', type=str,
                        help='which model to use.')
    parser.add_argument('--config_path', default='run_wizardcoder.yaml', type=str,
                        help='config path')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    args = parser.parse_args()

    main(args.model_type,
         args.config_path,
         args.use_parallel,
         args.device_id,
         args.checkpoint_path)
```

bash启动脚本，每张卡运行test_wizardcoder_pipeline_dist.py代码，加载不同的权重

```bash
# pipeline_dist.sh

CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=4
export START_RANK=0 # this server start rank
export END_RANK=4 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./test_wizardcoder_pipeline_dist.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done

```

执行命令bash pipeline_dist.sh hccl_4p_0123_127.0.0.1.json distribute_model_ckpt_path/

推理结果

```text
{'text_generation_text': ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']}
```

### mindspore lite推理

- step 1. yaml配置

修改run_wizardcoder.yaml中的配置项

```yaml
use_parallel: False
model:
  model_config:
    use_seq_parallel: False
    use_past: True # False为自回归推理，True为增量推理
    checkpoint_name_or_path: "/path/Wizardcoder-15B/wizardcoder_15b.ckpt"
...
```

- step 2. 生成mindir模型文件

基于配置文件run_wizardcoder.yaml生成自回归推理的mindir文件`wizardcoder-15b_mslite_autoregressive/prefill_2k_bs1_graph.mindir`

```text
# export_wizardcoder_autoregressive.py
import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
BS = 1
SEQ_LENGTH = 2048
config_path = 'run_wizardcoder.yaml'
config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
config.use_past = False # 自回归推理
model = WizardCoderLMHeadModel(config)
model.set_train(False)

model.add_flags_recursive(is_first_iteration=True)
input_ids = ms.Tensor(np.ones((BS, SEQ_LENGTH)), mstype.int32)
ms.export(model,
    input_ids,
    file_name=f"wizardcoder-15b_mslite_autoregressive/prefill_2k_bs{BS}",
    file_format="MINDIR")
```

基于配置文件run_wizardcoder.yaml生成增量推理的mindir文件`wizardcoder-15b_mslite_inc/prefill_2k_bs1_graph.mindir`和`wizardcoder-15b_mslite_inc/decode_2k_bs1_graph.mindir`

```text
# export_wizardcoder_inc.py
import os
import sys
sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
BS = 1
SEQ_LENGTH = 2048

# wizardcoder配置文件

config_path = 'run_wizardcoder.yaml'
config = WizardCoderConfig.from_pretrained(os.path.realpath(config_path))
config.use_past = True # 增量推理

model = WizardCoderLMHeadModel(config)
model.set_train(False)

# 全量推理 prefill

model.add_flags_recursive(is_first_iteration=True)
input_ids = ms.Tensor(np.ones((BS, SEQ_LENGTH)), mstype.int32)
input_position = ms.Tensor([127]*BS, mstype.int32)
init_reset = ms.Tensor([False], mstype.bool_)
batch_valid_length = ms.Tensor([[128]*BS], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length, file_name=f"wizardcoder-15b_mslite_inc/prefill_2k_bs{BS}", file_format="MINDIR")

# 增量推理 decode

model.add_flags_recursive(is_first_iteration=False)
input_ids = ms.Tensor(np.ones((BS, 1)), mstype.int32)
input_position = ms.Tensor([128]*BS, mstype.int32)
init_reset = ms.Tensor([True], mstype.bool_)
batch_valid_length = ms.Tensor([[129]*BS], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length, file_name=f"wizardcoder-15b_mslite_inc/decode_2k_bs{BS}", file_format="MINDIR")
```

- step 3. GE配置

GE配置文件context.cfg

```text
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
```

- step 4. mslite推理

指定使用tokenizer配置，配置步骤2生成的mindir文件，配置步骤3中的GE文件context.cfg

```text
# test_wizardcoder_mslite.py
import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.insert(0, os.getcwd().split('research')[0])

from mindspore import context
from mindformers.pipeline import pipeline

from wizardcoder_config import WizardCoderConfig
from wizardcoder_tokenizer import WizardCoderTokenizer


context.set_context(device_id=0, mode=0)
tokenizer_path = "/path/Wizardcoder-15B/tokenizer/" # Wizardcoder-15B tokenizer path
tokenizer = WizardCoderTokenizer(
    vocab_file=tokenizer_path + "vocab.json",
    merge_file=tokenizer_path + "merges.txt"
)
use_past = True # False为自回归推理，True为增量推理
if use_past:
    model_path = ("wizardcoder-15b_mslite_inc/prefill_2k_bs1_graph.mindir", "wizardcoder-15b_mslite_inc/decode_2k_bs1_graph.mindir")
else:
    model_path = ("wizardcoder-15b_mslite_autoregressive/prefill_2k_bs1_graph.mindir", None)
ge_config_path = "context.cfg"
pipeline_task = pipeline(task="text_generation", model=model_path, backend="mslite", tokenizer=tokenizer, ge_config_path=ge_config_path, model_type="mindir", infer_seq_length=2048, add_special_tokens=False)

input_data = ["使用python编写快速排序代码"]

pipeline_result = pipeline_task(input_data, do_sample=False, max_length=2048,  eos_token_id=0, pad_token_id=49152, skip_special_tokens=True)

print(pipeline_result[0])

# ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']
```

### 开源数据集评测

#### 评测结果

**注：** 评测结果基于开源的预训练模型

|                  | HumanEval Pass@1 | MBPP Pass@1 |
|------------------|------------------|-------------|
| 910B + Mindspore | 59.15            | 50.6        |
| A100 + Pytorch   | 59.75            | 50.6        |

#### HumanEval评测流程

- step 1: Install Human-Eval from OpenAI

```bash
git clone https://github.com/openai/human-eval.git
pip install -e human-eval
```

**注**: 需要将`human-eval/human_eval/execution.py`的第58行注释去掉

- step 2: 生成推理结果

```bash
model="mindspore_models"
device_id=0
temp=1
top_p=0.9
top_k=40
num_beams=1
max_len=1024
pred_num=1
num_seqs_per_iter=1

output_path=output_dir/910b_T${temp}_N${pred_num}_WizardCoder_Greedy_Decode

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems
echo 'Running process #'
python humaneval_generate.py --model ${model} \
  --temperature ${temp} \
  --num_seqs_per_iter ${num_seqs_per_iter} \
  --N ${pred_num} \
  --max_len ${max_len} \
  --output_path ${output_path} \
  --greedy_decode
```

- step 3: 生成测试分数

使用如下命令生成推理分数：

```bash
output_path=output_dir/910b_T0.1_N1

echo 'Output path: '$output_path
python humaneval_process.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

evaluate_functional_correctness ${output_path}.jsonl
```
