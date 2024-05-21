# WizardCoder

## 模型描述

WizardCoder是由WizardLM团队推出了一个新的指令微调代码大模型，打破了闭源模型的垄断地位，超越了闭源大模型Anthropic Claude和谷歌的Bard。WizardCoder大幅度地提升了开源模型的SOTA水平，创造了惊人的进步，提高了22.3%的性能，成为了开源领域的新时代引领者。
WizardCoder完全开源可商用，基于 Transformer 结构，上下文窗口长度为 2048，参数量为150亿。 本仓库提供了WizardCoder-15B预训练模型。

## 仓库介绍

`WizardCoder` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/wizardcoder`

    ```text
    wizardcoder
        ├── wizardcoder_tokenizer.py       # tokenizer
        ├── wizardcoder.py                 # 15B模型实现
        └── wizardcoder_modules.py         # self-attention模块实现
    ```

2. 模型配置：`research/wizardcoder`

    ```text
    wizardcoder
        └── run_wizardcoder.yaml           # 15B全量微调Atlas 800T A2启动配置
    ```

3. 数据处理脚本和任务启动脚本：`research/wizardcoder`

    ```text
    wizardcoder
        ├── wizardcoder_preprocess.py      # wizardcoder数据集预处理脚本
        └── run_wizardcoder.py             # wizardcoder高阶接口使用脚本
    ```

## 环境要求

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

设置环境变量

```shell
#!/bin/bash
export ASCEND_CUSTOM_PATH=/path/cann/ascend-toolkit
export ASCEND_HOME_PATH=$ASCEND_CUSTOM_PATH

#导入CANN基本环境变量
source $ASCEND_CUSTOM_PATH/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_PATH/latest/lib64:$ASCEND_CUSTOM_PATH/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH

#配置整网ND消除格式转换算子
export MS_ENABLE_FORMAT_MODE=1

#REF模式和CELL共享
export MS_DISABLE_REF_MODE=1

#内存优化：配置atomic内存单独清零
export MS_GE_ATOMIC_CLEAN_POLICY=1

#内存优化：配置内存扩展模式（实现纯静态图之间内存复用）
export GE_USE_STATIC_MEMORY=2
```

**注：** `ASCEND_CUSTOM_PATH`的`path`替换为CANN包的实际安装路径

### 模型权重下载与转换(mindformers权重或huggingface权重选择使用即可)

#### mindformers权重直接使用

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调。

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/wizardcoder_15B.ckpt
cd ../..
```

#### huggingface权重转换后使用

从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huffingface权重的链接如下：

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/added_tokens.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/generation_config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/merges.txt
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/pytorch_model.bin
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/special_tokens_map.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/tokenizer.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/tokenizer_config.json
wget https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/main/vocab.json
cd ..
```

**注**: 请安装torch=1.11.0和transformers=4.30.2版本; 进行模型转换后，需要重新根据本项目[requirements.txt](../../requirements.txt)恢复tokenizers版本

```shell
#!/bin/bash
pip install torch==1.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 后续转换任务完成后
pip install -r requirement.txt
```

下载完成后，运行`/research/wizardcoder/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
#!/bin/bash
python ./research/wizardcoder/convert_weight.py \
--torch_path ./ckpt/pytorch_model.bin \
--mindspore_path ./ckpt/rank_0/wizardcoder_15b.ckpt
```

```yaml
# 参数说明
torch_path: huggingface权重保存目录路径下任意权重bin文件，根据文件路径读取目录下全部权重
mindspore_path: mindspore格式的权重保存文件名，如'saved_dir/wizardcoder.ckpt'
```

## WizardCoder-15B

### 训练和微调性能

| config                                    | task            | Datasets | SeqLength | metric | phase           | score | performance(tokens/s/p) |
|-------------------------------------------|-----------------|----------|-----------|--------|-----------------|-------|-------------------------|
| [wizardcoder_15b](./run_wizardcoder.yaml) | text_generation | alpaca   | 2048      | -      | [train](#预训练)   | -     | 798.7                   |
| [wizardcoder_15b](./run_wizardcoder.yaml) | text_generation | alpaca   | 2048      | -      | [finetune](#微调) | -     | 798.7                   |

```shell
#!/bin/bash
pip install mindspore==2.2.0
pip install mindpet==1.0.2
```

### 预训练

#### 数据集准备-预训练数据集

当前提供Alpaca数据集的预处理和预训练样例，用于对wizardcoder-15B模型进行预训练。数据集的官方下载链接如下：

```shell
#!/bin/bash
mkdir dataset
cd ./dataset
wget https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
cd ..
```

修改`research/wizardcoder/wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```python
# 原代码 research/wizardcoder/wizardcoder_preprocess.py, 138行
# tokenize_qa()的入参if_jsonl需要设置为False
def tokenize_qa(tokenizer, file_path, max_length, if_jsonl=False):
    ...
```

```shell
#!/bin/bash
python research/wizardcoder/wizardcoder_preprocess.py \
--input_glob ./dataset/alpaca_data.json \
--vocab_file ./ckpt/vocab.json \
--merge_file ./ckpt/merges.txt \
--output_file ./dataset/alpaca_data.mindrecord \
--seq_length 2048
```

#### 预训练启动

- step 1. 修改`research/wizardcoder/run_wizardcoder.yaml`中相关配置

```yaml
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './ckpt'          # 添加预训练权重路径
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

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
#!/bin/bash
# 以wizardcoder模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
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

-[x] 2: 执行运行脚本。

```shell
#!/bin/bash
bash scripts/msrun_launcher.sh \
"python research/wizardcoder/run_wizardcoder.py \
--config research/wizardcoder/run_wizardcoder.yaml \
--load_checkpoint ./ckpt \
--use_parallel True \
--run_mode train \
--train_data ./dataset/alpaca_data.mindrecord" 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
run_mode: 运行模式，训练时设置为train
train_data: 训练数据集路径
```

### 微调

#### 数据集准备-SFT微调数据集

当前提供codealpaca数据集的预处理和微调样例，用于对wizardcoder-15B模型进行微调。数据集下载链接如下：

```shell
#!/bin/bash
mkdir finetune_dataset
cd ./finetune_dataset
wget https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json
cd ..
```

执行`research/wizardcoder/wizardcoder_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```python
# 原代码 research/wizardcoder/wizardcoder_preprocess.py, 138行
# tokenize_qa()的入参if_jsonl需要设置为False
def tokenize_qa(tokenizer, file_path, max_length, if_jsonl=False):
    ...
```

```bash
# 脚本路径：research/wizardcoder/wizardcoder_preprocess.py
python research/wizardcoder/wizardcoder_preprocess.py \
--input_glob ./finetune_dataset/code_alpaca_20k.json \
--vocab_file ./ckpt/vocab.json \
--merge_file ./ckpt/merges.txt \
--output_file ./finetune_dataset/code_alpaca.mindrecord \
--seq_length 2048
```

#### 全参微调

全参微调需要多卡启动，以`CodeAlpaca-20k`数据集为例,给出了默认配置文件`research/wizardcoder/run_wizardcoder.yaml`。

- step 1. 修改`research/wizardcoder/run_wizardcoder.yaml`中相关配置

```yaml
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './output/transformed_checkpoint/'          # 添加预训练权重路径
auto_trans_ckpt: False
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "./finetune_dataset/code_alpaca.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如code_alpaca数据集），input_columns: ["input_ids", "labels"]
```

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以wizardcoder模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
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

-[x] 2: 执行运行脚本。

```shell
bash scripts/msrun_launcher.sh \
"python research/wizardcoder/run_wizardcoder.py \
--config research/wizardcoder/run_wizardcoder.yaml \
--load_checkpoint ./output/transformed_checkpoint/ \
--use_parallel True \
--run_mode finetune \
--train_data ./finetune_dataset/code_alpaca.mindrecord" 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

### 快速推理

**注：** 推理部分需要更新如下环境变量

```shell
#!/bin/bash
unset MS_DISABLE_REF_MODE=1
export MS_ENABLE_REF_MODE=1
```

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
   vocab_file: 'vocab.json'        # 添加vocab.json的路径
   merge_file: 'merges.txt'        # 添加merges.txt的路径
   type: WizardCoderTokenizer
```

相关文件的下载链接如下：[vocab.json](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/vocab.json) 和 [merges.txt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/wizardcode/merges.txt)

- step 2. Trainer接口启动推理

wizardcoder的高阶接口使用脚本已集成在run_wizardcoder.py脚本中，运行此脚本命令示例：

其中`ckpt_path_or_dir`为模型文件地址，如：{path}/wizardcoder.ckpt

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

**注：** 使用如下脚本推理，其中`wizardcoder_model_path`是权重存放的地址，`tokenizer_path`是存放vocab.json和merges.txt的目录地址

```python
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

wizardcoder_model_path = "/path/wizardcoder_15b.ckpt" # 添加模型文件地址
wizardcoder_config = WizardCoderConfig(
    batch_size=1,
    seq_length=2048,
    n_position=8192,
    vocab_size=49153,
    hidden_size=6144,
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
    vocab_file=tokenizer_path + "vocab.json",    # tokenizer_path为存放vocab.json和merges.txt的地址
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

修改run_wizardcoder.yaml中的配置项，只需要修改如下yaml中的vocab_file和merge_file地址

```yaml
use_parallel: True   # 单机多卡或多机多卡必须设为True
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
model:
  model_config:
    use_seq_parallel: False
    use_past: True   # False为自回归推理，True为增量推理
    checkpoint_name_or_path: ""
processor:
  tokenizer:
    vocab_file: '/tokenizer_path/vocab.json'
    merge_file: '/tokenizer_path/merges.txt'
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

- step 3. 推理脚本

```shell
bash scripts/msrun_launcher.sh "research/wizardcoder/run_wizardcoder.py \
--config research/wizardcoder/run_wizardcoder.yaml \
--load_checkpoint ./output/transformed_checkpoint/ \
--use_parallel True \
--run_mode predict \
--predict_data '使用python编写快速排序代码，并分析其时间复杂度' \
--vocab_file vocab.json \
--merge_file merges.txt" 8
```

推理结果

```yaml
{'text_generation_text': ['使用python编写快速排序代码，并分析其时间复杂度。\r\n\r\n快速排序是一种分治算法，它的基本思想是：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。\r\n\r\n快速排序的步骤如下：\r\n\r\n1. 从数列中挑出一个元素，称为 “基准”（pivot）\r\n2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。\r\n3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。\r\n\r\n快速排序的时间复杂度为O(nlogn)，最坏情况下的时间复杂度为O(n^2)，平均情况下的时间复杂度为O(nlogn)。\r\n\r\n下面是Python代码实现的快速排序：\r\n\r\n```python\r\ndef quick_sort(arr):\r\n    if len(arr) <= 1:\r\n        return arr\r\n    else:\r\n        pivot = arr[0]\r\n        left = []\r\n        right = []\r\n        for i in range(1, len(arr)):\r\n            if arr[i] < pivot:\r\n                left.append(arr[i])\r\n            else:\r\n                right.append(arr[i])\r\n        return quick_sort(left) + [pivot] + quick_sort(right)\r\n```\r\n\r\n该代码的基本思路是：\r\n\r\n1. 如果数组的长度小于等于1，则直接返回数组。\r\n2. 选择数组的第一个元素作为基准值。\r\n3. 遍历数组，将比基准值小的元素放到左边，将比基准值大的元素放到右边。\r\n4. 递归地对左边和右边的子数组进行排序。\r\n5. 将左边子数组、基准值、右边子数组合并成一个新的数组。\r\n\r\n下面是该代码的时间复杂度分析：\r\n\r\n- 最坏情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行n-1次，因此最坏情况下的时间复杂度为O(n^2)。\r\n- 平均情况下的时间复杂度：每次选择的基准值都为数组的中间元素，每次递归都需要进行logn次，因此平均情况下的时间复杂度为O(nlogn)。\r\n- 最优情况下的时间复杂度：当数组的长度为n，且每次选择的基准值都为数组的第一个元素时，每次递归都需要进行logn次，因此最优情况下的时间复杂度为O(nlogn)。']}
```
