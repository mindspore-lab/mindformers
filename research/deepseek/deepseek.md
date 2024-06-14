# DeepSeek Coder

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 仓库介绍

`deepseek_33b` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/deepseek`

    ```bash
    deepseek
        ├── alpaca_converter.py          # 数据预处理
        ├── convert_reversed.py          # ckpt转huggingface
        ├── convert_weight.py            # huggingface转ckpt
        ├── deepseek_preprocess.py       # 数据预处理
        ├── deepseek.md                  # README
        ├── finetune_deepseek_33b.yaml   # 模型微调配置
        ├── predict_deepseek_33b.yaml    # 模型参数配置

## 前期准备

### 安装mindformers

参考[README](../../README.md#二、mindformers安装)安装mindformers。
本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore: 2.3.0
- MindFormers: dev

### deepseek-coder-33b-instruct 权重下载和转换

下载链接：

权重：https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/pytorch_model-00001-of-00007.bin

词表：https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json

linux可用如下命令下载。

```shell

mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/pytorch_model-00001-of-00007.bin
wget https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json
cd ../..

```

- 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[deepseek-coder-33b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell

git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct

```

执行权重转换脚本

```shell

cd research
python deepseek/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME

```

```text

# 参数说明
torch_ckpt_path: huggingface deepseek-coder-33b-instruct权重保存目录路径下任意权重bin文件，根据该文件路径读取目录下全部权重
mindspore_ckpt_path: mindspore权重文件保存路径

```

**注**: 请安装torch=1.13.1和transformers=4.30.2版本。如果执行报错，请检查并安装requests、decorator、pandas、sympy。

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

### 数据集准备

目前提供code_alpaca数据集的预处理脚本用于全参微调任务。
数据集下载链接如下：

- [code_alpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json)

执行`alpaca_converter.py`，将原始数据集转换为指定格式。

``` bash

cd research
python deepseek/alpaca_converter.py \
--data_path path/alpaca_data.json \
--output_path /path/alpaca-data-messages.json
# 参数说明
# data_path: 存放alpaca数据的路径
# output_path: 输出转换后对话格式的数据路径

```

转换后格式样例：

```text

  {
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nCreate an array of length 5 which contains all even numbers between 1 and 10.\n### Response:\n"
      },
      {
        "from": "gpt",
        "value": "arr = [2, 4, 6, 8, 10]"
      }
    ]
  },

```

执行`deepseek_preprocess.py`，进行数据预处理和Mindrecord数据生成。

```bash

cd research
python deepseek/deepseek_preprocess.py \
--dataset_type qa \
--input_glob /path/alpaca-data-messages.json \
--model_file /path/vocab.json \
--seq_length 4096 \
--output_file /path/alpaca-messages.mindrecord

```

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合算大算子降低推理时延，有效提升网络吞吐量。
在启动前，请先行在配置文件predict_deepseek_33b.yaml中按照如下配置

```yaml

load_checkpoint: '/path/to/ckpt_dir'  # 填写切分权重文件所在文件夹
auto_trans_ckpt: False                # 关闭自动权重转换
processor:
  return_tensors: ms
  tokenizer:
    ...
    tokenizer_file: '/path/deepseek_33b/tokenizer.json'  # 修改为实际路径
    ...
model:
  model_config:
    ...
    use_past: True
    extend_method: "PI"
    is_dyamic: True
    ...

```

- Trainer高阶接口推理

deepseek coder-33b，无法进行单卡推理，可使用多卡推理，如下脚本为4卡推理样例，
msrun_launcher.sh在mindformers的scripts目录下

```shell

cd {mindformers根目录}
bash scripts/msrun_launcher.sh "run_mindformer.py --config research/deepseek/predict_deepseek_33b.yaml --run_mode=predict --predict_data '#write a quick sort algorithm' --predict_length 100 --use_parallel True --use_past True" 4
# 运行结果：[{'text_generation_text': ['#write a quick sort algorithm\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a merge sort algorithm\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result += left[i:]\n    result += right[j:]\n    return result\n\nprint(merge_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a bubble sort algorithm\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, '']}]

```

## 全参微调

### 微调性能

| config                                        | task | Datasets | SeqLength | metric | phase |score | performance(tokens/s/p) |
|-----------------------------------------------|-------|-------|-----------|-------|-------|-------|-------------------------|
| [deepseek-33b](./finetune_deepseek_33b.yaml)  | text_generation | code_alpaca | 4096     | - | [finetune](#全参微调) | - | 565                   |

### 操作步骤

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的code_alpaca数据集，参照[deepseek-coder-33b-instruct 权重下载和转换](#deepseek-coder-33b-instruct 权重下载和转换)章节获取权重。

1. 当前支持模型已提供yaml文件，下文以deepseek-33b为例，即使用`finetune_deepseek_33b.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

   当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

2. 修改`finetune_deepseek_33b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

   ```yaml

   load_checkpoint: '/path/model_dir' # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
   src_strategy_path_or_dir: ''
   auto_trans_ckpt: True  # 打开自动权重转换
   only_save_strategy: False
   resume_training: False
   run_mode: 'finetune'

   model_config:
      seq_length: 4096 # 与数据集长度保持相同

   train_dataset: &train_dataset
     data_loader:
       type: MindDataset
       dataset_dir: "/path/alpaca.mindrecord"  # 配置训练数据集文件夹路径

   # 分布式策略配置
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 2
     micro_batch_num: 128
     vocab_emb_dp: True
     gradient_aggregation_group: 4

   ```

3. 启动微调任务。

在多机上同时拉起任务，将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，具体可参考[ms_run快速使用](https://gitee.com/mindspore/mindformers#%E5%9B%9B%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8)

   ```shell

   # 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config research/deepseek/finetune_deepseek_33b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300

   # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config research/deepseek/finetune_deepseek_33b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 300

   # 参数说明
   # config: 配置文件路径
   # load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
   # auto_trans_ckpt: 自动权重转换开关
   # run_mode: 运行模式，微调时设置为finetune
   # train_data: 训练数据集文件夹路径

   ```
