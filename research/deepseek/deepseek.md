# DeepSeek Coder

## 模型描述

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 模型性能

| config                                        | task | Datasets        | SeqLength | metric | phase            |score | performance(tokens/s/p) |
|-----------------------------------------------|-------|-----------------|-----------|-------|------------------|-------|-------------------------|
| [deepseek-33b](./finetune_deepseek_33b.yaml)  | text_generation | code_alpaca     | 4096      | - | [finetune](#全参微调) | - | 565                     |
| [deepseek-33b](./finetune_deepseek_33b.yaml)  | text_generation | wikitext-103-v1 | 16k       | - | train    | - | 421                     |

## 模型文件

`deepseek_33b` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型配置：

    ```text
    deepseek
        ├── finetune_deepseek_33b.yaml     # 全参微调启动配置
        ├── pretrain_deepseek_33b_16k.yaml # 预训练启动配置
        ├── predict_deepseek_33b.yaml      # huggingface转ckpt
        └── deepseek_preprocess.py         # 在线推理启动配置
    ```

2. 数据预处理脚本：

   ```text
    deepseek
        ├── alpaca_converter.py           # code_alpaca数据集格式转换脚本
        └── deepseek_preprocess.py        # 数据集预处理脚本
    ```

## 环境及数据准备

### 安装环境

### 环境参数设置

```shell
export MS_DEV_RUNTIME_CONF="inline:false"
```

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：Atlas 800T A2芯片支持33b单机4卡推理，全参微调至少需要2机16卡，预训练至少需要2机16卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext-103**作为[预训练](#预训练)数据集，**code_alpaca**作为[全参微调](#全参微调)数据集。

| 数据集名称        |                      适用模型                      |   适用阶段   |                                      下载链接                                       |
|:-------------|:----------------------------------------------:|:--------:|:-------------------------------------------------------------------------------:|
| Wikitext-103 | deepseek_33b  | Pretrain |                                                                                 |
| code_alpaca       | deepseek_33b | Finetune | [Link](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) |

数据预处理中所用的`vocab.json`和`merges.txt`可以参考[模型权重下载](#模型权重下载)进行下载。

- **Wikitext-103 数据预处理**

   使用`mindformers\research\deepseek\deepseek_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

   ```bash
   python deepseek_preprocess.py \
   --dataset_type 'wiki' \
   --input_glob /path/wiki.train.tokens \
   --model_file /path/vocab.json \
   --seq_length 16384 \
   --output_file /path/wiki.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.train.tokens的文件路径
  model_file:   vocab.json文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
   ```

- **code_alpaca 数据预处理**

    执行`mindformers\research\deepseek\alpaca_converter.py`，将原始数据集转换为指定格式。

    ```shell
    cd research
    python alpaca_converter.py \
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

    执行`mindformers\research\deepseek\deepseek_preprocess.py`，进行数据预处理和Mindrecord数据生成。

    ```shell
    cd research
    python deepseek/deepseek_preprocess.py \
    --dataset_type qa \
    --input_glob /path/alpaca-data-messages.json \
    --model_file /path/tokenizer.json \
    --seq_length 4096 \
    --output_file /path/alpaca-messages.mindrecord

    # 参数说明
    dataset_type: 预处理数据类型
    input_glob:   转换后的alpaca的文件路径
    model_file:   tokenizer.json文件路径
    seq_length:   输出数据的序列长度
    output_file:  输出文件的保存路径
    ```

### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，tokenizer.json文件也在链接中下载。

| 模型名称            |                           权重                           |
|:----------------|:---------------------------------------------------------------------:|
| deepseek-33b      |  [Link](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/pytorch_model-00001-of-00007.bin)  |
|   tokenizer.json   |  [Link](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/blob/main/tokenizer.json)  |

#### 模型权重转换

- **torch权重转mindspore权重**
  执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

    ```shell
    cd research
    python deepseek/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME

    # 参数说明
    torch_ckpt_dir: 预训练权重文件所在的目录, 此参数必须
    mindspore_ckpt_path: 转换后的输出文件存放路径, 默认为`./transform.ckpt`
    ```

- **[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)**

    从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

    通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

    以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 预训练

### 获取初始化权重配置

1. 登录网页[DeepSeek Coder 33B Weight](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct/tree/main)

2. 下载如下文件：

    1. config.json

    2. tokenizer.json

    3. tokenizer_config.json

3. 将文件放到项目根目录的`init_conf`文件夹下（手动创建该文件夹）

### 初始化随机权重

1. 进入代码根目录

2. 创建初始化随机脚本`init_weight.py`

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import torch

if __name__ == "__main__":
    config = AutoConfig.from_pretrained("./init_conf")
    model = AutoModelForCausalLM.from_config(config)
torch.save(model.state_dict(), 'deepseek_coder_16k_random.bin')
```

3. 执行命令获取初始化权重文件`deepseek_coder_16k_random.bin`

```shell
python init_weight.py
```

### 转换权重文件与生成分布式策略

参考[模型权重转换](#模型权重转换)章节

### 执行预训练

1. 进入项目根目录

2. 执行如下脚本进行预训练

   1. 当前脚本使用切分好的权重在2台机器上执行，需要手动将权重文件存放在2台机器项目根目录的`ckpt_trans`目录下

   2. 转换后的数据集，需要放到`./dataset/train_data`目录下

   3. {ip}参数：第0台机器的IP（确保两台机器网络互通且9543端口未被占用）

   4. {node}参数：第0台机器填写0，第1台机器填写1

```shell
bash ./scripts/msrun_launcher.sh "./run_mindformer.py \
--config ./research/deepseek/pretrain_deepseek_33b_16k.yaml \
--use_parallel True \
--load_checkpoint ./ckpt_trans \
--run_mode train \
--train_data ./dataset/train_data" \
16 8 {ip} 9543 {node} output/msrun_log False 3000;
```

## 全参微调

MindFormers提供`deepseek-33b`双机16卡的微调示例，使用**code_alpaca**数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 多机训练

1. 启动deepseek-33b微调任务

   修改`finetune_deepseek_33b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

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

   # 16卡分布式策略配置
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 2
     micro_batch_num: 128
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

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

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合算大算子降低推理时延，有效提升网络吞吐量。
在启动前，请先行在配置文件predict_deepseek_33b.yaml中按照如下配置

### 基于高阶接口推理

deepseek coder-33b，无法进行单卡推理，可使用多卡推理，如下脚本为4卡推理样例，
msrun_launcher.sh在mindformers的scripts目录下

#### 多卡推理

1. 主要参数配置参考：

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

2. 启动多卡推理：

  ```shell

  cd {mindformers根目录}
  bash scripts/msrun_launcher.sh "run_mindformer.py --config research/deepseek/predict_deepseek_33b.yaml --run_mode=predict --predict_data '#write a quick sort algorithm' --predict_length 100 --use_parallel True --use_past True" 4
  # 运行结果：[{'text_generation_text': ['#write a quick sort algorithm\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a merge sort algorithm\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result += left[i:]\n    result += right[j:]\n    return result\n\nprint(merge_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a bubble sort algorithm\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, '']}]

  ```
