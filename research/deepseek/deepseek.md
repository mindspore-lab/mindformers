# DeepSeek Coder

DeepSeek Coder由一系列代码语言模型组成，每个模型都在2T token上从零开始训练，其中87%的代码和13%的自然语言组成，英文和中文都有。在编码功能方面，DeepSeek Coder在多种编程语言和各种基准测试上的开源代码模型中实现了最先进的性能。

## 仓库介绍

`deepseek_33b` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/deepseek`

    ```bash
    deepseek
        ├── convert_reversed.py            # ckpt转huggingface
        ├── convert_weight.py              # huggingface转ckpt
        ├── deepseek.md                    # README
        ├── predict_deepseek_33b.yaml    # 模型参数配置

## 前期准备

### 安装mindformers

参考[README](../../README.md#二、mindformers安装)安装mindformers。
本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件: Ascend 910B
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

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合算大算子降低推理时延，有效提升网络吞吐量。
在启动前，请先行在配置文件predict_deepseek_33b.yaml中按照如下配置

```yaml

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
bash scripts/msrun_launcher.sh "run_mindformer.py --config research/deepseek/predict_deepseek_33b.yaml --run_mode=predict --predict_data "#write a quick sort algorithm" --predict_length 100 --use_parallel True --use_past True" 4
#运行结果：[{'text_generation_text': ['#write a quick sort algorithm\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\nprint(quick_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a merge sort algorithm\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] < right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result += left[i:]\n    result += right[j:]\n    return result\n\nprint(merge_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, 6, 8, 10]"\n\n#write a bubble sort algorithm\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr) - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\nprint(bubble_sort([3,6,8,10,1,2,1]))\n# Prints "[1, 1, 2, 3, '']}]

```
