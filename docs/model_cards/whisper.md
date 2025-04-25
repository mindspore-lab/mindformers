# whisper-large-v3

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 之后的版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

Whisper 是一种最先进的自动语音识别 (ASR) 和语音翻译模型，该模型由 OpenAI 的 Alec Radford 等人在论文[《Robust Speech Recognition via Large-Scale Weak Supervision》](https://huggingface.co/papers/2212.04356)中提出。Whisper 在超过 500 万小时的标注数据上进行了训练，在零样本设置下表现出对多种数据集和领域的强泛化能力。

```text
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

## 模型文件

`whisper-large-v3`基于`mindformers`实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/whisper
       ├── __init__.py
       ├── configuration_whisper.py    # 模型配置项
       ├── modeling_whisper.py         # 模型脚本
       ├── processing_whisper.py       # 语音数据处理
       └── tokenization_whisper.py     # tokenizer
   ```

2. 模型配置：

   ```text
   configs/whisper
       └── finetune_whisper_large_v3.yaml  # 模型训练启动配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 模型权重下载

MindFormers提供HuggingFace官方权重下载链接，用户可下载权重并经过[模型权重转换](#模型权重转换)后进行使用。

| 模型名称                   | MindSpore权重 |                              HuggingFace权重                               |
|:-----------------------|:-----------:|:------------------------------------------------------------------------:|
| whisper-large-v3 |      -      | [Link](https://huggingface.co/openai/whisper-large-v3) |

```text
                        下载清单
openai/whisper-large-v3
    ├── pytorch_model.bin           # 模型权重
    ├── added_tokens.json           # tokenizer相关文件
    ├── merges.txt                  # tokenizer相关文件
    ├── vocab.json                  # tokenizer相关文件
    ├── tokenizer.json              # tokenizer相关文件
    ├── special_tokens_map.json     # tokenizer相关文件
    └── tokenizer_config.json       # tokenizer相关文件
```

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
pip install transformers torch
python convert_weight.py --model whisper --input_path TORCH_CKPT_PATH --output_path {path}/MS_CKPT_NAME --dtype 'fp16'

# 参数说明
model:       模型名称
input_path:  HuggingFace权重文件pytorch_model.bin路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换后的MindSpore权重参数类型
```

#### 数据集下载与处理

Mindformers提供使用[common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)数据集进行微调的样例，需要下载mp3文件和对应的tsv文件。下面以印地语(Hindi)为例，进行数据处理。

1. 下载数据集文件
    * [hi_dev_0.tar](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/tree/main/audio/hi/dev)
    * [dev.tsv](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/tree/main/transcript/hi)

2. 生成MindRecord

    ```shell
    python mindformers/tools/dataset_preprocess/whisper/common_voice_preprocess.py \
    --mp3_dir ./hi_dev_0 \
    --tsv_file ./dev.tsv \
    --tokenizer_dir ./whisper-large-v3 \
    --output_file ./hindi.mindrecord \
    --seq_length 448

    # 参数说明
    mp3_dir:         解压后的mp3文件夹路径
    tsv_file:        tsv文件路径
    tokenizer_dir:   tokenizer相关文件所在文件夹
    output_file：    输出文件路径
    seq_length:      序列长度
    ```

## 全参微调

MindFormers提供`whisper-large-v3`的微调示例。

1. 修改模型配置文件`configs/whisper/finetune_whisper_large_v3.yaml`

    ```yaml
    load_checkpoint: "path_to_ckpt"            # 模型权重

    train_dataset: &train_dataset
      data_loader:
        type: MindDataset
        dataset_dir: "path_to_mindrecord"      # 训练数据集

    model:
      model_config:
        max_target_positions: 448              # 训练数据集的序列长度
    ```

2. 单机八卡启动训练

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config configs/whisper/finetune_whisper_large_v3.yaml"

   # 参数说明
   config:          模型配置文件路径
   ```
