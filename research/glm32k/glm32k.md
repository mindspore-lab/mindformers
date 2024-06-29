# ChatGLM3-32K

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型，更完整的功能支持，更全面的开源序列

ChatGLM3-6B-32K在ChatGLM3-6B的基础上进一步强化了对于长文本的理解能力，能够更好的处理最多32K长度的上下文。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 模型性能

| Config                                        |      Task       | Datasets  | SeqLength |      Phase      |       Performance       |
|:----------------------------------------------|:---------------:|:---------:|:---------:|:---------------:|:-----------------------:|
| [ChatGLM3-6B-32K](finetune_glm3_6b_bf16.yaml) | text_generation | LongBench |   32768   | [finetune](#微调) | 1758.00      tokens/s/p |

## 模型文件

`chatGLM3-6B-32K` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm2            # glm32k复用glm2的代码实现
        ├── __init__.py
        ├── glm2.py                    # 模型实现
        ├── glm2_config.py             # 模型配置项
        ├── glm2_modules.py            # 模组实现
        ├── glm2_tokenizer.py          # tokenizer
        └── glm2_transformer.py        # transformer层实现
    ```

2. 模型配置：

    ```text
    research/glm32k
        └── finetune_glm32k.yaml       # Atlas 800T A2最佳性能全量微调启动配置
        └── predict_glm32k.yaml        # Atlas 800T A2推理配置
    ```

3. 数据处理脚本和任务启动脚本：

    ```text
    research/glm32k
        └── glm32k_preprocess.py       # glm32k微调的数据前处理脚本
    ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](https://gitee.com/mindspore/mindformers/blob/dev/README.md#二mindformers安装)和[版本匹配关系](https://gitee.com/mindspore/mindformers/blob/dev/README.md#三版本匹配关系)。

### 数据集及权重准备

#### 数据集下载

| 数据集名称     |      适用模型       |   适用阶段   |                               下载链接                                |
|:----------|:---------------:|:--------:|:-----------------------------------------------------------------:|
| LongBench | ChatGLM3-6B-32K | Finetune | [Link](https://huggingface.co/datasets/THUDM/LongBench/tree/main) |

- **LongBench 数据预处理**

将`LongBench`数据集格式转换为`AdGen`数据集格式，以使用`ADGenDataLoader`进行数据读取。

下载链接：

1. [Longbench数据集](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)
2. [Longbench prompt file](https://github.com/THUDM/LongBench/blob/main/config/dataset2prompt.json)
3. Longbench数据集介绍请参见[官网地址](https://github.com/THUDM/LongBench)

```shell
cd research/glm32k
python glm32k_preprocess.py \
 --data_path INPUT_DATA_PATH \
 --output_path OUTPUT_PATH \
 --prompt_config_file PROMPT_PATH

# 参数说明
INPUT_DATA_PATH: 下载后LongBench数据文件夹路径
OUTPUT_PATH:     转换后的保存数据路径
PROMPT_PATH:     LongBench中不同数据对应的prompt
```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

| 模型名称            |                                               MindSpore权重                                               |                                 HuggingFace权重                                  |
|:----------------|:-------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|
| ChatGLM3-6B-32K |                                                    /                                                    |              [Link](https://huggingface.co/THUDM/chatglm3-6b-32k)              |
| tokenizer.model | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/tokenizer.model) | [Link](https://huggingface.co/THUDM/chatglm3-6b-32k/blob/main/tokenizer.model) |

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model glm-n --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 微调

MindFormers提供`ChatGLM3-6B-32K`的微调示例， 过程中使用`LongBench`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

1. 修改`research/glm32k/finetune_glm32k.yaml`配置文件。

   ```yaml
   load_checkpoint: 'path/to/glm3_32k.ckpt'                    # 预训练权重路径
   auto_trans_ckpt: False
   only_save_strategy: False
   resume_training: False
   use_parallel: True
   run_mode: 'finetune'

   train_dataset: &train_dataset
     data_loader:
       type: ADGenDataLoader
       dataset_dir: "/path/to/AdvertiseGen/train.json"
       shuffle: True
       phase: "train"
       version: 3
       origin_columns: ["content", "summary"]
     tokenizer:
       type: ChatGLM3Tokenizer
       vocab_file: "path/to/tokenizer.model"                    # 词表文件路径
     max_source_length: 30720                                   # 长序列源数据长度
     max_target_length: 2047                                    # 长序列目标数据长度
   ```

   > 注：长序列模型的训练，max_source_length和max_target_length数值较大，需要根据实际业务数据设置对应数值。

   修改并行配置为单机8卡。

   ```yaml
   parallel_config:
     data_parallel: 2
     model_parallel: 1
     pipeline_stage: 4
     micro_batch_num: 16
     vocab_emb_dp: False
     gradient_aggregation_group: 4
   ```

2. 执行启动命令。

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config research/finetune_glm32k.yaml \
    --run_mode finetune"
   ```

#### 多机训练

`ChatGLM3-6B-32K`多机多卡训练可以参考[多机多卡启动方式](../../README.md#多机多卡)。

### 分布式训练权重合并

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

1. 获取模型切分策略文件：

   在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

   > 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

   ```shell
   python transform_ckpt.py \
   --src_ckpt_strategy {path}/output/strategy/ \
   --src_ckpt_dir {path}/output/checkpoint/ \
   --dst_ckpt_dir {path}/target_checkpoint/ \
   --prefix glm2_6b

   # 参数说明
   src_ckpt_strategy: 步骤1中的切分策略文件路径
   src_ckpt_dir:      原切分权重文件夹
   dst_ckpt_dir:      目标路径
   prefix:            ckpt文件前缀名
   ```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本。

## 推理

MindFormers提供`ChatGLM3-6B-32K`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡多batch推理。

```shell
# 脚本使用
bash scripts/examples/glm32k/run_glm32k_predict.sh CONFIG_PATH CKPT_PATH TOKENIZER

# 参数说明
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
TOKENIZER:   模型tokenizer文件路径
```

运行如下命令进行推理：

```shell
bash scripts/examples/glm32k/run_glm32k_predict.sh \
 research/glm32k/predict_glm32k.yaml \
 path/to/glm32k_6b.ckpt
```
