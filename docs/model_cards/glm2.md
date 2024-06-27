# ChatGLM2

## 模型描述

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B引入了新特征：**更强大的性能**、**更长的上下文**、**更高效的推理**、**更开放的协议**。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 模型性能

- 以下模型性能均由Atlas 800硬件环境下测试得出。

GLM2_6b:

| config                                                           | task            | Datasets | metric                                  | phase                   | score                                  | performance                                    |
|------------------------------------------------------------------|-----------------|----------|-----------------------------------------|-------------------------|----------------------------------------|------------------------------------------------|
| [glm2_6b](../../configs/glm2/finetune_glm2_6b_fp16.yaml)         | text_generation | ADGEN    | -                                       | [finetune](#微调)         | -                                      | 815.2059134 tokens/s/p                         |
| [glm2_6b_lora](../../configs/glm2/lora_glm2_6b_fp16.yaml)        | text_generation | ADGEN    | -                                       | [finetune](#lora微调)     | -                                      | 3243.697479 tokens/s/p                         |
| [glm2_6b](../../configs/glm2/run_glm2_6b.yaml)                   | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)             | 30.7842<br>7.0734<br>24.7739<br>7.4661 | -                                              |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora_eval.yaml)    | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)             | 31.0563<br>7.1753<br>24.2296<br>7.2294 | -                                              |
| [glm2_6b](../../configs/glm2/predict_glm2_6b.yaml)               | text_generation | -        | -                                       | [predict](#推理)          | -                                      | 32.08 tokens/s (use_past=True, seq_length=512) |

## 模型文件

`chatGLM2-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm2`

    ```text
    glm2
        ├── __init__.py
        ├── glm2.py                  # 模型实现
        ├── glm2_config.py           # 模型配置项
        ├── glm2_modules.py          # 模组实现
        ├── glm2_tokenizer.py        # tokenizer
        └── glm2_transformer.py      # transformer层实现
    ```

2. 模型配置：`configs/glm2`

    ```text
    configs/glm2
      ├── run_glm2_6b.yaml
      ├── run_glm2_6b_finetune_2k_800T_A2_64G.yaml  # Atlas 800T A2 最佳性能全量微调启动配置
      ├── run_glm2_6b_finetune_2k_800_32G.yaml      # Atlas 800 最佳性能全量微调启动配置
      ├── run_glm2_6b_finetune_800T_A2_64G.yaml     # Atlas 800T A2 ADGEN全量微调启动配置
      ├── run_glm2_6b_finetune_800_32G.yaml         # Atlas 800 ADGEN全量微调启动配置
      ├── run_glm2_6b_finetune_eval.yaml            # 全量微调后评估配置
      ├── run_glm2_6b_lora_2k_800T_A2_64G.yaml      # Atlas 800T A2最佳性能 lora微调启动配置
      ├── run_glm2_6b_lora_2k_800_32G.yaml          # Atlas 800 最佳性能 lora微调启动配置
      ├── run_glm2_6b_lora_800T_A2_64G.yaml         # Atlas 800T A2 ADGEN lora微调启动配置
      ├── run_glm2_6b_lora_800_32G.yaml             # Atlas 800 ADGEN lora微调启动配置
      └── run_glm2_6b_lora_eval.yaml                # lora微调评估配置
    ```

## 环境及数据准备

### 安装环境

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 数据及权重准备

#### 数据集下载

模型使用 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集作为微调数据集。

| 数据集名称 |    适用模型     |   适用阶段   |                                下载链接                                |
|:------|:-----------:|:--------:|:------------------------------------------------------------------:|
| ADGEN | ChatGLM2-6b | Finetune | [Link](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) |

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

| 模型名称            |                                                   MindSpore权重                                                   |                  HuggingFace权重                   |
|:----------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------:|
| ChatGLM2-6b     |                                                        /                                                        | [Link](https://huggingface.co/THUDM/chatglm2-6b) |
| tokenizer.model | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/tokenizer.model) |                        /                         |

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

MindFormers提供`ChatGLM2-6B`的微调示例， 过程中使用`ADGEN`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

全参微调使用 `configs/glm2/finetune_glm2_6b_fp16.yaml` 配置文件，配置文件中定义了微调所需的各配置项。

修改数据集/模型权重配置路径：

- 数据集：修改 `configs/glm2/finetune_glm2_6b_fp16.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `configs/glm2/finetune_glm2_6b_fp16.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

```yaml
load_checkpoint: '{path}/glm2_6b.ckpt'  # 模型权重文件路径

model:
  model_config:
    seq_length: 192

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "{path}/AdvertiseGen/train.json" # 数据集路径
  tokenizer:
    vocab_file: "{path}/tokenizer.model"          # 词表路径
  max_source_length: 64
  max_target_length: 127
```

> 注：微调时模型的`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。 在配置文件中默认的`seq_length: 192`以及`max_source_length: 64`和`max_target_length: 127`适用于ADGEN数据集，
> 对于其他数据集，可以将数据集转换为`token_id`，使`seq_length`等于`token_id`的最大长度，`seq_length`太大影响训练性能，太小影响训练精度，需要做出权衡。

这里以单机8卡`glm2_6b`微调为例。

1. 修改配置文件`configs/glm2/finetune_glm2_6b_fp16.yaml`

   ```yaml
   parallel_config:
     data_parallel: 8
     model_parallel: 1
     pipeline_stage: 1
     expert_parallel: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

2. 执行训练命令

   ```text
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/glm2/finetune_glm2_6b_fp16.yaml \
    --run_mode finetune"

   # 参数说明
   config:   配置文件路径
   run_mode: 运行模式, 微调时设置为finetune

   # 补充说明
   训练的log日志路径：mindformers/output/log
   checkpoint(含优化器参数)存储路径：mindformers/output/checkpoint
   checkpoint(不含优化器参数)存储路径：mindformers/output/checkpoint_network
   若想合并ckpt用于后续评估，选择不含优化器参数的权重即可
   ```

### LoRA微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象。 因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象。

使用LoRA算法进行低参微调时，使用 `configs/glm2/lora_glm2_6b_fp16.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项。

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/lora_glm2_6b_fp16.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/lora_glm2_6b_fp16.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 单卡训练

```shell
python run_mindformer.py --config configs/glm2/lora_glm2_6b_fp16.yaml --run_mode finetune

# 参数说明
config:   配置文件路径
run_mode: 运行模式, 微调时设置为finetune
```

#### 多卡训练

训练过程与[全参微调](#全参微调)相同，执行命令时使用`configs/glm2/lora_glm2_6b_fp16.yaml`即可。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/glm2/lora_glm2_6b_fp16.yaml \
 --run_mode finetune"
```

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

### 边训边评估

1. 使用 `Rouge-1`、`Rouge-2` 等指标评测

   使用该指标评测时速度较慢，推荐使用 `PerplexityMetric` 评测。

   将训练配置文件的 `do_eval: False` 设置为 `do_eval: True`，并且需要将 `train_dataset` 和 `eval_dataset` 的 `max_source_length`、`max_target_length` 以及 `batch_size`项设置为相同值，并且保持 `max_source_length + max_target_length + 1 = seq_length`，如下所示：

   ```yaml
   do_eval: True
   eval_step_interval: 1788
   eval_epoch_interval: -1

   metric:
     type: ADGENMetric

   model:
     model_config:
       seq_length: 192
   train_dataset: &train_dataset
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   eval_dataset: &eval_dataset
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   ```

2. 使用 `PerplexityMetric` 指标评测

   将训练配置文件的 `do_eval: False` 设置为 `do_eval: True`，并且需要将 `train_dataset` 和 `eval_dataset` 的 `max_source_length`、`max_target_length` 、`phase` 以及 `batch_size`项设置为相同值，并且保持 `max_source_length + max_target_length + 1 = seq_length`，如下所示：

   ```yaml
   do_eval: True
   eval_step_interval: 1788
   eval_epoch_interval: -1

   metric:
     type: PerplexityMetric

   model:
     model_config:
       seq_length: 192
   train_dataset: &train_dataset
     data_loader:
       phase: "train"
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   eval_dataset: &eval_dataset
     data_loader:
       phase: "train"
     max_source_length: 64
     max_target_length: 127
     batch_size: 8
   ```

mindformers通过 `eval_step_interval` 和 `eval_epoch_interval` 两项配置参数来控制边训练边评估的执行间隔，参数含义如下：

- **eval_step_interval**: 评估step间隔, 默认为100，表示每100个step间隔执行一次评估；配置为大于0的数表示每隔所配置的step数后执行一次评估，配置为小于0的数则表示禁用step评估；注意：在数据下沉模式下，step间隔值建议配置为sink size的倍数。
- **eval_epoch_interval**: 评估epoch间隔, 默认为-1，表示禁用epoch结束时的评估；配置为大于0的数表示每隔所配置的epoch数后执行一次评估，配置为小于0的数则表示禁用epoch评估；注意：数据下沉模式下，epoch所包含的step数将从数据集大小变为sink size的大小，将在 `sink_size * eval_epoch_interval` 个step后执行一次评估。

## 推理

MindFormers提供`GLM2-6b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡多轮推理。

```shell
# 脚本使用
bash scripts/examples/glm2/run_glm2_predict.sh CONFIG_PATH CKPT_PATH

# 参数说明
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
```

运行如下命令进行推理：

```shell
bash scripts/examples/glm2/run_glm2_predict.sh \
 configs/glm2/predict_glm2_6b.yaml \
 path/to/glm2_6b.ckpt
```

## 评测

评测使用 `configs/glm2/run_glm2_6b_finetune_eval.yaml` 和`configs/glm2/run_glm2_6b_lora_eval.yaml`配置文件，配置文件中定义了评测所需的各配置项。

### 文本生成

评测数据集可参考[数据集下载](#数据集下载)。

配置文件修改部分如下：

```yaml
load_checkpoint: '{path}/glm2_6b.ckpt'  # 模型权重文件路径
# ==== model config ====
model:
  model_config:
    seq_length: 256
 # ==== dataset config ====
eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "{path}/AdvertiseGen/dev.json" # 数据集路径
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "{path}/tokenizer.model"        # 词表路径
  max_source_length: 256
  max_target_length: 256
```

> 注：评测时模型`seq_length`需要等于评测数据集的`max_source_length`和`max_target_length`。因此修改yaml中模型`seq_length`为256。

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_finetune_eval.yaml` glm2模型推理配置，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快。

```shell
python run_mindformer.py \
 --config configs/glm2/run_glm2_6b_finetune_eval.yaml \
 --run_mode eval \
 --load_checkpoint {path}/glm2_6b_finetune.ckpt \
 --device_id 0 \
 --use_parallel False
```

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_lora_eval.yaml` glm2_lora模型推理配置，此配置可用于lora模型，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快。

```shell
python run_mindformer.py \
 --config configs/glm2/run_glm2_6b_lora_eval.yaml \
 --run_mode eval \
 --load_checkpoint {path}/glm2_6b_lora.ckpt \
 --device_id 0 \
 --use_parallel False
```

**注意**：单卡评测时，应将yaml文件中 model:model_config:batch_size 修改为等于 runner_config:batch_size
