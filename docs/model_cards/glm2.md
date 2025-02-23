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

- 基于Atlas 800

**GLM2_6b**:

| config                                                           | task            | Datasets | metric                                  | phase                   | score                                  | performance                                    |
|------------------------------------------------------------------|-----------------|----------|-----------------------------------------|-------------------------|----------------------------------------|------------------------------------------------|
| [glm2_6b](../../configs/glm2/finetune_glm2_6b_fp16.yaml)         | text_generation | ADGEN    | -                                       | [finetune](#微调)         | -                                      | 815.2059134 tokens/s/p                         |
| [glm2_6b_lora](../../configs/glm2/lora_glm2_6b_fp16.yaml)        | text_generation | ADGEN    | -                                       | [finetune](#lora微调)     | -                                      | 3243.697479 tokens/s/p                         |
| [glm2_6b_ptuning2](../../configs/glm2/run_glm2_6b_ptuning2.yaml) | text_generation | ADGEN    | -                                       | [finetune](#P-Tuning微调) | -                                      | 4150.537634 tokens/s/p                         |
| [glm2_6b](../../configs/glm2/run_glm2_6b.yaml)                   | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)             | 30.7842<br>7.0734<br>24.7739<br>7.4661 | -                                              |
| [glm2_6b_lora](../../configs/glm2/run_glm2_6b_lora_eval.yaml)    | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)             | 31.0563<br>7.1753<br>24.2296<br>7.2294 | -                                              |
| [glm2_6b_ptuning2](../../configs/glm2/run_glm2_6b_ptuning2.yaml) | text_generation | ADGEN    | rouge-1<br>rouge-2<br>rouge-l<br>bleu-4 | [eval](#评测)             | 31.5933<br>7.4504<br>24.7071<br>7.3042 | -                                              |
| [glm2_6b](../../configs/glm2/predict_glm2_6b.yaml)               | text_generation | -        | -                                       | [predict](#推理)          | -                                      | 32.08 tokens/s (use_past=True, seq_length=512) |

## 仓库介绍

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

    ```bash
    configs/glm2
      ├── export_glm2_6b.yaml
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
      ├── run_glm2_6b_lora_eval.yaml                # lora微调评估配置
      └── run_glm2_6b_ptuning2.yaml                 # Atlas 800 ADGEN ptuning微调启动配置
    ```

## 前期准备

### 环境要求

**MindFormers安装**以及**软硬件配套关系**参考[MindFormers安装](../../README.md#二MindFormers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1. 使用官方权重进行转换

   克隆glm2-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm2-6b
   ```

   执行 python 脚本，合并模型权重。

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
   model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

   with open("pt_model_arch.txt", "w") as fp:
       print(model, file=fp, flush=True)
   with open("pt_ckpt.txt", "w") as fp:
       for name, param in model.named_parameters():
           fp.write(f"{name} {param.shape} {param.dtype}\n")
   torch.save(model.state_dict(), "glm2_6b.pth")
   ```

   执行转换脚本，得到转换后的输出文件`glm2_6b.ckpt`。

   ```python
   import mindspore as ms
   import torch as pt
   from tqdm import tqdm

   pt_ckpt_path = "glm2_6b.pth"
   pt_param = pt.load(pt_ckpt_path)

   type_map = {"torch.float16": "ms.float16",
               "torch.float32": "ms.float32"}
   ms_param = []
   with open("check_pt_ckpt.txt", "w") as fp:
       for k, v in tqdm(pt_param.items()):
           if "word_embeddings.weight" in k:
               k = k.replace("word_embeddings.weight", "embedding_table")
           fp.write(f"{k} {v.shape} {v.dtype}\n")
           ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

   ms.save_checkpoint(ms_param, "glm2_6b.ckpt")
   ```

2. 获取MindFormers提供的已转换权重

   可通过from_pretrained接口下载，也可直接从下面的链接获取

   [glm2_6b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/glm2_6b.ckpt)

   [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

> 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix glm2_6b
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

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('glm2_6b')

# 自定义修改配置后实例化,配置为yaml文件路径，示例yaml文件为configs/glm2/predict_glm2_6b.yaml
# 需要修改yaml中的checkpoint_name_or_path为权重下载章节下载的权重文件
config = AutoConfig.from_pretrained('/path/to/predict_glm2_6b.yaml')
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("你好")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=20, do_sample=True, top_k=3)
response = tokenizer.decode(outputs)
print(response)
# ['你好，作为一名人工智能助手，我欢迎您随时向我提问。']
```

### 基于Trainer的快速训练，微调，评测，推理

glm2_6b暂不支持使用Trainer进行单卡训练和微调，请参考多卡训练和微调。

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='glm2_6b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')

# 开启评测
# 需要在configs/glm2/run_glm2_6b.yaml中将seq_length修改为256
trainer.evaluate()

# 开启推理
# 需要在configs/glm2/run_glm2_6b.yaml中将param_init_type、compute_dtype修改为"float16"
predict_result = trainer.predict(input_data="你好")
print(predict_result)
```

### 基于Pipeline的快速推理

```python
import mindspore
mindspore.set_context(mode=0, device_id=0)

from mindformers import AutoModel, AutoTokenizer, TextGenerationPipeline
# 自定义修改配置后实例化,配置为yaml文件路径，示例yaml文件为configs/glm2/predict_glm2_6b.yaml
# 需要修改yaml中的checkpoint_name_or_path为权重下载章节下载的权重文件
config = AutoConfig.from_pretrained('/path/to/predict_glm2_6b.yaml')
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型
tokenizer = AutoTokenizer.from_pretrained('glm2_6b')
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
predict_result = pipeline("你好")
print(predict_result)
# [{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据集准备

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{"content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳", "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"}
```

参照 [ChatGLM仓库](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning) 的指引下载处理好的 ADGEN 数据集，目录结构为

```text
AdvertiseGen
  ├── train.json
  └── dev.json
```

修改配置文件 `configs/glm2/run_glm2_6b_*.yaml` 中的以下项：

```yaml
train_dataset: &train_dataset
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 64
  max_target_length: 127

eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "/path/to/AdvertiseGen/dev.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  max_source_length: 256
  max_target_length: 256
```

**注意**：微调时的模型`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。
yaml文件中默认的`seq_length: 192`以及`max_source_length: 64`和`max_target_length: 127`适用于ADGEN数据集，
其他数据集的`seq_length`设置，可以遍历并将数据集转换为token_id，取token_id最大长度，`seq_length`太大影响训练性能，
太小影响训练精度，需要做出权衡。

### 全参微调

全参微调使用 `configs/glm2/run_glm2_6b_finetune*.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `configs/glm2/run_glm2_6b_finetune*.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `configs/glm2/run_glm2_6b_finetune*.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单卡微调

由于glm2_6b模型较大，全量微调不支持单卡运行

#### 多卡微调

```shell
# 以glm2-6b模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：configs/glm2/run_glm2_6b_finetune*.yaml
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  expert_parallel: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

```shell
cd {mindformers根目录}
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/glm2/run_glm2_6b_finetune*.yaml --run_mode finetune"
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
```

> 训练的log日志路径：mindformers/output/log
>
> checkpoint(含优化器参数)存储路径：mindformers/output/checkpoint
>
> checkpoint(不含优化器参数)存储路径：mindformers/output/checkpoint_network
>
> 若想合并ckpt用于后续评估，选择不含优化器参数的权重即可。

### LoRA微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象。
因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象

使用LoRA算法进行低参微调时，使用 `configs/glm2/run_glm2_6b_lora*.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/run_glm2_6b_lora*.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/run_glm2_6b_lora*.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 单卡微调

```shell
cd {mindformers根目录}
python run_mindformer.py --config configs/glm2/run_glm2_6b_lora*.yaml --run_mode finetune
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
```

#### 多卡微调

```shell
# 以glm2-6b模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：configs/glm2/run_glm2_6b_lora*.yaml
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  expert_parallel: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

```shell
cd {mindformers根目录}
bash scripts/msrun_launcher.sh "run_mindformer.py --config configs/glm2/run_glm2_6b_lora*.yaml --run_mode finetune"
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
```

> 训练的log日志路径：mindformers/output/log
>
> checkpoint(含优化器参数)存储路径：mindformers/output/checkpoint
>
> checkpoint(不含优化器参数)存储路径：mindformers/output/checkpoint_network
>
> 若想合并ckpt用于后续评估，选择不含优化器参数的权重即可。

### P-Tuning微调

对于每个下游任务，在网络的每一层添加一份连续提示向量，冻结预训练模型的其他参数，只训练这些向量。

#### 单卡微调

使用P-Tuning算法进行低参微调时，使用 `configs/glm2/run_glm2_6b_ptuning2.yaml` 配置文件，该配置文件包含了P-Tuning低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/run_glm2_6b_ptuning2.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/run_glm2_6b_ptuning2.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

执行命令：

```shell
cd {mindformers根目录}
python run_mindformer.py --config configs/glm2/run_glm2_6b_ptuning2.yaml --run_mode finetune
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，微调时设置为finetune
```

> 训练的log日志路径：mindformers/output/log
>
> checkpoint(含优化器参数)存储路径：mindformers/output/checkpoint
>
> checkpoint(不含优化器参数)存储路径：mindformers/output/checkpoint_network
>
> 若想合并ckpt用于后续评估，选择不含优化器参数的权重即可。

### 边训边评估

#### 1. 使用 `Rouge-1`、`Rouge-2` 等指标评测

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

#### 2. 使用 `PerplexityMetric` 指标评测

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

- **eval_step_interval**: 评估step间隔, 默认为100，表示每100个step间隔执行一次评估；配置为大于0的数表示每隔所配置的step数后执行一次评估，配置为小于0的数则表示禁用step评估；注意：在数据下沉模式下，step间隔值建议配置为sink size的倍数
- **eval_epoch_interval**: 评估epoch间隔, 默认为-1，表示禁用epoch结束时的评估；配置为大于0的数表示每隔所配置的epoch数后执行一次评估，配置为小于0的数则表示禁用epoch评估；注意：数据下沉模式下，epoch所包含的step数将从数据集大小变为sink size的大小，将在 `sink_size * eval_epoch_interval` 个step后执行一次评估

## 评测

### 文本生成

### 数据集准备-文本生成

见微调章节的[数据集准备](#数据集准备)

评测时模型`seq_length`需要等于评测数据集的`max_source_length`和`max_target_length`。因此修改yaml中模型`seq_length`为256：

```yaml
model:
  model_config:
    seq_length: 256
```

### 单卡评测

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_finetune_eval.yaml` glm2模型推理配置，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b_finetune_eval.yaml--run_mode eval --load_checkpoint /path/to/glm2_6b_finetune.ckpt --device_id 0 --use_parallel False
```

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_lora_eval.yaml` glm2_lora模型推理配置，此配置可用于lora模型，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b_lora_eval.yaml --run_mode eval --load_checkpoint /path/to/glm2_6b_lora.ckpt --device_id 0 --use_parallel False
```

> 单卡评测时，应将yaml中 model:model_config:batch_size 修改为等于 runner_config:batch_size

## 推理

### 基于generate的推理

下面提供一个模型推理样例脚本 `infer.py`

```python
from mindformers import AutoConfig, AutoModel, AutoTokenizer, ChatGLM2Tokenizer
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 自定义修改配置后实例化,配置为yaml文件路径，示例yaml文件为configs/glm2/predict_glm2_6b.yaml
# 需要修改yaml中的checkpoint_name_or_path为权重下载章节下载的权重文件
config = AutoConfig.from_pretrained("/path/to/predict_glm2_6b.yaml")
config.seq_length = 1024
model = AutoModel.from_config(config)

# 本地加载方式
tokenizer = ChatGLM2Tokenizer("/path/to/your/tokenizer.model")

kwargs={}
gen_kwargs = {"max_length": config.seq_length, "num_beams": 1, "do_sample": False, "top_p": 3,"top_k": 0.7,
              "temperature": 1, **kwargs}

queries = ["你好", "请介绍一下杭州", "那里有什么好吃的吗"]
history = []
for query in queries:
    # 如果想关闭history，此处传入 `history=[]` 即可
    prompt = tokenizer.build_prompt(query, history=history)
    input_id = tokenizer(prompt)["input_ids"]

    output = model.generate([input_id], **gen_kwargs)

    # output 包括了[input_id, output]两个部分
    output = output[0][len(input_id):]
    response = tokenizer.decode(output)
    print(response)
    history += [(query, response)]

    '''
    response1:
    你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。

    response2:
    杭州是中国浙江省省会，位于浙江省东南部，地处浙江省北部，东临东海，南接福建省，北与江苏省毗邻，是中国著名的旅游城市之一。

    杭州有着悠久的历史和文化，被誉为“人间天堂”，被誉为“南宋都城”，是中国南方著名的历史文化名城之一。杭州还被誉为“全国最具幸福感城市”，具有丰富的历史遗存、优美的自然风光和浓郁的文化氛围。

    杭州的经济以服务业为主导产业，特别是交通运输、仓储和邮政业。同时，杭州也是中国重要的电子商务和互联网产业基地之一，被誉为“中国电子商务之都”。

    杭州的著名景点包括西湖、灵隐寺、千岛湖、钱塘江等。西湖是中国著名的风景名胜区之一，被誉为“人间天堂”，灵隐寺是中国著名的佛教寺庙之一，千岛湖和钱塘江是中国著名的自然风景区之一。

    杭州还拥有丰富的人文资源，被誉为“人间天堂”的杭州西湖、灵隐寺、千岛湖、钱塘江等景点，以及宋城、南宋御街等历史文化景点，都是游客前来杭州旅游的热门景点。

    response3:
    杭州是中国著名的美食城市之一，有许多特色美食和传统菜肴。以下是一些杭州的著名美食:

    1. 西湖醋鱼：这是杭州最著名的菜肴之一，鱼肉鲜美，入口即化，佐以香醋、糖、姜丝等调料，口感酸甜适中。

    2. 龙井虾仁：以当地特产的龙井茶为佐料，将鲜嫩的虾仁炒制而成，口感清香可口。

    3. 灌汤包：又称小笼包，是杭州的传统点心之一。包子的皮软馅鲜，汤汁鲜美，非常受欢迎。

    4. 姜母鸭：这是一道杭帮菜，以鸭肉、姜母、葱等调料烹制而成，口感鲜美。

    5. 老字号小吃：杭州还有很多老字号小吃店，如胡同口烤肉串、孔府家宴、宋嫂鱼羹等，是当地居民和游客的美食选择。

    此外，杭州还有许多特色小吃，如粽子、臭豆腐、糯米鸡、肉夹馍、鸭血粉丝汤等，让人垂涎欲滴。
    '''
```