# ChatGLM2-6B

## 模型描述

ChatGLM**2**-6B 是开源中英双语对话模型 [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM**2**-6B引入了新特征：**更强大的性能**、**更长的上下文**、**更高效的推理**、**更开放的协议**。

## 仓库介绍

`chatGLM2-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm2`

    ```bash
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
    glm2
        ├── run_glm2_6b_fintune.yaml  # 全量微调启动配置
        └── run_glm2_6b_lora.yaml     # lora低参微调启动配置
    ```

## 环境要求

- 硬件：Ascend 910A
- MindSpore：2.0

推理可在单机单卡上完成部署

全量微调训练需要最少单机8卡，Lora微调训练最少需要1卡

## 基线

测试环境同上述环境要求

### 性能

|          | data parallel | model parallel | pipeline parallel | batch size | sink size | sequence length | accumulate | per step time (ms) | tokens/s/p  | 优化器并行 | 重计算 | Memory (GB) |
| -------- | ------------- | -------------- | ----------------- | ---------- | --------- | --------------- | ---------- | ------------------ | ----------- | ---------- | ------ | ----------- |
| 全量微调 | 8             | 1              | 1                 | 8          | 4         | 193             | 1          | 1894               | 815.2059134 | True       | True   | 25.2        |
| LoRA微调 | 4             | 1              | 1                 | 8          | 4         | 193             | 1          | 476                | 3243.697479 | False      | False  | 22.38       |

### 评估指标

|          | rouge-1            | rouge-2           | rouge-l            | bleu-4            |
| -------- | ------------------ | ----------------- |--------------------| ----------------- |
| 全量微调 | 30.784298224299064 | 7.073415046728972 | 24.773958598130843 | 7.466147757009345 |
| LoRA微调 | 31.05639289719626  | 7.1753861682243   | 24.229674859813084 | 7.229435140186916 |

## ChatGLM2-6B推理

> 需开发者提前pip安装。具体接口说明请参[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

### AutoClass推理

可以使用AutoClass接口，通过模型名称获取相应的模型/tokenizer实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/glm2`

首次运行pipeline推理时需要进行模型编译，需等待一段时间

```python
from mindformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("glm2_6b")
model = AutoModel.from_pretrained("glm2_6b")

query = "你好"

prompted_inputs = tokenizer.build_prompt(query)
input_tokens = tokenizer([prompted_inputs])

outputs = model.generate(input_tokens["input_ids"], max_length=100)
response = tokenizer.decode(outputs)[0]
print(response)
```

### pipeline推理

也可以不实例化构造模型，直接通过指定任务模型与模型名的方式进行pipeline的构造

```python
>>> from mindformers import pipeline, TextGenerationPipeline
>>> task_pipeline = pipeline(task='text_generation', model='glm2_6b', max_length=2048)
>>> task_pipeline('你好')
[{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
>>> pipeline = TextGenerationPipeline(model='glm2_6b', max_length=2048)
>>> pipeline("你好")
[{'text_generation_text': ['你好，我是 ChatGLM2-6B， 一个人工智能助手。我背后使用的模型是 GLM2-6B， 是一种大型语言模型， 具有超过 2000 亿参数，支持多种任务。']}]
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据处理

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，目录结构为

```shell
AdvertiseGen
  ├── train.json
  └── dev.json
```

将任务配置文件 `configs/glm2/run_glm2_6b_*.yaml` 中的 `==== dataset config ====` 部分替换成：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 2
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM2Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 64
  max_target_length: 128
  ignore_pad_token_for_loss: True
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

train_dataset_task:
  type: KeyWordGenDataset
  dataset_config: *train_dataset

eval_dataset: &eval_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/dev.json"
    shuffle: False
    phase: "eval"
    version: 2
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM2Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  max_source_length: 256
  max_target_length: 256
  ignore_pad_token_for_loss: True
  input_columns: ["input_ids", "labels"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

eval_dataset_task:
  type: KeyWordGenDataset
  dataset_config: *eval_dataset
```

### 生成HCCL文件

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

```shell
# step1：机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

> 注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

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

### 全参微调

#### run_mindformers脚本启动全参微调

全参微调使用 `configs/glm2/run_glm2_6b.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/run_glm2_6b.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/run_glm2_6b.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

启动全参微调脚本：

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/glm2/run_glm2_6b.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的glm2/run_glm2_6b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

> 注：由于GLM2_6B的模型较大，无法在单卡上运行，此处仅提供分布式启动脚本

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

### LoRA低参微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象
因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象

#### run_mindformers脚本启动LoRA低参微调

使用LoRA算法进行低参微调时，使用 `configs/glm2/run_glm2_6b_lora.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm2/run_glm2_6b_lora.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm2/run_glm2_6b_lora.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 启动LoRA低参微调脚本(1卡)：

执行命令：

```shell
cd scripts
# Usage Help: bash run_stanalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS]
bash run_standalone.sh ../configs/glm2/run_glm2_6b_lora.yaml 0 finetune
```

训练的log日志路径：mindformers/scripts/mf_standalone/

checkpoint存储路径：mindformers/scripts/mf_standalone/output/checkpoint

#### Trainer高阶接口启动LoRA低参微调

示例脚本如下，需要指定训练数据集路径和微调权重。

```python
from mindformers import Trainer
trainer = Trainer(task="text_generation", model="glm2_6b", pet_method="lora",
                  train_dataset="/path/to/AdvertiseGen/train.json")
trainer.finetune(finetune_checkpoint="glm2_6b")
```

### 微调后推理

#### 推理样例脚本

下面提供一个模型推理样例脚本 `infer.py`

```python
from mindformers import AutoConfig, AutoModel, AutoTokenizer
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

config = AutoConfig.from_pretrained("glm2_6b")
config.checkpoint_name_or_path = "/path/to/glm2_6b_finetune.ckpt"
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("glm2_6b")

inputs = tokenizer(tokenizer.build_prompt("你好"))["input_ids"]
print(inputs)
print(tokenizer.decode(inputs))
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
inputs = tokenizer(tokenizer.build_prompt("请介绍一下华为"))["input_ids"]
print(inputs)
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
inputs = tokenizer(tokenizer.build_prompt("晚上睡不着应该怎么办"))["input_ids"]
print(inputs)
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
inputs = tokenizer(tokenizer.build_prompt("类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"))["input_ids"]
print(inputs)
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))
```

## 评估

### 模型权重文件合一

微调所得到的权重文件为根据模型切分策略切分后的权重，我们需要手动将切分权重合一，以用于评估和推理

1. 获取模型切分策略文件：
   在执行全参微调脚本时，模型完成编译后，将会在运行路径下，生成名为 `ckpt_strategy.ckpt` 的切分策略文件，该文件将用于第二步模型合成

2. MindSpore提供了根据切分策略转换模型权重切分的接口，[mindspore.transform_checkpoints](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.transform_checkpoints.html)，执行以下python脚本，将8份模型文件合成一份

    ```python
    from mindspore import transform_checkpoints
    transform_checkpoints(
        src_checkpoints_dir="./output/checkpoint/", # 原切分权重文件夹
        dst_checkpoints_dir="./target_checkpoint/", # 目标路径
        ckpt_prefix="glm2-6b", # .ckpt文件前缀名
        src_strategy_file="ckpt_stragery.ckpt", # 步骤1中的切分策略文件路径
        dst_strategy_file=None # None表示不切分，权重合一
    )
    ```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

### 使用全参微调权重

#### run_mindformers启动eval

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b.yaml` glm2模型推理配置，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b.yaml --run_mode eval --load_checkpoint /path/to/glm2_6b_finetune.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/ --device_id 0
```

> 注：使用离线生成数据方式时，将 `eval_dataset_dir` 一项指向`.mindrecord`文件，如 `/path/to/data/AdvertiseGen/adgen_dev.mindrecord`。

各项参数：

- `config`: 指定用于评估的配置文件名称，此处为`configs/glm2/run_glm2_6b.yaml`
- `run_mode`: 指定执行模式，此为`eval`，表示为评估模式
- `load_checkpoint`: 指定要加载的checkpoint路径，此处为`/path/to/glm2_6b_finetune.ckpt`，替换为需加载的权重的真实路径
- `eval_dataset_dir`: 评估数据集的路径
- `device_id`: 指定要使用的设备编号（从0开始）

评估完成后会打印评估指标 `bleu-4`、`rouge-1`、`rouge-2`、`rouge-l`

> 注：由于默认评估指标的获取方式为生成完整文本后与预期文本做比较，评估速度将受限于模型大小与文本生成速度，评估流程可能较为缓慢

#### Trainer高阶接口启动eval

与上文类似：

```bash
from mindformers import Trainer, ChatGLM2Config, ChatGLM2ForConditionalGeneration

# 开启增量推理使评估速度更快
config = ChatGLM2Config(use_past=True)
model = ChatGLM2ForConditionalGeneration(config)
trainer = Trainer(task="text_generation", model=model,
                  eval_dataset="/path/to/AdvertiseGen/dev.json")
trainer.evaluate(eval_checkpoint="/path/to/glm2_6b_finetune.ckpt")
```

### 使用LoRA低参微调权重

#### run_mindformers启动lora eval

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm2/run_glm2_6b_lora.yaml` glm2_lora模型推理配置，此配置可用于lora模型，修改其中`model`字段下`model_config`中`use_past: True`开启增量推理使评估速度更快

```bash
python run_mindformer.py --config configs/glm2/run_glm2_6b_lora.yaml --run_mode eval --load_checkpoint /path/to/glm2_6b_lora.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/ --device_id 0
```

各项参数同上，路径需替换为实际路径

#### Trainer高阶接口启动lora eval

与上文类似：

```bash
from mindformers import Trainer, ChatGLM2Config, ChatGLM2WithLora
from mindformers.pet.pet_config import LoraConfig

# 开启增量推理使评估速度更快
config = ChatGLM2Config(use_past=True)
config.pet_config = LoraConfig()
model = ChatGLM2WithLora(config)
trainer = Trainer(task="text_generation", model=model,
                  eval_dataset="/path/to/AdvertiseGen/dev.json")
trainer.evaluate(eval_checkpoint="/path/to/glm2_6b_lora.ckpt")
```

## 模型权重转化

本仓库中的`glm2`来自于HuggingFace的 [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)，基于下述的步骤获取：

1. 克隆chatglm2-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm2-6b
   ```

2. 执行 python 脚本，合并模型权重。

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

3. 执行转换脚本，得到转换后的输出文件`glm2_6b.ckpt`。

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
