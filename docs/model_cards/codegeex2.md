# CodeGeeX2-6B

## 模型描述

CodeGeeX**2**-6B 是多语言代码生成模型 CodeGeeX的第二代版本。不同于一代CodeGeeX，CodeGeeX2是基于ChatGLM2结构加入代码预训练实现，得益于ChatGLM2的更优性能，CodeGeeX2在多项指标上取得性能提升。

## 模型性能

- 基于Atlas 800T A2

|                                   config                                    |      task       | Datasets | metric |   score    | [train performance](#预训练) | [predict performance](#基于pipeline的推理) |
|:---------------------------------------------------------------------------:| :-------------: |:--------:| :----: | :--------: |:-------------------------:|:-------------------------------------:|
|   [codegeex2_6b](../../configs/codegeex2/run_codegeex2_6b_finetune.yaml)    | text_generation |  CodeAlpaca   |   -    |     -      |      1421 tokens/s/p      |   20.17 tokens/s/p (use past True)    |
| [codegeex2_6b](../../configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml) | text_generation |  CodeAlpaca   |   -    |     -      |     2167.2 tokens/s/p     |   20.31 tokens/s/p (use past True)    |

## 仓库介绍

`codegeex2-6B` 基于 `mindformers` 实现，主要涉及的文件有：

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

2. 模型配置：`configs/codegeex2`

    ```bash
    codegeex2
        ├── run_codegeex2_6b_fintune.yaml  # 全量微调启动配置
        └── run_codegeex2_6b.yaml     # 推理和lite推理配置
    ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

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

本仓库中的`codegeex2`来自于HuggingFace的 [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b)，基于下述的步骤获取：

1. 克隆codegeex2-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/codegeex2-6b
   ```

2. 执行 python 脚本，合并模型权重,模型转换权重需要依赖transformer版本为4.30.2，可参阅[CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b)。

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
   model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)

   with open("pt_model_arch.txt", "w") as fp:
       print(model, file=fp, flush=True)
   with open("pt_ckpt.txt", "w") as fp:
       for name, param in model.named_parameters():
           fp.write(f"{name} {param.shape} {param.dtype}\n")
   torch.save(model.state_dict(), "codegeex2_6b.pth")
   ```

3. 执行转换脚本，得到转换后的输出文件`codegeex2_6b.ckpt`。

   ```python
   import mindspore as ms
   import torch as pt
   from tqdm import tqdm

   pt_ckpt_path = "/path/to/codegeex2_6b.pth"
   pt_param = pt.load(pt_ckpt_path)

   type_map = {"torch.bfloat16": "ms.float32",
                "torch.float32": "ms.float32"}
   ms_param = []
   with open("./check_pt_ckpt.txt", "w") as fp:
       for k, v in tqdm(pt_param.items()):
           if v.dtype is pt.bfloat16:
               v = v.to(dtype = pt.float32)
           if "word_embeddings.weight" in k:
               k = k.replace("word_embeddings.weight", "embedding_table")
           fp.write(f"{k} {v.shape} {v.dtype}\n")
           ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

   ms.save_checkpoint(ms_param, "/path/to/codegeex2_6b.ckpt")
   ```

4. 也可获取MindFormers提供的已转换权重

   可通过from_pretrained接口下载，也可直接从下面的链接获取

   [codegeex2_6b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/codegeex2/codegeex2_6b.ckpt)

   [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/codegeex2/tokenizer.model)

### [模型权重切分与合并](../feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

## 基于API的快速使用

### AutoClass推理

可以使用AutoClass接口，通过模型名称获取相应的模型/tokenizer实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/codegeex2`

首次运行pipeline推理时需要进行模型编译，需等待一段时间

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("codegeex2_6b")
model = AutoModel.from_pretrained("codegeex2_6b")

prompt = "#language: Python\n# write a bubble sort function\n"
inputs = tokenizer.encode(prompt)
outputs = model.generate(inputs, max_length=256, top_k=1)
response = tokenizer.decode(outputs[0])
print(response)
```

**注：快速使用仅限单卡，该示例支持6B模型。**

### 基于Trainer的快速训练，微调，评测，推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer
from mindformers import AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
tokenizer = AutoTokenizer.from_pretrained("codegeex2_6b")
trainer = Trainer(task='text_generation',
                  model='codegeex2_6b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset',
                  tokenizer=tokenizer)

# 开启预训练
# trainer.train()

# 开启全量微调
# trainer.finetune()

# 开启推理
predict_result = trainer.predict(input_data="#language: Python\n# write a bubble sort function\n")
# output result is: [{'text_generation_text': ['#language: Python\n# write a bubble sort function\n\ndef bubble_sort(list):\n for i in range(len(list) - 1):\n for j in range(len(list) - 1):\n if list[j] > list[j + 1]:\n list[j], list[j + 1] = list[j + 1], list[j]\n return list\n\n\n print(bubble_sort([5, 2, 1, 8, 4]))']}]
```

**注：使用前请参照微调部分更改数据集设置，多卡请参考[使用高阶接口开发教程](https://mindformers.readthedocs.io/zh_CN/latest/docs/practice/Develop_With_Api.html)。**

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='codegeex2_6b', max_length=500)
pipeline_result = pipeline_task("#language: Python\n# write a bubble sort function\n", top_k=1)
print(pipeline_result)
# output result is: [{'text_generation_text': ['#language: Python\n# write a bubble sort function\n\ndef bubble_sort(list):\n for i in range(len(list) - 1):\n for j in range(len(list) - 1):\n if list[j] > list[j + 1]:\n list[j], list[j + 1] = list[j + 1], list[j]\n return list\n\n\n print(bubble_sort([5, 2, 1, 8, 4]))']}]
```

**注：快速使用仅限单卡，该示例支持6B模型。**
**注：多卡请参考[基于pipeline的推理](#基于pipeline的推理)。**

## 微调

### 数据集准备

数据处理方法同GLM2相同，其组织方式如下：

```json
{
    "PROMPT": "Create an array of 10 elements using java",
    "ANSWER": "int[] array = new int[10];"
}
```

从 [CodeAlpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) 或者 [Hugging Face](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) 下载数据集，并且处理其目录结构为

```shell
CodeAlpaca
  ├── train.json
  └── dev.json
```

处理脚本可以参考：`mindformers/tools/dataset_preprocess/codegeex2/codealpaca_preprocess.py`

将任务配置文件 `configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml` 中的 `==== dataset config ====` 部分替换成：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/CodeAlpaca/train.json"
    shuffle: True
    phase: "train"
    version: 2
    origin_columns: ["PROMPT", "ANSWER"]
  tokenizer:
    type: ChatGLM2Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 1023
  max_target_length: 1024
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
    dataset_dir: "/path/to/CodeAlpaca/dev.json"
    shuffle: False
    phase: "eval"
    version: 2
    origin_columns: ["PROMPT", "ANSWER"]
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

### 全参微调

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单卡微调

**注：在Atlas 800T A2上无法单卡全参微调codegeex2模型。**

#### 单机多卡全参微调

全参微调使用 `configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

启动全参微调脚本：

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的codegeex2/run_codegeex2_6b_finetune_2048.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

#### <span id="jump">多机多卡启动</span>

- step 1. 首先参考单机多卡启动方式，在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件。

```shell
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 合并每台机器上生成的`RANK_TABLE_FILE`。

将不同机器上生成的`RANK_TABLE_FILE`文件拷贝到一起，执行`merge_hccl.py`脚本进行合并，包括server_list合并，`server_count`设为机器数，`rank_id`顺序增加。

```shell
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

- step 4. 根据服务器节点数等信息，修改相应的配置。

```shell
# 以codegeex2-6b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。需注意，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml [0,8] finetune 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/codegeex2/run_codegeex2_6b_finetune_2048.yaml [8,16] finetune 16
```

## 推理

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


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


def main(use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["#language: Python\n# write a bubble sort function\n",
              "#language: Python\n# write a quick sort function\n",
              "#language: Python\n# write a heap sort function\n"]

    # set model config
    model_config = AutoConfig.from_pretrained("codegeex2_6b")
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codegeex2_6b")
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard codegeex2 and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

#### 单卡pipeline推理

```bash
python predict_custom.py
```

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


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


def main(use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["#language: Python\n# write a bubble sort function\n",
              "#language: Python\n# write a quick sort function\n",
              "#language: Python\n# write a heap sort function\n"]

    # set model config
    model_config = AutoConfig.from_pretrained("codegeex2_6b")
    model_config.batch_size = len(inputs)
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("codegeex2_6b")
    # build model from config
    model = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard codegeex2 and load sharded ckpt
        model = Model(model)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids, max_length=model_config.max_decode_length)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

#### 单卡generate推理

```bash
python predict_custom.py
```

### 脚本启动

#### 单卡推理

```bash
python run_mindformer.py --config configs/codegeex2/run_codegeex2_6b.yaml --run_mode predict --predict_data  #language: Python\n# write a bubble sort function\n --use_parallel False
# output result is: [{'text_generation_text': ['#language: Python\n# write a bubble sort function\n\ndef bubble_sort(list):\n for i in range(len(list) - 1):\n for j in range(len(list) - 1):\n if list[j] > list[j + 1]:\n list[j], list[j + 1] = list[j + 1], list[j]\n return list\n\n\n print(bubble_sort([5, 2, 1, 8, 4]))']}]
```

**注**：要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```yaml
# model config
use_past: True          # 开启增量推理
use_moe: False
checkpoint_name_or_path: "codegeex2_6b"
max_decode_length: 1024
top_k: 1
top_p: 1
do_sample: True
```

## Mindspore-Lite 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造的推理引擎 [MindSpore_lite](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　Lite 推理大致分两步：权重转换导出 MindIR -> Lite 推理，接下来分别描述上述两个过程。

### MindIR 导出

1. 修改模型相关的配置文件 configs/codegeex2/run_codegeex2_6b.yaml，其中需要关注这几项：

```yaml
# export
infer:
    prefill_model_path: "codegeex2_export/codegeex2_6b_prefill_seq1024.mindir" # 保存mindir的位置
    increment_model_path: "codegeex2_export/codegeex2_6b_inc_seq1024.mindir"   # 保存mindir的位置
    infer_seq_length: 1024 # 需要保持跟 model-model_config-seq_length 一致
    ge_config_path: "/path/ge_config.ini" # 需要保持跟下文lite.ini一致

# ==== model config ====
model:
  model_config:
    seq_length: 1024
    checkpoint_name_or_path: "/path/to/your/checkpoint"
```

2. 执行export.py，完成模型转换

```bash
python mindformers/tools/export.py --config_path configs/codegeex2/run_codegeex2_6b.yaml
```

### 执行推理

1. 新建推理配置文件：lite.ini

    ```ini
    [ascend_context]
    plugin_custom_ops=All
    provider=ge

    [ge_session_options]
    ge.exec.precision_mode=must_keep_origin_dtype

    [ge_graph_options]
    ge.exec.formatMode=1
    ```

2. 执行命令：

```bash
python run_infer_main.py --device_id 0 --model_name codegeex2_6b --prefill_model_path codegeex2_export/codegeex2_6b_prefill_seq1024_graph.mindir --increment_model_path codegeex2_export/codegeex2_6b_inc_seq1024_graph.mindir --config_path lite.ini --seq_length 1024 --max_length 512 --add_special_tokens True
```

　　输入：

```bash
#language: Python #write a bubble sort function
```

　　输出：

```bash
 ['#language: Python\n# write a bubble sort function\n\ndef bubble_sort(list):\n for i in range(len(list) - 1):\n for j in range(len(list) - 1):\n if list[j] > list[j + 1]:\n list[j], list[j + 1] = list[j + 1], list[j]\n return list\n\n\n print(bubble_sort([5, 2, 1, 8, 4]))']}]
```
