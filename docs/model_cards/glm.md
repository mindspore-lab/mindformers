# ChatGLM

## 模型描述

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考清华的[博客](https://chatglm.cn/blog)。在此仓中，提供ChatGLM6B的推理和微调能力。

## 仓库介绍

`chatGLM6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm`

    ```bash
    glm
        ├── __init__.py
        ├── attention.py            # 自注意力
        ├── chatglm_6b_tokenizer.py # tokenizer
        ├── glm_config.py           # 模型配置项
        ├── glm.py                  # 模型实现
        └── layers.py               # glm 层定义
    ```

2. 模型配置：`configs/glm`

    ```bash
    glm
        ├── run_glm_6b_fintune.yaml     # 全量微调启动配置
        ├── run_glm_6b_lora.yaml        # lora低参微调启动配置
        ├── run_glm_6b_infer.yaml       # 推理启动配置
        └── run_glm_6b_lora_infer.yaml  # lora模型推理启动配置
    ```

## 环境要求

- 硬件：Atlas 800
- MindSpore：2.0.0rc1 / 1.10.1
- MindFormers版本：dev

推理可在单机单卡上完成部署

全量微调训练需要最少单机8卡，Lora微调训练最少需要1卡

## ChatGLM6B推理

> 需开发者提前pip安装。具体接口说明请参[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

### AutoClass推理

可以使用AutoClass接口，通过模型名称获取相应的模型/tokenizer实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/glm`

首次运行pipeline推理时需要进行模型编译，需等待一段时间

```python
>>> import mindspore; mindspore.set_context(mode=0, device_id=0)
>>> from mindformers import AutoModel, AutoTokenizer, TextGenerationPipeline
>>> model = AutoModel.from_pretrained("glm_6b_chat")
>>> tokenizer = AutoTokenizer.from_pretrained("glm_6b")
>>> pipeline = TextGenerationPipeline(model, tokenizer, max_length=2048)
>>> pipeline("你好")
[{'text_generation_text': ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。']}]
```

> 注：`AutoModel.from_pretrained()` 接口当前支持 `glm_6b` 和 `glm_6b_chat` 两类模型，前者为通用模型，后者具备推理加速特性，仅用于推理，两者共享权重，在推理场景下建议使用后者，以获得更快的推理体验

### pipeline推理

也可以不实例化构造模型，直接通过指定任务模型与模型名的方式进行pipeline的构造

pipeline中，也可以使用 `glm_6b_chat` 模型加速推理

```python
>>> import mindspore; mindspore.set_context(mode=0, device_id=0)
>>> from mindformers import pipeline
>>> task_pipeline = pipeline(task='text_generation', model='glm_6b_chat', max_length=2048)
>>> task_pipeline('你好')
[{'text_generation_text': ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。']}]
```

### 基于API接口的推理

可使用如下`chat_glm.py`脚本：

```python
import time
import mindspore as ms
import numpy as np
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response

config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_sample_acceleration=True,
)

def chat_glm():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=7)
    model = GLMChatModel(config)
    ms.load_checkpoint("./checkpoint_download/glm/glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('./checkpoint_download/glm/ice_text.model')

    prompts = ["你好", "请介绍一下华为", "用python写一个快排"]
    history = []
    for query in prompts:
        input_ids = tokenizer(query)['input_ids']

        start_time = time.time()
        outputs = model.generate(input_ids, max_length=config.max_decode_length, do_sample=False)
        end_time = time.time()
        print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')

        response = tokenizer.decode(outputs)
        response = process_response(response[0])
        print(response)


if __name__ == "__main__":
    chat_glm()
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据处理（在线加载与离线生成二选一，优先推荐在线加载方式）

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。数据集可选离线生成 `Mindrecord` 或者实时生成两种方式，两种方式选其一即可。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 任意目录下

#### 1. 在线加载

将任务配置文件 `configs/glm/run_glm_6b_*.yaml` 中的 `==== dataset config ====` 部分中的 `dataset_dir` 指向 `*.json` 文件，`vocab_file` 指向词表文件，**跳过** “2. 离线生成” 步骤。

#### 2. 离线生成

将任务配置文件 `configs/glm/run_glm_6b_*.yaml` 中的 `==== dataset config ====` 部分替换成：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids", "labels", "position_ids", "attention_mask"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 1
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset

eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
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
  type: CausalLanguageModelDataset
  dataset_config: *eval_dataset
```

使用 `mindformers/tools/dataset_preprocess/glm/adgen_dataset.py` 脚本将数据集处理成mindrecord格式。

执行命令生成训练数据集：

```bash
python adgen_dataset.py \
    --input_file /path/to/AdvertiseGen/train.json \
    --vocab_file /path/to/ice_text.model\
    --output_file /path/to/AdvertiseGen/train_0604_128.mindrecord \
    --max_source_length 64 \
    --max_target_length 64 \
    --mode train
```

执行命令生成评估数据集：

```bash
python adgen_dataset.py \
    --input_file /path/to/AdvertiseGen/dev.json \
    --vocab_file /path/to/ice_text.model \
    --output_file /path/to/AdvertiseGen/eval_0604_256.mindrecord \
    --max_source_length 256 \
    --max_target_length 256 \
    --mode eval
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

全参微调使用 `configs/glm/run_glm_6b_finetune.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm/run_glm_6b_finetune.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm/run_glm_6b_finetune.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

启动全参微调脚本：

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/glm/run_glm_6b_finetune.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的glm/run_glm_6b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

> 注：由于GLM6B的模型较大，无法在单卡上运行，此处仅提供分布式启动脚本

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

#### Trainer高阶接口启动全参微调

下面提供一个使用高阶接口进行GLM模型开发的样例脚本 `task.py`，用户可参照以下步骤熟悉如何使用高阶接口进行GLM模型的训练开发

```python
import argparse

from mindformers import Trainer, TrainingArguments
from mindformers import init_context, ContextConfig, ParallelContextConfig

def context_init(use_parallel=False, optimizer_parallel=False):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    rank_id, device_num = init_context(use_parallel=use_parallel,
                                       context_config=context_config,
                                       parallel_config=parallel_config)

def main(use_parallel=False,
         run_mode='train',
         task='text_generation',
         model_type='glm_6b',
         checkpoint_path='./glm_6b.ckpt',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='你好',
         batch_size=4,
         dp=1, mp=1, pp=1, micro_size=1, op=False):
    if use_parallel.lower() == "true":
        use_parallel = True
    else:
        use_parallel = False
    # 环境初始化
    context_init(use_parallel, op)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=batch_size, learning_rate=5e-5, warmup_steps=100, sink_mode=True, sink_size=4)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task, model=model_type, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    task.set_parallel_config(data_parallel=dp,
                             model_parallel=mp,
                             pipeline_stage=pp,
                             micro_batch_num=micro_size)
    if run_mode == 'train':
        # 训练
        task.train()
    elif run_mode == 'finetune':
        # 微调
        task.finetune(checkpoint_path)
    elif run_mode == 'eval':
        # 评估
        task.evaluate(checkpoint_path)
    elif run_mode == 'predict':
        # 推理，仅支持单卡推理
        assert use_parallel == False, "only support predict under stand_alone mode."
        result = task.predict(input_data=predict_data)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='train', required=True, help='set run mode for model.')
    parser.add_argument('--use_parallel', default=False, help='open parallel for model.')
    parser.add_argument('--task', default='text_generation', required=True, help='set task type.')
    parser.add_argument('--model_type', default='glm_6b', required=True, help='set model type.')
    parser.add_argument('--checkpoint_path', default=None, help='set checkpoint path.')
    parser.add_argument('--train_dataset', default=None, help='set train dataset.')
    parser.add_argument('--eval_dataset', default=None, help='set eval dataset.')
    parser.add_argument('--batch_size', default=4, help='batch size of dataset.')
    parser.add_argument('--data_parallel', default=1, type=int,help='set data parallel number. Default: None')
    parser.add_argument('--model_parallel', default=1, type=int, help='set model parallel number. Default: None')
    parser.add_argument('--pipeline_parallel', default=1, type=int, help='set pipeline parallel number. Default: None')
    parser.add_argument('--micro_size', default=1, type=int, help='set micro batch number. Default: None')
    parser.add_argument('--optimizer_parallel', default=False, type=bool, help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()
    print(args)
    main(run_mode=args.run_mode,
         task=args.task,
         use_parallel=args.use_parallel,
         model_type=args.model_type,
         checkpoint_path=args.checkpoint_path,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         batch_size=int(args.batch_size),
         dp=args.data_parallel,
         mp=args.model_parallel,
         pp=args.pipeline_parallel,
         micro_size=args.micro_size,
         op=args.optimizer_parallel)
```

因GLM模型过大，**无法在单卡上启动训练**，因此需要**通过分布式脚本拉起多卡训练任务**

在此提供 `run_distribute_single_node.sh` 单机多卡标准启动脚本，用户可用其拉起分布式训练

```bash
#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 4 ]
then
  echo "Usage Help: bash run_distribute_single_node.sh [EXECUTE_ORDER] [RANK_TABLE_PATH]  [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices In Single Machine"
  exit 1
fi

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

EXECUTE_ORDER=$1
RANK_TABLE_PATH=$(check_real_path $2)
DEVICE_RANGE=$3

DEVICE_RANGE_LEN=${#DEVICE_RANGE}
DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
PREFIX=${DEVICE_RANGE%%","*}
INDEX=${#PREFIX}
START_DEVICE=${DEVICE_RANGE:0:INDEX}
END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}

if [ ! -f $RANK_TABLE_PATH ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
exit 1
fi


if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
    echo "error: start_device=$START_DEVICE is not a number"
exit 1
fi

if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
    echo "error: end_device=$END_DEVICE is not a number"
exit 1
fi

ulimit -u unlimited

export RANK_SIZE=$4
export RANK_TABLE_FILE=$RANK_TABLE_PATH

shopt -s extglob

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((i-START_DEVICE))
    mkdir -p ./output/log/rank_$RANK_ID
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
done

shopt -u extglob
```

全参微调分布式拉起命令(8卡)：

```bash
bash run_distribute_single_node.sh "python task.py --task text_generation --model_type glm_6b --checkpoint_path ./glm_6b.ckpt --train_dataset ./train --run_mode finetune --use_parallel True --data_parallel 1 --model_parallel 8" /path/to/hccl_8p_xxx.json '[0,8]' 8
```

参数含义:

- `"python task.py --task text_generation --model_type glm_6b --checkpoint_path ./glm_6b.ckpt --train_dataset ./train --run_mode finetune --use_parallel True --data_parallel 1 --model_parallel 8"`: 需执行的命令，此处完整输入task.py的启动命令

python task.py 各项参数含义：

- `task`: 需运行的训练任务，此处为 `text_generation` 文本生成任务
- `model_type`: 模型类型，此处选择 `glm_6b` 模型
- `checkpoint_path`: 权重路径，此处替换为实际需加载的权重路径
- `train_dataset`: 训练数据集路径，替换为实际路径
- `run_mode`: 启动模式，train——训练，finetune——微调，eval——评估，predict——推理，此处选择 `finetune`
- `use_parallel`: 是否使用多卡并行训练，此处为 `True`
- `data_parallel`: 数据并行数，此处为1表示不开启
- `model_parallel`: 模型并行数，此处为8表示8卡并行

bash 脚本其余参数：

- `/path/to/hccl_4p_xxx.json`: rank table file路径，替换为之前准备的rank table file的实际路径
- `'[0,8]'`: 占用的卡范围，0包含，8不包含，表示使用 `0~7` 8张卡并行训练
- `8`: rank size，一共使用了多少张卡，此处为8

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

### LoRA低参微调

全参微调能够在微调数据集上取得良好效果，但存在遗忘预训练知识的现象
因此推荐使用低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，在微调数据集上取得良好效果的同时，缓解模型遗忘现象

#### run_mindformers脚本启动LoRA低参微调

使用LoRA算法进行低参微调时，使用 `configs/glm/run_glm_6b_lora.yaml` 配置文件，该配置文件包含了lora低参微调算法所需的配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `mindformers/configs/glm/run_glm_6b_lora.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm/run_glm_6b_lora.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 启动LoRA低参微调脚本(1卡)：

执行命令：

```shell
cd scripts
# Usage Help: bash run_standalone.sh [CONFIG_PATH] [DEVICE_ID] [RUN_STATUS]
bash run_standalone.sh ../configs/glm/run_glm_6b_lora.yaml 0 finetune
```

训练的log日志路径：mindformers/scripts/mf_standalone/

checkpoint存储路径：mindformers/scripts/mf_standalone/output/checkpoint

#### 启动LoRA低参微调脚本(4卡)：

> 注：如果需要进行多卡训练，则需要对`glm/run_glm_6b_lora.yaml`配置文件对应参数进行修改，以4卡为例，需要重新生成4卡的HCCL文件：

```shell
data_parallel: 4
```

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_4_0123_xxx.json ../configs/glm/run_glm_6b_lora.yaml '[0,4]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明：

对比全参微调启动方式，仅将 `CONFIG_PATH` 项修改为configs文件夹下面的 `glm/run_glm_6b_lora.yaml` 配置文件，表示使用该接口进行

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

#### Trainer高阶接口启动LoRA低参微调

可复用全参微调部分所提供的 `task.py` 和 `run_distribute_single_node.sh` 脚本

4卡分布式启动命令：

```bash
bash run_distribute_single_node.sh "python task.py --task text_generation --model_type glm_6b_lora --checkpoint_path ./glm_6b.ckpt --train_dataset ./train --run_mode finetune --use_parallel True --data_parallel 4 --model_parallel 1" /path/to/hccl_4p_xxx.json '[0,4]' 4
```

参数说明：对比全参微调启动，仅改动以下几点：

- `model_type`: 指定模型类型为 `glm_6b_lora`，表示使用低参微调算法
- `data_parallel`: 4卡启动，数据并行改为4
- `/path/to/hccl_4p_xxx.json`: 使用4卡的rank_table_file
- `'[0,4]' 4`: 使用0~3共4卡

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

1卡启动命令：

```shell
python task.py --task text_generation --model_type glm_6b_lora --checkpoint_path ./glm_6b.ckpt --train_dataset ./train --run_mode finetune --use_parallel False --data_parallel 1 --model_parallel 1
```

### 多机多卡微调训练

多机多卡启动
首先在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

将不同机器上生成的RANK_TABLE_FILE文件中的server_list合并，server_count设为机器数，rank_id顺序增加，并保证不同机器上的RANK_TABLE_FILE相同；

在多机上同时拉起任务，拉起方式为

cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_MODE RANK_SIZE

#### 参数说明

- RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
- CONFIG_PATH: 为configs文件夹下面的gpt2/run_gpt2*.yaml配置文件
- DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
- RUN_MODE: 为任务运行状态，支持关键字 train 预训练、predict（文本生成预测）
- RANK_SIZE: 总运行卡数

#### 4机32卡参考RANK_TABLE_FILE样例

```text
{
  "version": "1.0",
  "server_count": "4",
  "server_list": [
    {
      "server_id": "10.155.111.140",
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
    },
    {
      "server_id": "10.155.111.141",
      "device": [
        {"device_id": "0","device_ip": "192.1.27.8","rank_id": "8"},
        {"device_id": "1","device_ip": "192.2.27.8","rank_id": "9"},
        {"device_id": "2","device_ip": "192.3.27.8","rank_id": "10"},
        {"device_id": "3","device_ip": "192.4.27.8","rank_id": "11"},
        {"device_id": "4","device_ip": "192.1.27.9","rank_id": "12"},
        {"device_id": "5","device_ip": "192.2.27.9","rank_id": "13"},
        {"device_id": "6","device_ip": "192.3.27.9","rank_id": "14"},
        {"device_id": "7","device_ip": "192.4.27.9","rank_id": "15"}],
      "host_nic_ip": "reserve"
    },
    {
      "server_id": "10.155.111.142",
      "device": [
        {"device_id": "0","device_ip": "192.1.27.10","rank_id": "16"},
        {"device_id": "1","device_ip": "192.2.27.10","rank_id": "17"},
        {"device_id": "2","device_ip": "192.3.27.10","rank_id": "18"},
        {"device_id": "3","device_ip": "192.4.27.10","rank_id": "19"},
        {"device_id": "4","device_ip": "192.1.27.11","rank_id": "20"},
        {"device_id": "5","device_ip": "192.2.27.11","rank_id": "21"},
        {"device_id": "6","device_ip": "192.3.27.11","rank_id": "22"},
        {"device_id": "7","device_ip": "192.4.27.11","rank_id": "23"}],
      "host_nic_ip": "reserve"
    },
    {
      "server_id": "10.155.111.143",
      "device": [
        {"device_id": "0","device_ip": "192.1.27.12","rank_id": "24"},
        {"device_id": "1","device_ip": "192.2.27.12","rank_id": "25"},
        {"device_id": "2","device_ip": "192.3.27.12","rank_id": "26"},
        {"device_id": "3","device_ip": "192.4.27.12","rank_id": "27"},
        {"device_id": "4","device_ip": "192.1.27.13","rank_id": "28"},
        {"device_id": "5","device_ip": "192.2.27.13","rank_id": "29"},
        {"device_id": "6","device_ip": "192.3.27.13","rank_id": "30"},
        {"device_id": "7","device_ip": "192.4.27.13","rank_id": "31"}],
      "host_nic_ip": "reserve"
    }
  ],
  "status": "completed"
}
```

#### 任务拉起命令示例

```shell
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/glm/run_glm_6b_lora.yaml [0,8] train 32
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/glm/run_glm_6b_lora.yaml [8,16] train 32
# 第三台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/glm/run_glm_6b_lora.yaml [16,24] train 32
# 第四台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/glm/run_glm_6b_lora.yaml [24,32] train 32
```

### 微调后推理

#### 推理样例脚本

下面提供一个模型推理样例脚本 `infer.py`

```python
import time
import mindspore as ms
import numpy as np
import argparse
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response
from mindformers.pet.pet_config import LoraConfig
from mindformers.pet import get_pet_model

parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', default=1024, type=int, help='Which device to run service.')
parser.add_argument('--device_id', default=0, type=int, help='Which device to run service.')
parser.add_argument('--checkpoint_path', type=str, default='/path/chatglm6b.ckpt', help='Checkpoint file to load on.')
parser.add_argument('--vocab_path', type=str, default='/path/ice_text.model', help='Vocab file to load on.')
parser.add_argument('--is_lora', type=str, default='false',help='Whether is lora model.')

args = parser.parse_args()

if args.is_lora.lower() == "true":
    is_lora = True
else:
    is_lora = False

config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_sample_acceleration=True,
)

pet_config = LoraConfig(
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules = '.*query_key_value*'
)


def chat_glm():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
    model = GLMChatModel(config)
    if is_lora:
       config.pet_config = pet_config
       model = get_pet_model(model, pet_config)
    ms.load_checkpoint(args.checkpoint_path, model)
    tokenizer = ChatGLMTokenizer(args.vocab_path)

    inputs = ["你好",
              "请介绍一下华为",
              "用Python写一个快排",
              "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"]

    for query in inputs:
        input_ids = tokenizer(query)['input_ids']

        start_time = time.time()
        outputs = model.generate(input_ids, max_length=config.max_decode_length, do_sample=False)
        end_time = time.time()
        print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')

        response = tokenizer.decode(outputs)
        response = process_response(response[0])
        print(response)


if __name__ == "__main__":
    chat_glm()
```

#### 运行命令

```shell
python infer.py --seq_length 1024 --device_id 0  --checkpoint_path /path/chatglm6b.ckpt --vocab_path /path/ice_text.model --is_lora True
```

参数说明：

- `seq_length`: 用于指定推理输入长度
- `device_id`: 指定推理在那张设备运行
- `checkpoint_path`: 指定训练出来的模型文件路径用于推理
- `vocab_path`: 模型词表
- `is_lora`: 用于区分是否是lora模型，设置为true表示为lora微调训练模型

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
        ckpt_prefix="glm-6b", # .ckpt文件前缀名
        src_strategy_file="ckpt_stragery.ckpt", # 步骤1中的切分策略文件路径
        dst_strategy_file=None # None表示不切分，权重合一
    )
    ```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

### 使用全参微调权重

#### run_mindformers启动eval

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm/run_glm_6b_infer.yaml` glm模型推理配置，此配置下评估速度更快

```bash
python run_mindformer.py --config configs/glm/run_glm_6b_infer.yaml --run_mode eval --load_checkpoint /path/to/glm_6b.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/ --device_id 0
```

> 注：使用离线生成数据方式时，将 `eval_dataset_dir` 一项指向`.mindrecord`文件，如 `/path/to/data/AdvertiseGen/adgen_dev.mindrecord`。

各项参数：

- `config`: 指定用于评估的配置文件名称，此处为`configs/glm/run_glm_6b_infer.yaml`
- `run_mode`: 指定执行模式，此为`eval`，表示为评估模式
- `load_checkpoint`: 指定要加载的checkpoint路径，此处为`/path/to/glm_6b.ckpt`，替换为需加载的权重的真实路径
- `eval_dataset_dir`: 评估数据集的路径
- `device_id`: 指定要使用的设备编号（从0开始）

评估完成后会打印评估指标 `bleu-4`、`rouge-1`、`rouge-2`、`rouge-l`

> 注：由于默认评估指标的获取方式为生成完整文本后与预期文本做比较，评估速度将受限于模型大小与文本生成速度，评估流程可能较为缓慢

#### Trainer高阶接口启动eval

仍然可复用 `task.py` 脚本，启动命令：

```bash
python task.py --task text_generation --model_type glm_6b_chat --checkpoint_path /path/to/glm_6b.ckpt --eval_dataset /path/to/data/AdvertiseGen/ --run_mode eval --batch_size 1
```

> 1. 当前评估时，batch_size需为1，否则评估速度下降严重
> 2. 使用离线生成数据方式时，将 `eval_dataset` 一项指向`.mindrecord`文件，如 `/path/to/data/AdvertiseGen/adgen_dev.mindrecord`。

### 使用LoRA低参微调权重

#### run_mindformers启动lora eval

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm/run_glm_6b_lora_infer.yaml` glm_lora模型推理配置，此配置可用于lora模型，并且评估速度更快

```bash
python run_mindformer.py --config configs/glm/run_glm_6b_lora_infer.yaml --run_mode eval --load_checkpoint /path/to/glm_6b_lora.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/ --device_id 0
```

各项参数同上，路径需替换为实际路径

> 使用离线生成数据方式时，将 `eval_dataset_dir` 一项指向`.mindrecord`文件，如 `/path/to/data/AdvertiseGen/adgen_dev.mindrecord`。

#### Trainer高阶接口启动lora eval

仍然可复用 `task.py` 脚本，启动命令：

```bash
python task.py --task text_generation --model_type glm_6b_lora_chat --checkpoint_path /path/to/glm_6b_lora.ckpt --eval_dataset /path/to/data/AdvertiseGen/ --run_mode eval --batch_size 1
```

> 1. 当前评估时，batch_size需为1，否则评估速度下降严重
> 2. 使用离线生成数据方式时，将 `eval_dataset_dir` 一项指向`.mindrecord`文件，如 `/path/to/data/AdvertiseGen/adgen_dev.mindrecord`。

## 模型权重转化

本仓库中的`glm`来自于HuggingFace的[chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)，基于下述的步骤获取：

1. 克隆chatglm-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm-6b
   ```

2. 执行转换脚本，得到转换后的输出文件`ms_glm_6b.ckpt`。

   ```shell
   python mindformers/models/glm/convert_weight.py --pt_ckpt_path "replace your ptroch pth path" --ms_ckpt_path ./ms_glm_6b.ckpt
   ```

   ```shell
   # 参数说明
   pt_ckpt_path: huggingface权重保存目录下的任意权重bin文件,根据该文件路径读取目录下全部权重
   ms_ckpt_path: 权重保存文件名，可以指定自定义保存路径
   ```

