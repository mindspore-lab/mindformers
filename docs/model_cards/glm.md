# ChatGLM6B

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

- 硬件：Ascend 910A
- MindSpore：2.0.0rc1 / 1.10.1
- MindFormers版本：dev

推理可在单机单卡上完成部署

训练需要最少单机8卡

## ChatGLM6B推理

> 需开发者提前pip安装。具体接口说明请参[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

### AutoClass推理

可以使用AutoClass接口，通过模型名称获取相应的模型/tokenizer实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/glm`

首次运行pipeline推理时需要进行模型编译，需等待一段时间

```python
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
    is_npu_acceleration=True,
)

def chat_glm():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=7)
    model = GLMChatModel(config)
    ms.load_checkpoint("./checkpoint_download/glm/glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('./checkpoint_download/glm/ice_text.model')

    prompts = ["你好", "请介绍一下华为"]
    history = []
    for query in prompts:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer(prompt)

        start_time = time.time()
        outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                                    max_length=config.max_decode_length, do_sample=False, top_p=0.7, top_k=1)
        end_time = time.time()
        print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')
        response = tokenizer.decode(outputs)
        response = process_response(response[0])
        history = history + [(query, response)]
        print(response)

if __name__ == "__main__":
    chat_glm()
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据处理

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。数据集可选离线生成 `Mindrecord` 或者实时生成两种方式

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 任意目录下

#### 离线生成

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

#### 在线加载

在线加载数据集的方式目前正在开发中，建议使用生成MindRecord的方式处理数据集

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
    # 环境初始化
    context_init(use_parallel, op)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=batch_size, learning_rate=5e-5, warmup_steps=100, sink_mode=True, sink_size=4)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task, model=model_type, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    task.set_parallel_config(data_parallel=dp,
                             model_parallel=mp,
                             pipeline_stage=pp,
                             optimizer_shard=op,
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

启动LoRA低参微调脚本(4卡)：

> 注：因低参微调所需内存减少，此处用4卡并行即可训练，需重新生成4卡训练所需的rank table file

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_4p_0123_xxx.json ../configs/glm/run_glm_6b_lora.yaml '[0,4]' finetune
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
bash run_distribute_single_node.sh "python task.py --task text_generation --model_type glm_6b_lora --checkpoint_path ./glm_6b.ckpt --train_dataset ./train --run_mode finetune --use_parallel True --data_parallel 1 --model_parallel 4" /path/to/hccl_4p_xxx.json '[0,4]' 4
```

参数说明：对比全参微调启动，仅改动以下几点：

- `model_type`: 指定模型类型为 `glm_6b_lora`，表示使用低参微调算法
- `model_parallel`: 4卡启动，模型并行数改为4
- `/path/to/hccl_4p_xxx.json`: 使用4卡的rank_table_file
- `'[0,4]' 4`: 使用0~3共4卡

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

## 评估

### 模型权重文件合一

微调所得到的权重文件为根据模型切分策略切分后的权重，我们需要手动将切分权重合一，以用于评估和推理

1. 获取模型切分策略文件：
   在执行全参微调脚本时，模型完成编译后，将会在运行路径下，生成名为 `ckpt_strategy.ckpt` 的切分策略文件，将其保存

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
>
> 此外，非2.0版本的mindspore，在低参微调时，生成的切分策略文件将不包含被冻结的权重，导致权重文件合并失败；此时，需将 `mindformers/models/glm/glm.py` 文件中有关LoRA冻结权重的代码注释后，重新运行微调脚本，获取到正确的切分策略文件后停止训练进程；相关代码如下

```python
@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTrainingWithLora(GLMForPreTraining):
    """GLM Model for pretraining with LoRA

Args:
    config (GLMConfig): The config of network.
"""

def __init__(self, config: GLMConfig = None, pet=None, **kwargs):
    _ = kwargs
    super().__init__(config)
    # get Pet tuning model.
    self.pet = pet
    self.pet.pet_config.reg_rules = r'.*query_key_value*'
    self.transformer = LoraAdapter.get_pet_model(self.transformer, self.pet.pet_config)
    # freeze pretrained model
    PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)     # 注释此行以生成新的策略文件
```

### 使用全参微调权重

#### run_mindformers启动eval

使用全参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm/run_glm_6b_infer.yaml` glm模型推理配置，此配置下评估速度更快

```bash
python run_mindformer.py --config configs/glm/run_glm_6b_infer.yaml --run_mode eval --load_checkpoint /path/to/glm_6b.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/adgen_dev.mindrecord --device_id 0
```

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
python task.py --task text_generation --model_type glm_6b_chat --checkpoint_path /path/to/glm_6b.ckpt --eval_dataset /path/to/data/AdvertiseGen/adgen_dev.mindrecord --run_mode eval --batch_size 1
```

> 注：当前评估时，batch_size需为1，否则评估速度下降严重

### 使用LoRA低参微调权重

#### run_mindformers启动lora eval

使用LoRA低参微调权重时，启动如下shell脚本，执行单卡评估

配置文件选择 `configs/glm/run_glm_6b_lora_infer.yaml` glm_lora模型推理配置，此配置可用于lora模型，并且评估速度更快

```bash
python run_mindformer.py --config configs/glm/run_glm_6b_lora_infer.yaml --run_mode eval --load_checkpoint /path/to/glm_6b_lora.ckpt --eval_dataset_dir /path/to/data/AdvertiseGen/adgen_dev.mindrecord --device_id 0
```

各项参数同上，路径需替换为实际路径

#### Trainer高阶接口启动lora eval

仍然可复用 `task.py` 脚本，启动命令：

```bash
python task.py --task text_generation --model_type glm_6b_lora_chat --checkpoint_path /path/to/glm_6b_lora.ckpt --eval_dataset /path/to/data/AdvertiseGen/adgen_dev.mindrecord --run_mode eval --batch_size 1
```

> 注：当前评估时，batch_size需为1，否则评估速度下降严重

## 模型权重转化

本仓库中的`glm`来自于HuggingFace的[chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)，基于下述的步骤获取：

1. 克隆chatglm-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm-6b
   ```

2. 执行 python 脚本，合并模型权重。

   ```python
   from transformers import AutoModel
   import torch as pt

   pt_ckpt_path="Your chatglm-6b path"
   model = AutoModel.from_pretrained(pt_ckpt_path, trust_remote_code=True).half()
   pt_pth_path = "pt_glm_6b.pth"
   pt.save(model.state_dict(), pt_pth_path)
   ```

3. 执行转换脚本，得到转换后的输出文件`ms_glm_6b.ckpt`。

   ```shell
   python mindformers/models/glm/convert_weight.py --pt_ckpt_path "replace your ptroch pth path" --ms_ckpt_path ./ms_glm_6b.ckpt
   ```
