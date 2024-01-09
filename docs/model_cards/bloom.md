# Bloom

## 模型描述

Bloom (BigScience Large Open-science Open-access Multilingual) 是一个开源的开放权限的自回归大语言模型(LLM)，用于对用自然语言表达的多种下游任务进行文本生成。Bloom系列模型涵盖从560M到176B的多种规模，其中176B千亿级参数的大模型的预训练基于工业级的计算机集群，在46种语言和13种编程语言的文本生成中达到比拟人类写作的SOTA效果。对于训练数据集中没有显式包括的下游任务，Bloom也可以通过指令的方式，给出令人满意的zero-shot回答。

[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

## 模型性能

|       config        |      task       | Datasets | metric | score | [train performance](#预训练) | [predict performance](#基于pipeline的推理) |
|:-------------------:|:---------------:|:--------:|:------:|:-----:|:-------------------------:|:-------------------------------------:|
| run_bloom_560m.yaml | text_generation |    -     |   -    |   -   |             -             |                   -                   |
| run_bloom_7.1b.yaml | text_generation |  Alpaca  |   -    |   -   |   1063tokens/s/p - Atlas 800   |  21.33tokens/s(use_past True) - Atlas 800  |
| run_bloom_65b.yaml  | text_generation |    -     |   -    |   -   |             -             |                   -                   |
| run_bloom_176b.yaml | text_generation |    -     |   -    |   -   |             -             |                   -                   |

## 仓库介绍

`Bloom` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/bloom`

    ```
    bloom
        ├── __init__.py
        ├── convert_weight.py         # 权重转换脚本
        ├── bloom.py                  # 模型实现
        ├── bloom_config.py           # 模型配置项
        ├── layers.py                 # bloom 层定义
        ├── bloom_tokenizer.py        # tokenizer
    ```

2. 模型配置：`configs/bloom`

    ```
    bloom
        ├── run_bloom_560m.yaml     # 560m 用于推理
        ├── run_bloom_7.1b.yaml     # 7.1b 用于8卡(Atlas 800)训练
        ├── run_bloom_7.1b_910b.yaml      # 7.1b 用于8卡(Atlas 800T A2)训练
        └── run_bloom_7.1b_910b_fa.yaml     # 7.1b 用于8卡(Atlas 800T A2)训练，并使用Flash Attention
    ```

    其中Bloom_7.1b可在单机单卡上推理，在单机8卡上训练。

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(**多卡运行必须环节**)

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

### 多机RANK_TABLE_FILE合并(**多机多卡必备环节**)

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

作为参考，这里描述CheckPoint在HuggingFace或者官方开源github仓库和MindSpore间的转换，在不同分布式策略间的转换。

如果不需要加载权重，或者使用from_pretrained功能自动下载，则可以跳过此章节。

Mindformers可以直接通过高级接口from_pretrained下载转换好的560M和7.1B两种规模的ckpt,无需手动转换。如需手动下载，下面提供手动下载链接。

| | huggingface | mindspore ckpt | mindspore tokenizer |
|-|-|-|-|
|bloom_560m| [bloom_560m](https://huggingface.co/bigscience/bloomz-560m) | [bloom_560m.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/bloom/bloom_560m.ckpt) | [tokenizer.json]() |
|bloom_7.1b| [bloom_7.1b](https://huggingface.co/bigscience/bloomz-7b1-mt) | [bloom_7.1b.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/bloom/bloom_7.1b.ckpt) | 同上 |

higgingface到mindformers的CheckPoint转换由以下命令完成。

```bash
cd mindformers/models/bloom
python convert_weight.py --n_head=xx --hidden_size=xx --torch_path=path_to_hf_bin_file_or_folder --mindspore_path=output_path
```

其中`--n_head=xx --hidden_size=xx`根据模型定义，bloom_560m的分别为16/1024；bloom_7.1b的分别为32/4096.

### [模型权重切分与合并](../feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

## 基于API的快速使用

### 基于AutoClass的使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/model_name`

```python
import mindspore as ms
from mindformers import AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
ms.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
model = AutoModel.from_pretrained("bloom_560m")

inputs = tokenizer("what color is the sky?")

outputs = model.generate(inputs["input_ids"], max_length=100)
response = tokenizer.decode(outputs, skip_special_tokens=True)[0]
print(response)
# output
# what color is the sky? blue
```

### 基于Pipeline的快速推理

```python
from mindformers import AutoModel, AutoTokenizer, TextGenerationPipeline
import mindspore as ms

# 指定图模式，指定使用训练卡id
ms.set_context(mode=0, device_id=0)

model = AutoModel.from_pretrained("bloom_560m")
tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
bloom_ppl = TextGenerationPipeline(model, tokenizer, max_length=256)

result = bloom_ppl([
    "what color is the sky?",
    "Translate to English: Je t’aime."
    ])
print(result)

# expect print result
# {'text_generation_text': ['what color is the sky? blue</s>']}, {'text_generation_text': ['Translate to English: Je t’aime. I love you.</s>']}]

```

## 预训练

### 数据集准备-预训练

这里以Alpaca为例，数据大概21MB,用于调试。
首先去官方下载[alpaca_data.json文件](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)
然后调用`mindformers/tools/dataset_preprocess/bloom/make_mindrecord.py`脚本将json转换成mindrecord文件。

```bash
python mindformers/tools/dataset_preprocess/bloom/make_mindrecord.py --input_dataset_file=XXX/alpaca_data.json --output_path=XXX --N=51200
```

其中`--N=51200`表示将json中的52002条数据中的前51200转换成mindrecord(推荐)，`--N=-1`将转换全部json中的数据. 在执行此脚本时，对于每个prompt如下操作将被执行：

- 将问题和回答按照模板制作成prompt text;
- 使用BloomTokenizer将prompt从text转成token ids;
- 添加eos_token_id直到seq_length。

执行文本后，`--output_path`目录下将生成mindrecord文件。

### 脚本启动

#### 多卡训练

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/bloom/run_bloom_7.1b.yaml [0,8] train 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE ../configs/bloom/run_bloom_65b.yaml [0,8] train $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE ../configs/bloom/run_bloom_65b.yaml [$rank_start,$rank_end] train $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 微调

### 数据集准备

参考[数据集准备-预训练](#数据集准备-预训练)

### 全参微调

当前模型已支持使用**Flash Attention算法**进行全参微调，请使用`configs/bloom/run_bloom_7.1b_910b_fa.yaml`替换下述说明中的配置文件以使能Flash Attention。关于Flash Attention，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

通过yaml配置文件中的`load_checkpoint:`字段来控制是否加载CKPT

#### 多卡微调

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE../configs/bloom/run_bloom_7.1b.yaml [0,8] finetune 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE ../configs/bloom/run_bloom_65b.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE ../configs/bloom/run_bloom_65b.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

### 微调后对话效果

在`mindformers/scripts`路径下执行以下脚本`combine_ckpt.py`.这个脚本会

- 对strategy进行合并
- 清理微调ckpt文件中的优化器状态
- 合并微调ckpt文件用于单机推理

```python
# combine_ckpt.py
import os
import mindspore as ms

CKPT_SUFFIX = "300_8" # 300(sink number) * 8 (sink size) = 2400 step
CLEANED_CKPT_DIR = "../output/checkpoint_cleaned"
COMBINED_CKPT_DIR = "../output/checkpoint_combined"
COMBINED_STGY = "../output/strategy/ckpt_strategy.ckpt"


# combine straegies
ms.merge_pipeline_strategys("../output/strategy", COMBINED_STGY)


# clean ckpt by removing optimizer states
for rank_id in range(8):
    input_file_name = f"../output/checkpoint/rank_{rank_id}/mindformers_rank_{rank_id}-{CKPT_SUFFIX}.ckpt"
    params = ms.load_checkpoint(input_file_name)
    new_params = [{"name": key, "data": val}  for key, val in params.items() if not ("accu_grads" in key or "adam_" in key) ]

    save_path = os.path.join(CLEANED_CKPT_DIR, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(new_params, f"{save_path}/cleaned.ckpt")
    print(f"saved {save_path}")


# combine ckpt
ms.transform_checkpoints(CLEANED_CKPT_DIR, COMBINED_CKPT_DIR, ckpt_prefix = "combined_", src_strategy_file = COMBINED_STGY)
```

然后执行以下脚本进行新的对话。
> 以下脚本针对Alpaca数据集的prompt模板。如果使用其他数据集微调，请更换对应模板。

```python
import numpy as np
import mindspore as ms
from mindformers import AutoTokenizer
from mindformers.models.bloom import BloomConfig, BloomLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

alpaca_prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n")

# 7B
CKPT_FILE = "xxx/mindformers/output/checkpoint_combined/rank_0/combined_0.ckpt"
SEQ_LENGTH = 1024
config = BloomConfig(
    param_init_type="float16",
    embedding_init_type="float32",
    checkpoint_name_or_path=CKPT_FILE,
    max_decode_length=SEQ_LENGTH,
    seq_length=SEQ_LENGTH,
    hidden_size=4096,
    num_layers=30,
    num_heads=32,
    hidden_dropout_rate=0.0,
    attention_dropout_rate=0.0,
    batch_size = 1,
    use_past = True
)


def chat():
    tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
    model = BloomLMHeadModel(config)
    model.set_train(False)

    question_list = [
        "why the earth is unique?",
        "why the sky is blue?",
        "write a job application for a data scientist and explain your related work experience."
        ]

    while True:
        if question_list:
            question = question_list.pop(0)
        else:
            question = input("please input your question: ")
        question = alpaca_prompt.format_map({"instruction":question})
        inputs = tokenizer.encode(question)
        inputs = np.array([inputs]).astype(np.int32) # add batch dim
        outputs = model.generate(inputs, max_length=None, do_sample=False, eos_token_id=2)
        outputs = outputs[0] # remove batch dim
        print(tokenizer.decode(outputs))

if __name__ == "__main__":
    chat()

```

预期的对话效果大致为下表所示
|                                                                                        |                      Before                     |                                                                                                                                                                                                                                   After                                                                                                                                                                                                                                  |
|:--------------------------------------------------------------------------------------:|:-----------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| why the sky is blue?                                                                   | light from the sun is scattered<EOS>              | The sky is blue because of the presence of water droplets in the atmosphere. These droplets reflect light back to the sky, causing the sky to appear blue.<EOS>                                                                                                                                                                                                                                                                                                           |
| what would be the best way to travel from San Fransisco to New York?                   | take a flight<EOS>                                | The best way to travel from San Francisco to New York is by taking the flight. The flight is the fastest and most convenient way to travel from San Francisco to New York.<EOS>                                                                                                                                                                                                                                                                                            |
| write a job application for a data scientist and explain your related work experience. | <EOS>                                             | Dear Employer, I am writing to apply for the position of Data Scientist. I have over 5 years of experience in data science and machine learning, and I am excited to join your team. I have experience in supervised and unsupervised machine learning algorithms, data visualization, and data cleaning. I am also proficient in Python, R, and SQL. I am looking forward to discussing my qualifications further and hearing from you soon. Sincerely, [Your Name]<EOS>  |
| why the earth is unique?                                                               | it is the only planet with a liquid surface<EOS>  | The Earth is unique because it is the only planet with a liquid surface, a magnetic field, and a protective atmosphere. It is also the only planet with a life-supporting atmosphere and a diverse and abundant life-supporting ecosystem.<EOS>                                                                                                                                                                                                                            |

## 推理

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡多batch推理。

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
    inputs = ["what color is the sky?",
              "Translate to English: Je t’aime."]

    # set model config
    model_config = AutoConfig.from_pretrained("bloom_7.1b")
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bloom_7.1b")
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard bloom and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
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

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path CHECKPOINT_PATH &> minformers_$RANK_ID.log &
done
```

#### 单卡pipeline推理

```bash
python predict_custom.py
```

#### 多卡pipeline推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/bloom_7.1b_shard_checkpoint_dir
```

#### 单卡与多卡pipeline推理预期输出为

- what color is the sky? _**blue**_
- Translate to English: Je t’aime. _**I love you.**_

### 基于generate的推理

#### 单卡generate推理

```python
import mindspore as ms
from mindformers import AutoTokenizer
from mindformers.models.bloom import BloomConfig, BloomLMHeadModel

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# ##############################
# # bloom_560m config
# CKPT_FILE="bloom_560m"
# SEQ_LENGTH = 256
# config = BloomConfig(
#     param_init_type="float16",
#     embedding_init_type="float16",
#     checkpoint_name_or_path=CKPT_FILE,
#     max_decode_length=SEQ_LENGTH,
#     seq_length=SEQ_LENGTH,
#     hidden_size=1024,
#     num_layers=24,
#     num_heads=16,
#     hidden_dropout_rate=0.0,
#     attention_dropout_rate=0.0,
#     batch_size = 1,
#     use_past = True
#
# )
# ##############################

# 7B
CKPT_FILE = "bloom_7.1b"
# CKPT_FILE also takes absolute path to ckpt file, e.g.
# "/home/xxx/mindformers/checkpoint_download/bloom/bloom_7.1b.ckpt"
SEQ_LENGTH = 256
config = BloomConfig(
    param_init_type="float16",
    embedding_init_type="float16",
    checkpoint_name_or_path=CKPT_FILE,
    max_decode_length=SEQ_LENGTH,
    seq_length=SEQ_LENGTH,
    hidden_size=4096,
    num_layers=30,
    num_heads=32,
    hidden_dropout_rate=0.0,
    attention_dropout_rate=0.0,
    batch_size=1,
    use_past=True
)


def chat():
    tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
    model = BloomLMHeadModel(config)
    model.set_train(False)

    question_list = [
        "what color is the sky?",
        "Translate to English: Je t’aime.",
        ]

    for question in question_list:
        inputs = tokenizer.encode(question)
        inputs = [inputs]  # add batch dim
        outputs = model.generate(inputs, max_length=100, do_sample=False)
        outputs = outputs[0]  # remove batch dim
        print(tokenizer.decode(outputs, skip_special_tokens=True))


if __name__ == "__main__":
    chat()
```

- Bloom_560m的预期输出为:

    - what color is the sky? _**blue**_
    - Translate to English: Je t’aime. _**I love you.**_

- Bloom_7.1B的预期输出为:

    - what color is the sky? _**blue**_
    - Translate to English: Je t’aime. _**I love you.**_

#### 多卡generate推理

这里我们以1机器8卡推理bloom_7.1B为例。涉及两个文件`chat.py`和`run_chat.py`。

```text
/SOME/PATH/
    ├── chat.py # 负责定义一个并行进程
    └── run_chat.py # 负责多次执行chat.py并拉起分布式
```

加载ckpt有两种方式, 由`run_chat.py` 中的`DISTRIBUTED_CKPT_PATH`变量来控制。这里默认使用`DISTRIBUTED_CKPT_PATH=""`代表的`Load then Shard`的方式加载ckpt.

| 分布式加载ckpt的方法   | Load then Shard | Shard then Load |
|-----------------------:|:-----------------|:-----------------|
| DISTRIBUTED_CKPT_PATH= | "" | "/path/to/distributed/ckpt/" |
|说明| 先加载全量ckpt，然后切分模型。|先切分模型，然后加载分布式ckpt。|
|| 不用预先按照策略切分ckpt，推理时可以灵活调整策略。|需要确定推理的分布式策略，并按照策略预先切分ckpt。 |
|| 对host内存的占用较高。| 对host内存的占用较低。|
|适用| 适用于较小模型，如`560m`，`7.1b`。|适用于较大模型，如`65b`, `176b`。 |

```python
# >>> `chat.py`文件

import os
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers import BloomLMHeadModel, BloomConfig, AutoTokenizer
from mindformers import init_context
from mindformers.modules import TransformerOpParallelConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools import logger

SEQ_LENGTH = 256
DISTRIBUTED_CKPT_PATH = os.getenv("DISTRIBUTED_CKPT_PATH", "")


# set context
context_config = {"device_target": "Ascend", "mode": 0,  "max_device_memory": "31GB"}
parallel_context_config = {"parallel_mode": 1, "gradients_mean": False, "full_batch": True}
rank_id, device_num = init_context(use_parallel=True, context_config=context_config, parallel_config=parallel_context_config)
set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=True)
_set_multi_subgraphs()


# config blooom 7.1b
config = BloomConfig(
    embedding_init_type="float32" if DISTRIBUTED_CKPT_PATH else "float16",
    checkpoint_name_or_path="" if DISTRIBUTED_CKPT_PATH else "bloom_7.1b",
    seq_length=SEQ_LENGTH,
    hidden_size=4096,
    num_layers=30,
    num_heads=32,
    hidden_dropout_rate=0.0,
    attention_dropout_rate=0.0,
    top_k=1, top_p=1, do_sample=True,
    parallel_config=TransformerOpParallelConfig(
        data_parallel=1,
        model_parallel=8,
        pipeline_stage=1
        )
    )

def chat():
    # init bloom
    tokenizer = AutoTokenizer.from_pretrained("bloom_560m")
    bloom = BloomLMHeadModel(config)
    bloom.set_train(False)

    if DISTRIBUTED_CKPT_PATH:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(DISTRIBUTED_CKPT_PATH, "rank_{}".format(rank_id))
        ckpt_path = get_last_checkpoint(ckpt_path)
        logger.info("ckpt path: %s", str(ckpt_path))

        # shard bloom and load sharded ckpt
        m = Model(bloom)
        m.infer_predict_layout(ms.Tensor(np.ones(shape=(1, SEQ_LENGTH)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(bloom, checkpoint_dict)
        logger.info("Network parameters are not loaded: %s", str(not_load_network_params))

    question_list = [
        "what color is the sky?",
        "Translate to English: Je t’aime.",
        ]

    for question in question_list:
        inputs = tokenizer.encode(question)
        inputs = [inputs]  # add batch dim
        outputs = bloom.generate(inputs, max_length=100, do_sample=False)
        outputs = outputs[0]  # remove batch dim
        print(tokenizer.decode(outputs, skip_special_tokens=True))


if __name__ == "__main__":
    chat()

```

```bash
# >>> `run_chat.sh`文件

# define variable
export RANK_SIZE=8
export RANK_TABLE_FILE="../hccl_8p.json" # <<< change to yours

# distributed ckpt path to load after sharding model.
# use "" if load full ckpt before sharding model.
export DISTRIBUTED_CKPT_PATH=""

export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./chat.py &> minformers_$RANK_ID.log &
done

```

使用一下命令拉起分布式推理:

```bash
bash run_chat.sh
```

日志可以通过`tail -f mindformers_0.log`查看。预期结果与单机单卡`bloom_7.1b`推理相同：

- what color is the sky? _**blue**_

- Translate to English: Je t’aime. _**I love you.**_

## mindspore-lite

使用mindspore-lite推理共分两步，首先将ckpt文件导出为mindir文件，然后调用`run_infer_main.py`脚本进行推理。

第一步将bloom 7.1B的ckpt权重文件转换成2个共享参数的mindir文件用于lite离线推理:

- `./mindirs/bloom7b_prefill_seq1024_bs1_graph.mindir` 用于首token的全量推理

- `./mindirs/bloom7b_decode_seq1024_bs1_graph.mindir`用于后续token的增量推理。

导出mindir的命令如下：

```bash
python mindformers/tools/export.py --config_path ./configs/bloom/infer_bloom_7.1b.yaml
```

其中infer_bloom_7.1b.yaml以下配置：

```yaml

# trainer config

trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'bloom_7.1b'

model:
  model_config:
    type: BloomConfig
    batch_size: 1
    seq_length: 1024
    hidden_size: 4096
    num_layers: 30
    num_heads: 32
    hidden_act: "gelu"
    param_init_type: "float16"
    embedding_init_type: "float16"
    layernorm_compute_type: "float32"
    softmax_compute_type: "float16"
    compute_dtype: "float16"
    checkpoint_name_or_path: "bloom_7.1b"
    repetition_penalty: 1
    top_k: 1
    top_p: 1
    use_past: True
  arch:
    type: BloomLMHeadModel

# lite

infer:
  prefill_model_path: "./mindirs/bloom7b_prefill_seq1024_bs1"
  increment_model_path: "./mindirs/bloom7b_decode_seq1024_bs1"
  infer_seq_length: 1024
  model_type: mindir
```

第二步，导出mindir后，用以下命令调用mindformers中的`run_infer_main.py`脚本进行推理。

```bash
python run_infer_main.py \
--device_id 7 \
--model_name bloom \
--prefill_model_path ./mindirs/bloom7b_prefill_seq1024_bs1_graph.mindir \
--increment_model_path ./mindirs/bloom7b_decode_seq1024_bs1_graph.mindir \
--config_path ./mindirs/context.cfg \
--is_sample_acceleration False \
--seq_length 1024
```

其中`./mindirs/context.cfg`为lite推理的配置文件，具体内容如下：

```[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.formatMode=1
```

最后，lite推理的预期结果为

```Please enter your predict data:
介绍一下什么是人工智能。
one token takes 0.09584355354309082 s
one token takes 0.02199864387512207 s
one token takes 0.02149224281311035 s
...
['介绍一下什么是人工智能。 人工智能（AI），或机器学习，是研究、开发、应用机器学习技术，让机器像人一样智能。']
```

## 附录

### 附录A BELLE

[BELLE](https://github.com/LianjiaTech/BELLE)（Be Everyone's Large Language model Engine）是一个旨在促进中文对话大模型开源社区发展的组织。BELLE-7B是基于Bloomz-7B-mt，使用中文问答数据集微调出来开源的中文对话模型。根据微调所使用的中文数据大小分为0.2M, 0.6M, 1M, 2M四个权重。
微调的模板为
> Human: {input} \n\nAssistant:{output}

原始的开源BELLE的数据集和权重可以通过以下链接获得

|          | 文件 | 链接                                                     |
|----------| --- |--------------------------------------------------------|
| 2M SFT数据集 | train_2M_CN.json | https://huggingface.co/BelleGroup/BELLE-7B-2M          |
| 2M 模型权重   | pytorch_model.bin | https://huggingface.co/datasets/BelleGroup/train_2M_CN |

数据集和权重的转换参考2.1章和2.2章，命令如下：

```bash
# 数据集转换
python mindformers/tools/dataset_preprocess/bloom/make_mindrecord.py --input_dataset_file=XXX/train_2M_CN.json --output_path=XXX

# 权重转换
cd mindformers/models/bloom
python convert_weight.py --n_head=32 --hidden_size=4096 --torch_path=xxx/pytorch_model.bin --mindspore_path=output_path

```

数据集和权重的转换到mindspore后，可以按照Bloom的方式进行推理和微调。
