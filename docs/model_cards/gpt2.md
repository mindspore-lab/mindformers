# GPT2

## 模型描述

GPT-2由OpenAI于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，原生GPT-2模型可分为small（124M）、medium（355M）、large（774M）、xlarge（1.5B），但在此仓中，基于GPT2扩展了13B，52B等规格。

[论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)A Radford，et al., Language Models are Unsupervised Multitask Learners, 2019

## 模型性能

```txt
Mindspore: 2.0.0rc1
Ascend: Atlas 800
```

|               config                |        task         |              Datasets              | [metric](#评测) |                score                | [train performance](#预训练) |         [predict performance](#推理)         |
| :---------------------------------: | :-----------------: | :--------------------------------: |:-----------:| :---------------------------------: |:-------------------------:|:----------------------------------------:|
|    [gpt2](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2.yaml)     |   text_generation   |             wikitext2              |     ppl     |                22.11                |       1265 tokens/s       | 4.66/11.37 tokens/s(use past True/False) |
|  [gpt2_lora](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_lora.yaml)  |   text_generation   |             wikitext2              |      -      |                  -                  |      33573 tokens/s       |                    -                     |
| [gpt2_txtcls](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_txtcls.yaml) | text_classification | SST-2<br/>IMDB<br/>AGNews<br/>COLA |  accuracy   | 0.908<br/>0.934<br/>0.941<br/>0.693 |             -             |                    -                     |

## 仓库介绍

1、模型具体实现：`mindformers/models/gpt2`

`gpt2`基于`mindformers`实现，主要涉及的文件有：

```bash
gpt2
    ├── __init__.py
    ├── convert_weight.py           # 权重转换脚本
    ├── gpt2.py                     # 模型实现
    ├── gpt2_config.py              # 模型配置项
    ├── gpt2_processor.py           # gpt2预处理
    ├── gpt2_tokenizer.py           # tokenizer
    └── gpt2_modules.py             # transformer层实现
```

2、模型配置：`configs/gpt2`

```bash
gpt2
    ├── run_gpt2.yaml           # gpt2 small模型启动配置
    ├── run_gpt2_13b.yaml       # gpt 13b模型启动配置
    ├── run_gpt2_52b.yaml       # gpt 52b模型启动配置
    ├── run_gpt2_lora.yaml      # gpt2 small lora低参微调启动配置
    ├── run_gpt2_txtcls.yaml    # gpt2 small文本分类模型启动配置
    ├── run_gpt2_xl.yaml        # gpt2 xlarge模型启动配置
    └── run_model_xl_lora.yaml  # gpt2 xlarge lora低参微调启动配置
```

3、预处理脚本和任务启动脚本：`mindformers/tools/dataset_preprocess/gpt2`

```bash
gpt2
    ├── txtcls_dataset_to_mindrecord.py     # 文本分类数据集预处理
    └── wikitext2_data_process.py           # wikitext2数据集预处理
```

## 前期准备

### [mindformers安装](https://gitee.com/mindspore/mindformers/tree/dev#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)

### 生成RANK_TABLE_FILE(多卡运行必备环节)

运行`mindformers/tools/hccl_tools.py`生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

**其中`server_id`是机器ip地址**

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

以下`server_id`为机器ip地址，不同机器需设置不同的值

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

Huggingface权重：
    [gpt2 small](https://huggingface.co/gpt2/resolve/main/pytorch_model.bin)
    [gpt2 xlarge](https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin)
    [gpt2 13b](https://huggingface.co/cerebras/Cerebras-GPT-13B/tree/main)

其中，13b的权重需要将上述链接下的`pytorch_model-00001-of-00002.bin`、`pytorch_model-00002-of-00002.bin`、`pytorch_model.bin.index.json
`、`config.json`下载并存到一个文件夹`torch_weights`中，然后使用如下命令将Huggingface的权重进行合并

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("torch_weights")
model.save_pretrained("gpt_13b.bin", max_shard_size="60GB")
```

权重转换：

```bash
# gpt2 small
python mindformers/models/gpt2/convert_weight.py --layers 12 --torch_path gpt2_small.bin --mindspore_path ./gpt2_small.ckpt
# gpt2 xlarge
python mindformers/models/gpt2/convert_weight.py --layers 48 --torch_path gpt2_small.bin --mindspore_path ./gpt2_xlarge.ckpt
# gpt2 13b
python mindformers/models/gpt2/convert_weight.py --layers 40 --torch_path gpt_13b.bin --mindspore_path ./gpt_13b.ckpt
```

另，`mindformers`已经提供转换好的权重（其中lora权重为mindformers训练得到，非Huggingface官方权重转化得到）：
    [gpt2 small](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2.ckpt)
    [gpt2 small lora](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_lora.ckpt)
    [gpt2 xlarge](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_xl.ckpt)
    [gpt2 xlarge lora](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_xl_lora.ckpt)
    [gpt2 13b](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/gpt2_13b.ckpt)

### [模型权重切分与合并](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md)

## 基于API的快速使用

### 基于AutoClass的使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/gpt2`

```python
# 以gpt2 small为例
import mindspore
from mindformers import AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
model.set_train(False)
inputs = tokenizer("An increasing sequence: one,")
outputs = model.generate(inputs["input_ids"], max_length=20, do_sample=20)
response = tokenizer.decode(outputs)[0]
print(response)
# An increasing sequence: one, two, three, four, five, six, seven, eight,
```

### 基于Trainer的训练，微调，评测，推理

```python
# 以gpt2 small为例
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='gpt2',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')
# 开启预训练
trainer.train()

# 开启全量微调
trainer.finetune()

# 开启评测
trainer.evaluate()

 # 开启推理
predict_result = trainer.predict(input_data="An increasing sequence: one,", do_sample=False, max_length=20)
print(predict_result)
# output result is: [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight,']}]
# Lora微调
trainer = Trainer(task="text_generation", model="gpt2", pet_method="lora",
                  train_dataset="path/to/train_dataset")
trainer.finetune(finetune_checkpoint="gpt2")
```

### 基于Pipeline的推理

```python
# 以gpt2 small为例
# 单卡推理支持gpt2、gpt2 xl、gpt2 lora三个模型
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)
pipeline_task = pipeline(task="text_generation", model="gpt2")
pipeline_result = pipeline_task("An increasing sequence: one,", do_sample=False, max_length=20)
print(pipeline_result)
# [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight,']}]
```

## 预训练

### 数据集准备-预训练

以Wikitext2数据集为例

1、数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

2、词表下载：[vocab.json](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/vocab.json)，[merges.txt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/gpt2/merges.txt)

3、参考[wikitext-2处理脚本](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/gpt2/wikitext2_data_process.py#)，将数据处理成Mindrecord格式。

**注：除使用`configs/gpt2/run_gpt2_txtcls.yaml`配置文件外，预训练或者微调时，数据需处理为`configs/gpt2/run_gpt2_*.yaml`中`model.model_config.seq_length`的值加1，如下，当使用`run_gpt2.yaml`配置文件执行训练时，`max_length`需设为1025。**

```bash
# 训练
python mindformers/tools/dataset_preprocess/gpt2/wikitext2_data_process.py \
                              --input_file ./wikitext-2/wiki.train.tokens \
                              --output_file ./wikitext-2.train.mindrecord \
                              --max_length 1025
```

### 脚本启动

#### 单卡训练

- python启动

```bash
# dataset_dir可指定文件目录或文件路径，指定文件路径时，读取单文件，
# 指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --run_mode train \
                         --train_dataset_dir ./wikitext-2.train.mindrecord
```

- bash启动

```bash
cd scripts
bash run_standalone.sh ../configs/gpt2/run_gpt2.yaml [DEVICE_ID] train
```

#### 多卡训练

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必备环节)

- 单机多卡

请提前在yaml文件中修改并行模式`parallel_mode`为`1`，即`半自动并行模式`，并修改相应的分布式配置，配置参考[并行配置](https://gitee.com/mindspore/mindformers/blob/dev/docs/readthedocs/source_zh_cn/docs/design/Parallel_Design.md#config-%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)

**请提前将yaml文件中train_dataset配置中的dataset_dir设置为处理好的mindrecord数据路径**

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/gpt2/run_gpt2.yaml [0,8] train
```

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

请提前在yaml文件中修改并行模式`parallel_mode`为`1`，即`半自动并行模式`，并修改相应的分布式配置，配置参考[并行配置](https://gitee.com/mindspore/mindformers/blob/dev/docs/readthedocs/source_zh_cn/docs/design/Parallel_Design.md#config-%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)

**请提前将yaml文件中train_dataset配置中的dataset_dir设置为处理好的mindrecord数据路径**

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] train $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] train $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 微调

### 数据集准备-微调数据集

- [参考预训练数据集制作](#数据集准备-预训练)

### 全参微调

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单卡微调

- python启动

**微调需传入`load_checkpoint`入参，可以是已经预训练好的模型权重（以.ckpt结尾），也可以是模型名，如`gpt2`。**

```bash
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --run_mode finetune \
                         --load_checkpoint "the path of pretrained ckpt or gpt2" \
                         --train_dataset_dir ./wikitext-2.train.mindrecord
```

- bash启动

**请提前将yaml文件中`train_dataset`配置中的`dataset_dir`设置为处理好的`mindrecord`数据路径，并指定`load_checkpoint`为预训练权重路径。**

```bash
cd scripts
bash run_standalone.sh ../configs/gpt2/run_gpt2.yaml [DEVICE_ID] finetune
```

#### 多卡微调

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必备环节)

- 单机多卡

**请提前将yaml文件中`train_dataset`配置中的`dataset_dir`设置为处理好的`mindrecord`数据路径，并指定`load_checkpoint`为预训练权重路径。**

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/gpt2/run_gpt2.yaml [0,8] finetune 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

**请提前将yaml文件中`train_dataset`配置中的`dataset_dir`设置为处理好的`mindrecord`数据路径，并指定`load_checkpoint`为预训练权重路径。**

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

### Lora微调

#### 单卡微调

```bash
python run_mindformer.py --config configs/gpt2/run_gpt2_lora.yaml --run_mode finetune
```

```bash
cd scripts
bash run_standalone.sh ../configs/gpt2/run_gpt2_lora.yaml [DEVICE_ID] finetune
```

#### 多卡微调

请提前在yaml文件中修改并行模式`parallel_mode`为`1`，即`半自动并行模式`，并修改相应的分布式配置，配置参考[并行配置](https://gitee.com/mindspore/mindformers/blob/dev/docs/readthedocs/source_zh_cn/docs/design/Parallel_Design.md#config-%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必备环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE path/to/config_lora.yaml [0,8] finetune 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机生成rank_table_file合并多机多卡必备环节)

- 多机多卡

请提前在yaml文件中修改并行模式`parallel_mode`为`1`，即`半自动并行模式`，并修改相应的分布式配置，配置参考[并行配置](https://gitee.com/mindspore/mindformers/blob/dev/docs/readthedocs/source_zh_cn/docs/design/Parallel_Design.md#config-%E5%B9%B6%E8%A1%8C%E9%85%8D%E7%BD%AE)

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config_lora.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config_lora.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 评测

GPT2支持文本生成和文本分类两个任务的评测。

**注：以下以`GPT2`12层模型举例，数据处理脚本的`max_length`入参默认是`configs/gpt2/run_gpt2.yaml`中的`seq_length`，即`1024`。如更换使用模型，需设置数据处理脚本的`max_length`为对应yaml文件中的`seq_length`。**

### 文本生成

#### 获取数据集

- [WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)是从维基百科上经过验证的优质文章集中提取的超过1亿个token的集合。

#### 处理数据成mindrecord格式

```bash
cd mindformers/tools/dataset_preprocess/gpt2
python wikitext2_data_process.py --input_file {your_path/wiki.valid.tokens} \
                             --output_file {your_path/wikitext-2.valid.mindrecord}
```

#### 开启评测

```bash
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --eval_dataset_dir {your_path/wikitext-2.valid.mindrecord} \
                         --run_mode eval \
                         --epochs 1
# gpt2: PerplexityMetric: {'PerplexityMetric': {'loss': 3.24, 'PPL': 25.55}
# gpt2_13b(需替换yaml文件): PerplexityMetric: {'PerplexityMetric': {'loss': 2.35, 'PPL': 10.49}
```

### 文本分类

#### 获取数据集

- [SST-2数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)数据集包含电影评论中的句子和它们情感的人类注释。类别分为两类正面情感（positive，样本标签对应为1）和负面情感（negative，样本标签对应为0）

- [IMDB数据集](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)影评数据集，包含5万条IMDB影评，评论的情绪是二元的，专门用于情绪分析。

- [AG-News数据集](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)数据集包含496,835条来自AG新闻语料库4大类别超过2000个新闻源的新闻文章。

- [COLA数据集](https://nyu-mll.github.io/CoLA/)数据集来自语言理论的书籍和期刊，每个句子被标注为是否合乎语法的单词序列。

#### 处理数据成mindrecord格式

```bash
# 因评测前需要微调模型，所以需要生成训练/评测数据集。注：生成的数据集文件需以.mindrecord结尾
cd mindformers/tools/dataset_preprocess/gpt2
python txtcls_dataset_to_mindrecord.py --dataset_name {select one from ['cola', 'sst_2', 'ag_news', 'imdb']}
                                     --input_file {your_path/train.tsv} \
                                     --output_file {your_path/dataset_name.train.mindrecord}
python txtcls_dataset_to_mindrecord.py --dataset_name {the same as above}
                                     --input_file {your_path/dev.tsv} \
                                     --output_file {your_path/dataset_name.dev.mindrecord}
```

#### 开启微调

- 因为原始权重中不包含隐向量向类别映射的参数，所以无法进行zero-shot，评测前需要事先进行微调。

```bash
# 运行前请确保run_gpt2_txtcls.yaml中的model.model_config.num_labels准确，具体的，
# sst2/cola/imdb: num_labels = 2, agnews: num_labels = 4
python run_mindformer.py --config configs/gpt2/run_gpt2_txtcls.yaml \
                       --train_dataset_dir {your_path/dataset_name.train.mindrecord} \
                       --load_checkpoint {the path of pretrained ckpt} \
                       --run_mode finetune
```

#### 开启评测

- 评测指标为ACC

```bash
# 运行前请确保run_gpt2_txtcls.yaml中的model.model_config.num_labels准确，具体的，
# sst2/cola/imdb: num_labels = 2, agnews: num_labels = 4
python run_mindformer.py --config configs/gpt2/run_gpt2_txtcls.yaml \
                       --eval_dataset_dir {your_path/dataset_name.dev.mindrecord} \
                       --run_mode eval \
                       --epochs 1
# ACC: COLA-0.693, SST-2-0.908, IMDB-0.934, AG-News-0.941
```

## 推理

### 基于pipeline的推理

Pipeline接口进行推理

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.pipeline import pipeline

pipeline_task = pipeline("text_generation", model='gpt2', max_length=20)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ["I love Beijing, because it's a beautiful city. It's a beautiful city. It's a"]}]
```

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
         use_past=True,
         do_sample=False,
         max_decode_length=30):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["I love Beijing, because",
              "An increasing sequence: one,"]

    # set model config
    model_config = AutoConfig.from_pretrained("gpt2")
    # if use parallel, data_parallel * model_parallel = device_num
    model_config.parallel_config.data_parallel = 1
    model_config.parallel_config.model_parallel = 1
    model_config.use_past = use_past
    model_config.do_sample = do_sample
    model_config.max_decode_length = max_decode_length
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # build model from config
    network = AutoModel.from_config(model_config)
    model = Model(network)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard gpt2 and load sharded ckpt
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
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
    parser.add_argument('--do_sample', default=False, type=str2bool,
                        help='whether enable do_sample.')
    parser.add_argument('--max_decode_length', default=30, type=int,
                        help='the length of generated text.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past,
         args.do_sample,
         args.max_decode_length)
# {'text_generation_text': ["I love Beijing, because it's a beautiful city. It's a beautiful city. It's a beautiful city. It's a beautiful city. It"]}
# {'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen,']}
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0
export END_RANK=8

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done
```

#### 单卡pipeline推理

```bash
python predict_custom.py
```

#### 多卡pipeline推理

- 修改yaml文件中分布式配置及并行模式，参考[模型权重切分与合并](../feature_cards/Transform_Ckpt.md)进行离线权重切分。**注**：推理暂不支持流水线并行

- 将上述`predict_custom.py`中的分布式配置更改为上步的分布式配置

```text
model_config.parallel_config.data_parallel = 1
model_config.parallel_config.model_parallel = 1
```

- 配置上述sh脚本中的卡数设置，默认是0-8卡

```text
export RANK_SIZE=8  # 总卡数
export START_RANK=0 # 起始卡序号
export END_RANK=8   # 结束卡序号
```

- 运行如下命令进行推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/gpt2_shard_checkpoint_dir
```

### 基于generate的推理

Generate接口进行推理

```python
# 使用AutoModel.from_pretrained实例化模型
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("I love Beijing, because")["input_ids"]
generated_ids = model.generate(input_ids, do_sample=False, max_length=30)
generate_result = tokenizer.decode(generated_ids)
print(generate_result)
# ["I love Beijing, because it's a beautiful city. It's a beautiful city. It's a beautiful city. It's a beautiful city. It"]
```

```python
# 使用AutoModel.from_config实例化模型
from mindformers import AutoModel, AutoTokenizer, AutoConfig

# 可对model_config进行修改，如model_config.do_sample = False
model_config = AutoConfig.from_pretrained("gpt2")
model = AutoModel.from_config(model_config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("I love Beijing, because")["input_ids"]
generated_ids = model.generate(input_ids, do_sample=False, max_length=30)
generate_result = tokenizer.decode(generated_ids)
print(generate_result)
# ["I love Beijing, because it's a beautiful city. It's a beautiful city. It's a beautiful city. It's a beautiful city. It"]
```

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

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
         use_past=True,
         do_sample=False,
         max_decode_length=30):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["I love Beijing, because",
              "An increasing sequence: one,"]

    # set model config
    model_config = AutoConfig.from_pretrained("gpt2")
    # if use parallel, data_parallel * model_parallel = device_num
    model_config.parallel_config.data_parallel = 1
    model_config.parallel_config.model_parallel = 1
    model_config.batch_size = len(inputs)
    model_config.use_past = use_past
    model_config.do_sample = do_sample
    model_config.max_decode_length = max_decode_length
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # build model from config
    network = AutoModel.from_config(model_config)
    model = Model(network)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard gpt2 and load sharded ckpt
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.max_decode_length, padding="max_length")["input_ids"]
    outputs = network.generate(inputs_ids, max_length=model_config.max_decode_length)
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
    parser.add_argument('--do_sample', default=False, type=str2bool,
                        help='whether enable do_sample.')
    parser.add_argument('--max_decode_length', default=30, type=int,
                        help='the length of generated text.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past,
         args.do_sample,
         args.max_decode_length)
# I love Beijing, because it's a beautiful city. It's a beautiful city. It's a beautiful city. It's a beautiful city. It
# An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen,
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0
export END_RANK=8

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path $CHECKPOINT_PATH &> mindformers_$RANK_ID.log &
done
```

#### 单卡generate推理

```bash
python predict_custom.py
```

#### 多卡generate推理

- 修改yaml文件中分布式配置及并行模式，参考[模型权重切分与合并](../feature_cards/Transform_Ckpt.md)进行离线权重切分。**注**：推理暂不支持流水线并行

- 将上述`predict_custom.py`中的分布式配置更改为上步的分布式配置

```text
model_config.parallel_config.data_parallel = 1
model_config.parallel_config.model_parallel = 1
```

- 配置上述sh脚本中的卡数设置，默认是0-8卡

```text
export RANK_SIZE=8  # 总卡数
export START_RANK=0 # 起始卡序号
export END_RANK=8   # 结束卡序号
```

- 运行如下命令进行推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/gpt2_shard_checkpoint_dir
```

### 脚本启动

#### 单卡推理

```bash
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml --run_mode predict --predict_data "An increasing sequence: one," --use_parallel False
# 以下结果是在do_sample=False，max_decode_length=30的配置下跑出的，两处配置可在yaml文件中进行设置。
# output result is: [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen,']}]
```

**注**：要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```yaml
# model config
use_past: True          # 开启增量推理
use_moe: False
expert_num: 1
per_token_num_experts_chosen: 1
checkpoint_name_or_path: "gpt2"
repetition_penalty: 1
max_decode_length: 1024
top_k: 3
top_p: 1
do_sample: True
```

## Mindspore-Lite 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造的推理引擎 [MindSpore_lite](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite)，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　Lite 推理大致分两步：权重转换导出 MindIR -> Lite 推理，接下来分别描述上述两个过程。

### MindIR 导出

1. 执行run_mindformer.py导出MindIR模型，参考如下命令

```shell
python run_mindformer.py --config run_gpt2_13b_910b.ymal --run_mode export --load_checkpoint /path/to/gpt2_13b.ckpt --device_id 7 --batch_size 1 --use_parallel False --output_dir /path/to/export/
```

### 执行推理

1. 新建推理配置文件, 配置参数请参考特性文档[如何配置GE图引擎配置](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Inference.md#%E5%A6%82%E4%BD%95%E9%85%8D%E7%BD%AEge%E5%9B%BE%E5%BC%95%E6%93%8E%E9%85%8D%E7%BD%AE)

```bash
lite.ini
```

2. 执行命令：

```bash
python run_infer_main.py --device_id 0 --model_name gpt2 --prefill_model_path gpt2_export/gpt2_13b_prefill_seq2048_graph.mindir --increment_model_path gpt2_export/gpt2_13b_inc_seq2048_graph.mindir --config_path lite.ini --is_sample_acceleration False --seq_length 2048 --add_special_tokens True
```

　　等待模型载入、编译后，出现：

```bash
Please enter your predict data:
```

　　输入：

```bash
An increasing sequence: one,
```

　　输出：

```bash
['An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, ...']
```
