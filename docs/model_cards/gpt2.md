# GPT2

## 模型描述

GPT-2由OpenAI于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，GPT-2模型可分为small（124M）、medium（355M）、large（774M）、xlarge（1.5B）。

[论文](https://arxiv.org/abs/1810.04805)J Devlin，et al., Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019

## 数据集准备

以Wikitext2数据集为例

- 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

- 参考[数据处理](https://gitee.com/mindspore/models/tree/master/research/nlp/gpt2#language-modeling-%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1%E4%BB%BB%E5%8A%A1)，将数据处理成Mindrecord格式。注：训练数据处理时，长度应等于模型接收长度加一

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

示例命令如下，将会执行一个12层的GPT2模型训练

#### 单卡启动

```shell
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --run_mode train \
                         --device_target Ascend \
                         --dataset_dir /your_path/wikitext-2-mindrecord
```

#### 单机多卡启动

```shell
# 8卡分布式运行， DEVICE_RANGE = [0, 8], 不包含8本身
cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_STATUS
```

```text
# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的gpt2/run_gpt2*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\predict
```

其中`device_target`根据用户的运行设备不同，可选`GPU/Ascend`。

#### 多机多卡启动

- 首先参考单机多卡启动方式，在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

- 将不同机器上生成的RANK_TABLE_FILE文件中的server_list合并，server_count设为机器数，rank_id顺序增加，并保证不同机器上的RANK_TABLE_FILE相同；

- 在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。

```text
# RANK_TABLE_FILE 参考样例
# 单机8卡
{
    "version": "1.0",
    "server_count": "1",
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
        }
    ],
    "status": "completed"
}
```

```text
# 两机16卡
{
  "version": "1.0",
  "server_count": "2",
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
    }
  ],
  "status": "completed"
}

# 以gpt2-13b模型两机训练为例，首先修改../configs/gpt2/run_gpt2_13b.yaml中默认的并行配置

parallel_config:
  data_parallel: 2
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4

# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/gpt2/run_gpt2_13b.yaml [0,8] train
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/gpt2/run_gpt2_13b.yaml [8,16] train
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.set_train(False)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
inputs = tokenizer(["hello world"],
                 padding='max_length',
                 max_length=model.config.seq_length,
                 return_tensors='ms')
output = model(inputs["input_ids"])
print(output)  # 计算输出的logits

model.set_train(True)
inputs = tokenizer(["hello world"],
                   padding='max_length',
                   max_length=model.config.seq_length+1,
                   return_tensors='ms')
output = model(inputs["input_ids"])
print(output)  # 计算loss
```

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task='text_generation', model='gpt2', train_dataset="your data file path")
trainer.train() # 开启预训练

res = trainer.predict(input_data="I love Beijing, because")
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
pipeline_task = pipeline("text_generation", model='gpt2', max_length=20)
pipeline_result = pipeline_task("I love Beijing, because", top_k=3)
print(pipeline_result)
```

## 模型权重

本仓库中的`gpt2`来自于HuggingFace的[gpt2](https://huggingface.co/gpt2/blob/main/pytorch_model.bin), 基于下述的步骤获取：

- 从上述的链接中下载`gpt2`的HuggingFace权重，文件名为`pytorch_model.bin`

- 执行转换脚本，得到转换后的输出文件`mindspore_gpt2.ckpt`

```shell
python mindformers/models/gpt2/convert_weight.py --layers 12 --torch_path pytorch_model.bin --mindspore_path ./mindspore_gpt2.ckpt
```
