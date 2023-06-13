# GPT2

## 模型描述

GPT-2由OpenAI于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，原生GPT-2模型可分为small（124M）、medium（355M）、large（774M）、xlarge（1.5B），但在此仓中，基于GPT2扩展了13B，52B等规格。

[论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)A Radford，et al., Language Models are Unsupervised Multitask Learners, 2019

## 数据集准备

以Wikitext2数据集为例

- 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

- 词表下载：[vocab.json](https://huggingface.co/gpt2/blob/main/vocab.json)，[merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt)

- 参考[ModelZoo](https://gitee.com/mindspore/models/tree/master/research/nlp/gpt2#language-modeling-%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1%E4%BB%BB%E5%8A%A1)，将数据处理成Mindrecord格式。注：训练数据处理时，长度应等于模型接收长度加一

```bash
# 数据预处理示例代码，代码来源于ModelZoo
# 1、数据清洗
python task_dataset_preprocess.py --task "LanguageModeling" --input_file /{path}/wiki.train.tokens --dataset "wikitext2" --output_file /{path}/{cleaned_data_name}
# 2、生成Mindrecord数据，其中output_file需以字符串mindrecord结尾
python create_lm_data.py --input_file /{path}/{cleaned_data_name} --output_file /{path}/{data_name.mindrecord} --num_splits 1 --max_length 1025 --vocab_file={path of vocab.json} --merge_file={path of merges.txt}
```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

示例命令如下，将会执行一个12层的GPT2模型训练

#### 单卡启动

```shell
# dataset_dir可指定文件目录或文件路径，指定文件路径时，读取单文件，
# 指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --run_mode train \
                         --device_target Ascend \
                         --train_dataset_dir /your_path/wikitext-2-mindrecord
```

其中`device_target`根据用户的运行设备不同，可选`CPU/Ascend`。另，模型和训练等相关配置可在`configs/gpt2`目录下的yaml文件中配置。

#### 单机多卡启动

- 运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

```shell

# step1：机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：# 执行运行脚本：8卡分布式运行， DEVICE_RANGE = [0, 8]， 不包含8本身。
cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_STATUS

```

```python
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
# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的gpt2/run_gpt2*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\predict
```

其中，模型和训练等相关配置可在`configs/gpt2`目录下的yaml文件中配置，如数据集路径，可在`configs/gpt2/run_gpt2_***.yaml`中配置`dataset_dir`参数。
`dataset_dir`可指定文件目录或文件路径，指定文件路径时，读取单文件，指定目录时，读取目录下所有以字符串mindrecord结尾的数据文件

#### 多机多卡启动

- 首先参考单机多卡启动方式，在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

- 执行merge_hccl.py脚本将不同机器上生成的RANK_TABLE_FILE文件中的hccl*.json进行合并，包括server_list合并，server_count设为机器数，rank_id顺序增加，并保证不同机器上的RANK_TABLE_FILE相同；

- 在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式，需注意的是，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# step1：在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json

# step3：将step2得到的合并后的RANK_TABLE_FILE文件分别复制到所有的机器上。

# step4：根据服务器节点数等信息，修改相应的配置
'''
以gpt2-13b模型四机训练为例，默认配置4机32卡，如果节点数有变，需要修改相应的配置。配置文件在../configs/gpt2/run_gpt2_13b.yaml

parallel_config:
  data_parallel: 4
  model_parallel: 2
  pipeline_stage: 4
  optimizer_shard: True
  micro_batch_num: 24
  vocab_emb_dp: True
  gradient_aggregation_group: 4
'''

# step5：执行运行脚本
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/gpt2/run_gpt2_13b.yaml [0,8] train 32
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/gpt2/run_gpt2_13b.yaml [8,16] train 32
# 第三台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the third device} ../configs/gpt2/run_gpt2_13b.yaml [16,24] train 32
# 第四台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the forth device} ../configs/gpt2/run_gpt2_13b.yaml [24,32] train 32
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
output = model(input_ids=inputs["input_ids"], input_mask=inputs["attention_mask"])
print(output)  # 计算输出的logits

model.set_train(True)
inputs = tokenizer(["hello world"],
                   padding='max_length',
                   max_length=model.config.seq_length+1,
                   return_tensors='ms')
output = model(input_ids=inputs["input_ids"], input_mask=inputs["attention_mask"])
print(output)  # 计算loss
```

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task='text_generation', model='gpt2', train_dataset="your data file path")
# 方式1: 开启训练，并使用训练好的权重进行推理
trainer.train()
res = trainer.predict(predict_checkpoint=True, input_data="I love Beijing, because")

# 方式2： 从obs下载训练好的权重并进行推理
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
