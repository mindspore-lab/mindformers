# GPT2

## 模型描述

GPT-2由Open于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，GPT-2模型可分为small（124M）、medium（355M）、large（774M）、xlarge（1.5B）。

[论文](https://arxiv.org/abs/1810.04805)J Devlin，et al., Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019

## 数据集准备

1. 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)
2. 参考[数据处理](https://gitee.com/mindspore/models/tree/master/research/nlp/gpt2#language-modeling-%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1%E4%BB%BB%E5%8A%A1)，将数据处理成Mindrecord格式。注：训练数据处理时，长度应等于模型接收长度加一

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

示例命令如下，将会执行一个12层的GPT2模型训练

```shell
python run_mindformer.py --config configs/gpt2/run_gpt2.yaml \
                         --run_mode train \
                         --device_target Ascend \
                         --dataset_dir /your_path/wikitext-2-mindrecord
```

其中`device_target`根据用户的运行设备不同，可选`GPU/Ascend/CPU`。

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import GPT2LMHeadModel, Gpt2Tokenizer
from mindspore.context import set_context

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.set_train(False)
tokenizer = Gpt2Tokenizer.from_pretrained('gpt2')
inputs = tokenizer(["hello world"],
                 padding='max_length',
                 max_length=model.config.seq_length,
                 return_tensors='ms')
output = model(inputs["input_ids"], inputs["attention_mask"])
print(output)  # 计算输出的logits

model.set_train(True)
inputs = tokenizer(["hello world"],
                   padding='max_length',
                   max_length=model.config.seq_length+1,
                   return_tensors='ms')
output = model(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])
print(output)  # 计算loss
```

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task='text_generation', model='gpt2', train_dataset="your data file path")
trainer.train() # 开启预训练
```

## 模型权重

本仓库中的`gpt2`来自于HuggingFace的[`gpt2`](https://huggingface.co/gpt2/blob/main/pytorch_model.bin), 基于下述的步骤获取：

- 从上述的链接中下载`gpt2`的HuggingFace权重，文件名为`pytorch_model.bin`

- 执行转换脚本，得到转换后的输出文件`mindspore_gpt2.ckpt`

```shell
python mindformers/models/gpt2/convert_weight.py --layers 12 --torch_path pytorch_model.bin --mindspore_path ./mindspore_gpt2.ckpt
```
