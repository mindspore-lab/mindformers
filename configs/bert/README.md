# BERT

## 模型描述

BERT:全名`Bidirectional Encoder Representations from Transformers`模型是谷歌在2018年基于Wiki数据集训练的Transformer模型。

[论文](https://arxiv.org/abs/1810.04805)J Devlin，et. al., 2019

## 数据集准备

1. 从[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)中下载数据集。
2. 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，执行命令如下：

```shell
pip install wikiextractor
python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

#### 计算Loss

```python
from mindformers import BertForPretraining, BertTokenizer
model = BertForPretraining.from_pretrained('bert_base_uncased')
tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
data = tokenizer("Paris is the [MASK] of France.")
input_ids = data['input_ids']
attention_mask = input_ids['attention_mask']
token_type_ids = input_ids['token_type_ids']
masked_lm_positions = Tensor([[4]], mstype.int32)
next_sentence_labels = Tensor([[1]], mstype.int32)
masked_lm_weights = Tensor([[1]], mstype.int32)
masked_lm_ids = Tensor([[3007]], mstype.int32)
output = model(input_ids, attention_mask, token_type_ids, next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights)
print(output)
#[0.6706]
```

#### 推理

## 模型权重

本仓库中的`bert_base_uncased`来自于HuggingFace的[`bert_base_uncased`](https://huggingface.co/bert-base-uncased), 基于下述的步骤获取：

1. 从上述的链接中下载`bert_base_uncased`的HuggingFace权重，文件名为`pytorch_model.bin`

2. 执行转换脚本，得到转换后的输出文件`mindspore_t5.ckpt`

```shell
python mindformers/models/bert/convert_weight.py --layers 12 --torch_path pytorch_model.bin --mindspore_path ./mindspore_bert.ckpt
```
