# BERT

## 模型描述

BERT:全名`Bidirectional Encoder Representations from Transformers`模型是谷歌在2018年基于Wiki数据集训练的Transformer模型。

[论文](https://arxiv.org/abs/1810.04805)J Devlin，et al., Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019

## 预训练数据集下载

1. 从[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)中下载数据集。
2. 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，执行命令如下：

```shell
pip install wikiextractor
python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
```

### 数据处理

#### TFRecord类型BERT预训练数据

用户可以参考[BERT](https://github.com/google-research/bert#pre-training-with-bert)代码仓中的create_pretraining_data.py文件，
进行`TFRecord`格式文件的生成，
如果出现下述报错

```bash
AttributeError: module 'tokenization' has no attribute 'FullTokenizer'
```

请安装`bert-tensorflow`。注意，用户需要同时下载对应的`vocab.txt`文件。

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

示例命令如下，将会执行一个12层的BERT模型训练

```shell
python run_mindformer.py --config configs/bert/run_bert_base_uncased.yaml --run_mode train  \
                         --device_target Ascend \
                         --train_dataset_dir /your_path/wiki_data
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import BertForPreTraining, BertConfig

BertForPreTraining.show_support_list()
# 输出：
# - support list of BertForPreTraining is:
# -    ['bert_base_uncased']
# - -------------------------------------

# 模型标志加载模型
model = BertForPreTraining.from_pretrained("bert_base_uncased")

#模型配置加载模型
config = BertConfig.from_pretrained("bert_base_uncased")
# {'model_config': {'use_one_hot_embeddings': False, 'num_labels': 1, 'dropout_prob': 0.1,
# 'batch_size': 128, seq_length: 128, vocab_size: 30522, embedding_size: 768, num_layers: 12,
# num_heads: 12, expand_ratio: 4, hidden_act: "gelu", post_layernorm_residual: True,
# hidden_dropout_prob: 0.1, attention_probs_dropout_prob: 0.1, max_position_embeddings: 512,
# type_vocab_size: 2, initializer_range: 0.02, use_relative_positions: False,
# use_past: False, checkpoint_name_or_path: "bert_base_uncased"}}
model = BertForPreTraining(config)
```

- Trainer接口开启训练/评估/推理：

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.trainer import Trainer

# 初始化预训练任务
trainer = Trainer(task='fill_mask',
    model='bert_base_uncased',
    train_dataset='/your_path/wiki_data')
trainer.train() # 开启预训练
```

### 多卡训练

- 单机8卡数据并行训练BERT-base模型

```bash
RANK_SIZE=$1
HOSTFILE=$2

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_bert \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
python run_mindformer.py --config ./configs/bert/run_bert_base_uncased.yaml --use_parallel True --run_mode train  > distribute_train_gpu_log.txt 2>&1 &
```

其中各个参数的含义：

- RANK_SIZE：总共使用的卡的数量，采用单机8卡训练时，设为8

- HOSTFILE：一个文本文件，格式如下

```text
10.1.2.3 slots=8
```

表示节点ip为10.1.2.3的服务器拥有8张设备卡。用户应该将自己的实际IP替换掉10.1.2.3。

日志会重定向到`distribute_train_gpu_log.txt`中。可以通过`tail -f distribute_train_gpu_log.txt`的
命令及时刷新日志。注意此时8张卡的日志都会输出到上述的文件中，造成重复输出。用户在如下的位置查看每卡的输出

```bash
tail -f run_distributed_train_bert/1/rank.0/stdout
```

要完成上述训练只需输入命令

```bash
bash scripts/examples/masked_language_modeling/bert_pretrain_distributed_gpu.sh RANK_SIZE hostfile
```

即可。

#### 计算Loss

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import BertForPreTraining, BertTokenizer
from mindspore import Tensor
import mindspore.common.dtype as mstype
model = BertForPreTraining.from_pretrained('bert_base_uncased')
tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
data = tokenizer("Paris is the [MASK] of France.",
                 max_length=128, padding="max_length")
input_ids = Tensor([data['input_ids']], mstype.int32)
attention_mask = Tensor([data['attention_mask']], mstype.int32)
token_type_ids = Tensor([data['token_type_ids']], mstype.int32)
masked_lm_positions = Tensor([[4]], mstype.int32)
next_sentence_labels = Tensor([[1]], mstype.int32)
masked_lm_weights = Tensor([[1]], mstype.int32)
masked_lm_ids = Tensor([[3007]], mstype.int32)
output = model(input_ids, attention_mask, token_type_ids, next_sentence_labels, masked_lm_positions, masked_lm_ids, masked_lm_weights)
print(output)
#[0.6706]
```

## 模型权重

本仓库中的`bert_base_uncased`来自于HuggingFace的[`bert_base_uncased`](https://huggingface.co/bert-base-uncased), 基于下述的步骤获取：

1. 从上述的链接中下载`bert_base_uncased`的HuggingFace权重，文件名为`pytorch_model.bin`

2. 执行转换脚本，得到转换后的输出文件`mindspore_t5.ckpt`

```shell
python mindformers/models/bert/convert_weight.py --layers 12 --torch_path pytorch_model.bin --mindspore_path ./mindspore_bert.ckpt
```
