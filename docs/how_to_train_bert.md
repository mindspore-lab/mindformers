# BERT模型预训练微调

## 数据准备

### 数据下载

#### 预训练数据集

如下步骤展示了将中文或者英文的维基百科中作为预训练预料的步骤处理：

1. 从[zhwiki](https://dumps.wikimedia.org/zhwiki/)或[enwiki](https://dumps.wikimedia.org/enwiki/)中下载数据集。
2. 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，执行命令如下：

```bash
pip install wikiextractor
python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
```

#### 微调数据下载

以下述的`SQuAD`和`GLUE`数据集为例，用户可以点击下述的链接下载数据集：

- 英文问答任务[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)和[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- 英文分类任务[GLUE](https://gluebenchmark.com/tasks)数据集下载

### 数据处理

MindSpore支持用户传入`TFRecord`和`MindRecord`两种类型的数据。

#### TFRecord类型BERT预训练数据

用户可以参考[BERT](https://github.com/google-research/bert#pre-training-with-bert)代码仓中的create_pretraining_data.py文件，
进行`TFRecord`格式文件的生成，
如果出现下述报错

```bash
AttributeError: module 'tokenization' has no attribute 'FullTokenizer'
```

请安装`bert-tensorflow`。注意，用户需要同时下载对应的`vocab.txt`文件。

#### MindRecord类型BERT预训练数据

如果已按上面[步骤](#预训练数据集)下载原始预训练数据集，并使用`WikiExtractor`提取文本数据，你可以按以下操作获取对应的`MindRecord`数据集

```bash
cd examples/generate_mindrecord/
bash ./generate_pretrain_mindrecords.sh INPUT_FILES_PATH OUTPUT_FILES_PATH VOCAB_FILE
```

例如下述的命令：

```bash
cd examples/generate_mindrecord/
bash ./generate_pretrain_mindrecords.sh /path/wiki-clean-aa /path/output/ /path/bert-base-uncased-vocab.txt
```

如果已将`json`格式的数据转换为`TFRecord`数据集，你也可以通过以下方式将`TFRecord`转换成对应的`MindRecord`格式

```bash
cd tools
python parallel_tfrecord_to_mindrecord.py --input_tfrecord_dir /path/tfrecords_path --output_mindrecord_dir /path/save_mindrecord_path
```

同时也可以按以下操作对`TFRecord`或`MindRecord`数据进行可视化

```bash
cd tools
python vis_tfrecord_or_mindrecord.py --file_name /path/train.mindrecord --vis_option vis_mindrecord > mindrecord.txt
```

`vis_option` 需要从`["vis_tfrecord", "vis_mindrecord"]`中选择。注：在执行之前，需要确保需要已安装tensorflow==1.15.0

#### TFRecord类型SQuAD和GLUE数据

将数据集文件从`json`格式转换为`TFRecord`格式。详见[BERT](https://github.com/google-research/bert)代码仓中的`run_classifier.py`或`run_squad.py`文件。

#### MindRecord类型SQuAD和GLUE数据

- `SQuAD`任务：给定`SQuAD`原始数据集和[vocab.txt](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/pretrained_model/uncased_L-12_H-768_A-12/vocab.txt)，生成`MindRecord`格式数据

```bash
cd examples/generate_mindrecord/
python generate_squad_mindrecord.py --vocab_file /path/squad/vocab.txt --train_file /path/squad/train-v1.1.json --predict_file /path/squad/dev-v1.1.json --output_dir /path/squad
```

- `GLUE`任务：给定GLUE原始数据集和[vocab.txt](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/pretrained_model/uncased_L-12_H-768_A-12/vocab.txt)，生成`MindRecord`格式数据

```bash
cd examples/generate_mindrecord/
python generate_glue_mindrecord.py --task_name=MNLI --vocab_path /path/glue/vocab.txt --input_dir /path/glue/MRPC/train.tsv --predict_file /path/glue/MRPC/train.tsv --output_dir /path/MRPC
```

## BERT模型预训练

### 单卡训练

执行下述的命令，单卡训练BERT-base模型。

```bash
DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python -m mindtransformer.models.bert.bert_trainer \
    --train_data_path=$DATA_DIR \
    --optimizer="adam" \
    --seq_length=128 \
    --max_position_embeddings=512 \
    --parallel_mode="stand_alone" \
    --checkpoint_prefix="bert" \
    --global_batch_size=32 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
tail -f standalone_train_gpu_log.txt
```

其中各个参数的含义：

- DEVICE_ID是期望运行的卡号。例如0、1、2等等
- EPOCH_SIZE表示设置的数据训练轮次。例如0、1、2等等
- DATA_DIR表示处理完毕的数据集路径。例如/home/data/

单卡训练时，需要设置parallel_mode参数为"stand_alone"。

日志会重定向到`standalone_train_gpu_log.txt`中。可以通过`tail -f standalone_train_gpu_log.txt`的
命令及时刷新日志。

要完成上述训练只需输入命令

```bash
bash examples/pretrain/pretrain_bert.sh DEVICE_ID EPOCH_SIZE DATA_DIR
```

即可。

### 多卡训练

- 单机8卡数据并行训练BERT-base模型

```bash
RANK_SIZE=$1
HOSTFILE=$2
DATASET=$3
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_bert \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --merge-stderr-to-stdout \
python -m mindtransformer.models.bert.bert_trainer \
    --device_num=$RANK_SIZE \
    --train_data_path=$DATASET \
    --seq_length=128 \
    --global_batch_size=32 \
    --vocab_size=30522 \
    --parallel_mode="data_parallel" \
    --checkpoint_prefix="bert" \
    --full_batch=False \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --data_parallel=8 \
    --model_parallel=1 \
    --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &
```

其中各个参数的含义：

- RANK_SIZE：总共使用的卡的数量，采用单机8卡训练时，设为8

- HOSTFILE：一个文本文件，格式如下

```text
10.1.2.3 slots=8
```

表示节点ip为10.1.2.3的服务器拥有8张设备卡。用户应该将自己的实际IP替换掉10.1.2.3。

数据并行模式下parallel_mode参数为"data_parallel"。

日志会重定向到`distribute_train_gpu_log.txt`中。可以通过`tail -f distribute_train_gpu_log.txt`的
命令及时刷新日志。注意此时8张卡的日志都会输出到上述的文件中，造成重复输出。用户在如下的位置查看每卡的输出

```bash
tail -f run_distributed_train_bert/1/rank.0/stdout
```

要完成上述训练只需输入命令

```bash
bash examples/pretrain/pretrain_bert_distributed.sh RANK_SIZE hostfile DATA_DIR
```

即可。

如果要多机训练BERT模型，只需要设置RANK_SIZE为总的卡数，并在hostfile文件中输入所有机器的ip和对应的卡数。

## HuggingFace Checkpoint转换

我们提供将`HuggingFace`的预训练权重转换成`MindSpore`模型权重的方法。只需执行下述命令：

```bash
python transformer/models/bert/convert_bert_weight.py --layers 12 --torch_path pytorch_model.bin --mindspore_path ./bertbase.ckpt
```

其中参数 `layers`表示模型的层数。

## BERT模型下游微调

微调同样支持单卡和分布式微调。这边以分布式微调为例。

### 文本分类任务GLUE微调

以MRPC任务为例

```bash
RANK_SIZE=$1
HOSTFILE=$2
export NCCL_IB_HCA=mlx5_
mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self  --merge-stderr-to-stdout \
python -m mindtransformer.trainer.trainer  \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="data_parallel" \
    --data_parallel=8 \
    --model_parallel=1 \
    --full_batch=False \
    --epoch_size=3 \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --max_position_embeddings=512 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --use_one_hot_embeddings=False \
    --model_type="bert" \
    --dropout_prob=0.1 \
    --train_data_shuffle="true" \
    --global_batch_size=4 \
    --start_lr=5e-5 \
    --save_checkpoint_path="./glue_ckpt/MRPC" \
    --load_checkpoint_path="./checkpoint_path/bertbase.ckpt" \
    --checkpoint_prefix="MRPC" \
    --train_data_path="/GLUE_path/MRPC/train.tf_record" \
```

也可以直接运行脚本

```bash
bash examples/finetune/run_classifier_distributed.sh RANK_SIZE hostfile TASK_NAME
```

在完成微调之后，可以通过如下命令对结果进行评估

```bash
python -m mindtransformer.tasks.text_classification \
    --auto_model="bert_glue" \
    --device_target="GPU" \
    --dataset_format="tfrecord" \
    --assessment_method="accuracy" \
    --parallel_mode="stand_alone" \
    --num_labels=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=128 \
    --max_position_embeddings=512 \
    --use_one_hot_embeddings=False \
    --model_type="bert" \
    --dropout_prob=0.0 \
    --eval_data_shuffle="false" \
    --eval_batch_size=16 \
    --load_checkpoint_path="./glue_ckpt/MRPC" \
    --checkpoint_prefix="MRPC" \
    --eval_data_path="/GLUE_path/MRPC/eval.tf_record"
```

也可以直接运行脚本:

```bash
bash examples/inference/predict_bert_classifier.sh RANK_ID TASK_NAME
```

这里TASK_NAME表示具体的GLUE任务，如果要微调MPRC任务只需令TASK_NAME为MRPC即可。

### 英文问答任务SQuAD微调

类似地，SQuAD任务微调需要执行如下命令：

```bash
RANK_SIZE=$1
HOSTFILE=$2
export NCCL_IB_HCA=mlx5_

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_classifier \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
      --mca btl tcp,self --mca btl_tcp_if_include 10.90.43.0/24,enp177s0f0 --merge-stderr-to-stdout \
python -m  mindtransformer.trainer.trainer \
    --auto_model="bert_squad" \
    --device_target="GPU" \
    --device_num=$RANK_SIZE \
    --parallel_mode="data_parallel" \
    --full_batch=False \
    --epoch_size=3 \
    --num_class=2 \
    --vocab_size=30522 \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12 \
    --seq_length=384 \
    --max_position_embeddings=512 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --data_parallel=8 \
    --model_parallel=1 \
    --global_batch_size=12 \
    --vocab_file_path="./vocab.txt" \
    --save_checkpoint_path="./squad_ckpt" \
    --load_checkpoint_path="/checkpoint_path/bertbase.ckpt" \
    --checkpoint_prefix="squad" \
    --train_data_path="/squad_path/train.mindrecord" \
    --schema_file_path="" \
```

可以直接运行脚本

```bash
bash examples/finetune/run_squad_distributed.sh RANK_SIZE hostfile
```

微调后评估如下：

```bash
python -m mindtransformer.tasks.question_answering \
    --auto_model="bert_squad" \
    --eval_data_path="/squad_path/dev-v1.1.json" \
    --load_checkpoint_path="./squad_ckpt" \
    --vocab_file_path="./vocab.txt" \
    --checkpoint_prefix="squad" \
    --embedding_size=768 \
    --num_layers=12 \
    --num_heads=12
```

可以直接运行脚本：

```bash
bash examples/inference/predict_bert_squad.sh
```

### 下游任务精度

下表给出基于HuggingFace中BERT-base的[checkpoint](https://huggingface.co/bert-base-uncased/tree/main)进行微调后的评估结果

| Task        | MNLI | QNLI   | SST-2 | MRPC | RTE |  Squad|
|-------------|------|--------|-------|------| ------|------|
| Result        | 84.3 | 90.4 |     93.6  |  88.6  | 67.2 | 88.8 |

### 环境

| 项目   | 值   |
|------| --------- |
| 模型规模 | bert-base |
| 环境   | A100     |
| MindSpore | 2.0.0 |
