# 快速上手

下面展示如何基于GPT模型进行一个训练的基本示例。

## GPT模型预训练

### 数据处理

目前提供了如下数据集的处理：

- [GPT](./examples/preprocess/gptpreprocess/README.md)

### 开始训练

- 单卡训练gpt模型

执行下述的命令，开始训练一个1.3B的GPT模型。

```bash
bash examples/pretrain/pretrain_gpt.sh DEVICE_ID EPOCH_SIZE DATA_DIR
```

其中各个参数的含义：

- DEVICE_ID是期望运行的卡号。例如0、1、2等等
- EPOCH_SIZE表示设置的数据训练轮次。例如0、1、2等等
- DATA_DIR表示处理完毕的数据集路径。例如/home/data/

日志会重定向到`standalone_train_gpu_log.txt`中。可以通过`tail -f standalone_train_gpu_log.txt`的
命令及时刷新日志。

- 单机8卡训练gpt模型

执行下述命令会开始训练一个10B的GPT模型。

```bash
bash examples/pretrain/pretrain_gpt_distributed.sh EPOCH_SIZE hostfile DATA_DIR
```

其中各个参数的含义：

- hostfile：一个文本文件，格式如下

```text
10.1.2.3 slots=8
```

表示节点ip为10.1.2.3的服务器拥有8张设备卡。用户应该将自己的实际IP替换掉10.1.2.3。

日志会重定向到`distribute_train_gpu_log.txt`中。可以通过`tail -f distribute_train_gpu_log.txt`的
命令及时刷新日志。注意此时8张卡的日志都会输出到上述的文件中，造成重复输出。用户在如下的位置查看每卡的输出

```bash
tail -f run_distributed_train_gpt/1/rank.0/stdout
```

### GPT下游任务微调

#### 数据预处理

下载GLUE数据集，参考[google](https://github.com/google-research/bert)下载GLUE数据集。

#### 下载词表文件

在数据预处理中需要词表文件和SentencePiece model文件(可选)

#### 执行预处理脚本

下述的命令需要词表文件和SentencePiece model文件。用户可以从[albert](https://github.com/google-research/albert)下载

```bash
TASK_NAME=CoLA
VOCAB_PATH=/albert_base/30k-clean.vocab
SPM_MODEL=/albert_base/30k-clean.model
SRC_DATA_PATH=xx/xxx
OUTPUT_PATH=xxx/xxx
SHARD_NUM=1
python tools/glue_to_mindrecord.py  \
    --task_name=$TASK_NAME \
    --vocab_path=${VOCAB_PATH} \
    --spm_model_file=${SPM_MODEL} \
    --max_seq_length=512 \
    --do_lower_case="true" \
    --input_dir=${SRC_DATA_PATH} \
    --output_dir=${OUTPUT_PATH} \
    --shard_num=$SHARD_NUM \
    --do_train="true" \
    --do_eval="true" \
    --do_pred="true" \
```

如果不提供SPM_MODEL路径，将使用[google/bert](https://github.com/google-research/bert)的tokenization版本。只需要提供Vocab文件即可。

```bash
TASK_NAME=CoLA
VOCAB_PATH=/albert_base/vocab.txt
SRC_DATA_PATH=xx/xxx
OUTPUT_PATH=xxx/xxx
SHARD_NUM=1
python tools/glue_to_mindrecord.py  \
    --task_name=$TASK_NAME \
    --vocab_path=${VOCAB_PATH} \
    --max_seq_length=512 \
    --do_lower_case="true" \
    --input_dir=${SRC_DATA_PATH} \
    --output_dir=${OUTPUT_PATH} \
    --shard_num=$SHARD_NUM \
    --do_train="true" \
    --do_eval="true" \
    --do_pred="true" \
```
