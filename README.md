# transformer

## 介绍

{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

## 软件架构

```text
tasks: 下游任务
examples:运行脚本
```

## 快速上手

### 1. 数据预处理

下载GLUE数据集，参考[google](https://github.com/google-research/bert)下载GLUE数据集，数据集下载后的目录如下

```text
├── CoLA
│   └── original
│       ├── raw
│       └── tokenized
├── diagnostic
├── MNLI
│   └── original
├── MRPC
├── QNLI
├── QQP
├── RTE
├── SST-2
│   └── original
├── STS-B
│   └── original
└── WNLI
```

### 2. 下载词表文件

在数据预处理中需要词表文件和SentencePiece model文件(可选)

### 3. 执行预处理脚本

下述的命令需要词表文件和SentencePiece model文件。用户可以从[albert](https://github.com/google-research/albert)下载

```bash
TASK_NAME=CoLA
VOCAB_PATH=/albert_base/30k-clean.vocab
SPM_MODEL=/albert_base/30k-clean.model
SRC_DATA_PATH=xx/xxx
OUTPUT_PATH=xxx/xxx
SHARD_NUM=1
python tasks/glue/generate_records.py  \
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

```text
TASK_NAME=CoLA
VOCAB_PATH=/albert_base/vocab.txt
SRC_DATA_PATH=xx/xxx
OUTPUT_PATH=xxx/xxx
SHARD_NUM=1
python tasks/glue/generate_records.py  \
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

3. 单卡训练gpt模型

```bash
bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR
```

3. 单机8卡训练gpt模型

```bash
bash examples/pretrain/pretrain_gpt_distributed.sh 8 hostfile /path/dataset
```
