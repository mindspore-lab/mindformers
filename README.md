# transformer

## 介绍

Transformer套件可以轻松的实现大模型训练流程。目前支持的并行策略和模型如下：

并行策略：

- 数据并行
- 模型并行
- 流水线并行

支持的模型：

- GPT
- BERT
- VIT
- T5

## 软件架构

```text
.
├── examples
│   └── pretrain  # 预训练的脚本示例。包含GPT、BERT、T5等模型
├── knowledge_distillation
├── tasks # 下游任务微调和处理
│   ├── nlp
│   │   └── glue
│   └── vision
└── transformer
    ├── configs # 模型的配置文件
    │   ├── bert
    │   ├── gpt
    │   ├── t5
    │   └── vit
    ├── data # 数据集
    ├── loss
    ├── models # 模型脚本
    │   ├── bert
    │   ├── gpt
    │   ├── t5
    │   └── vit
    ├── modules # 自定义的网络组件
    │   └── attention
    ├── optim # 优化器类定义
    ├── tokenization
    └── trainer # 自定义的训练过程
```

## 快速上手

### GPT模型预训练

#### 数据准备

1. 获取数据集

2. 清洗数据

3. 构建词典

4. 执行预处理

#### 开始训练

单卡训练gpt模型

```bash
bash examples/pretrain/pretrain_gpt.sh  DEVICE_ID EPOCH_SIZE DATA_DIR
```

其中各个参数的含义：

- DEVICE_ID是期望运行的卡号。例如0、1、2等等
- EPOCH_SIZE表示设置的数据训练轮次。例如0、1、2等等
- DATA_DIR表示处理完毕的数据集路径。例如/home/data/

单机8卡训练gpt模型

```bash
bash examples/pretrain/pretrain_gpt_distributed.sh EPOCH_SIZE hostfile DATA_DIR
```

其中各个参数的含义：

- hostfile：一个文本文件，格式如下

```text
10.1.2.3 slots=8
```

表示节点ip为10.1.2.3的服务器拥有8张设备卡。用户应该将自己的实际IP替换掉10.1.2.3

### GPT下游任务微调

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

## 配置文件

模型的配置文件位于`transformer/configs/`，每个模型单独拥有自己的文件夹。以`gpt_base.yaml`配置文件为例，介绍其中每个字段关键含义：

```text
arch: 'gpt'  # 必选字段，用来区分加载的模型名字。在每个目录下面
model:
  micro_batch_size: 4
  global_batch_size: 4
  seq_length: 1024
  vocab_size: 50304
  embedding_size: 1024
  num_layers: 2
  num_heads: 32
  expand_ratio: 4
  post_layernorm_residual: False
  dropout_rate: 0.1
  compute_dtype: fp16

seed: 1234
context:
  device_target: 'GPU'
  save_graphs: False
  mode: 0
  graph_kernel_flags: "--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true"

parallel_mode: "semi_auto_parallel"

speed_up:
  micro_batch_num: 1
  flatten_weights: False
  fused_kernel: False

moe_config:
  expert_num: 1
  capacity_factor: 1.05
  aux_loss_factor: 0.05
  num_experts_chosen: 1

recompute_config:
  recompute: True
  parallel_optimizer_comm_recompute: False
  mp_comm_recompute: True
  recompute_slice_activation: False

parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  expert_parallel: 1
  vocab_emb_dp: False

optimizer: adam

acc_step: 1
grad_sync_dtype: fp16
data_url: /your/data/path
epoch_size: 1
start_lr: 1e-4
end_lr: 1e-5
warmup_step: 1000
opt_offload: False
sink_size: 10
ckpt_save_dir: ./ckpt
init_loss_scale_value: 65536
scale_factor: 2
scale_window: 1000
```

### 自定义参数

#### 添加自己的自定义参数

#### 添加自定义模型

#### 添加自定义数据集

### 运行模式

#### 单卡训练

#### 数据并行

#### 优化器并行

#### 模型并行
