# 欢迎来到MindSpore Transformer

## 介绍

MindSpore Transformer套件的目标是构建一个大模型训练、推理、部署的全流程套件：
提供业内主流的Transformer类预训练模型，
涵盖丰富的并行特性。 期望帮助用户轻松的实现大模型训练。

MindSpore Transformer基于MindSpore内置的并行技术，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换。
- 提供灵活易用的个性化并行配置。
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略；实现单卡到大规模集群的无缝切换。

如果您对MindSpore Transformer有任何建议，请通过Gitee或MindSpore与我们联系，我们将及时处理。

目前支持的并行策略如下：

- 数据并行
- 模型并行
- 优化器并行
- 多副本并行

目前支持的模型列表如下：

- GPT
- BERT
- OPT
- T5

如何迁移HuggingFace权重，请查看[此文件](./tools/README.md)

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
    │   ├── opt
    │   └── vit
    ├── data # 数据集
    ├── loss
    ├── models # 模型脚本
    │   ├── bert
    │   ├── gpt
    │   ├── t5
    │   ├── opt
    │   └── vit
    ├── modules # 自定义的网络组件
    │   └── attention
    ├── optim # 优化器类定义
    ├── tokenization
    └── trainer # 自定义的训练过程
```

## 快速上手

### GPT模型预训练

#### 数据处理

目前提供了如下数据集的处理：

- [GPT](./examples/preprocess/gptpreprocess/README.md)
- [BERT](./examples/preprocess/bertpreprocess/README.md)

#### 开始训练

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

### 数据预处理

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

```bash
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

```yaml
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

目前参数的传入主要采用传入`yaml`文件+命令行参数覆盖的方式。例如下文所示

```bash
python -m transformer.train \
    --config='./transformer/configs/gpt/gpt_base.yaml' \
    --epoch_size=$EPOCH_SIZE \
    --data_url=$DATA_DIR \
    --optimizer="adam"
    --custom_args="test" \
```

`config`作为命令行解析的第一个参数，将从指定的文件中加载所有的参数。然后开始解析其后面的
参数`epoch_size`、`data_url`、`optimizer`和`custom_args`等参数。
由于前三个参数已经在`gpt_base.yaml`文件中定义，所以这些参数会被命令行中传入的参数覆盖。

而`custom_args`中没有在配置文件中定义，会被添加到解析的参数中去。用户可以在`train.py`中通过`opt.custom_args`获取。
其中`opt`是`run_train`的入参。

#### 自定义配置文件

用户可以直接从`./transformer/configs/gpt/gpt_base.yaml`中复制一份自己的配置文件。然后修改其中的`arch: 'gpt'`参数。代码中根据`arch`
关键字来识别将要初始化的模型和数据实例。

#### 添加自定义模型

用户可以在`transformer/models`目录下创建自己的模型文件夹。构建好模型代码后，需要在`tranformer/models/build_model.py`中加入自己模型的
构建接口。

#### 添加自定义数据集

用户可以在`transformer/data`目录下创建自己的数据集处理文件。然后在`tranformer/data/build_dataset.py`中加入数据集的构建接口。
构建接口。

### 运行模式

目前脚本根据传入的`parallel_mode`参数来决定运行的模式。目前`parallel_mode`的可选入参为如下:

#### 单卡运行

`stand_alone`: 单卡模式。示例脚本可以参考`examples/pretrain_gpt.sh`。此时`parallel_config`中的参数并不会生效。

#### 数据并行

`data_parallel`: 数据并行模式。示例脚本可以参考`examples/pretrain_gpt_distributed.sh`。用户需要手动修改`--parallel_mode=data_parallel`
此时`parallel_config`中的参数并不会生效。

#### 自动并行模式：

`semi_auto_parall`: 半自动并行模式。此模式下可以使能目前MindSpore提供的所有并行能力。
模型将根据传入的`parallel_config`中配置的模型并行数目对权重进行切分。
用户可以根据自己的需要，在`parallel_mode`为`semi_auto_parall`的模式下，逐步开启如下的并行配置。

##### 自动并行下的数据并行

>--parallel_mode=semi_auto_parallel --data_parallel=总卡数

用户需要在启动脚本中增加参数。
其中`data_parallel`表示数据并行度，默认值在`gpt_base.yaml`的配置文件中给定，值为1。
此参数下和`--parallel_mode=data_parallel`的区别如下：

- ReduceSum、ReduceMean等操作在`axis`轴进行聚合时，其结果等价在所有卡的输入数据在单卡上的运行结果。

##### 优化器并行

>--parallel_mode=semi_auto_parallel --data_parallel=总卡数 --optimizer_shard=True

用户可以在启动脚本中增加入参来使能此配置。
模型的参数、优化器状态将会进一步在数据并行维度上进行切分，将进一步减少模型参数在每卡的占用。
在使能此项配置后，每卡保存的模型权重是整体的一个切片。

##### 模型并行

>--parallel_mode=semi_auto_parallel --data_parallel=4 --model_parallel=2

当用户需要对模型中的权重进行切分，以进一步减少模型在每卡中占用的内存时，可以增加上述入参。
此时模型中的所有权重将会被切分为`model_parallel`份数。用户需要确保`data_parallel`和`model_parallel`
的乘积等于总卡数。**注意**，由于模型并行会在前向计算和反向计算中引入额外的通信。
推荐的`model_parallel`可选值为2/4/8，此时将确保模型并行产生的通信在单机内。

### 开启重计算

用户可以在启动脚本中增加如下参数开启重计算。开启后，程序能够运行更大的Batch Size或者更大的模型，但是代价是增加更多的计算时间。

>--recompute=True

## 贡献

欢迎参与社区贡献，详情参考[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 许可证

[Apache 2.0许可证](LICENSE)
