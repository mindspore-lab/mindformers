# 星辰语义大模型 Telechat

## 模型描述

星辰语义大模型Telechat是由中电信人工智能科技有限公司研发训练的大语言模型，采用3万亿Tokens中英文高质量语料进行训练。目前开源模型：Telechat-7B，Telechat-12B模型，本仓库已支持7B和12B模型的微调权重，权重文件来源于中电信人工智能科技有限公司。

基于GPU，Torch版本的Telechat链接：

[Telechat](https://github.com/Tele-AI/Telechat)

[TeleChat Technical Report](https://arxiv.org/abs/2401.03804)

``` text
@article{wang2024telechat,
      title={TeleChat Technical Report},
      author={Zihan Wang and Xinzhang Liu and Shixuan Liu and Yitong Yao and Yuyao Huang and Zhongjiang He and Xuelong Li and Yongxiang Li and Zhonghao Che and Zhaoxi Zhang and Yan Wang and Xin Wang and Luwen Pu and Huihan Xu and Ruiyu Fang and Yu Zhao and Jie Zhang and Xiaomeng Huang and Zhilong Lu and Jiaxin Peng and Wenjun Zheng and Shiquan Wang and Bingkai Yang and Xuewei he and Zhuoru Jiang and Qiyi Xie and Yanhan Zhang and Zhongqiu Li and Lingling Shi and Weiwei Fu and Yin Zhang and Zilu Huang and Sishi Xiong and Yuxiang Zhang and Chao Wang and Shuangyong Song},
      journal={arXiv preprint arXiv:2401.03804},
      year={2024}
}
```

## 模型性能

基于910B

telechat_7b:

| config                                                | task                  | Datasets   | SeqLength | phase           | performance  |
|-------------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [telechat_7b](./run_telechat_7b_910b.yaml)            | text_generation       | example_dataset | 2048      | [train](#预训练)   | 1940 tks/s/p |
| [telechat_7b](./run_telechat_7b_910b_finetune.yaml)   | text_generation       | example_dataset     | 2048      | [finetune](#微调) | 1925 tks/s/p |
| [telechat_7b](./run_telechat_7b_910b_finetune.yaml)   | text_generation       | example_dataset     | 2048      | [predict](#推理)  | 27 tks/s/p   |

telechat_12b:

| config                                                | task                  | Datasets   | SeqLength | phase           | performance  |
|-------------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [telechat_12b](./run_telechat_12b_910b.yaml)          | text_generation       | example_dataset | 1024      | [train](#预训练)   | 1433 tks/s/p |
| [telechat_12b](./run_telechat_12b_910b_finetune.yaml) | text_generation       | example_dataset     | 1024      | [finetune](#微调) | 1433 tks/s/p |
| [telechat_12b](./run_telechat_12b_910b_finetune.yaml) | text_generation       | example_dataset     | 1024      | [predict](#推理)  | 20 tks/s/p   |

## 仓库介绍

`Telechat` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/research/telechat`

   ```bash
   telechat
       ├── convert_weight_ms_to_torch.py         # ms->torch权重转换脚本
       ├── convert_weight_torch_to_ms.py         # torch->ms权重转换脚本
       ├── telechat_preprocess.py                # telechat模型的mindrecord数据处理脚本
       ├── telechat.py                           # 模型实现
       ├── telechat_config.py                    # 模型配置项
       ├── telechat_layer.py                     # telechat网络层定义
       ├── telechat_predict_utils.py             # telechat推理模块
       ├── telechat_tokenizer.py                 # telechat tokenizer
       └── telechat_transformer.py               # transformer层实现
   ```

2. 模型配置：`mindformers/research/telechat`

   ```bash
   telechat
       ├── run_telechat_7b_910b.yaml             # 7b模型预训练启动配置
       ├── run_telechat_7b_finetune_910b.yaml    # 7b全量微调启动配置
       ├── run_telechat_12b_910b.yaml            # 12b模型预训练启动配置
       └── run_telechat_12b_finetune_910b.yaml   # 12b全量微调启动配置
   ```

3. 任务启动脚本：`mindformers/research/telechat`

   ```text
   telechat
       ├── run_telechat_predict.py              # 推理脚本
       └── run_telechat.py                      # telechat高阶接口使用脚本
   ```

## 前期准备

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.2.11
- CANN: 7.1
- MindFormers版本：dev

注：Atlas 800T A2芯片：7b, 12b推理可在单机单卡上完成部署。

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
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

### 多机RANK_TABLE_FILE合并(多机多卡必备环节)

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

RANK_TABLE_FILE 双机16卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换（Telechat-7B为例）

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1.torch模型权重及词模型下载链接：

- [telechat-7b](https://huggingface.co/Tele-AI/Telechat-7B/)
- [telechat-12b](https://huggingface.co/Tele-AI/TeleChat-12B)

下载完成后，运行如下转换脚本，将全量微调的权重转换为完整的ckpt权重。

```shell
python mindformers/research/telechat/convert_weight_torch_to_ms.py \
--torch_path TORCH_CKPT_DIR \
--mindspore_path {path}/MS_CKPT_NAME \
--model_name 'telechat_7b' \
```

```text
# 参数说明
torch_path: torch版本权重保存目录路径
mindspore_path: 权重保存文件名，可以指定自定义保存路径
model_name: 模型的名称
```

2.获取MindFormers提供的已转换权重，可直接从下面的链接获取。

- [telechat-7b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/mindspore.ckpt)
- [telechat-12b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/mindspore_12B.ckpt)

### [分布式训练/微调权重合并](../../docs/feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix telechat_7b
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 预训练（Telechat-7B为例）

### 数据集准备

step 1. 获取数据集

[数据集](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/example_dataset.jsonl)

数据集的格式：

```text
# input_dataset examples:
    {"input": "电信主卡和副卡的区别在哪里？", "output": "主卡和副卡的主要区别在于，主卡只能使用一张手机号码。<_end>"}
```

step 2. 处理数据成mindrecord格式

```bash
# 使用mindformers/research/telechat/telechat_preprocess.py进行数据预处理+Mindrecord数据生成
# 由于此工具依赖AutoTokenizer，所以需要提前下载transformers
python telechat_preprocess.py \
--input_dataset_file /{path}/input_dataset.jsonl \
--vocab_file_path /{path}/tokenizer.model \
--max_length 2048 \
--output_path /{path}/output_dataset.mindrecord
```

```text
# 参数说明
input_dataset_file: 预训练的数据集
vocab_file_path: 词模型文件路径(如使用上述链接下载，指定到对应路径下即可)
max_length: 数据集长度
output_path: 生成数据集的路径
```

### 脚本启动

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

#### 多卡训练

##### 单机多卡

- step 1. 修改模型对应的配置文件。

在模型对应的配置文件`research/telechat/run_telechat_7b_910b.yaml`中，用户可自行修改模型、训练相关参数(推荐开启flash_attention，可加速训练)，并通过`train_dataset`的`dataset_dir`参数，指定训练数据集的路径。

配置文件中各参数含义详见[Config配置说明文档](https://gitee.com/mindspore/mindformers/blob/master/configs/README.md)。auto_parallel说明详见[自动并行](../../docs/feature_cards/Auto_Parallel.md)。

- step2. 设置环境变量，变量配置如下：

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  # 推荐开启饱和模式
```

- step3. 启动训练任务，在单机上拉起任务。

```shell
cd mindformers/research

bash run_singlenode.sh \
"python telechat/run_telechat.py \
--config telechat/run_telechat_7b_910b.yaml \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，预训练时设置为train
train_data: 训练数据集文件夹路径
RANK_TABLE_FILE: 生成的rank_table文件
```

##### 多机多卡

- step 1. 多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

> **注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

- step 2. 根据服务器节点数等信息，修改相应的配置。

```yaml
# 以telechat-7b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：run_telechat_7b_910b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 3. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式。

```shell
cd mindformers/research

# 第一台机器
bash run_multinode.sh \
"python telechat/run_telechat.py \
--config telechat/run_telechat_7b_910b.yaml \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# 第二台机器
bash run_multinode.sh \
"python telechat/run_telechat.py \
--config telechat/run_telechat_7b_910b.yaml \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16
```

```text
# 参数说明
config: 配置文件路径
run_mode: 运行模式，预训练时设置为train
train_data: 训练数据集文件夹路径
RANK_TABLE_FILE: 生成的rank_table文件
```

## 微调（Telechat-7B为例）

### 数据集准备

目前使用的数据集样例由中电信人工智能科技有限公司提供，该样例的预处理脚本可用于全参微调任务，详细数据集格式以及数据处理参考预训练格式样例。

### 全参微调

当前模型已支持使用**Flash Attention算法**进行全参微调，推荐开启flash_attention，可加速训练。详请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 参考`research/telechat/run_telechat_7b_910b_finetune.yaml`中训练数据集路径为微调数据集路径。

```python
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}/"
    shuffle: True
```

- step 2. 修改微调时学习率, 优化器参数，`seq_length`, 新增 `context`中参数, 与预训练不同，微调配置如下：

```python
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.95
  eps: 1.e-5
  learning_rate: 1.e-5

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  lr_end: 0
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset

# model config
model:
  model_config:
    type: TelechatConfig
    model_name: 'telechat_7b'
    batch_size: 1 # add for increase predict
    seq_length: 2048

# context
context:
  runtime_num_threads: 1
```

- step 3. 微调`telechat-7b`时修改并行策略配置，配置如下：

```python
# parallel_config
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
```

- step4. 设置环境变量，变量配置如下：

```bash
export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3  # 去除TensorMove
export MS_MEMORY_POOL_RECYCLE=1  # 内存优化
export GE_NOT_CUT=1   # 内存优化
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"  # telechat_7b 不用设置该项
```

- step 5. 添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。
- step 6. 启动微调任务，telechat-7b模型以单机八卡为例进行微调，命令如下：

```shell
cd mindformers/research

bash run_singlenode.sh \
"python telechat/run_telechat.py \
--config telechat/run_telechat_7b_910b_finetune.yaml \
--load_checkpoint model_dir \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
load_checkpoint: 预训练模型权重文件
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

## 推理（Telechat-7B为例）

推理时将配置文件中`param_init_type`修改为和全量微调一致的数据类型。

```python
# context_config 910B推理添加ascend_config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
```

### 单卡generate推理

1. telechat用于在线推理，输入按照 "question"的模板格式输入，910B支持单卡推理。主要参数配置参考:

```yaml
load_checkpoint: 'path/to/telechat.ckpt'            # 填写权重路径
use_past: True                                      # 使用增量推理
use_parallel: False                                 # 关闭并行模式
```

2. 启动推理

```shell
cd research
python telechat/run_telechat_predict.py --input_file /path/to/infer_file.jsonl --vocab_file path/to/tokenizer.model --yaml_file path/to/config_yaml

# 参数说明
input_file: 输入的问题文件
yaml_file: 模型的配置文件
vocab_file: 配置词表路径
```

7B 模型推理结果如下：

```text
生抽与老抽的区别？ 生抽和老抽是两种不同的酱油，它们的区别如下：
1. 原料不同：生抽是用大豆、小麦等谷物为原料制成的；而老抽则是用豆酱、面酱等发酵后的调味品为原料制成的。
2. 制作工艺不同：生抽是通过将大豆浸泡在水中，然后经过蒸煮、发酵等过程制成的；而老抽则是在生抽的基础上加入一定比例的盐、糖、味精等调料，再进行发酵制成的。
3. 口感和风味不同：生抽具有咸鲜的味道，口感比较清爽；而老抽则具有特殊的香味和味道，口感相对较重。
总的来说，生抽和老抽都是酱油的不同种类，它们在原料、制作工艺和口感等方面都有所不同。
```

12B 模型推理结果如下：

```text
生抽与老抽的区别？ 生抽和老抽是两种不同的酱油，它们在风味、色泽和用途上都有所区别。
1. 颜色：生抽的颜色比较淡，而老抽的颜色较深。生抽的颜色呈红褐色或棕红色，而老抽的颜色则更偏向棕黑色。
2. 味道：生抽具有鲜美的咸味和微甜的味道，而老抽则具有浓郁的酱香味和深厚的味道。由于生抽的含盐量较低，所以它更适合用于调味和提鲜，而老抽则更适合用于炖煮和烧煮菜肴。
3. 用途：生抽通常用于调味，如炒菜、拌菜、腌制等，而老抽则更适合用于烧肉、炖菜、烧鱼等需要突出酱香味的菜肴。
总之，生抽和老抽在颜色、味道和用途上都有所不同，可以根据个人口味和烹饪需求选择适合的酱油品种。
```
