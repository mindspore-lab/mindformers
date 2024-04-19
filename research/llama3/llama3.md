# Llama 3

## 模型描述

Llama 3，是开源Llama系列的最新产品，目前有二个版本：Llama3-8B，Llama 3-70B。Llama 3在来自公开可用来源的超过15T的数据上进行了预训练。微调数据包括公开可用的指令数据集，以及超过1000万个人工标注的示例。模型支持上下文窗口长度8K，并使用了新的分词器，词汇表大小达到128256个，采用了分组查询注意力机制(GQA)。Llama 3模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。目前Mindformers支持Llama 3-8B。

## 仓库介绍

`Llama 3` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`research/llama3`

   ```bash
   llama3
       ├── convert_weight.py                        # 权重转换脚本
       ├── predict_llama3_8b_8k_800T_A2_64G.yaml    # 8B推理配置
       └── run_llama3_8b_8k_800T_A2_64G.yaml        # 8B全量微调Atlas 800 A2启动配置
   ```

3. 数据预处理脚本和任务启动脚本：`research/llama3`

   ```bash
   llama3
       ├── run_llama3.py           # 7B全量微调Atlas 800 A2启动配置
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.3
- MindFormers版本：dev
- 硬件支持矩阵

|     模型      | 硬件 | 全量微调 | lora微调 | 推理 |
| :-----------: | :--: | :------: | :------: | :--: |
| Llama3-8b | Atlas 800T A2 |  单节点  |  单节点  | 单卡 |

### RANK_TABLE_FILE准备

- **单机8卡**

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```shell
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

- **2机16卡**

1. 在每个机器上运行`mindformers/tools/hccl_tools.py`，生成各自的`RANK_TABLE_FILE`文件。

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

2. 将不同机器的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上，运行`mindformers/tools/merge_hccl.py`合并`RANK_TABLE_FILE`文件

```shell
# 运行如下命令，合并每个机器的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

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

**注：多机多卡获取`RANK_TABLE_FILE`步骤同2机16卡。**

### 数据集准备

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

- 分词模型下载：例如下载申请通过后huggingface里对应Files 中的tokenizer.model

- 使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 8192 \
--output_file /{path}/wiki8192.mindrecord
```

数据处理时候注意bos，eos，pad等特殊ids要和yaml配置中model_config里保持一致。

### 模型权重准备

选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

**注**: 请安装transformers=4.40版本

下载完成后，运行`mindformers/models/llama/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path {path}/MS_CKPT_NAME
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径
```

### [模型权重转换](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

## Llama3-8B

### 全参微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的wiki数据集，参照[模型权重准备](#模型权重准备)章节获取Baichuan2-7B-Base权重。

Llama3-8B在Atlas 800T A2上训练，支持**单机/多机训练**。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)：

使用`run_llama3_8b.yaml`进行训练，或修改默认配置文件中的`model_config.seq_length`，使数据集与训练配置的`seq_length`保持一致。

- **单机训练**

Llama3-8B用于微调，seq_length默认为8192，分布式微调训练在Atlas 800T A2上单节点即可启动。以`wiki`数据集为例，给出了默认配置文件`run_llama3_8b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备 ：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-单机8卡章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_llama3_8b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: 'model_dir/xxx.ckpt'  # 使用完整权重路径
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids"]
# input_colums按照数据集中的字段指定（如wikitext2数据集），input_columns: ["input_ids"]

# 8卡分布式策略配置
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机上拉起任务。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python llama3/run_llama3.py \
--config llama3/run_llama3_8b_8k_800T_A2_64G.yaml \
--load_checkpoint model_dir/xxx.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

### MindSpore推理

Llama3-8b用于在线推理，Atlas 800T A2支持**单卡推理**。

以下提供了基于高阶接口推理：基于trainer推理，支持传入单句或多句列表。

#### 基于高阶接口推理

1. 主要参数配置参考

```yaml
load_checkpoint: 'path/to/llama3_8b.ckpt'   # 填写权重路径
auto_trans_ckpt: False                              # 关闭自动权重转换
use_past: True                                      # 使用增量推理
vocab_file: 'path/to/tokenizer.model'               # 配置词表路径
use_parallel: False                                 # 关闭并行模式
```

2. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数，基于释出的权重。
python llama3/run_llama3.py \
--config llama3/predict_llama3_8b_800T_A2_64G.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/llama3_8b.ckpt \
--vocab_file path/to/tokenizer.model \
--auto_trans_ckpt False \
--predict_data "I love Beijing,because"

# output: [{'text_generation_text': ['I love Beijing,because it is a city of history and culture. I love Beijing,because it is a city of modernization. I love Beijing, because it is a city of the future. I love Beijing,because it is a city of my heart.']}]
```
