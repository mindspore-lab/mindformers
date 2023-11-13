# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## 模型性能

|                            config                            |      task       | Datasets | [train performance](#全参微调) |  [predict performance](#推理)  |
| :----------------------------------------------------------: | :-------------: | :------: | :----------------------------: | :----------------------------: |
| [baichuan2_7b](../../research/baichuan2/run_baichuan2_7b.yaml) | text_generation |  belle   |         550 tokens/s         | 20.54 tokens/s (use_past=True) |
| [baichuan2_13b](../../research/baichuan2/run_baichuan2_13b.yaml) | text_generation |  belle   |          379 tokens/s          | 17.75 tokens/s (use_past=True, 2卡) |
| [baichuan2_7b_910b](../../research/baichuan2/run_baichuan2_7b_910b.yaml) | text_generation |  belle   |        1264 tokens/s         | 23.69 tokens/s (use_past=True) |
| [baichuan2_13b_910b](../../research/baichuan2/run_baichuan2_13b_910b.yaml) | text_generation |  belle   |          867 tokens/s          | 16.65 tokens/s (use_past=True)  |

## 仓库介绍

`Baichuan2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/baichuan2`

   ```text
   baichuan2
       ├── baichuan2_tokenizer.py    # tokenizer
       ├── baichuan2_7b.py           # 7B模型实现
       └── baichuan2_13b.py          # 13B模型实现
   ```

2. 模型配置：`research/baichuan2`

   ```text
   baichuan2
       ├── run_baichuan2_7b.yaml               # 7B全量微调910a启动配置
       ├── run_baichuan2_13b.yaml              # 13B全量微调910a启动配置
       ├── run_baichuan2_7b_910b.yaml          # 7B全量微调910b启动配置
       ├── run_baichuan2_13b_910b.yaml         # 13B全量微调910b启动配置
       ├── run_baichuan2_7b_lora_910b.yaml     # 7BLora微调910b启动配置
       └── run_baichuan2_13b_lora_910b.yaml    # 13BLora微调910b启动配置
   ```

3. 数据处理脚本和任务启动脚本：`research/baichuan2`

   ```text
   baichuan2
       ├── belle_preprocess.py     # belle数据集预处理脚本
       └── run_baichuan2.py        # baichuan2高阶接口使用脚本
       └── run_baichuan2_chat.py   # baichuan2 chat推理使用脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Ascend 910A/B
- MindSpore：2.2
- MindFormers版本：0.8
- 硬件支持矩阵

|     模型      | 硬件 | 全量微调 | lora微调 | 推理 |
| :-----------: | :--: | :------: | :------: | :--: |
| Baichuan2-7b  | 910A |  ≥2节点  |  单节点  | 单卡 |
| Baichuan2-7b  | 910B |  单节点  |  单节点  | 单卡 |
| Baichuan2-13b | 910A |  ≥2节点  |  单节点  | ≥2卡 |
| Baichuan2-13b | 910B |  单节点  |  单节点  | 单卡 |

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

当前提供belle_chat_ramdon数据集的预处理和微调样例，用于对Baichuan2-7B-Base，Baichuan2-13B-Base模型进行微调。数据集下载链接如下：

- [belle_chat_ramdon_10k](https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/data/belle_chat_ramdon_10k.json)

执行`belle_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```shell
# 脚本路径：research/baichuan2/belle_preprocess.py
python research/baichuan2/belle_preprocess.py \
--input_glob /{path}/belle_chat_ramdon_10k.json \
--model_file /{path}/tokenizer.model \
--output_file /{path}/belle_512.mindrecord \
--seq_length 512
```

### 模型权重准备

本仓库提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

- [Baichuan2-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Base.ckpt)
- [Baichuan2-7B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt)
- [Baichuan2-13B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_13B_Base.ckpt)
- [Baichuan2-13B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt)
- [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

**注**: 请安装torch=2.0.0和transformers=4.30.2版本

```shell
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/baichuan/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python ./research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME

# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

### [模型权重转换](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

- ## Baichuan2-7B

### 全参微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的belle数据集，参照[模型权重准备](#模型权重准备)章节获取Baichuan2-7B-Base权重。

- **910A**

Baichuan2-7B-Base用于微调，seq_length默认为512，分布式微调训练在910A上需要2节点16卡启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_7b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备 ：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-2机16卡章节，获取两节点合并后的`RANK_TABLE_FILE`文件。

2. 权重准备：将完整权重转为16卡分布式权重。
   ① 若所有节点之间无共享盘，请参考[权重转换文档-离线权重转换](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md#%E7%A6%BB%E7%BA%BF%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)章节，将完整权重转为16卡分布式权重；
   ② 若所有节点之间有共享盘，Mindformer支持自动权重转换，请参考[权重转换文档-物理机多机多卡训练](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md#%E7%89%A9%E7%90%86%E6%9C%BA%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E6%A1%88%E4%BE%8B)案例，可跳过该步骤。

3. 修改`run_baichuan2_7b.yaml`中相关配置，默认节点之间无共享盘，不开启自动权重切分，完整权重已经离线切分为16卡分布式权重。

```yaml
output_dir: './output'          # 默认路径，若需要自动权重转换，请配置为共享盘输出路径
load_checkpoint: 'model_dir'    # 使用分布式权重，权重按照`model_dir/rank_x/xxx.ckpt`格式存放
auto_trans_ckpt: False          # 关闭自动权重转换，若需要自动权重转换，则改为True
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

# 16卡分布式策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

4. 启动微调任务，在多机上同时拉起任务。

```shell
# node 1
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

- **910B**

Baichuan2-7B-Base用于微调，seq_length默认为512，分布式微调训练在910B上单节点即可启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_7b_910b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备 ：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-单机8卡章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_baichuan2_7b_910b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

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
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b_910b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

### Lora微调

Baichuan2-7B-Base用于Lora微调，seq_length默认为512。Lora微调支持910A/B，配置文件基本相同。以`belle_chat_ramdon_10k.json`数据集为例，给出910B的默认配置文件`run_baichuan2_7b_lora_910b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备 ：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-单机8卡章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_baichuan2_7b_lora_910b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```shell
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

# 8卡分布式策略配置
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4

# 权重保存配置参考
parallel:
  strategy_ckpt_config:
    save_file: "./ckpt_strategy.ckpt"
    only_trainable_params: False # 配置为False，保存完整权重

# model增加pet_config
model:
  model_config:
    pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 1
      lora_alpha: 32
      lora_dropout: 0.1
      target_modules: '.*wq|.*wk|.*wv'
```

3. 启动Lora微调任务，在单机上拉起任务。

```bash
cd mindformers/research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b_lora_910b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

- 若要在910A上运行，只需修改配置文件如下：

```yaml
# research/baichuan2/run_baichuan2_7b_lora_910b.yaml
max_device_memory: "31GB"    # 910A将最大内存改为31GB即可
```

### MindSpore推理

Baichuan2-7B-Chat用于在线推理，输入按照`<reserved_106>question<reserved_107>`的模板格式输入，910A/B均支持单卡推理。

以下给出了四种推理方式，仅供参考：

- **基于高阶接口推理**：基于trainer推理，不支持batch推理；
- **基于Pipeline推理**：基于pipeline推理，不支持batch推理；
- **基于Generate推理**：基于generate推理，支持batch推理；
- **chat多轮对话推理**：基于generate推理，支持单卡交互式多轮对话；

请下载词表文件：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

**注：**如需使用多卡推理，请参考[Baichuan2-13B](#Baichuan2-13B)的910A多卡推理示例。

#### 基于高阶接口推理

1. 主要参数配置参考

```yaml
load_checkpoint: 'path/to/Baichuan2_7B_Chat.ckpt'   # 填写权重路径
auto_trans_ckpt: False                              # 关闭自动权重转换
use_past: True                                      # 使用增量推理
vocab_file: 'path/to/tokenizer.model'               # 配置词表路径
use_parallel: False                                 # 关闭并行模式
```

2. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/Baichuan2_7B_Chat.ckpt \
--auto_trans_ckpt False \
--predict_data <reserved_106>你是谁？<reserved_107>

# output: [{'text_generation_text': ['<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

#### 基于Pipeline推理

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: 'path/to/Baichuan2-7B-Chat.ckpt'     # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: 'path/to/tokenizer.model'                         # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

2. 运行run_baichuan2_pipeline.py

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=0, mode=0)

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_7b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_config.model.model_config.batch_size = 1
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model = Baichuan7BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
outputs = pipeline_task(inputs,
                        do_sample=False,
                        top_k=1,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        temperature=1.0,
                        max_length=64)
for output in outputs:
    print(output)

# 推理输出
# {'text_generation_text': ['<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}
# {'text_generation_text': ['<reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。']}
# {'text_generation_text': ['<reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。']}
```

#### 基于Generate推理

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: 'path/to/Baichuan2-7B-Chat.ckpt'     # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: 'path/to/tokenizer.model'                         # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

2. 运行run_baichuan2_generate.py

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=0, mode=0)

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]
batch_size = len(inputs)

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_7b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_config.model.model_config.batch_size = batch_size
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_network = Baichuan7BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
inputs_ids = tokenizer(inputs, max_length=64, padding="max_length")["input_ids"]
outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_length=64)
for output in outputs:
    print(tokenizer.decode(output))

# 推理输出
# <reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>
# <reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。</s>
# <reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>
```

#### chat多轮对话推理

Baichuan2-7B-Chat支持910A/B单卡多轮对话推理，使用`research/baichuan2/run_baichuan2_chat.py`。

```shell
cd research/baichuan2
python run_baichuan2_chat.py \
--config run_baichuan2_7b_910b.yaml \
--use_parallel False \
--load_checkpoint '/path/to/Baichuan2-7B-Chat.ckpt' \
--auto_trans_ckpt False \
--vocab_file '/path/to/tokenizer.model'
```

### MindSpore Lite推理

本章节提供Baichuan2-7B在MindSpore Lite上进行推理的基本使用流程，更多详细的特性介绍可以参考[Mindspore Lite特性文档](../../docs/feature_cards/Inference.md)

#### 模型导出

step 1. 准备好Baichuan2-7B模型相关的配置文件、权重文件放置在同一个文件夹下

```text
infer_baichuan2_7b_dir
    ├── run_baichuan2_7b_910b.yaml    # 推理模型的配置文件
    └── Baichuan2-7B-Chat.ckpt        # 推理模型的权重文件
```

step 2. 修改配置文件，在配置文件中新增infer配置项，在run_baichuan2_7b_910b.yaml中添加如下配置

```yaml
infer:
  prefill_model_path: "/path/to/baichuan2_7b_export/baichuan2_7b_prefill.mindir"
  increment_model_path: "/path/to/baichuan2_7b_export/baichuan2_7b_inc.mindir"
  infer_seq_length: 512
  model_type: mindir

# 参数说明：
prefill_model_path: 全量图路径
increment_model_path: 增量图路径
infer_seq_length: 推理序列长度
model_type: 推理模型类型

# 注意与export.py中的batch_size设置保持一致，如下所示，否则用导出的MindIR图进行单卡推理可能会出现out of memory的问题
# 由于配置了prefill_model_path和increment_model_path两个路径，需要导出增量图，因此在模型配置中打开增量开关，如下所示
# 使用后处理加速需要配置is_sample_acceleration开关，注意与推理脚本中的设置保持一致，如下所示
model:
  model_config:
    batch_size: 1                    # 单batch推理设置为1，多batch推理设置为相应的batch数
    use_past: True
    is_sample_acceleration: False    # 后处理加速开关，当前baichuan2模型暂不支持，设置为False
```

step 3. 执行export.py，完成模型转换

```shell
python mindformers/tools/export.py --model_dir /path/to/infer_baichuan2_7b_dir
```

#### 模型推理

step 1. 利用`模型导出`章节得到的MindIR图，如果是增量模型则会得到两个MindIR图（baichuan2_7b_prefill.mindir和baichuan2_7b_inc.mindir）

step 2. 执行run_infer_main.py脚本，修改相关配置启动推理

```shell
python run_infer_main.py \
--device_id 0 \
--model_name baichuan2_7b \
--seq_length 512 \                            # 注意与export导出时的推理序列长度保持一致
--tokenizer_path path/to/tokenizer.model \    # 不设置时，以from_pretrained的方式自动加载tokenizer（research模型不支持）
--prefill_model_path /path/to/baichuan2_7b_export/baichuan2_7b_prefill.mindir \
--increment_model_path /path/to/baichuan2_7b_export/baichuan2_7b_inc.mindir \
--config_path /path/to/910b_ge_default.cfg \
--do_sample False \
--top_k 1 \
--top_p 1.0 \
--repetition_penalty 1.0 \
--temperature 1.0 \
--max_length 512 \
--is_sample_acceleration False \              # 后处理加速开关，当前baichuan2模型暂不支持，设置为False
--add_special_tokens False \
--stream False

# 参数说明
device_id: 设备物理ID
model_name: 模型名称
seq_length: 推理序列长度
tokenizer_path: 模型tokenizer路径
prefill_model_path: 全量图路径
increment_model_path: 增量图路径
config_path: GE配置文件路径
do_sample: 是否对候选id进行采样
top_k: 选择top_k个token id作为候选
top_p: 将累积概率小于top_k的token id作为候选
repetition_penalty: 生成单词的惩罚因子，设置为1时不打开
temperature: 温度系数，用来调整下个token的概率
max_length: 能够生成的最大语句长度
is_sample_acceleration: 后处理加速开关
add_special_tokens: 对输入token化时是否添加特殊字符
stream: 是否采用流式结果返回
prompt: 输入中加入prompt的内容，Baichuan2可以选择不设置，按默认的prompt进行推理

# output
# ['<reserved_106>解释一下“温故而知新”<reserved_107>“温故而知新”是一个中国古代成语，出自《论语·为政》。它的意思是通过回顾和了解过去的事情，可以从中获得新的知识和启示。这个成语强调了学习和思考的重要性，鼓励人们在不断积累知识的过程中，不断地回顾和总结，从而实现自我提升和成长。']
```

Baichuan2-7B在910B上推荐的GE配置（910b_ge_default.cfg）如下：

```ini
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

# 参数说明
# provider=ge：采用GE接口
# ge.externalWeight=1：将网络中Const/Constant节点的权重保存在单独的文件中
# ge.exec.atomicCleanPolicy=1：不集中清理网络中atomic算子占用的内存
# ge.exec.staticMemoryPolicy=2：网络运行使用动态扩展内存方式
# ge.exec.precision_mode=must_keep_origin_dtype：选择算子精度模式
```

- ## Baichuan2-13B

### 全参微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的belle数据集，参照[模型权重准备](#模型权重准备)章节获取Baichuan2-13B-Base权重。

- **910A**

Baichuan2-13B-Base用于微调，seq_length默认为512，分布式微调训练在910A上需要2节点多卡启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_13b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-2机16卡章节，获取两节点合并后的`RANK_TABLE_FILE`文件。

2. 权重准备：将完整权重转为16卡分布式权重。
   ① 若所有节点之间无共享盘，请参考[权重转换文档-离线权重转换](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md#%E7%A6%BB%E7%BA%BF%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)章节，将完整权重转为16卡分布式权重；
   ② 若所有节点之间有共享盘，Mindformer支持自动权重转换，请参考[权重转换文档-物理机多机多卡训练](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Transform_Ckpt.md#%E7%89%A9%E7%90%86%E6%9C%BA%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E6%A1%88%E4%BE%8B)案例，可跳过该步骤。

3. 修改`run_baichuan2_13b.yaml`中相关配置，默认节点之间无共享盘，不开启自动权重切分，完整权重已经离线切分为16卡分布式权重。

```yaml
output_dir: './output'          # 默认路径，若需要自动权重转换，请配置为共享盘输出路径
load_checkpoint: 'model_dir'    # 使用分布式权重，权重按照`model_dir/rank_x/xxx.ckpt`格式存放
auto_trans_ckpt: False          # 关闭自动权重转换，若需要自动权重转换，则改为True
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

# 16卡分布式策略配置
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

4. 启动微调任务，在多机上同时拉起任务。

```shell
# node 1
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

- **910B**

Baichuan2-13B-Base用于微调，seq_length默认为512，分布式微调训练在910B上单节点即可启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_13b_910b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-单机8卡章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_baichuan2_13b_910b.yaml`中相关配置，默认使用完整权重，开启自动权重转换。

```yaml
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

# 8卡分布式策略配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 4
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 启动微调任务，在单机上拉起任务。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b_910b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

### Lora微调

Baichuan2-13B-Base用于Lora微调，seq_length默认为512。Lora微调支持910A/B，配置文件基本相同。以`belle_chat_ramdon_10k.json`数据集为例，给出910B的默认配置文件`run_baichuan2_13b_lora_910b.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)-单机8卡章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_baichuan2_13b_lora_910b.yaml`中相关配置，默认使用完整权重，开启自动权重转换。

```shell
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]

# 8卡分布式策略配置
parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4

# 权重保存配置参考
parallel:
  strategy_ckpt_config:
    save_file: "./ckpt_strategy.ckpt"
    only_trainable_params: False # 配置为False，保存完整权重

# model增加pet_config
model:
  model_config:
    pet_config:
      pet_type: lora
      # configuration of lora
      lora_rank: 1
      lora_alpha: 32
      lora_dropout: 0.1
      target_modules: '.*wq|.*wk|.*wv'
```

3. 启动Lora微调任务，在单机上拉起任务。

```bash
cd mindformers/research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b_lora_910b.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_x/xxx.ckpt'格式存放
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

- 若要在910A上运行，只需修改配置文件如下：

```yaml
# research/baichuan2/run_baichuan2_13b_lora_910b.yaml
max_device_memory: "31GB"    # 910A将最大内存改为31GB即可

# 910A的8卡分布式策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 1
  pipeline_stage: 4
  micro_batch_num: 4
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

### MindSpore推理

Baichuan2-13B-Chat用于在线推理，输入按照`<reserved_106>question<reserved_107>`的模板格式输入，910B支持单卡推理，910A支持2卡分布式推理。

以下给出了四种推理方式，仅供参考：

- **基于高阶接口推理**：基于trainer推理，不支持batch推理；
- **基于Pipeline推理**：基于pipeline推理，不支持batch推理；
- **基于Generate推理**：基于generate推理，支持batch推理；
- **chat多轮对话推理**：基于generate推理，支持单卡交互式多轮对话；

请下载词表文件：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

#### 基于高阶接口推理

- **910A**

Baichuan2-13B在910A推理至少需要2卡，以单机2卡为例：

1. 生成2卡的`RANK_TABLE_FILE`文件

```shell
python mindformers/tools/hccl_tools.py --device_num [0,2]
```

2. 主要参数配置参考

```yaml
load_checkpoint: 'model_dir'             # 使用完整权重，权重存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True                    # 打开自动权重转换
use_past: True                           # 使用增量推理
use_parallel: True                       # 使用并行模式
vocab_file: 'path/to/tokenizer.model'    # 配置词表路径

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
bash ./run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--predict_data <reserved_106>你是谁？<reserved_107>" RANK_TABLE_FILE [0,2] 2

# output: [{'text_generation_text': ['<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

- **注：推理结束后，保存`output/transformed_checkpoint`到自定义文件夹下，后续分布式推理可以直接加载`transformed_checkpoint`里面的2卡分布式权重**，只需修改如下配置：

```yaml
load_checkpoint: 'transformed_checkpoint'    # 使用转换后的权重，权重存放格式为"transformed_checkpoint/rank_x/xxx.ckpt"
auto_trans_ckpt: False                       # 关闭自动权重转换
```

- **910B**

Baichuan2-13B在910B支持单卡推理。

1. 主要参数配置参考

```yaml
load_checkpoint: 'path/to/Baichuan2_13B_Chat.ckpt'  # 填写权重路径
auto_trans_ckpt: False                              # 关闭自动权重转换
use_past: True                                      # 使用增量推理
vocab_file: 'path/to/tokenizer.model'               # 配置词表路径
use_parallel: False                                 # 关闭并行模式
```

2. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b_910b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint path/to/Baichuan2_13B_Chat.ckpt \
--auto_trans_ckpt False \
--predict_data <reserved_106>你是谁？<reserved_107>

# output: [{'text_generation_text': ['<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

#### 基于Pipeline推理

- **910A**

1. 生成2卡的`RANK_TABLE_FILE`文件

```shell
python mindformers/tools/hccl_tools.py --device_num [0,2]
```

2. 主要参数配置参考

```yaml
load_checkpoint: 'model_dir'             # 使用完整权重，权重存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True                    # 打开自动权重转换
use_parallel: True                       # 使用并行模式
use_past: True                           # 使用增量推理
vocab_file: 'path/to/tokenizer.model'    # 配置词表路径

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 构建run_baichuan2_pipeline.py

```python
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)
build_context(baichuan2_config)

build_parallel_config(baichuan2_config)

baichuan2_config.model.model_config.parallel_config = baichuan2_config.parallel_config
baichuan2_config.model.model_config.batch_size = 1
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model_config.checkpoint_name_or_path = None
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

baichuan2_model = Model(baichuan2_network)

if baichuan2_config.load_checkpoint:
    print("----------------Transform and load checkpoint----------------")
    seq_length = baichuan2_config.model.model_config.seq_length
    infer_data = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
    transform_and_load_checkpoint(baichuan2_config, baichuan2_model, baichuan2_network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
outputs = pipeline_task(inputs,
                        do_sample=False,
                        top_k=1,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        temperature=1.0,
                        max_length=64)
for output in outputs:
    print(output)
```

4. 启动2卡分布式pipeline推理

```shell
cd research
bash run_singlenode.sh "python baichuan2/run_baichuan2_pipeline.py" RANK_TABLE_FILE [0,2] 2

# 推理输出
# {'text_generation_text': [<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>]}
# {'text_generation_text': [<reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>]}
# {'text_generation_text': [<reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>]}
```

- **910B**

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: 'path/to/Baichuan2-13B-Chat.ckpt'    # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: 'path/to/tokenizer.model'                         # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

2. 运行run_baichuan2_pipeline.py

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=0, mode=0)

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b_910b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_config.model.model_config.batch_size = 1
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
outputs = pipeline_task(inputs,
                        do_sample=False,
                        top_k=1,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        temperature=1.0,
                        max_length=64)
for output in outputs:
    print(output)

# 推理输出
# {'text_generation_text': [<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>]}
# {'text_generation_text': [<reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>]}
# {'text_generation_text': [<reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>]}
```

#### 基于Generate推理

- **910A**

1. 生成2卡的`RANK_TABLE_FILE`文件

```shell
python mindformers/tools/hccl_tools.py --device_num [0,2]
```

2. 主要参数配置参考

```yaml
load_checkpoint: 'model_dir'             # 使用完整权重，权重存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True                    # 打开自动权重转换
use_parallel: True                       # 使用并行模式
use_past: True                           # 使用增量推理
vocab_file: 'path/to/tokenizer.model'    # 配置词表路径

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

3. 构建run_baichuan2_generate.py

```python
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]
batch_size = len(inputs)

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)
build_context(baichuan2_config)

build_parallel_config(baichuan2_config)

baichuan2_config.model.model_config.parallel_config = baichuan2_config.parallel_config
baichuan2_config.model.model_config.batch_size = batch_size
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

baichuan2_model = Model(baichuan2_network)

if baichuan2_config.load_checkpoint:
    print("----------------Transform and load checkpoint----------------")
    seq_length = baichuan2_config.model.model_config.seq_length
    infer_data = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
    transform_and_load_checkpoint(baichuan2_config, baichuan2_model, baichuan2_network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
inputs_ids = tokenizer(inputs, max_length=64, padding="max_length")["input_ids"]
outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_length=64)
for output in outputs:
    print(tokenizer.decode(output))
```

4. 启动2卡分布式generate推理

```shell
cd research
bash run_singlenode.sh "python baichuan2/run_baichuan2_generate.py" RANK_TABLE_FILE [0,2] 2

# 推理输出
# <reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>
# <reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>
# <reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>
```

- **910B**

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: 'path/to/Baichuan2-13B-Chat.ckpt'    # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: 'path/to/tokenizer.model'                         # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

2. 运行run_baichuan2_generate.py

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=0, mode=0)

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]
batch_size = len(inputs)

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b_910b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_config.model.model_config.batch_size = batch_size
baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
inputs_ids = tokenizer(inputs, max_length=64, padding="max_length")["input_ids"]
outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_length=64)
for output in outputs:
    print(tokenizer.decode(output))

# 推理输出
# <reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>
# <reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>
# <reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>
```

### chat多轮对话推理

Baichuan2-13B-Chat仅支持910b单卡多轮对话推理，使用`research/baichuan2/run_baichuan2_chat.py`。

```shell
cd research/baichuan2
python run_baichuan2_chat.py \
--config run_baichuan2_13b_910b.yaml \
--use_parallel False \
--load_checkpoint '/path/to/Baichuan2-13B-Chat.ckpt' \
--auto_trans_ckpt False \
--vocab_file '/path/to/tokenizer.model'
```

- **注：910A需开启2卡分布式推理，不支持交互。**

### MindSpore Lite推理

本章节提供Baichuan2-13B在MindSpore Lite上进行推理的基本使用流程，更多详细的特性介绍可以参考[Mindspore Lite特性文档](../../docs/feature_cards/Inference.md)

#### 模型导出

step 1. 准备好Baichuan2-13B模型相关的配置文件、权重文件放置在同一个文件夹下

```text
infer_baichuan2_13b_dir
    ├── run_baichuan2_13b_910b.yaml    # 推理模型的配置文件
    └── Baichuan2-13B-Chat.ckpt        # 推理模型的权重文件
```

step 2. 修改配置文件，在配置文件中新增infer配置项，在run_baichuan2_13b_910b.yaml中添加如下配置

```yaml
infer:
  prefill_model_path: "/path/to/baichuan2_13b_export/baichuan2_13b_prefill.mindir"
  increment_model_path: "/path/to/baichuan2_13b_export/baichuan2_13b_inc.mindir"
  infer_seq_length: 512
  model_type: mindir

# 参数说明：
prefill_model_path: 全量图路径
increment_model_path: 增量图路径
infer_seq_length: 推理序列长度
model_type: 推理模型类型

# 注意与export.py中的batch_size设置保持一致，如下所示，否则用导出的MindIR图进行单卡推理可能会出现out of memory的问题
# 由于配置了prefill_model_path和increment_model_path两个路径，需要导出增量图，因此在模型配置中打开增量开关，如下所示
# 使用后处理加速需要配置is_sample_acceleration开关，注意与推理脚本中的设置保持一致，如下所示
model:
  model_config:
    batch_size: 1                    # 单batch推理设置为1，多batch推理设置为相应的batch数
    use_past: True
    is_sample_acceleration: False    # 后处理加速开关，当前baichuan2模型暂不支持，设置为False
```

step 3. 执行export.py，完成模型转换

```shell
python mindformers/tools/export.py --model_dir /path/to/infer_baichuan2_13b_dir
```

#### 模型推理

step 1. 利用`模型导出`章节得到的MindIR图，如果是增量模型则会得到两个MindIR图（baichuan2_13b_prefill.mindir和baichuan2_13b_inc.mindir）

step 2. 执行run_infer_main.py脚本，修改相关配置启动推理

```shell
python run_infer_main.py \
--device_id 0 \
--model_name baichuan2_13b \
--seq_length 512 \                            # 注意与export导出时的推理序列长度保持一致
--tokenizer_path path/to/tokenizer.model \    # 不设置时，以from_pretrained的方式自动加载tokenizer（research模型不支持）
--prefill_model_path /path/to/baichuan2_13b_export/baichuan2_13b_prefill.mindir \
--increment_model_path /path/to/baichuan2_13b_export/baichuan2_13b_inc.mindir \
--config_path /path/to/910b_ge_default.cfg \
--do_sample False \
--top_k 1 \
--top_p 1.0 \
--repetition_penalty 1.0 \
--temperature 1.0 \
--max_length 512 \
--is_sample_acceleration False \              # 后处理加速开关，当前baichuan2模型暂不支持，设置为False
--add_special_tokens False \
--stream False

# 参数说明
device_id: 设备物理ID
model_name: 模型名称
seq_length: 推理序列长度
tokenizer_path: 模型tokenizer路径
prefill_model_path: 全量图路径
increment_model_path: 增量图路径
config_path: GE配置文件路径
do_sample: 是否对候选id进行采样
top_k: 选择top_k个token id作为候选
top_p: 将累积概率小于top_k的token id作为候选
repetition_penalty: 生成单词的惩罚因子，设置为1时不打开
temperature: 温度系数，用来调整下个token的概率
max_length: 能够生成的最大语句长度
is_sample_acceleration: 后处理加速开关
add_special_tokens: 对输入token化时是否添加特殊字符
stream: 是否采用流式结果返回
prompt: 输入中加入prompt的内容，Baichuan2可以选择不设置，按默认的prompt进行推理

# output
# ['<reserved_106>解释一下“温故而知新”<reserved_107>“温故而知新”是一句源自中国古代的成语，出自《论语·为政》篇。这句话的意思是：通过回顾过去，我们可以发现新的知识和理解。具体来说，它鼓励我们在学习或工作中，不仅要关注新的知识和技能，还要回顾和巩固已经学过的内容。这样，我们既能保持对旧知识的熟练度，又能不断发现新的启示和进步。\n\n这句话强调了学习和成长的一个重要原则：不断回顾和反思。通过回顾过去的经验，我们可以发现新的视角、灵感和方法，从而更好地理解和掌握知识。同时，这也提醒我们要保持谦逊和好奇心，不断地在学习中寻求进步。']
```

Baichuan2-13B在910B上推荐的GE配置（910b_ge_default.cfg）如下：

```ini
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

# 参数说明
# provider=ge：采用GE接口
# ge.externalWeight=1：将网络中Const/Constant节点的权重保存在单独的文件中
# ge.exec.atomicCleanPolicy=1：不集中清理网络中atomic算子占用的内存
# ge.exec.staticMemoryPolicy=2：网络运行使用动态扩展内存方式
# ge.exec.precision_mode=must_keep_origin_dtype：选择算子精度模式
```
