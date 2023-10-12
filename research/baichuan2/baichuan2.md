# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## 模型性能

|                            config                            |      task       | Datasets | [train performance](#全参微调) |  [predict performance](#推理)  |
| :----------------------------------------------------------: | :-------------: | :------: | :----------------------------: | :----------------------------: |
| [baichuan2_7b](../../research/baichuan2/run_baichuan2_7b.yaml) | text_generation |  belle   |         513.8 tokens/s         | 20.83 tokens/s (use past True) |
| [baichuan2_13b](../../research/baichuan2/run_baichuan2_13b.yaml) | text_generation |  belle   |          393 tokens/s          | 15.77 tokens/s (use past True) |
| [baichuan2_7b_910b](../../research/baichuan2/run_baichuan2_7b_910b.yaml) | text_generation |  belle   |        2037.8 tokens/s         | 14.71 tokens/s (use past True) |
| [baichuan2_13b_910b](../../research/baichuan2/run_baichuan2_13b_910b.yaml) | text_generation |  belle   |          525 tokens/s          | 9.25 tokens/s (use past True)  |

## 仓库介绍

`Baichuan2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/baichuan2`

   ```text
   baichuan2
       ├── baichuan2_tokenizer.py       # tokenizer
       ├── baichuan2_7b.py              # 7B模型实现
       └── baichuan2_13b.py             # 13B模型实现
   ```

2. 模型配置：`research/baichuan2`

   ```text
   baichuan2
       ├── run_baichuan2_7b.yaml             # 7B全量微调910a启动配置
       ├── run_baichuan2_13b.yaml            # 13B全量微调910a启动配置
       ├── run_baichuan2_7b_910b.yaml        # 7B全量微调910b启动配置
       └── run_baichuan2_13b_910b.yaml       # 13B全量微调910b启动配置
   ```

3. 数据处理脚本和任务启动脚本：`research/baichuan2`

   ```text
   baichuan2
       ├── belle_preprocess.py            # belle数据集预处理脚本
       └── run_baichuan2.py               # baichuan2高阶接口使用脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Ascend 910A/B
- MindSpore：2.0.0 / 1.10.1
- MindFormers版本：dev

注：Baichuan2-7B推理可在单机单卡上完成部署，全量微调至少需要16卡。Baichuan2-13B推理至少需要4卡，全量微调至少需要16卡。

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

### 多机RANK_TABLE_FILE合并(多机多卡必备环)

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

### 模型权重下载与转换

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

- [Baichuan2-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Base.ckpt)
- [Baichuan2-7B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt)
- [Baichuan2-13B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_13B_Base.ckpt)
- [Baichuan2-13B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huffingface权重的链接如下：

- [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

**注**: 请安装torch=2.0.0和transformers=4.29.2版本

```bash
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.29.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/baichuan/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```bash
python ./research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

- ## Baichuan2-7B

### 微调

#### 数据集准备-SFT微调数据集

当前提供belle_chat_ramdon数据集的预处理和微调样例，用于对Baichuan2-7B-Base，Baichuan2-13B-Base模型进行微调。数据集下载链接如下：

- [belle_chat_ramdon_10k](https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/data/belle_chat_ramdon_10k.json)

执行`belle_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：research/baichuan2/belle_preprocess.py
python belle_preprocess.py \
--input_glob /{path}/belle_chat_ramdon_10k.json \
--model_file /{path}/tokenizer.model \
--output_file /{path}/belle_512.mindrecord \
--seq_length 512
```

#### 全参微调

全参微调需要多卡启动，以`belle_chat_ramdon_10k.json`数据集为例,给出了默认配置文件`run_baichuan2_7b.yaml`。

1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── baichuan2_7b.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入权重路径文件夹即可。

2. 修改`run_baichuan2_7b.yaml`中相关配置

```yaml
output_dir: './output'
load_checkpoint: '{path}/'        # 添加预训练权重路径
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/belle512.mindrecord"        # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]
```

3. 启动微调任务，以默认配置2机16卡为例，按照以下步骤启动：

- step 1. 首先参考在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件

```bash
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 合并每台机器上生成的`RANK_TABLE_FILE`

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

将不同机器上生成的`RANK_TABLE_FILE`文件拷贝到一起，执行`merge_hccl.py`脚本进行合并，包括server_list合并，`server_count`设为机器数，`rank_id`顺序增加。

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，**需要保证执行的节点和`RANK_TABLE_FILE`的节点顺序保持一致，即rank_id匹配**

- step 4. 根据服务器节点数等信息，修改相应的配置

```yaml
# 以Baichuan2-7B模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/baichuan2/run_baichuan2_7b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本

在多机上同时拉起任务，每台机器拉起方式如下：

```bash
# node 1
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [8,16] 16
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

### 推理

#### 基于高阶接口的推理

1. 配置文件设置，添加tokenizer路径`vocab_file`，并设置`batch_size`值为`1`

在使用Trainer接口进行推理时，若用户自行下载Baichuan2-7B权重，请在启动前先在配置文件中将tokenizer.model的路径自行配置，配置项为vocab_file。

```yaml
# research/baichuan2/run_baichuan2_7b.yaml
# runner config
runner_config:
  epochs: 1
  batch_size: 1                 # batch_size设为1
  sink_mode: True
  sink_size: 2
...
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<unk>'
   bos_token: '<s>'
   eos_token: '</s>'
   pad_token: '</s>'
   vocab_file: '/path/Baichuan2-7b/tokenizer.model'        # 添加tokenizer路径
   type: Baichuan2Tokenizer
```

2. Trainer接口启动推理

Baichuan2-7B的高阶接口使用脚本已集成在run_baichuan2.py脚本中，运行此脚本命令示例：

```bash
python run_baichuan2.py \
--config "run_baichuan2_7b.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '将以下内容翻译成英文：你今天真好看。' \
--device_id 0

# output: [{'text_generation_text': ['将以下内容翻译成英文：你今天真好看。 \nYou look really nice today.']}]
```

#### 基于Pipeline的推理

```python
# predict_custom.py 文件
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=0, mode=0)
# init model
baichuan2_model_path = "/path/Baichuan2-7B/baichuan2_7b.ckpt" # Baichuan2-7B ckpt path
baichuan2_config = LlamaConfig(
    vocab_size=125696,
    pad_token_id=0,
    rms_norm_eps=1.0e-6,
    checkpoint_name_or_path=baichuan2_model_path,
    use_past=True
)
baichuan2_model = Baichuan7BV2ForCausalLM(
    config=baichuan2_config
)
# init tokenizer
tokenizer_path = "/path/Baichuan2-7B/tokenizer.model" # Baichuan2-7B tokenizer.model path
tokenizer = Baichuan2Tokenizer(
    vocab_file=tokenizer_path
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("诗词接龙：白日依山尽的下一句是什么？",
                                do_sample=False,
                                repetition_penalty=1.05,
                                max_length=64)

print(pipeline_result)

# output: [{'text_generation_text': ['诗词接龙：白日依山尽的下一句是什么？ \n答：黄河入海流。']}]
```

以下为多卡运行自定义多batch推理的脚本

```python
# run_predict.sh 文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0    # this server start rank
export END_RANK=8      # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path CHECKPOINT_PATH &> minformers_$RANK_ID.log &
done
```

##### 单卡pipeline推理

```bash
python predict_custom.py
```

##### 多卡pipeline推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/baichuan2_7b_shard_checkpoint_dir
```

- ## Baichuan2-13B

### 微调

#### 数据集准备-SFT微调数据集

参考[Baichuan2-7B-微调-数据集准备](##Baichuan2-7B)

#### 全参微调

- ##### 910A

Baichuan2-13B-Base用于微调，seq_length默认为512，分布式微调训练在910A上需要2节点多卡启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_13b.yaml`。

1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── baichuan2_13b.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入权重路径文件夹即可。

2. 修改`run_baichuan2_13b.yaml`中相关配置

```yaml
output_dir: './output'
load_checkpoint: '{path}/'      # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True           # 打开权重自动转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/belle512.mindrecord"       # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]
```

3. 启动微调任务，以默认配置2机16卡为例，按照以下步骤启动：

- step 1. 首先参考在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件

```bash
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 合并每台机器上生成的`RANK_TABLE_FILE`

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

将不同机器上生成的`RANK_TABLE_FILE`文件拷贝到一起，执行`merge_hccl.py`脚本进行合并，包括server_list合并，`server_count`设为机器数，`rank_id`顺序增加。

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，**需要保证执行的节点和`RANK_TABLE_FILE`的节点顺序保持一致，即rank_id匹配**

- step 4. 根据服务器节点数等信息，修改相应的配置

```yaml
# 以Baichuan2-13B模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/baichuan2/run_baichuan2_13b.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 2
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本

在多机上同时拉起任务，每台机器拉起方式如下：

```bash
# node 1
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--load_checkpoint path/to/baichuan2_13b_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--load_checkpoint path/to/baichuan2_13b_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [8,16] 16
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

- ##### 910B

Baichuan2-13B-Base用于微调，seq_length默认为512，分布式微调训练在910B上单节点即可启动。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`run_baichuan2_13b_910b.yaml`。

1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── baichuan2_13b.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入权重路径文件夹即可。

2. 修改`run_baichuan2_13b_910b.yaml`中相关配置

```yaml
output_dir: './output'
load_checkpoint: '{path}/'      # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True           # 打开权重自动转换
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/belle512.mindrecord"       # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 指令微调时（如belle数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]
```

3. 启动微调任务，以默认配置2机16卡为例，按照以下步骤启动：

- step 1. 首先参考在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件

```bash
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 根据服务器节点数等信息，修改相应的配置

```yaml
# 以Baichuan2-13B模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/baichuan2/run_baichuan2_13b_910b.yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 4
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本

在单机上拉起任务：

```bash
cd mindformers/research
bash run_multinode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b_910b.yaml \
--load_checkpoint path/to/baichuan2_13b_910b_ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 8
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

### 推理

#### 基于高阶接口的推理

- ##### 910A

**注1**：Baichuan2-13B-Chat用于推理，seq_length默认为512，推理需要2卡，不支持单卡推理。

**注2**：由于Baichuan2-13B基于高阶接口的形式开发，存放于research文件夹下，使用时需要将mindformers安装为python的包，才能直接进入research目录下执行相关命令。

**注3**：当前`run_baichuan2_13b.yaml`文件默认为train配置，用于eval和predict时需要修改并行策略。

以单机2卡为例：

1. 主要参数配置参考

```yaml
load_checkpoint: '{path}/'   # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True        # 打开权重自动转换
use_past: True               # 打开增量推理
vocab_file: 'path/to/tokenizer.model'

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

2. 生成2卡的rank_table_file

```bash
python mindformers/tools/hccl_tools.py --device_num [0,2]
```

3. 启动推理

```bash
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
bash ./run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint path/to/baichuan2-13b-chat.ckpt \
--auto_trans_ckpt True \
--predict_data 你是谁？" rank_table_file [0,2] 2

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

- 注：推理结束后，保存`output/transformed_checkpoint`到自定义文件夹下，后续分布式推理可以直接加载`transformed_checkpoint`里面的4卡分布式权重，配置修改如下：

```yaml
load_checkpoint: 'transformed_checkpoint' # 完整模型存放格式为"transformed_checkpoint/rank_x/xxx.ckpt"
auto_trans_ckpt: False                    # 关闭权重自动转换
```

- ##### 910B

**注1**：Baichuan2-13B-Chat用于推理，seq_length默认为512，支持单卡推理。

**注2**：由于Baichuan2-13B基于高阶接口的形式开发，存放于research文件夹下，使用时需要将mindformers安装为python的包，才能直接进入research目录下执行相关命令。

**注3**：当前`run_baichuan2_13b.yaml`文件默认为train配置，用于eval和predict时需要修改并行策略。

1. 主要参数配置参考

```yaml
load_checkpoint: '{path}/'   # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: False       # 关闭权重自动转换
use_past: True               # 打开增量推理
vocab_file: 'path/to/tokenizer.model'

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

2. 启动推理

```bash
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_13b_910b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint path/to/baichuan2-13b-chat.ckpt \
--auto_trans_ckpt True \
--predict_data 你是谁？

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

#### 基于Pipeline的推理

- ##### 910A

1. 主要参数配置参考

```yaml
load_checkpoint: '{path}/'   # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True        # 打开权重自动转换
use_past: True               # 打开增量推理
vocab_file: 'path/to/tokenizer.model'
use_parallel: True

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

2. 构建run_baichuan2_pipeline.py

```python
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)
build_context(baichuan2_config)

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model_config.checkpoint_name_or_path = None
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

baichuan2_model = Model(baichuan2_network)

print("----------------Transform and load checkpoint----------------")
seq_length = baichuan2_config.model.model_config.seq_length
infer_data = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
transform_and_load_checkpoint(baichuan2_config, baichuan2_model, baichuan2_network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("你是谁？",
                                do_sample=False,
                                top_k=1,
                                top_p=1.0,
                                repetition_penalty=1.0,
                                temperature=1.0,
                                max_length=64)

print(pipeline_result)
```

3. 启动2卡分布式pipeline推理

```bash
cd research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2_pipeline.py" \
path/to/rank_table_file [0,2] 2

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

- ##### 910B

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                        # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                     # 关闭权重自动转换
checkpoint_name_or_path: 'path/to/baichuan2-13B-Chat.ckpt' # 填写权重绝对路径
use_past: True                                             # 打开增量推理
vocab_file: 'path/to/tokenizer.model'
use_parallel: False
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
# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b_910b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
pipeline_task = pipeline(task="text_generation", model=baichuan2_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("你是谁？",
                                do_sample=False,
                                top_k=1,
                                top_p=1.0,
                                repetition_penalty=1.0,
                                temperature=1.0,
                                max_length=64)

print(pipeline_result)

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

#### 基于Generate的推理

- ##### 910A

1. 主要参数配置参考

```yaml
load_checkpoint: '{path}/'   # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True        # 打开权重自动转换
use_past: True               # 打开增量推理
vocab_file: 'path/to/tokenizer.model'
use_parallel: True

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

2. 构建run_baichuan2_generate.py

```python
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers import MindFormerConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context

from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)
build_context(baichuan2_config)

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_model_config.checkpoint_name_or_path = None
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

baichuan2_model = Model(baichuan2_network)

print("----------------Transform and load checkpoint----------------")
seq_length = baichuan2_config.model.model_config.seq_length
infer_data = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
transform_and_load_checkpoint(baichuan2_config, baichuan2_model, baichuan2_network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
inputs_ids = tokenizer("你是谁？", max_length=baichuan2_model_config.max_decode_length, padding="max_length")["input_ids"]
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

3. 启动2卡分布式generate推理

```bash
cd research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2_generate.py" \
path/to/rank_table_file [0,2] 2

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

- ##### 910B

1. 主要参数配置参考

```yaml
load_checkpoint: ''                                        # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                     # 关闭权重自动转换
checkpoint_name_or_path: 'path/to/baichuan2-13B-Chat.ckpt' # 填写权重绝对路径
use_past: True                                             # 打开增量推理
vocab_file: 'path/to/tokenizer.model'
use_parallel: False
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
# init model
baichuan2_config_path = "research/baichuan2/run_baichuan2_13b_910b.yaml"
baichuan2_config = MindFormerConfig(baichuan2_config_path)

baichuan2_model_config = LlamaConfig(**baichuan2_config.model.model_config)
baichuan2_network = Baichuan13BV2ForCausalLM(
    config=baichuan2_model_config
)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=baichuan2_config.processor.tokenizer.vocab_file
)
inputs_ids = tokenizer("你是谁？", max_length=baichuan2_model_config.max_decode_length, padding="max_length")["input_ids"]
outputs = baichuan2_network.generate(inputs_ids,
                                     do_sample=False,
                                     top_k=1,
                                     top_p=1.0,
                                     repetition_penalty=1.0,
                                     temperature=1.0,
                                     max_length=64)
for output in outputs:
    print(tokenizer.decode(output))

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```