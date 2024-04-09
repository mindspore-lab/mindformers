# iFlytekSpark

## 模型描述

讯飞星火开源-13B（iFlytekSpark-13B）拥有130亿参数，新一代认知大模型，一经发布，众多科研院所和高校便期待科大讯飞能够开源。 为了让大家使用的更加方便，科大讯飞增加了更多的数据，并针对工具链进行了优化。此次正式开源拥有130亿参数的iFlytekSpark-13B模型（讯飞星火开源-13B），也是首个基于全国产化算力平台“飞星一号”的大模型，正式开源！

iFlytekSpark-13B不仅具备通用任务处理能力如聊天、问答、文本提取和分类等，还具备数据分析和代码生成等生产力功能。我们特别在学习辅助、数学、推理等领域进行了深度优化，大幅提升模型的实用性和易用性。详细的评测结果见下面评测部分。

本次开源，既包含基础模型iFlytekSpark-13B-base、精调模型iFlytekSpark-13B-chat，也开源了微调工具iFlytekSpark-13B-Lora、人设定制工具iFlytekSpark-13B-Charater，让企业和学术研究可以基于这些全栈自主创新的星火优化套件方便地训练自己的专用大模型。

星火开源-13B在多项知名公开评测任务中名列前茅，在文本生成、语言理解、文本改写、行业问答、机器翻译等企业典型场景中，通过对学习辅助、语言理解等领域的深入研究和优化，大幅提升了其实用性，在处理复杂的自然语言任务时更加得心应手，确保了其在面对多样化和专业化的应用场景时能够保持高效和准确，效果显著优于其他同等尺寸的开源模型。

这对于追求高性能而对成本敏感的企业来说，无疑是一个巨大的吸引力，也为各行各业的企业提供了一种性价比高的解决方案。

## 仓库介绍

`iFlytekSpark` 基于 [MindFormers](https://gitee.com/mindspore/mindformers) 套件实现，主要涉及的文件有：

1. 模型具体实现：`research/iflytekspark`

   ```text
       ├── iflytekspark_config.py             # 模型配置基类
       ├── iflytekspark_infer.py              # 在线推理脚本
       ├── iflytekspark_layers.py             # 模型基本模块实现
       ├── iflytekspark_model.py              # 模型实现
       ├── iflytekspark_sampler.py            # 在线推理后处理采样实现
       ├── iflytekspark_streamer.py           # 流式推理实现
       ├── iflytekspark_text_generator.py     # 在线推理API
       ├── iflytekspark_tokenizer.py          # tokenizer
       ├── optim.py                           # 优化器实现
       └── repetition_processor.py            # Repetition算法实现
   ```

2. 模型配置文件：`research/iflytekspark`

   ```text
       ├── run_iflytekspark_13b_infer_800T_A2_64G.yaml            # 13B在线推理配置（适用Atlas 800T A2）
       ├── run_iflytekspark_13b_infer_800_32G.yaml                # 13B在线推理配置（适用Atlas 800）
       ├── run_iflytekspark_13b_infer_lora_800T_A2_64G.yaml       # 13BLora模型在线推理配置（适用Atlas 800T A2）
       ├── run_iflytekspark_13b_infer_lora_800_32G.yaml           # 13BLora模型在线推理配置（适用Atlas 800）
       ├── run_iflytekspark_13b_lora_800T_A2_64G.yaml             # 13BLora微调配置（适用Atlas 800T A2）
       ├── run_iflytekspark_13b_lora_800_32G.yaml                 # 13BLora微调配置（适用Atlas 800）
       ├── run_iflytekspark_13b_pretrain_800T_A2_64G.yaml         # 13B预训练配置（适用Atlas 800T A2）
       ├── run_iflytekspark_13b_pretrain_800_32G.yaml             # 13B预训练配置（适用Atlas 800）
       ├── run_iflytekspark_13b_sft_800T_A2_64G.yaml              # 13B全量微调配置（适用Atlas 800T A2）
       └── run_iflytekspark_13b_sft_800_32G.yaml                  # 13B全量微调配置（适用Atlas 800）
   ```

3. 数据处理脚本、权重处理脚本及任务启动脚本：`research/iflytekspark`

   ```text
       ├── pretrain_data_preprocess.py      # 预训练数据处理脚本
       ├── run_iflytekspark.py              # 高阶接口使用脚本
       ├── sft_data_preprocess.py           # 微调数据处理脚本
       └── weight_convert.py                # mindspore BF16格式权重转换脚本
   ```

    `run_iflytekspark.py`接受以下参数，通过Shell脚本传入参数时，**传入参数优先级高于配置文件中对应的配置项**。
    - task：任务类型，默认值：`text_generation`。
    - config：配置文件路径，必须指定。
    - run_mode：运行模式，包括`train`，`finetune`，`eval`，`predict`和`export`，默认值：`train`。
    - use_parallel：是否使能并行，默认值：`False`。
    - load_checkpoint：加载checkpoint路径，默认值：`None`。
    - auto_trans_ckpt：自动根据当前并行策略进行checkpoint切分，默认值：`None`。
    - resume：断点续训，默认值：`False`。
    - train_dataset：训练数据集路径，默认值：`None`。
    - eval_dataset：验证数据集路径，默认值：`None`。
    - predict_data：推理数据集路径，默认值：`None`。
    - predict_length：模型的最大推理长度，默认值：`512`。
    - predict_batch：模型推理的batch数，默认值：`1`。
    - optimizer_parallel：是否使能优化器并行，默认值：`False`。
    - device_id：指定设备id，仅在非并行模式下生效，默认值：`0`。
    - prompt：推理使用的语料模版，默认值：`None`，表示不使用模板。
    - tokenizer_file：tokenizer文件路径，进行在线推理时需要指定。
    - mindir_save_dir：导出mindir模型文件路径，默认当前路径。
    - streamer：是否使能流式推理，默认值：`False`。

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800/Atlas 800T A2
- MindSpore：2.2.13
- MindFormers版本：r1.0
- 硬件支持矩阵如下

|     模型      | 硬件 | 预训练 | 全量微调 | lora微调 | 推理 |
| :-----------: | :--: | :------: | :------: | :------: | :--: |
| iFlytekSpark-13b  | Atlas 800 | ≥2节点 |  ≥2节点  |  单节点  | ≥2卡 |
| iFlytekSpark-13b  | Atlas 800T A2 | 单节点 |  单节点  |  单节点  | 单卡 |

### [acctransformer安装](https://gitee.com/mindspore/acctransformer/tree/fa1_for_ms2.2.11/)

在Atlas 800机型上进行iFlytekSpark模型的训练、微调、推理，需要安装acctransformer套件使能FlashAttention。执行以下命令克隆源码到本地：

```shell
git clone -b fa1_for_ms2.2.11 https://gitee.com/mindspore/acctransformer.git
```

安装方法如下：

1. 直接克隆源码使用，使用源码方式调用时设置PYTHONPATH。

```shell
export PYTHONPATH=/yourcodepath/acctransformer/train:$PYTHONPATH
```

2. 安装whl包使用。

```shell
cd train
python setup.py install
```

或

```shell
cd train
bash build.sh
pip install dist/acctransformer-1.0.0-py3-none-any.whl
```

### RANK_TABLE_FILE准备

- **单节点**

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

`RANK_TABLE_FILE`` 单机8卡参考样例:

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

- **多节点**

以2机16卡为例：

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

`RANK_TABLE_FILE` 双机16卡参考样例:

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

本章节以alpaca_gpt4数据集为例，介绍如何使用本仓提供的脚本制作数据集，用于对 iFlytekSpark-13B 模型进行预训练和微调。alpaca_gpt4数据集下载链接如下：

- [alpaca_gpt4_data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data)

执行`pretrain_data_preprocess.py`，进行预训练数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。
执行`sft_data_preprocess.py`，进行微调数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```shell
# 预训练
python ./research/iflytekspark/pretrain_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH

# SFT&Lora
python ./research/iflytekspark/sft_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH \
--pad_id PAD_ID

# 参数说明
tokenizer: Tokenizer文件路径，指定至含有 .model 文件的文件夹
raw_data_path: 原始文本数据文件路径
output_filename: 数据集Mindrecord保存文件名，需指定到以.mindrecord结尾的文件名。可选参数，默认值为dataset.mindrecord
seq_length: 数据集单个样本句长，默认值：32768
pad_id: SFT数据集用于padding到指定句长的padding token id，默认值：0
```

### 模型权重准备

本仓提供支持MindSpore框架的预训练权重、微调权重和词表文件用于训练/微调/推理。

**预训练权重**

- [iFlytekSpark_13b_base_fp32](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_base_fp32.ckpt)
- [iFlytekSpark_13b_base_bf16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_base_bf16.ckpt)
- [iFlytekSpark_13b_base_fp16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_base_fp16.ckpt)

**微调权重**

- [iFlytekSpark_13b_chat_fp32](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_chat_fp32.ckpt)
- [iFlytekSpark_13b_chat_bf16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_chat_bf16.ckpt)
- [iFlytekSpark_13b_chat_fp16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/iflytekspark_13b_chat_fp16.ckpt)

**Tokenizer：**

- [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/tokenizer.model)
- [tokenizer.vocab](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/iflytekspark/tokenizer.vocab)

本仓开源的权重包含Float32，BFloat16和Float16三种格式。也可使用`/research/iflytekspark/weight_convert.py`脚本将已有MindSpore权重进行数据类型的转换。

**注**：BFloat16数据类型训练仅在Atlas 800T A2型号服务器上支持。在线推理不支持BFloat16数据类型。

```shell
python ./research/iflytekspark/weight_convert.py \
--src_ckpt /{ORIGIN_CKPT} \
--dst_ckpt /{TARGET_CKPT} \
--dtype {TARGET_DTYPE} \
--embed_bf16         # (optional)
--layernorm_bf16     # (optional)

# 参数说明
src_ckpt: 原始MindSpore权重保存路径。
dst_ckpt: 转换后Bfloat16数据类型权重保存路径。
dtype: 转换后权重的数据类型（embedding、layernorm除外），支持float16、float32和bfloat16，默认bfloat16。
embed_bf16: embedding层采用Bfloat16数据类型计算，当前版本不支持。默认不开启。
layernorm_bf16: layernorm层采用Bfloat16数据类型计算。默认不开启。
```

例：Bfloat16训练权重转换为Float16数据格式

```shell
python ./research/iflytekspark/weight_convert.py \
--src_ckpt /{ORIGIN_CKPT} \
--dst_ckpt /{TARGET_CKPT} \
--dtype float16
```

### [模型权重转换](../../docs/feature_cards/Transform_Ckpt.md)

本仓提供的权重下载链接是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。

- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

**示例**：

- 单卡权重转换为多卡权重

本仓提供的默认配置脚本和教程中的启动方式已默认开启自动权重转换，使用时只需从[模型权重准备](#模型权重准备)章节给出的链接下载完整权重并以`model_dir/rank_0/xxx.ckpt`的路径形式存储。`yaml`配置文件中`load_checkpoint`字段指定至`model_dir`目录层级。

- 多卡权重转换为单卡权重

在进行分布式训练后，需要将训练保存的分布式权重合并为完整权重进行在线推理。使用本仓提供的默认配置及启动命令，训练结束后会在`research`路径下生成`output`文件夹：

```text
    ├── checkpoint                # 训练过程中保存的checkpoint（包含优化器状态等信息）
    ├── checkpoint_network        # 训练结束保存的checkpoint（仅包含模型权重）
    ├── log                       # 训练日志
    ├── strategy                  # 模型切分策略文件
    └── transformed_checkpoint    # 开启auto_trans_ckpt保存的切分后初始权重
```

以单机8卡，模型并行`model_parallel=8`为例，执行以下脚本将多卡权重合并为单卡权重：

```bash
cd mindformers
python mindformers/tools/transform_ckpt.py \
--src_ckpt_strategy {path_to_mindformers}/research/output/strategy/ckpt_strategy_rank_0_rank_0.ckpt \
--dst_ckpt_strategy None \
--src_ckpt_dir {path_to_mindformers}/research/output/checkpoint_network \
--dst_ckpt_dir {checkpoint_save_path} \
--prefix "checkpoint_"
```

## 预训练

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。
执行`pretrain_data_preprocess.py`脚本制作预训练mindrecord格式数据集需要。

```shell
python ./research/iflytekspark/pretrain_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH
```

- **单机训练**

iFlytekSpark-13B用于预训练，seq_length默认为32768，分布式预训练在Atlas 800T A2上训练，单节点8卡即可启动。Atlas 800T A2上默认使用BFloat16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_pretrain_800T_A2_64G.yaml`。

**步骤**

1. 多卡运行需要 RANK_TABLE_FILE,  请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节生成对应文件。

2. 修改模型对应的配置, 在 `run_iflytekspark_13b_pretrain_800T_A2_64G.yaml`中, 用户可以自行修改模型配置、训练相关参数。

```yaml
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'train'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动运行脚本, 进行单节点8卡分布式运行。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekfpark_13b_pretrain_800T_A2_64G.yaml \
--load_checkpoint model_dir \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

**注**：

- `run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- `load_checkpoint`指定ckpt的加载路径，也可通过`yaml`配置文件中`load_checkpoint`字段确定。路径径格式需满足`{model_dir}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{model_dir}`该层目录即可

- **多机训练**

iFlytekSpark-13B用于预训练，seq_length默认为32768，分布式预训练在Atlas 800上训练，至少需要2节点16卡启动。Atlas 800上默认使用Float16类型训练，需要开启序列并行（配置文件中设置`seq_parallel=True`）。本仓给出了默认配置文件`run_iflytekspark_13b_pretrain_800_32G.yaml`。

**注：**此处以在Atlas 800上基于2节点16卡进行训练为例，使用更多的节点或在Atlas 800T A2上进行多机训练参考以下样例修改对应配置文件和参数。

**步骤**

1. 多卡运行需要 RANK_TABLE_FILE,  请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-多节点章节生成对应文件。

2. 修改模型对应的配置, 在 `run_iflytekspark_13b_pretrain_800_32G.yaml`中, 用户可以自行修改模型配置、训练相关参数。

```yaml
oad_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'train'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

3. 启动运行脚本, 进行2节点16卡分布式运行。

```shell
# node 1
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_pretrain_800_32G.yaml \
--load_checkpoint model_dir \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_pretrain_800_32G.yaml \
--load_checkpoint model_dir \
--use_parallel True \
--run_mode train \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16
```

**注**：

- `run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- `load_checkpoint`指定ckpt的加载路径，也可通过`yaml`配置文件中`load_checkpoint`字段确定。路径径格式需满足`{model_dir}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{model_dir}`该层目录即可

## 全参微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。
执行`sft_data_preprocess.py`脚本制作预训练mindrecord格式数据集需要。

```shell
python ./research/iflytekspark/sft_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH \
--pad_id PAD_ID
```

- **单机训练**

iFlytekSpark-13B用于全参微调，seq_length默认为32768，分布式微调在Atlas 800T A2上训练，单节点8卡即可启动。Atlas 800T A2上默认使用BFloat16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_sft_800T_A2_64G.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改`run_iflytekspark_13b_sft_800T_A2_64G.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

1. 启动微调任务，在单机8节点上拉起任务。

```shell
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800T_A2_64G.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

**注**：

- `run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- `load_checkpoint`指定ckpt的加载路径，也可通过`yaml`配置文件中`load_checkpoint`字段确定。路径径格式需满足`{model_dir}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{model_dir}`该层目录即可

- **多机训练**

iFlytekSpark-13B用于全参微调，seq_length默认为32768，分布式微调在Atlas 800上训练，至少需要2节点16卡启动。Atlas 800上默认使用Float16类型训练，需要开启序列并行（配置文件中设置`seq_parallel=True`）。本仓给出了默认配置文件`run_iflytekspark_13b_sft_800_32G.yaml`。

**注：**此处以在Atlas 800上基于2节点16卡进行训练为例，使用更多的节点或在Atlas 800T A2上进行多机训练参考以下样例修改对应配置文件和参数。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-多节点章节，获取多节点的`RANK_TABLE_FILE`文件。

2. 修改`run_iflytekspark_13b_sft_800_32G.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: True
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 2
  model_parallel: 8
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

1. 启动微调任务，在单机8节点上拉起任务。

```shell
cd mindformers/research
# node 1
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800_32G.yaml \
--load_checkpoint model_dir \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [0,8] 16

# node 2
cd mindformers/research
bash run_multinode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_sft_800_32G.yaml \
--load_checkpoint model_dir \
--use_parallel True \
--run_mode finetune \
--train_data dataset_dir" \
RANK_TABLE_FILE [8,16] 16
```

**注**：

- `run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- `load_checkpoint`指定ckpt的加载路径，也可通过`yaml`配置文件中`load_checkpoint`字段确定。路径径格式需满足`{model_dir}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{model_dir}`该层目录即可

## Lora微调

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的数据集，参照[模型权重准备](#模型权重准备)章节获取iFlytekSpark-13B权重。
执行`sft_data_preprocess.py`脚本制作预训练mindrecord格式数据集需要。

```shell
python ./research/iflytekspark/sft_data_preprocess.py \
--tokenizer /{TOKENIZER_PATH} \
--raw_data_path /{RAW_DATA_PATH} \
--output_filename /{OUTPUT_FILE_PATH} \
--seq_length SEQ_LENGTH \
--pad_id PAD_ID
```

- **单机训练**

iFlytekSpark-13B用于Lora微调，seq_length默认为32768，分布式Lora微调在Atlas 800T A2和Atlas 800上训练训均单节点即可启动。Atlas 800T A2上默认使用BFloat16类型训练，Atlas 800上默认使用Float16类型训练。本仓给出了默认配置文件`run_iflytekspark_13b_lora_800T_A2_64G.yaml`和`run_iflytekspark_13b_lora_800_32G.yaml`。

**步骤**：

1. RANK_TABLE_FILE准备，请参照[RANK_TABLE_FILE准备](#rank_table_file准备)-单节点章节，获取单节点的`RANK_TABLE_FILE`文件。

2. 修改配置文件中相关配置，默认开启自动权重转换，使用完整权重。

```yaml
load_checkpoint: ''    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
auto_trans_ckpt: True           # 打开自动权重转换
use_parallel: True
run_mode: 'finetune'
...
# 数据集配置
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "dataset_dir" # 配置训练数据集文件夹路径
    shuffle: False
  input_columns: ["input_ids", "loss_mask"]
...
# 并行训练策略配置
parallel_config:
  data_parallel: 4
  model_parallel: 2
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
```

1. 启动微调任务，在单机上拉起任务。

```shell
# Atlas 800
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_lora_800_32G.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8

# Atlas 800T A2
cd mindformers/research
bash run_singlenode.sh \
"python iflytekspark/run_iflytekspark.py \
--config iflytekspark/run_iflytekspark_13b_lora_800T_A2_64G.yaml \
--load_checkpoint model_dir \
--auto_trans_ckpt True \
--use_parallel True \
--train_dataset dataset_dir" \
RANK_TABLE_FILE [0,8] 8
```

**注**：

- `run_iflytekspark.py`启动脚本支持的参数列表参考[仓库介绍](#仓库介绍)-数据处理脚本、权重处理脚本及任务启动脚本章节。

- `load_checkpoint`指定ckpt的加载路径，也可通过`yaml`配置文件中`load_checkpoint`字段确定。路径径格式需满足`{model_dir}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{model_dir}`该层目录即可

## 在线推理

**配置文件**

在线推理任务中使用的yaml文件为`run_iflytekspark_13b_infer_800_32G.yaml`与`run_iflytekspark_13b_infer_800T_A2_64G.yaml`，在`model_config`中包含了一些关键参数：

- `seq_length`: 最大推理长度。
- `batch_size`: 推理batch数。
- `sparse_local_size`: sparse attention的局部长度。
- `use_past`: 是否使能增量推理。
- `do_sample`: 推理时是否进行随机采样。
- `is_dynamic`: 是否开启动态shape推理（尚未使能）。
- 各类采样参数: `top_k`, `top_p`, `temperature`, `repetition_penalty`等，采样参数仅当`do_sample=True`时生效。

```yaml
model:
  model_config:
    type: IFlytekSparkConfig
    seq_length: 32768
    batch_size: 1
    hidden_size: 5120
    ffn_hidden_size: 28672
    num_layers: 40
    num_heads: 40
    vocab_size: 60000
    layernorm_epsilon: 1.0e-5
    bos_token_id: 1
    eos_token_id: 5
    pad_token_id: 0
    ignore_token_id: -100
    compute_type: "float16"
    softmax_compute_type: "float16"
    layernorm_compute_type: "float32"
    embedding_init_type: "float16"
    dropout_rate: 0.0
    hidden_act: "fast_gelu"
    sparse_local_size: 8192
    seq_parallel: False
    is_reward_model: False
    offset: 0
    checkpoint_name_or_path: ""
    use_past: True
    do_sample: False
    is_dynamic: False
    top_k: 1
    top_p: 1.0
    temperature: 1.0
    repetition_penalty: 1.0
    repetition_penalty_increase: 0.1
  arch:
    type: IFlytekSparkModelForCasualLM
```

此外，在使用分布式推理时，需要关注`parallel_config`中的并行策略:

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  ...
```

如在Atlas 800上，单卡的显存不足以执行推理任务时，可修改`model_parallel`的并行策略以启动分布式推理。如上例中`model_parallel`的值修改为了2，则代表预计使用2卡进行分布式推理。当前Atlas 800的yaml文件中均默认使用2卡的模型并行。

**注：在线推理目前不支持`bfloat16`类型，权重请使用SFT权重，即微调权重。**

**启动脚本**

在线推理的入口脚本为`run_infer.sh`，核心内容如下：

```shell
if [ $# != 0 ]  && [ $# != 3 ]
then
  echo "Usage Help: bash run_distribute.sh For Single Devices"
  echo "Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices"
  exit 1
fi

...

# According to the different machines, the yaml file that should be used is:
# Atlas 800 32G -> run_iflytekspark_13b_infer_800_32G.yaml
# Atlas 800T A2 64G -> run_iflytekspark_13b_infer_800T_A2_64G.yaml
export CONFIG_PATH=run_iflytekspark_13b_infer_800T_A2_64G.yaml

# You can modify the following inference parameters
export PY_CMD="python run_iflytekspark.py \
               --config $CONFIG_PATH \
               --run_mode predict \
               --use_parallel $PARALLEL \
               --load_checkpoint '{your_ckpt_path}' \
               --predict_data '[为什么地球是独一无二的？##请问生抽和老抽有什么区别？]' \
               --predict_length 32768 \
               --predict_batch 1 \
               --prompt '<User> {}<end><Bot> ' \
               --tokenizer_file '{your_tokenizer_path}' \
               --streamer False"

...
```

参数说明：

- `config`: 配置文件路径
- `run_mode`: 运行模式，推理使用`predict`字段
- `use_parallel`: 是否开启并行推理，当前脚本会根据执行脚本的入参个数自行设置
- `load_checkpoint`: ckpt的加载路径，路径格式需满足`{your_path}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{your_path}`该层目录即可
- `predict_data`：
  1. 格式为`[{question1}##{question2}...]`的问题列表，每个问题之间使用`##`进行分隔
  2. `.json`或.`jsonl`格式的文件路径，要求文件中每一行应为一个问题，每行问题为字典格式：`{input: your_question}`
- `predict_length`: 实际推理的最大长度
- `predict_batch`: 每次推理的batch数
- `prompt`: 推理使用的语料模版，该模板中需包含一个大括号`{}`，每个`predict_data`中的问题都会填入`{}`中，生成新的输入数据。
- `tokenizer_file`: tokenizer文件路径，该路径应包含`.vocab`与`.model`文件
- `streamer`: 是否使用流式返回

请根据实际的ckpt路径与真实使用的推理内容修改上述参数。

**推理启动方式**

目前支持两种命令格式执行`run_infer.sh`启动推理。

- **单卡推理**

当仅使用单卡进行推理时，可直接执行如下命令：

```shell
bash run_infer.sh
```

推理的输出结果会打印在`./log/infer.log`日志中，此外每个问题的输出结果还会存储在`./log/infer_result_num{question_num}_rank0.json`中便于批量处理。

- **多卡推理**

当使用多卡进行推理时，推荐使用单卡权重直接加载，此外需要准备`RANK_TABLE_FILE`，具体过程请参照[RANK_TABLE_FILE准备](#rank_table_file准备)中的单节点章节，生成对应的文件，下面以两卡推理作为例子，相应的`RANK_TABLE_FILE`内容应如下：

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

然后执行如下命令：

```shell
# 入参格式：
# bash run_infer.sh [RANK_TABLE_FILE_PATH] [DEVICE_RANGE] [RANK_SIZE]
# 此例中[RANK_TABLE_FILE_PATH]假设为./hccl_2p.json

bash run_infer.sh ./hccl_2p.json [0,2] 2
```

执行命令后，推理任务会转至后台执行，每张卡的推理结果会打印在`./log/infer_{device_id}.log`日志中，每张卡的输出结果也会存储在`./log/infer_result_num{question_num}_rank{device_id}.json`中。

示例中的两个问题及其回答可参考下表：
|问题   |                      回答|
|:---------------:|:---------------:|
为什么地球是独一无二的？| 地球是独一无二的，因为它是太阳系中唯一已知存在生命的行星。地球上的生命形式多种多样，包括植物、动物和微生物。地球上的生态系统和气候条件为生命的存在提供了适宜的环境。此外，地球上的水资源丰富，为生命提供了必要的条件。\<ret>\<ret>地球的独特性还体现在其地质历史和地理特征上。地球的地质历史长达46亿年，经历了多次的地壳运动和板块构造。地球上的山脉、河流、湖泊和海洋等地理特征也是独一无二的。\<ret>\<ret>此外，地球的自转和公转运动也使其成为独一无二的行星。地球的自转轴倾斜，导致季节的变化，而公转轨道的椭圆形状则导致了四季的变化。\<ret>\<ret>总之，地球的独特性在于其适宜的生命存在条件、地质历史和地理特征，以及自传和公转运动。这使得地球成为太阳系中独一无二的行星。|
请问生抽和老抽有什么区别？| 生抽是老抽都是酱油的一种，但它们在制作工艺、颜色、味道和用途上有所不同。\<ret>\<ret>1. 制作工艺：生抽是酱油的一种，是通过发酵大豆、小麦和盐制成的。老抽则是在生抽的基础上，经过长时间的发酵和陈化，使酱油的颜色更深，味道更醇厚。\<ret>\<ret>2. 颜色：生抽的颜色较浅，呈淡黄色或淡红色，而老抽的颜色较深，呈深红色或深褐色。\<ret>\<ret>3. 味道：生抽的味道较淡，主要用于烹饪，如炒菜、炖汤等。老抽主要用于烹饪，如炖肉、炖鱼等。\<ret>\<ret>4. 用途：生抽主要用于烹饪，如炒菜、炖汤等。老抽主要用于烹饪，如炖肉、炖鱼等。\<ret>\<ret>5. 营养价值：生抽和老抽的营养价值相似，都含有丰富的氨基酸、矿物质和维生素B群等营养成分。|

 **注：在 Atlas 800 32G设备上默认使用分布式推理**

- **Lora微调后推理**

进行lora微调后，需要将微调后权重进行调整：

1. 如为`Bfloat16`格式，需转换为`float16`或`float32`类型。

2. lora微调后得到的权重为分布式权重，需参考[模型权重转换](#模型权重转换)章节将分布式权重转换为单卡权重进行推理。

与普通推理流程不同的是，lora微调后推理需要在yaml中增加与lora相关的推理内容，相关yaml文件为`run_iflytekspark_13b_infer_lora_800_32G.yaml`和`run_iflytekspark_13b_infer_lora_800T_A2_64G.yaml`，配置文件的新增内容如下：

```yaml
model:
  model_config:
    type: IFlytekSparkConfig
    seq_length: 32768
    batch_size: 1
    hidden_size: 5120
    ffn_hidden_size: 28672
    num_layers: 40
    num_heads: 40
    vocab_size: 60000
    layernorm_epsilon: 1.0e-5
    bos_token_id: 1
    eos_token_id: 5
    pad_token_id: 0
    ignore_token_id: -100
    compute_type: "float16"
    softmax_compute_type: "float16"
    layernorm_compute_type: "float32"
    embedding_init_type: "float16"
    dropout_rate: 0.0
    hidden_act: "fast_gelu"
    sparse_local_size: 8192
    seq_parallel: False
    is_reward_model: False
    offset: 0
    use_past: True
    do_sample: False
    is_dynamic: False
    top_k: 1
    top_p: 1.0
    temperature: 1
    repetition_penalty: 1.0
    repetition_penalty_increase: 0.1
    pet_config:
      pet_type: lora
      # configuration of lora
      param_init_type: "float16"
      compute_dtype: "float16"
      lora_rank: 3
      lora_alpha: 128
      lora_dropout: 0.0
      target_modules: "q_proj|v_proj|k_proj|out_proj|fc0|fc1|fc2"
  arch:
    type: IFlytekSparkModelForCasualLM
```

将`run_infer.sh`中的`--config`项修改为`run_iflytekspark_13b_infer_lora_800_32G.yaml`或`run_iflytekspark_13b_infer_lora_800T_A2_64G.yaml`，然后执行如下命令：

```shell
# 入参格式：
# bash run_infer.sh [RANK_TABLE_FILE_PATH] [DEVICE_RANGE] [RANK_SIZE]
# 此例中[RANK_TABLE_FILE_PATH]假设为./hccl_2p.json
# 更改run_infer.sh中的--config项为：run_iflytekspark_13b_infer_lora_800_32G.yaml / run_iflytekspark_13b_infer_lora_800T_A2_64G.yaml

# 单卡推理
bash run_infer.sh

# 分布式推理
bash run_infer.sh ./hccl_2p.json [0,2] 2
```

执行结束后存储路径与内容同上章节。

## 离线推理

**配置文件**

离线推理任务中使用的yaml文件为`run_iflytekspark_13b_lite_infer_800_32G.yaml`与`run_iflytekspark_13b_lite_infer_800T_A2_64G.yaml`，在`model_config`中包含了一些关键参数：

- `seq_length`: 最大推理长度。
- `batch_size`: 推理batch数。
- `sparse_local_size`: sparse attention的局部长度。
- `use_past`: 是否使能增量推理。
- `do_sample`: 推理时是否进行随机采样。
- `is_dynamic`: 是否开启动态shape推理（单卡使能）。
- 各类采样参数: `top_k`, `top_p`, `temperature`, `repetition_penalty`等，采样参数仅当`do_sample=True`时生效。

```yaml
model:
  model_config:
    type: IFlytekSparkConfig
    seq_length: 32768
    batch_size: 1
    hidden_size: 5120
    ffn_hidden_size: 28672
    num_layers: 40
    num_heads: 40
    vocab_size: 60000
    layernorm_epsilon: 1.0e-5
    bos_token_id: 1
    eos_token_id: 5
    pad_token_id: 0
    ignore_token_id: -100
    compute_type: "float16"
    softmax_compute_type: "float16"
    layernorm_compute_type: "float32"
    embedding_init_type: "float16"
    dropout_rate: 0.0
    hidden_act: "fast_gelu"
    sparse_local_size: 8192
    seq_parallel: False
    is_reward_model: False
    offset: 0
    checkpoint_name_or_path: ""
    use_past: True
    do_sample: False
    is_dynamic: True
    top_k: 1
    top_p: 1.0
    temperature: 1.0
    repetition_penalty: 1.0
    repetition_penalty_increase: 0.1
  arch:
    type: IFlytekSparkModelForCasualLM
```

此外，在使用分布式推理时，需要关注`parallel_config`中的并行策略:

```yaml
parallel_config:
  data_parallel: 1
  model_parallel: 2
  ...
```

同时分布式推理时使用需run_iflytekspark_13b_lite_infer_800T_A2_64G_dis.yaml,要将配置yaml文件中is_dynamic设置为False,当前动态shape仅支持单卡推理

**注：离线推理目前不支持`bfloat16`类型**

**导出脚本**

离线推理的步骤为先导出mindir模型，然后执行；第一步为导出模型

导出模型的入口脚本为`run_export.sh`，核心内容如下：

```shell
if [ $# != 0 ]  && [ $# != 3 ]
then
  echo "Usage Help: bash run_export.sh For Single Devices"
  echo "Usage Help: bash run_export.sh [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices"
  exit 1
fi

...

# According to the different machines, the yaml file that should be used is:
# Atlas 800 32G -> run_iflytekspark_13b_infer_800_32G.yaml
# Atlas 800T A2 64G -> run_iflytekspark_13b_infer_800T_A2_64G.yaml
# Atlas 800T A2 64G with distribution -> run_iflytekspark_13b_infer_800T_A2_64G_dis.yaml
export CONFIG_PATH=run_iflytekspark_13b_lite_infer_800T_A2_64G.yaml

# You can modify the following parameters
export PY_CMD="python run_iflytekspark.py \
               --config $CONFIG_PATH \
               --run_mode export \
               --mindir_save_dir $mindir_save_dir \
               --use_parallel $PARALLEL \
               --load_checkpoint '{your_ckpt_path}' \
               --predict_length $max_seq_len \
               --predict_batch $batch_size"

...
```

参数说明：

- `config`: 配置文件路径
- `run_mode`: 运行模式，推理使用`export`字段
- `use_parallel`: 是否开启并行推理，当前脚本会根据执行脚本的入参个数自行设置
- `load_checkpoint`: ckpt的加载路径，路径格式需满足`{your_path}/rank_{0...7}/{ckpt_name}.ckpt`，只需指定到`{your_path}`该层目录即可
- `predict_length`: 实际推理的最大长度
- `predict_batch`: 每次推理的batch数

请根据实际的ckpt路径与真实使用的推理参数修改上述内容。多卡导出时需要设置CONFIG_PATH为run_iflytekspark_13b_lite_infer_800T_A2_64G_dis.yaml。

**推理启动方式**

导出模型以后，目前支持两种命令格式执行`run_lite.sh`启动单卡推理,`run_lite_dis.sh`启动多卡推理 。

- **单卡推理**

当仅使用单卡进行推理时，可直接执行如下命令：

```shell
bash run_lite.sh
```

推理的输出结果会打印在`./log/lite.log`日志中。

- **多卡推理**

```shell
bash run_lite_dis.sh
```

当使用多卡进行推理时，推荐使用单卡权重直接加载，此外需要准备`RANK_TABLE_FILE`，具体过程请参照[RANK_TABLE_FILE准备](#rank_table_file准备)中的单节点章节，生成对应的文件，下面以两卡推理作为例子，相应的`RANK_TABLE_FILE`内容应如下：

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

然后执行如下命令导出模型：

```shell
# 入参格式：
# bash run_export.sh [RANK_TABLE_FILE_PATH] [DEVICE_RANGE] [RANK_SIZE]
# 此例中[RANK_TABLE_FILE_PATH]假设为./hccl_2p.json

bash run_export.sh ./hccl_2p.json [0,2] 2
```

准备好RANK_TABLE_FILE以后将路径填入lite分布式推理的配置文件中，修改lite_config/full_dyn_infer_cfg_dis.ini文件和lite_config/inc_dyn_infer_cfg_akg_dis.ini文件中的rank_table_file字段,两个文件分别代表全量推理和增量推理的ge(计算底座)配置。
然后在run_lite_dis.sh 配置start_device_id表示多节点对应的起始device id, --device 已经废弃，填0即可。

输入文件支持json和文本文件两种，json文件需要以json为后缀，其他默认为文本文件，json输入需要含有'input'字段。

执行命令后，执行分布式推理脚本`run_lite_dis.sh`推理任务会转至后台执行，所有卡的推理结果会打印在`./log/lite.log`日志中。

## 协议

请您知悉，无论您是否已实际阅读[星火开源-13B大模型许可协议](https://gitee.com/iflytekopensource/iFlytekSpark-13B/blob/master/LICENSE_MODEL.md)，当您通过部署及使用该模型服务即表示确认同意本协议或实际使用、复制、分发、修改本协议中的讯飞星火认知大模型-13B模型时，均表示您与科大讯飞股份有限公司（以下称“许可方”）已就本协议达成一致，本协议具有合同效力。如果您不同意本协议的任一内容，或者无法准确理解许可方对[本协议条款](https://gitee.com/iflytekopensource/iFlytekSpark-13B/blob/master/LICENSE_MODEL.md)的解释，请停止使用本服务。否则，即表示您已接受本协议所述的所有条款及其适用条件，同意受本协议约束。
