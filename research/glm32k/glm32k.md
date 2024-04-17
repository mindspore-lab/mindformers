# ChatGLM3-32K

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型，更完整的功能支持，更全面的开源序列

ChatGLM3-6B-32K在ChatGLM3-6B的基础上进一步强化了对于长文本的理解能力，能够更好的处理最多32K长度的上下文。

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 仓库介绍

`chatGLM3-6B-32K` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm2`, `mindformers/models/glm3`

    ```bash
    glm2
        ├── __init__.py
        ├── glm2.py                  # 模型实现
        ├── glm2_config.py           # 模型配置项
        ├── glm2_modules.py          # 模组实现
        ├── glm2_tokenizer.py        # glm2的tokenizer
        └── glm2_transformer.py      # transformer层实现
    ```

   ```bash
    glm3
        ├── __init__.py
        └── glm3_tokenizer.py       # glm3和glm32k共用的tokenizer
    ```

2. 模型配置：`research/glm32k`

    ```bash
    glm32k
        ├── export_glm32k_pa.yaml     # Atlas 800T A2的glm32k lite推理模型导出配置
        ├── 910b_ge_prefill_pa.cfg    # 全量推理的ge配置文件
        ├── 910b_ge_inc_pa.cfg        # 增量推理的ge配置文件
        └── run_glm32k.yaml           # Atlas 800T A2最佳性能全量微调启动配置
    ```

3. 数据处理脚本和任务启动脚本：`research/glm32k`

    ```bash
    glm32k
        ├── run_glm32k.py                  # glm32k高阶接口使用脚本
        ├── convert_weight.py              # 模型权重转换脚本
        └── glm32k_preprocess.py           # glm32k微调的数据前处理脚本
    ```

4. 评测脚本：`research/glm32k`

    ```bash
    glm32k
        ├── eval_start.sh                 # 多卡启动推理的脚本
        ├── eval_end.sh                   # 多卡启动评估的脚本
        ├── eval_gen_mslite.py            # 生成推理结果的脚本
        └── eval_postprocess.py           # 生成最终评估结果的脚本
    ```

## 前期准备

### 安装mindformers

参考[README](../../README.md#二、mindformers安装)安装mindformers。
本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件: Atlas 800T A2
- MindSpore: 2.2.13
- MindSpore Lite: 2.2.13
- MindFormers: 1.0
- Mindpet: 1.0.3

### 生成RANK_TABLE_FILE

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

### 模型权重下载与转换(mindformers权重或huggingface权重选择使用即可)

#### mindformers权重直接使用

本仓库提供已经转换完成的预训练权重用于微调/推理，用户可自行从下方链接拉取后直接使用。

下载链接：

权重：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/glm32k.ckpt

词表：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/tokenizer.model

linux可用如下命令下载。

```shell
#!/bin/bash
mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/glm32k/glm32k.ckpt
wget https://huggingface.co/THUDM/chatglm3-6b-32k/tree/main/tokenizer.model
cd ../..
```

#### 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell
git lfs install
git clone https://huggingface.co/THUDM/chatglm3-6b-32k
```

```shell
#!/bin/bash
pip install torch==1.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/glm32k/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
#!/bin/bash
python ./research/glm32k/convert_weight.py --torch_path TORCH_CKPT_DIR --mindspore_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
MS_CKPT_NAME: mindspore格式的权重保存文件名，如'saved_dir/glm32k.ckpt'
```

**注**: 请安装torch=2.1.1和transformers=4.33.0版本。

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## ChatGLM3-6B-32K

### 微调性能

| config                               | task              | Datasets  | SeqLength | metric | phase             | score     | performance(tokens/s/p) |
|--------------------------------------|-------------------|-----------|-----------| ------ | ----------------- | --------- |-------------------------|
| [ChatGLM3-6B-32K](./run_glm32k.yaml) | text_generation   | longbench | 32768     | -      | [finetune](#微调)  | -         | 906.98                  |

### 微调

#### 数据集准备-SFT微调数据集

当前提供[LongBench](https://huggingface.co/datasets/THUDM/LongBench/tree/main)长序列数据集的预处理和微调样例，用于对ChatGLM3-6B-32K模型进行微调。

LongBench数据集样式

```text
{
    "input": "任务的输入/指令，通常较短，比如QA中的问题、Few-shot任务中的提问等",
    "context": "任务所需的长语境文本，比如文档、跨文件代码、Few-shot任务中的few-shot样本",
    "answers": "由所有标准答案组成的列表",
    "length": "前三项文本的总长度（中、英文分别用字、词数统计）",
    "dataset": "本条数据所属数据集名称",
    "language": "本条数据的语言",
    "all_classes": "分类任务中的所有类别，非分类任务则为null",
    "_id": "每条数据的随机id"
}
```

#### 数据集处理

将LongBench数据集格式转换为AdGen数据集格式，以便复用mindformers的ADGenDataLoader来转换为微调使用的数据样式。启动命令：

```shell
cd research/glm32k
python glm32k_preprocess.py \
--data_path INPUT_DATA_PATH \
--output_path OUTPUT_PATH \
--prompt_config_file PROMPT_PATH
```

```text
# 参数说明
INPUT_DATA_PATH: 原始longbench数据所处的文件夹路径
OUTPUT_PATH：转换格式后的数据存储路径
PROMPT_PATH：longbench中不同数据对应的prompt
```

**注意**：

- Longbench数据集链接：[Longbench](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)
- prompt_config_file链接：[prompt_config_file](https://github.com/THUDM/LongBench/blob/main/config/dataset2prompt.json)
- 具体Longbench数据集介绍请参见[官网](https://github.com/THUDM/LongBench)地址

#### 全参微调

全参微调需要多卡启动，以`LongBench`数据集为例，给出了默认配置文件`research/glm32k/run_glm32k.yaml`。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

- step 1. 修改`research/glm32k/run_glm32k.yaml`中相关配置

```text
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: './output/transformed_checkpoint/'          # 添加预训练权重路径
auto_trans_ckpt: False
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 3
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM3Tokenizer
    vocab_file: "/path/to/tokenizer.model"                   # 添加字典文件
  max_source_length: 30720                                   # 长序列源数据长度
  max_target_length: 2047                                    # 长序列目标数据长度
```

**注意**：长序列模型的训练，max_source_length和max_target_length数值较大，需要根据实际业务数据设置对应数值

- step 2. 启动微调任务，按照以下步骤启动：

-[x] 1: 根据服务器节点数等信息，修改相应的配置。

```shell
# 以glm-6b-32k模型为例，默认配置单机8卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：./research/glm32k/run_glm32k.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 1
  pipeline_stage: 4
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

-[x] 2: 执行运行脚本。

```shell
cd research
bash run_singlenode.sh \
"python glm32k/run_glm32k.py \
--config glm32k/run_glm32k.yaml \
--run_mode finetune \
--train_dataset /path/to/train.json \
--use_parallel True" \
path/to/rank_table_file [0,8] 8
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
path/to/rank_table_file: 生成RANK_TABLE_FILE步骤中生成的hccl_***_json文件路径
```

### 快速推理

**注意**：

- 目前可以根据实际业务调整yaml文件中的`seq_length`参数以降低推理时延(如设置`seq_length=4096`，最高到4096)
- 长序列推理研发正在适配中

在启动前，请先行在配置文件run_glm32k.yaml中将processor.tokenizer.vocab_file的路径配置为实际路径；如果使用增量推理，需要在配置文件中将model.model_config.use_past值设置为True。例如：

```yaml
processor:
  return_tensors: ms
  tokenizer:
    ...
    vocab_file: '/path/tokenizer.model'  # 修改为实际路径
    ...
model:
  model_config:
    ...
    use_past: True
    ...
```

相关文件的下载链接如下：[tokenizer.model](https://huggingface.co/THUDM/chatglm3-6b-32k/blob/main/tokenizer.model)

**注意**：目前长序列的增量推理所需的IFA算子还未适配，等待海思算子版本上新后再适配使用，所以`use_past`目前需设置为`False`

#### 基于generate接口的推理

- 代码见`research/glm32k/infer_generate.py`
- 启动命令：

```shell
python infer_generate.py --checkpoint_path CKPT_PATH --device_id DEVICE_ID --user_query "晚上睡不着应该怎么办"
```

```text
# 参数说明
checkpoint_path: 权重文件夹路径
device_id: NPU卡号
user_query: 用户输入问题
```

```text
# 输出结果
output:
[gMASK]sop<|user|>
 晚上睡不着应该怎么办<|assistant|>
 晚上睡不着,可以参考下述建议:
1. 建立规律的睡眠时间表:每天在相同的时间上床和起床,有助于身体建立规律的睡眠时间表,更容易入睡。
2. 创造舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗,凉爽,有助于入睡。
3. 避免刺激性物质:避免饮用咖啡因和酒精等刺激性物质,这些物质会影响睡眠。
4. 放松身心:在睡前放松身心,例如泡个热水澡,听些轻柔的音乐,读本书等,有助于入睡。
5. 避免使用电子设备:在睡前避免使用电子设备,例如手机,平板电脑等,这些设备发出的蓝光会抑制睡眠激素的分泌,影响睡眠。
6. 锻炼身体:适度的锻炼身体有助于睡眠,但避免在睡前进行剧烈运动。
如果以上方法都无法解决问题,建议咨询医生或专业睡眠专家,获得更具体的建议和治疗方案。
```

### MindSpore Lite推理

本章节提供ChatGLM3-6B-32K在MindSpore Lite上进行推理的基本使用流程，更多详细的特性介绍可以参考[Mindspore Lite特性文档](../../docs/feature_cards/Inference.md)

#### 单卡导出与PA推理

##### Step1. 模型导出MindIR

修改模型导出相关的配置文件 export_glm3_32k_pa.yaml，其中需要关注以下几项：

```yaml
# model config
model:
  model_config:
    seq_length: 32640
    checkpoint_name_or_path: "/path/to/glm32k.ckpt"
    use_past: True              # 开启增量推理
    is_dynamic: True            # 使用PA推理时设置为True
    use_paged_attention: True   # 使用PA推理时设置为True
    block_size: 128             # PA推理的参数设置
    num_blocks: 512             # PA推理的参数设置
```

执行run_glm32k.py，完成MindIR导出，得到全量minder_full_checkpoint/rank_0_graph.mindir和增量minder_inc_checkpoint/rank_0_graph.mindir两个MindIR图

```bash
python glm32k/run_glm32k.py
--config glm32k/export_glm3_32k_pa.yaml
--load_checkpoint /path/to/glm32k.ckpt
--run_mode=export --use_parallel False --use_past True --batch_size 1 --device_id 0
```

##### Step2. 执行MS Lite推理

新建推理配置文件，ChatGLM3-6B-32K在Atlas 800T A2上推荐的GE配置如下：

- 全量mindir模型PA推理配置（910b_ge_prefill_pa.cfg）

```ini
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.deterministic=1
[ge_graph_options]
ge.inputShape=batch_valid_length:1;tokens:1,32640;slot_mapping:32640
[graph_kernel_param]
opt_level=2
disable_cluster_ops=MatMul,Reshape
enable_cce_lib=true
enable_cluster_ops_only="paged_attention"
enable_expand_ops_only="paged_attention"
disable_cce_lib_ops=MatMul
```

- 增量mindir模型PA推理配置（910b_ge_inc_pa.cfg）

```ini
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.deterministic=1
[ge_graph_options]
ge.inputShape=batch_valid_length:-1;block_tables:-1,255;slot_mapping:-1;tokens:-1,1
ge.dynamicDims=1,1,1,1;2,2,2,2;4,4,4,4
ge.dynamicNodeType=1
[graph_kernel_param]
opt_level=2
disable_cluster_ops=MatMul,Reshape
enable_cce_lib=true
enable_cluster_ops_only="paged_attention"
enable_expand_ops_only="paged_attention"
disable_cce_lib_ops=MatMul
```

执行run_infer_main.py脚本，修改相关配置启动推理：

- PA推理执行命令如下：

```bash
python run_infer_main.py
--batch_size 1
--device_id 0
--model_name glm3
--prefill_model_path /path/to/mindir_full_checkpoint/rank_0_graph.mindir
--increment_model_path /path/to/mindir_inc_checkpoint/rank_0_graph.mindir
--tokenizer_path /path/to/tokenizer.model
--config_path "/glm32k/910b_ge_prefill_pa.cfg,/glm32k/910b_ge_inc_pa.cfg"
--seq_length 32640
--max_length 32640
--dynamic False
--paged_attention True
--pa_block_size 128
--pa_num_blocks 512

# 参数说明
batch_size: 推理多batch设置
device_id: 设备物理ID
model_name: 模型名称
prefill_model_path: 全量图路径
increment_model_path: 增量图路径
tokenizer_path: 模型tokenizer路径
config_path: GE配置文件路径
seq_length: 推理序列长度
max_length: 能够生成的最大语句长度
dynamic: 是否采用双动态推理,执行PA推理时设置为False
paged_attention: 是否执行PA推理
pa_block_size: PA推理的参数
pa_num_blocks: PA推理的参数
```

**注：** 使用run_infer_main.py脚本执行推理时需要安装tiktoken包，不然会出现报错；同时tiktoken包需要配套python=3.9的环境。安装命令：pip install tiktoken

#### 多batch推理流程（以batch_size=2为例）

##### Step1. 模型导出batch_size=2情况下的MindIR

修改模型导出相关的配置文件 export_glm3_32k_pa.yaml，其中需要关注以下几项：

```yaml
# model config
model:
  model_config:
    type: ChatGLM2Config
    batch_size: 2         # 此处设置batch size值
```

执行run_glm32k.py，完成MindIR导出，得到全量minder_full_checkpoint/rank_0_graph.mindir和增量minder_inc_checkpoint/rank_0_graph.mindir两个MindIR图

```bash
python glm32k/run_glm32k.py \
--config glm32k/export_glm3_32k_pa.yaml \
--load_checkpoint /path/to/glm32k.ckpt \
--run_mode=export --use_parallel False --use_past True --batch_size 2 --device_id 0
```

##### Step2. 执行MS Lite推理

- 全量mindir模型PA推理配置（910b_ge_prefill_pa.cfg）：slot_mapping的值等于batch_size*seq_len=2*32640

```ini
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.deterministic=1
[ge_graph_options]
ge.inputShape=batch_valid_length:2;tokens:2,32640;slot_mapping:65280
[graph_kernel_param]
opt_level=2
disable_cluster_ops=MatMul,Reshape
enable_cce_lib=true
enable_cluster_ops_only="paged_attention"
enable_expand_ops_only="paged_attention"
disable_cce_lib_ops=MatMul
```

- 增量mindir模型PA推理配置（910b_ge_inc_pa.cfg）

```ini
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.deterministic=1
[ge_graph_options]
ge.inputShape=batch_valid_length:-1;block_tables:-1,255;slot_mapping:-1;tokens:-1,1
ge.dynamicDims=1,1,1,1;2,2,2,2;4,4,4,4
ge.dynamicNodeType=1
[graph_kernel_param]
opt_level=2
disable_cluster_ops=MatMul,Reshape
enable_cce_lib=true
enable_cluster_ops_only="paged_attention"
enable_expand_ops_only="paged_attention"
disable_cce_lib_ops=MatMul
```

执行run_infer_main.py脚本，修改相关配置启动推理：

- PA推理执行命令如下：

```bash
python run_infer_main.py \
--batch_size 2 \
--device_id 0 \
--model_name glm3 \
--prefill_model_path /path/to/mindir_full_checkpoint/rank_0_graph.mindir \
--increment_model_path /path/to/mindir_inc_checkpoint/rank_0_graph.mindir \
--tokenizer_path /path/to/tokenizer.model \
--config_path "/glm32k/910b_ge_prefill_pa.cfg,/glm32k/910b_ge_inc_pa.cfg" \
--seq_length 32640 \
--max_length 32640 \
--dynamic False \
--paged_attention True \
--pa_block_size 128 \
--pa_num_blocks 512

# 参数说明
batch_size: 推理多batch设置
```

**注：** 为适配batch_size=2，需要修改run_infer_main.py部分代码，如下所示：

```python
def infer_main(args_):
    ...
    if args_.distributed:
        ...
    else:
        while True:
        user_input = input("Please enter your predict data: \n")
        if user_input == "exit":
            print("Task is over.")
            sys.exit()
        user_input = [user_input] * args_.batch_size   # 此处新增一行代码，用于多batch推理
        ...
```

### 评测

#### 评测结果

**注：** 评测结果基于开源的预训练模型

|                                 | DuReader rouge |
|---------------------------------|----------------|
| Atlas 800T A2 + Mindspore (PA推理) | 44.83          |
| A100 + Pytorch                  | 44.59          |

#### 评测流程

评测中的推理过程使用MindSpore-Lite。所用数据集为长序列DuReader评测集，评测时需要安装`rouge`三方包来替换原环境中的`rouge-chinese`包

安装命令如下：

```shell
pip uninstall rouge-chinese
pip install rouge==1.0.1
```

- step 1: 模型导出MindIR

参考[上一节模型导出部分](#step1-模型导出mindir)。

- step 2: 多卡对评测数据集进行推理，生成结果：

修改eval_start.sh中各文件路径：

```bash
input_dataset_file=/path/to/eval_dataset/dureader.jsonl  # 数据集文件路径
checkpoint_path=/path/to/glm3_32k.ckpt  # 模型权重文件路径
ge_config_path="/research/glm32k/910b_ge_prefill_pa.cfg,/research/glm32k/910b_ge_inc_pa.cfg"  # GE配置文件路径，注意第一个为全量（prefill）模型配置，第二个为增量（inc）模型配置
tokenizer_path=/path/to/tokenizer.model  # 词表文件路径
full_model_path=/research/output/mindir_full_checkpoint/rank_0_graph.mindir  # 全量模型MindIR路径
inc_model_path=/research/output/mindir_inc_checkpoint/rank_0_graph.mindir  # 增量模型MindIR路径
```

运行eval_start.sh

```shell
bash eval_start.sh
```

- step 3: 将多张卡生成的结果汇总到一起，生成测试分数，命令如下：

```shell
bash eval_end.sh
# - INFO - evaluate score is: 44.83
# evaluation completed!
```