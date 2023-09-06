# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## <span id="jump">权重准备</span>

从huggingface下载预训练权重用于训练/微调/推理，需要下载整个工程，Base用于微调，Chat用于推理：

- [baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- [baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

下载完成后，运行`/research/baichuan/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python ./research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

## 代码结构介绍

`Baichuan2` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/baichuan2`

    ```bash
    baichuan2
        ├── baichuan2_tokenizer.py       # tokenizer
        ├── baichuan2_7b.py              # 7B模型实现
        └── baichuan2_13b.py             # 13B模型实现
    ```

2. 模型配置：`research/baichuan2`

    ```bash
    baichuan2
        ├── run_baichuan2_7b.yaml             # 7B全量微调910a启动配置
        ├── run_baichuan2_13b.yaml            # 13B全量微调910a启动配置
        ├── run_baichuan2_7b_910b.yaml        # 7B全量微调910b启动配置
        └── run_baichuan2_13b_910b.yaml       # 13B全量微调910b启动配置
    ```

3. 任务启动脚本：`research/baichuan2`

    ```bash
    baichuan2
        └── run_baichuan2.py               # baichuan2高阶接口使用脚本
    ```

## 环境要求

- 硬件：Ascend 910A
- MindSpore：2.0.0 / 1.10.1
- MindFormers版本：dev

注：Baichuan2-7B推理可在单机单卡上完成部署，全量微调至少需要16卡。Baichuan2-13B推理至少需要4卡，全量微调至少需要16卡。

## Baichuan2-7B

### 快速推理

#### 基于高阶接口的推理

1. 配置文件设置，添加tokenizer路径`vocab_file`，并设置`batch_size`值为`1`

在使用Trainer接口进行推理时，若用户自行下载Baichuan2-7B权重，请在启动前先在配置文件中将tokenizer.model的路径自行配置，配置项为vocab_file。

```python
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

```shell
python run_baichuan2.py \
--config "run_baichuan2_7b.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '将以下内容翻译成英文：你今天真好看。' \
--device_id 0

# output: [{'text_generation_text': ['将以下内容翻译成英文：你今天真好看。 \nYou look really nice today.']}]
```

### Pipeline推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig

from baichuan2 import Baichuan27BForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

context.set_context(device_id=1)
# init model
baichuan2_model_path = "/path/Baichuan2-7B/baichuan2_7b.ckpt" # Baichuan2-7B ckpt path
baichuan2_config = LlamaConfig(
    vocab_size=125696,
    pad_token_id=0,
    rms_norm_eps=1.0e-6,
    checkpoint_name_or_path=baichuan2_model_path,
    use_past=True
)
baichuan2_model = Baichuan27BForCausalLM(
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

### 全参微调

全参微调需要多卡启动，以`wikitext2`数据集为例,给出了默认配置文件`run_baichuan2_7b.yaml`对于`wikitext2`数据集的预处理及格式转换，请参考`llama`模型指导文档中的[数据集准备章节](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%87%86%E5%A4%87)：

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

```python
output_dir: './output'
load_checkpoint: '{path}/'          # 添加预训练权重路径
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/wiki2048.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids"]
# 指令微调时（如alpaca数据集），input_columns: ["input_ids", "labels"]
# 继续预训练时（如wikitext2数据集），input_columns: ["input_ids"]
```

2. 启动微调任务，以默认配置2机16卡为例，按照以下步骤启动：

- step 1. 首先参考在每台机器上运行`mindformers/tools/hccl_tools.py`生成`RANK_TABLE_FILE`的json文件。

```shell
# 在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num [0,8]
```

- step 2. 合并每台机器上生成的`RANK_TABLE_FILE`。

将不同机器上生成的`RANK_TABLE_FILE`文件拷贝到一起，执行`merge_hccl.py`脚本进行合并，包括server_list合并，`server_count`设为机器数，`rank_id`顺序增加。

```shell
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

- step 4. 根据服务器节点数等信息，修改相应的配置。

```shell
# 以baichuan2-7B模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
# 配置文件路径：../configs/llama/run_llama_13b.yaml
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 8
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

- step 5. 执行运行脚本。

在多机上同时拉起任务，每台机器拉起方式如下。

```shell
# node 1
cd mindformers/research
bash run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_ckpt \
--auto_trans_ckpt True \
--run_mode=finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [0,8] 16

# node 2
cd mindformers/research
bash run_singlenode.sh \
"python baichuan/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_ckpt \
--auto_trans_ckpt True \
--run_mode=finetune \
--train_data path/to/mindrecord_dir" \
path/to/rank_table_file [8,16] 16
```

```text
# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件夹路径
auto_trans_ckpt: 是否进行权重自动切分
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集路径
```

## baichuan2-13B

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://mindformers.readthedocs.io/zh_CN/latest/docs/api_python/README.html)。

请根据[权重准备](#jump)章节获取baichuan2_13B的完整权重。

### 快速使用

`Baichuan2-13B`的高阶接口使用脚本已集成在`run_baichuan2.py`脚本中

**注1**：由于模型较大，不支持单卡训练以及单卡推理

**注2**: 由于baichuan2-13B基于高阶接口的形式开发，存放于research文件夹下，使用时需要将mindformers安装为python的包，才能直接进入research目录下执行相关命令。

**注3**: 当前`run_baichuan2_13b.yaml`文件默认为train配置，用于eval和predict时需要修改并行策略。

- **单机多卡运行推理**

1. 主要参数配置参考

```shell
load_checkpoint: 'model_dir' # 完整模型存放格式为"model_dir/rank_0/xxx.ckpt"
auto_trans_ckpt: True # 打开权重自动转换
use_past: True # 打开增量推理
vocab_file: 'path/to/tokenizer.model'

# 分布式配置
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

2. 生成4卡的rank_table_file

```shell
python mindformers/tools/hccl_tools.py --device_num [0,4]
```

3. 启动推理

```shell
cd research
# 推理命令中参数会覆盖yaml文件中的相同参数
./run_singlenode.sh "python baichuan2/run_baichuan2.py --config baichuan2/run_baichuan2_13b.yaml --run_mode predict --use_parallel True --load_checkpoint model_dir --auto_trans_ckpt True --predict_data 你是谁？" rank_table_file [0,4] 4

# output: [{'text_generation_text': ['你是谁？ \n我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问']}]
```

- 注：推理结束后，保存`output/transformed_checkpoint`到自定义文件夹下，后续分布式推理可以直接加载`transformed_checkpoint`里面的4卡分布式权重，配置修改如下：

```shell
load_checkpoint: 'transformed_checkpoint' # 完整模型存放格式为"transformed_checkpoint/rank_x/xxx.ckpt"
auto_trans_ckpt: False # 关闭权重自动转换
```

- **多机多卡运行训练**

seq_length默认为4096，分布式训练需要4节点。

```shell
# node 1
cd mindformers/research
bash run_singlenode.sh "python baichuan2/run_baichuan2.py --config baichuan2/run_baichuan2_13b.yaml --load_checkpoint path/to/baichuan2_13b_ckpt --run_mode=train --train_data path/to/mindrecord_dir" path/to/rank_table_file [0,8] 32
# node 2
cd mindformers/research
bash run_singlenode.sh "python baichuan2/run_baichuan2.py --config baichuan2/run_baichuan2_13b.yaml --load_checkpoint path/to/baichuan2_13b_ckpt --run_mode=train --train_data path/to/mindrecord_dir" .path/to/rank_table_file [8,16] 32
# node 3
cd mindformers/research
bash run_singlenode.sh "python baichuan2/run_baichuan2.py --config baichuan2/run_baichuan2_13b.yaml --load_checkpoint path/to/baichuan2_13b_ckpt --run_mode=train --train_data path/to/mindrecord_dir" path/to/rank_table_file [16,24] 32
# node 4
cd mindformers/research
bash run_singlenode.sh "python baichuan2/run_baichuan2.py --config baichuan2/run_baichuan2_13b.yaml --load_checkpoint path/to/baichuan2_13b_ckpt --run_mode=train --train_data path/to/mindrecord_dir" .path/to/rank_table_file [24,32] 32
```
