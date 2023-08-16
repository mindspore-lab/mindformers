# InternLM

## 模型描述

InternLM ，即书生·浦语大模型，是由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。包含面向实用场景的70亿参数基础模型与对话模型 （InternLM-7B）。模型具有以下特点：

- 使用上万亿高质量语料，建立模型超强知识体系；
- 支持8k语境窗口长度，实现更长输入与更强推理体验；
- 通用工具调用能力，支持用户灵活自助搭建流程；

本仓库目前能够支持上述特性1，暂未支持特性2、3。

本仓库支持InternLM-7B和InternLM-chat-7B预训练模型。由于InternLM与llama结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上llama的代码。

注: 由于InternLM基于高阶接口的形式开发，存放于research文件夹下，使用时需要将mindformers[安装](https://gitee.com/mindspore/mindformers#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)为python包，才能直接进入research/internlm目录下执行相关命令。

``` text
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```

## 代码结构介绍

`InternLM` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/internlm`

    ```bash
    internlm
        ├── internlm_tokenizer.py       # tokenizer
        ├── internlm_transformer.py     # transformer层实现
        └── internlm.py                 # 模型实现
    ```

2. 模型配置：`research/internlm`

    ```bash
    internlm
        ├── run_internlm_7b.yaml             # 全量微调启动配置
        └── run_internlm_7b_lora.yaml        # lora低参微调启动配置
    ```

3. 预处理脚本和任务启动脚本：`research/internlm`

    ```bash
    internlm
        ├── alpaca_data_preprocess.py     # alpaca数据集预处理
        ├── wiki_data_preprocess.py       # wikitext2数据集预处理
        ├── convert_weight.py             # 权重转换
        └── run_internlm.py               # 高阶接口使用脚本
    ```

## <span id="jump">权重转换</span>

从huggingface下载预训练权重用于训练/微调/推理，需要下载整个工程，包含对应的分词模型：

- [internlm-7b](https://huggingface.co/internlm/internlm-7b)

- [internlm-chat-7b](https://huggingface.co/internlm/internlm-chat-7b)

注：internlm-7b权重用于训练/微调，internlm-chat-7b用于直接开启快速推理。

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python ./research/internlm/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

## 快速推理

### 基于高阶接口的推理

1. 配置文件设置，添加tokenizer路径`vocab_file`，并设置`rms_norm`，`batch_size`的值

在使用Trainer接口进行推理时，由于InternLM-7b的tokenizer需要用户自行下载，因此在启动前，请先在配置文件中将tokenizer.model的路径自行配置，配置项为vocab_file。

```python
# research/internlm/run_internlm_7b.yaml
# runner config
runner_config:
  epochs: 1
  batch_size: 1                 # batch_size设为1
  sink_mode: True
  sink_size: 2
...
# model config
model:
  model_config:
    type: LlamaConfig
    ...
    rms_norm_eps: 1.0e-6        # rms_norm_eps设为1.0e-6
...
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<unk>'
   bos_token: '<s>'
   eos_token: '</s>'
   pad_token: '</s>'
   vocab_file: '/path/Internlm-7b/tokenizer.model'        # 添加tokenizer路径
   type: InternLMTokenizer
```

2. Trainer接口启动推理

InternLM-7b的高阶接口使用脚本已集成在run_internlm.py脚本中，运行此脚本命令示例：

```shell
python run_internlm.py \
--config "run_internlm_7b.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '我们来对对联吧！生意如春意 的下联是' \
--device_id 0

# output: [{'text_generation_text': ['<|User|>:我们来对对联吧！生意如春意 的下联是<eoh>\n<|Bot|>:财源似水流<eoa>\n']}]
```

### Pipeline推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig

from internlm import InternLMForCausalLM
from internlm_tokenizer import InternLMTokenizer

context.set_context(device_id=1)
# init model
internlm_model_path = "/path/InternLM-7B/internlm.ckpt" # InternLM ckpt path
internlm_config = LlamaConfig(
    vocab_size=103168,
    pad_token_id=0,
    rms_norm_eps=1.0e-6,
    checkpoint_name_or_path=internlm_model_path,
    use_past=True
)
internlm_model = InternLMForCausalLM(
    config=internlm_config
)
# init tokenizer
tokenizer_path = "/path/InternLM-7B/tokenizer.model" # InternLM-7B tokenizer.model path
tokenizer = InternLMTokenizer(
    vocab_file=tokenizer_path
)
pipeline_task = pipeline(task="text_generation", model=internlm_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("<s><s><|User|>:你好<eoh>\n<|Bot|>:",
                                do_sample=False,
                                repetition_penalty=1.0,
                                max_length=256)

print(pipeline_result)

# output: [{'text_generation_text': ['<|User|>:你好<eoh>\n<|Bot|>:你好，有什么我可以帮助你的吗？<eoa>\n']}]
```

## 微调

### 数据集准备

本仓库提供了WikiText2、Alpaca数据集的预处理脚本，用于生成mindrecord训练数据。

1. 数据集下载：

- [WikiText2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

- [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

- [alpaca-gpt4-data-zh](https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh/blob/main/alpaca_gpt4_data_zh.json)

2. 分词模型下载：

从huggingface下载预训练权重时，同时下载对应的tokenizer.model。参考[权重转换](#jump)中提供的链接进行下载。

3. 使用预处理脚本生成mindrecord训练数据：

- WikiText2数据集预处理指令示例：

```shell
python wiki_data_preprocess.py \
--mindrecord_schema internlm_wiki \
--input_glob {path}/wikitext-2/wiki.train.tokens \
--output_file {path}/wiki_processed/wiki.mindrecord \
--model_file {path}/tokenizer.model \
--seq_length 2048 \
--min_length 50  # 过滤token长度小于min_length的数据，default=50
```

- Alpaca数据集预处理指令示例：（同时适用于alpaca_data和alpaca-gpt4-data-zh数据集）

```shell
python alpaca_data_preprocess.py \
--mindrecord_schema internlm_alpaca \
--input_glob {path}/alpaca_data.json \
--output_file {path}/alpaca_processed/alpaca.mindrecord \
--model_file {path}/tokenizer.model \
--seq_length 2048
```

### 全参微调

全参微调需要多卡启动，以alpaca_data数据集为例,给出了默认配置文件`run_internlm_7b.yaml`：

1. 修改`run_internlm_7b.yaml`中相关配置

```python
output_dir: './output'
load_checkpoint: './internlm.ckpt'          # 添加预训练权重路径
auto_trans_ckpt: False
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "{path}/alpaca.mindrecord"   # 修改训练数据集路径
    shuffle: True
  input_columns: ["input_ids", "labels"]
# 用alpaca数据集指令微调时，input_columns: ["input_ids", "labels"]
# 用wiki数据集微调时，input_columns: ["input_ids"]
```

2. 启动微调任务，以单机八卡为例，指令如下：

```shell
bash run_singlenode.sh \
"python internlm/run_internlm.py \
--run_mode=finetune \
--use_parallel True \
--config internlm/run_internlm_7b.yaml \
--load_checkpoint ckpt_path_or_dir \
--train_dataset {path}/train_data" \
hccl_xp_xxx.json [0,8] 8
```

### Lora微调

Lora微调支持单卡/多卡启动，以alpaca-gpt4-data-zh数据集为例,给出了默认配置文件`run_internlm_7b_lora.yaml`：

1. 参考全参微调任务修改配置文件中的预训练权重路径、数据集路径。

2. 启动lora微调任务。

单卡启动指令如下：

```shell
python run_internlm.py \
--config run_internlm_7b_lora.yaml \
--run_mode finetune \
--pet_method lora \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--train_dataset {path}/train_data \
--device_id 0
```

多卡启动以单机八卡为例，指令如下：

```shell
bash run_singlenode.sh \
"python internlm/run_internlm.py \
--config internlm/run_internlm_7b_lora.yaml \
--run_mode=finetune \
--pet_method lora \
--use_parallel True \
--load_checkpoint ckpt_path_or_dir \
--train_dataset {path}/train_data" \
hccl_xp_xxx.json [0,8] 8
```