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

## 模型性能

|                                     config                                      |      task       | Datasets | [train performance](#全参微调) | [predict performance](#推理) |
|:-------------------------------------------------------------------------------:|:---------------:|:--------:|:--------------------------:|:--------------------------:|
|      [internlm_7b(Atlas 800T A2)](../../research/internlm/run_internlm_7b_910b.yaml)      | text_generation |  alpaca  |       1802 tokens/s        | 7 tokens/s (use_past=True) |
| [internlm_7b_lora(Atlas 800T A2)](../../research/internlm/run_internlm_7b_lora_910b.yaml) | text_generation |  alpaca  |       2211 tokens/s        |             -              |

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
        ├── run_internlm_7b.yaml                  # 全量微调Atlas 800启动配置
        ├── run_internlm_7b_910b.yaml             # 全量微调Atlas 800T A2启动配置
        ├── run_internlm_7b_lora.yaml             # lora低参微调Atlas 800启动配置
        └── run_internlm_7b_lora_910b.yaml        # lora低参微调Atlas 800T A2启动配置
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

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理，tokenizer.model为词表文件。

- [InternLM-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm.ckpt)
- [InternLM-7B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm-chat.ckpt)
- [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/tokenizer.model)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，包含对应的分词模型，需要下载整个工程，huggingface权重的链接如下：

- [InternLM-7B-Base](https://huggingface.co/internlm/internlm-7b)

- [InternLM-7B-Chat](https://huggingface.co/internlm/internlm-chat-7b)

- [InternLM-20B-Base](https://huggingface.co/internlm/internlm-20b)

- [InternLM-20B-Chat](https://huggingface.co/internlm/internlm-chat-20b)

注：InternLM-7B-Base权重用于训练/微调，InternLM-7B-Chat用于直接开启快速推理，InternLM-20B同上。

原始权重下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
# 请安装torch=2.0.0和transformers=4.30.2版本:
# pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
python ./research/internlm/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
# 参数说明
torch_ckpt_path: huggingface权重保存目录路径下任意权重bin文件，将根据该文件路径读取目录下全部权重
mindspore_ckpt_path: 转换后mindspore权重的保存路径
```

## InternLM-7B

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
from internlm_config import InternLMConfig

context.set_context(device_id=0, mode=0)
# init model
internlm_model_path = "/path/InternLM-7B/internlm-chat.ckpt" # InternLM ckpt path
internlm_config = InternLMConfig(
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

- [WikiText2](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

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

internlm-7b用于微调，seq_length默认为2048，分布式微调训练在Atlas 800 / Atlas 800T A2上均可在单机八卡上启动。以alpaca_data数据集为例,给出了Atlas 800上的默认配置文件`run_internlm_7b.yaml`。若使用Atlas 800T A2机器，使用`run_internlm_7b_910b.yaml`配置文件即可，其他步骤与Atlas 800一致。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

1. 权重准备

权重支持在线/离线切分方式。在线切分则会在启动微调任务后自动按照分布式策略进行权重切分，离线切分需要在任务前手动进行切分。

若使用在线切分，则需要将完整权重文件按如下路径放置，并将启动配置参数`auto_trans_ckpt`置为`True`。

```text
    └── path of ckpt
        └── rank_0
            └── internlm_7b_base.ckpt
```

若使用离线切分，配置参数`auto_trans_ckpt`置为`False`，`load_checkpoint`传入切分好的权重路径文件夹即可。

2. 修改`run_internlm_7b.yaml`中相关配置

```python
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: 'path/of/ckpt'          # 添加预训练权重路径
auto_trans_ckpt: True                       # 开启权重自动切分
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
--run_mode finetune \
--use_parallel True \
--config internlm/run_internlm_7b.yaml \
--load_checkpoint path/of/ckpt \
--auto_trans_ckpt True \
--train_dataset {path}/train_data" \
hccl_xp_xxx.json [0,8] 8
```

### Lora微调

Lora微调支持Atlas 800 / Atlas 800T A2上的单卡/多卡启动，以alpaca-gpt4-data-zh数据集为例,给出了Atlas 800的默认配置文件`run_internlm_7b_lora.yaml`。若使用Atlas 800T A2机器，使用`run_internlm_7b_lora_910b.yaml`配置文件即可，其他步骤与Atlas 800一致。

1. 参考全参微调任务修改配置文件中的预训练权重路径、数据集路径。

2. 启动lora微调任务。

单卡启动指令如下：

```shell
python run_internlm.py \
--config run_internlm_7b_lora.yaml \
--run_mode finetune \
--use_parallel False \
--load_checkpoint path/of/ckpt \
--auto_trans_ckpt True \
--train_dataset {path}/train_data \
--device_id 0
```

多卡启动以单机八卡为例，指令如下：

```shell
bash run_singlenode.sh \
"python internlm/run_internlm.py \
--config internlm/run_internlm_7b_lora.yaml \
--run_mode finetune \
--use_parallel True \
--load_checkpoint path/of/ckpt \
--auto_trans_ckpt True \
--train_dataset {path}/train_data" \
hccl_xp_xxx.json [0,8] 8
```

## InternLM-20B

### MindSpore推理

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造了全新的训推一体高性能推理引擎，保证训练与推理使用同一套脚本，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　MindSpore 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

#### 基于高阶接口推理

- 修改yaml配置文件设置

```bash
auto_trans_ckpt: False                              # 关闭自动权重转换
use_past: True                                      # 使用增量推理
vocab_file: '/path/to/tokenizer.model'              # 配置词表路径
```

- Trainer接口启动推理

InternLM-20B的高阶接口使用脚本已集成在run_internlm.py脚本中，运行此脚本命令示例：

```bash
python run_internlm.py \
--config 'run_internlm_20b_910b.yaml' \
--run_mode predict \
--use_parallel False \
--load_checkpoint '/path/to/InternLM-20B-Chat.ckpt' \
--predict_data '你是谁？' \
--device_id 0

# output: [{'text_generation_text': ['<|User|>:你是谁？<eoh>\n<|Bot|>:我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我的设计理念是有用、诚实并且无害。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<eoa>\n']}]
```

#### 基于Pipeline推理

- 构建run_internlm_pipeline.py，该脚本提供了加载**完整权重**进行**单卡pipeline推理**的简单示例。

```python
# run_internlm_pipeline.py
import mindspore as ms
from mindspore import context
from mindformers.pipeline import pipeline

from mindformers import MindFormerConfig
from internlm import InternLMForCausalLM
from internlm_config import InternLMConfig
from internlm_tokenizer import InternLMTokenizer


# init context
context.set_context(device_id=0, mode=0)

# init config
internlm_config_path = "/path/to/run_internlm_20b.yaml"
config = MindFormerConfig(internlm_config_path)
internlm_config = InternLMConfig(**config.model.model_config)
internlm_model = InternLMForCausalLM(
    config=internlm_config
)

# init tokenizer
tokenizer_path = "/path/to/InternLM-20B/tokenizer.model" # InternLM-20B tokenizer path
tokenizer = InternLMTokenizer(
    vocab_file=tokenizer_path
)

# init and run pipeline
pipeline_task = pipeline(task="text_generation", model=internlm_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("<s><s><|User|>:你是谁？<eoh>\n<|Bot|>:",
                                do_sample=False,
                                repetition_penalty=1.0,
                                max_length=256)
print(pipeline_result)

# 推理输出
# [{'text_generation_text': ['<|User|>:你是谁？<eoh>\n<|Bot|>:我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我的设计理念是有用、诚实并且无害。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<eoa>\n']}]
```

- 修改yaml配置文件，以下为主要参数设置参考：

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: '/path/to/InternLM-20B-Chat.ckpt'    # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: '/path/to/tokenizer.model'                        # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

- 运行run_internlm_pipeline.py

```bash
python internlm/run_internlm_pipeline.py
```

#### 基于Generate推理

- 构建run_internlm_generate.py，该脚本提供了加载**完整权重**进行**单卡generate推理**的简单示例。

```python
# run_internlm_generate.py
import mindspore as ms
from mindspore import context

from mindformers import MindFormerConfig
from internlm import InternLMForCausalLM
from internlm_config import InternLMConfig
from internlm_tokenizer import InternLMTokenizer


# init context
context.set_context(device_id=0, mode=0)

# init config
internlm_config_path = "/path/to/run_internlm_20b.yaml"
config = MindFormerConfig(internlm_config_path)
internlm_config = InternLMConfig(**config.model.model_config)
internlm_model = InternLMForCausalLM(
    config=internlm_config
)

# init tokenizer
tokenizer_path = "/path/to/InternLM-20B/tokenizer.model" # InternLM-20B tokenizer path
tokenizer = InternLMTokenizer(
    vocab_file=tokenizer_path
)

# predict using generate
input_ids = tokenizer("<s><s><|User|>:你是谁？<eoh>\n<|Bot|>:",
                      max_length=64, padding="max_length")["input_ids"]
generate_ids = internlm_model.generate(input_ids,
                                       do_sample=False,
                                       top_k=1,
                                       top_p=1.0,
                                       repetition_penalty=1.0,
                                       temperature=1.0,
                                       max_length=64)
generate_result = tokenizer.decode(generate_ids)
print(generate_result)

# 推理输出
# [{'text_generation_text': ['<|User|>:你是谁？<eoh>\n<|Bot|>:我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我的设计理念是有用、诚实并且无害。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<eoa>\n']}]
```

- 修改yaml配置文件，以下为主要参数设置参考：

```yaml
load_checkpoint: ''                                           # 单卡推理时，只需配置checkpoint_name_or_path
auto_trans_ckpt: False                                        # 关闭自动权重转换
checkpoint_name_or_path: '/path/to/InternLM-20B-Chat.ckpt'    # 填写权重绝对路径
use_past: True                                                # 使用增量推理
vocab_file: '/path/to/tokenizer.model'                        # 配置词表路径
use_parallel: False                                           # 关闭并行模式
```

- 运行run_internlm_generate.py

```bash
python internlm/run_internlm_generate.py
```