# Yi大模型

Yi系列是由零一万物研究的大规模语言预训练模型，目前开源的有Yi-6B/34B-Base/Chat，Yi-VL-6B/34B，MindFormers已支持Yi-6B-Base。

## 前期准备

### 安装mindformers

参考[README](../../README.md#二、mindformers安装)安装mindformers。
本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件: Atlas 800T A2
- MindSpore: 2.3.0
- MindFormers: dev

**注** yi-6b推理可以在单卡上完成部署，全量微调至少需要4卡。

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，在当前路径生成该机器的RANK_TABLE_FILE的json文件，生成的文件名形如hccl_8p_01234567_127.0.0.1.json
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注** 若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

### Yi-6B-Base 预训练权重下载和转换

- 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[Yi-6B-Base](https://huggingface.co/01-ai/Yi-6B)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B
```

执行权重转换脚本

```shell
cd research
python yi/convert_ckpt_bf16.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface Yi-6B-Base权重保存目录路径
MS_CKPT_NAME: 自定义mindspore权重文件保存路径和名称
```

**注**: 请安装torch>=2.2.0和transformers>=4.37.2版本。如果执行报错，请检查并安装requests、decorator、pandas、sympy。

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

# Yi-6B-Base

Yi-6B-Base 模型以双语语言模型为目标，并在3T多语言语料库上进行了训练，成为世界上最强大的LLM之一，在语言理解、常识推理、阅读理解等方面显示出前景。

## 微调

目前提供了模型的基础配置文件`research/yi/run_yi_6b_finetune.yaml`。使用前请将配置文件中路径相关参数修改为实际路径。

## 模型性能

| config                                                       | task                  | Datasets  | SeqLength | metric | phase             | score     | performance(tokens/s/p)  |
| ------------------------------------------------------------ | --------------------- | --------- | --------- | ------ | ----------------- | --------- | ------------ |
| [yi_6b](./run_yi_6b_finetune.yaml)    | text_generation       | Yi-demo-data    | 2048      | -      | [finetune](#微调)  | -         | 3324  |

### 数据集准备

使用Yi-6B-Base进行训练或者微调时，需要使用Yi-6B-Base配套的tokenizer.model处理数据集，以及选用Yi-6B-Base的yaml配置文件进行任务启动。

目前提供[alpaca_gpt4_data_zh数据集](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_gpt4_data_zh.json) （jsonl格式）数据集的预处理脚本用于全参微调任务。

alpaca数据集样式

```text
  {
    "instruction": "保持健康的三个提示。",
    "input": "",
    "output": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"
  }
```

- step 1. 执行`alpaca_converter.py`，将原始数据集转换为对话格式。

``` bash
# 脚本路径：yi/alpaca_converter.py
# 执行转换脚本
python alpaca_converter.py \
--data_path /{path}/alpaca_gpt4_data_zh.json \
--output_path /{path}/alpaca_gpt4_data_zh-conversation.json
```

```text
# 参数说明
data_path: 存放原始数据的路径
output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
[
  {
    "id": "1",
    "conversations": [
      {
        "from": "human",
        "value": "保持健康的三个提示。"
      },
      {
        "from": "gpt",
        "value": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"
      }
    ]
  }
]
```

- step 2. 执行`yi_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```bash
# 脚本路径：yi/yi_preprocess.py
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
python yi_preprocess.py \
--dataset_type qa \
--input_glob /{path}/alpaca_gpt4_data_zh-conversation.json \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/alpaca_gpt4_data_zh.mindrecord
```

```text
# 参数说明
input_file_path：数据集输入文件路径
output_file：生成的mindrecord目标文件路径
dataset_type：数据集类型，目前仅支持"text"和"qa"
model_file：tokenizer.model文件路径
seq_length：数据长度
```

<!-- ### 预训练

- 单机多卡预训练示例

```shell
cd research
# Usage Help: bash run_singlenode.sh [START_CMD] [RANK_TABLE_FILE] [DEVICE_RANGE] [DEVICE_NUM]
bash run_singlenode.sh \
"python yi/run_yi.py \
--config yi/run_yi_6b_pretrain.yaml \
--run_mode train \
--train_dataset /{path}/wiki4096.mindrecord \
--auto_trans_ckpt True \
--use_parallel True" \
../hccl_8p_01234567_127.0.0.1.json [0,8] 8
```

**参数说明**

```text
START_CMD：Python启动命令，其中
 config：为research/yi文件夹下面的run_yi_6b_*.yaml配置文件，配置文件参数请按需修改
 run_mode：任务运行状态，支持关键字train/finetune/eval/predict/export
 train_dataset：训练数据集路径
 auto_trans_ckpt：是否自动转换ckpt
 use_parallel：是否使用并行模式
RANK_TABLE_FILE：由 mindformers/tools/hccl_tools.py 生成的分布式json文件
DEVICE_RANGE：为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
DEVICE_NUM：使用的卡的个数
```

**注**：由于模型较大，未切分的模型当seq_length为4096时，仅能进行batch_size为1的单机8卡训练。如果要使用其他并行策略训练，请参考 [多卡权重切分](../../docs/feature_cards/Transform_Ckpt.md) -->

### 微调

- 单机多卡微调示例

```shell
cd research
# Usage Help: bash run_singlenode.sh [START_CMD] [RANK_TABLE_FILE] [DEVICE_RANGE] [DEVICE_NUM]
bash run_singlenode.sh \
"python yi/run_yi.py \
--config yi/run_yi_6b_finetune.yaml \
--run_mode finetune \
--load_checkpoint  /{path}/ \
--train_dataset /{path}/alpaca_gpt4_data_zh.mindrecord \
--auto_trans_ckpt True \
--use_parallel True" \
../hccl_8p_01234567_127.0.0.1.json [0,8] 8
```

**参数说明**

```text
START_CMD：Python启动命令，其中
 config：为research/yi文件夹下面的run_yi_6b_*.yaml配置文件，配置文件参数请按需修改
 run_mode：任务运行状态，支持关键字train/finetune/eval/predict/export
 load_checkpoint：权重路径。例如路径形式为/path/ckpt/rank_0/yi_6b.ckpt，则参数填写为/path/ckpt
 train_dataset：训练数据集路径
 auto_trans_ckpt：是否自动转换ckpt
 use_parallel：是否使用并行模式
RANK_TABLE_FILE：由 mindformers/tools/hccl_tools.py 生成的分布式json文件
DEVICE_RANGE：为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
DEVICE_NUM：使用的卡的个数
```

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

### 设置推理配置

以6b推理为例，在启动前，请先行在配置文件predict_yi_6b.yaml中将processor.tokenizer.vocab_file的路径配置为实际路径,
model_config按如下配置

```yaml

processor:
  return_tensors: ms
  tokenizer:
    ...
    vocab_file: '/path/Yi-6B/tokenizer.model'  # 修改为实际路径
    ...
model:
  model_config:
    ...
    use_past: True
    is_dynamic: True
    ...

```

- generate接口推理：

```python
from mindspore import context
from mindformers.generation import GenerationConfig
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

# init yi-6b-Base model
yi_model_path = "/xxx/save_checkpoint/yi_6b.ckpt"  # 填写实际路径
config_path = '/xxx/xxx/predict_yi_6b.yaml'  # 填写实际路径

config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = yi_model_path
yi_config = LlamaConfig(**config.model.model_config)

yi_model = LlamaForCausalLM(config=yi_config)

# init yi-6b-Base tokenizer
tokenizer_path = config.processor.tokenizer.vocab_file
bos_token = config.processor.tokenizer.bos_token
eos_token = config.processor.tokenizer.eos_token
tokenizer = LlamaTokenizer(vocab_file=tokenizer_path, bos_token=bos_token,  eos_token=eos_token, add_bos_token=False)
generation_config = GenerationConfig(
    temperature=0.7,
    top_p=0.8,
    top_k=40,
    num_beams=1,
    eos_token_id=2,
    pad_token_id=0,
    bos_token_id=1,
    do_sample=False,
    max_length=100,
    repetition_penalty=1.3,
    _from_model_config=True,
)

inputs = tokenizer("以雷霆之力")["input_ids"]
outputs = yi_model.generate(inputs, generation_config=generation_config)

print(tokenizer.decode(outputs))


# 运行结果
# ['<s>以雷霆之力，将这股力量化为一道道剑气。\n“噗！”\n一柄长枪被斩断成两截后，那名大汉的脸上露出惊恐之色，他连忙向后退去，想要逃走。\n可是他的速度哪里比得上叶星辰的速度？\n只见叶星辰的身影出现在了他的面前，然后一脚踩在了这名大汉的手臂上，将他整个人都给踢飞了出去。\n这一脚的力量']
```

- pipeline接口推理：

```python

from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

yi_model_path = "/xxx/save_checkpoint/yi_6b.ckpt"  # 填写实际路径
config_path = '/xxx/xxx/predict_yi_6b.yaml'  # 填写实际路径
config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = yi_model_path
yi_config = LlamaConfig(**config.model.model_config)

yi_model = LlamaForCausalLM(yi_config)

# init yi-6b-Base tokenizer
tokenizer_path = config.processor.tokenizer.vocab_file
bos_token = config.processor.tokenizer.bos_token
eos_token = config.processor.tokenizer.eos_token
tokenizer = LlamaTokenizer(vocab_file=tokenizer_path, bos_token=bos_token,  eos_token=eos_token, add_bos_token=False)
pipeline_task = pipeline("text_generation", model=yi_model, tokenizer=tokenizer, max_length=32)
pipeline_result = pipeline_task(
    "以雷霆之力",
    temperature=0.7,
    top_p=0.8,
    top_k=40,
    num_beams=1,
    eos_token_id=2,
    pad_token_id=0,
    bos_token_id=1,
    do_sample=False,
    max_length=100,
    repetition_penalty=1.3,
    _from_model_config=True,
    use_past=config.model.model_config.use_past)

print(pipeline_result)

# 运行结果
[{'text_generation_text': ['以雷霆之力，将这股力量化为一道道剑气。\n“噗！”\n一柄长枪被斩断成两截后，那名大汉的脸上露出惊恐之色，他连忙向后退去，想要逃走。\n可是他的速度哪里比得上叶星辰的速度？\n只见叶星辰的身影出现在了他的面前，然后一脚踩在了这名大汉的手臂上，将他整个人都给踢飞了出去。\n这一脚的力量']}]

```

- 分布式推理：

参数量较大的模型，如yi-34b，可能无法进行单卡推理，可使用多卡推理，如下脚本为4卡推理样例，
msrun_launcher.sh在mindformers的scripts目录下

```shell

cd {mindformers根目录}
bash scripts/msrun_launcher.sh "research/yi/run_yi.py --config research/yi/predict_yi_34b.yaml --run_mode=predict --predict_data "DNA分子具有双螺旋结构" --predict_length 4096 --use_parallel True --use_past True" 4

# 运行结果
[{'text_generation_text': ['DNA分子具有双螺旋结构，磷酸和脱氧核糖交替连接，排列在外侧，构成基本骨架，碱基排列在内侧，两条链上的碱基通过氢键连接起来，A与T配对，G与C配对，A、C、T、G、U五种碱基的排列顺序不同，构成了DNA分子的多样性．']}]

```

## 推理性能评测

### 评测结果

|batch_size|seq_length|Atlas 800T A2（400T）tokens/s|A100（首次） tokens/s|对比
|----------|----------|----------|----------|----------|
|1|512|39.5741|35.0316|1.1297
|2|512|71.4809|77.2835|0.9249