# Code Llama

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

Code Llama是基于Llama 2的一系列大型代码语言模型，它在开源模型中提供了最先进的性能、填充能力、对大型输入上下文的支持以及zero-shot指令跟随能力，用于编程任务。现有多种不同版本来覆盖广泛的应用领域：基础模型（Code Llama）、Python专业化模型（Code Llama - Python）和指令跟随模型（Code Llama - Instruct），每个模型分别具有7B、13B和34B个参数。所有模型都是在16k标记序列上进行训练，并对高达100k标记的输入显示出改进效果。7B和13B版本的Code Llama以及Code Llama - Instruct变体支持基于周围内容的填充功能。Code Llama是通过对Llama 2进行更高比例的代码取样进行微调而开发的。

[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                                       |      Task       | SeqLength | Datasets |  Performance   |  Phase   |
|:-----------------------------------------------------------------------------|:---------------:|:---------:|:--------:|:--------------:|:--------:|
| [codellama_34b_32p](../../configs/codellama/finetune_codellama_34b_32p.yaml) | text_generation |   4096    |  belle   | 667 tokens/s/p | Finetune |
| [codellama_34b](../../configs/codellama/predict_codellama_34b.yaml)          | text_generation |   4096    |    /     |  139 tokens/s  | Predict  |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                                       |      Task       | SeqLength |  Datasets   |  Performance   |  Phase   |
|:-----------------------------------------------------------------------------|:---------------:|:---------:|:-----------:|:--------------:|:--------:|
| [codellama_34b_16p](../../configs/codellama/finetune_codellama_34b_16p.yaml) | text_generation |   4096    | code-alpaca | 669 tokens/s/p | Finetune |
| [codellama_34b_32p](../../configs/codellama/finetune_codellama_34b_32p.yaml) | text_generation |   4096    | code-alpaca | 747 tokens/s/p | Finetune |

## 模型文件

`Code Llama` 基于 `mindformers` 实现，本仓库当前支持34b模型配置，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/llama
       ├── __init__.py
       ├── convert_reversed.py         # 权重逆向转换
       ├── convert_weight.py           # llama权重转换
       ├── llama.py                    # 模型实现
       ├── llama_config.py             # 模型配置项
       ├── llama_interleave.py         # interleave优化
       ├── llama_layer.py              # llama网络层定义
       ├── llama_processor.py          # llama预处理
       ├── llama_tokenizer.py          # tokenizer
       ├── llama_tokenizer_fast.py     # fast tokenizer
       └── llama_transformer.py        # transformer层实现
   ```

2. 模型配置：

   ```text
   configs/codellama
       ├── pretrain_codellama_34b.yaml             # 34b模型预训练启动配置
       ├── finetune_codellama_34b_16p.yaml         # 34b模型2机16p微调启动配置
       ├── finetune_codellama_34b_32p.yaml         # 34b模型4机32p微调启动配置
       └── predict_codellama_34b.yaml              # 34b模型推理配置
   ```

3. 数据预处理脚本：

   ```text
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       ├── conversation.py         # 对话提示模板生成脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

> 注：34b推理使用Atlas 800T A2 至少使用2卡，全量微调至少需要2机16卡，建议4机32卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供`Wikitext2`作为[预训练](#预训练)数据集，`code-alpaca`作为[微调](#微调)数据集。

| 数据集名称       |     适用模型      |   适用阶段   |                                          下载链接                                           |
|:------------|:-------------:|:--------:|:---------------------------------------------------------------------------------------:|
| Wikitext2   | CodeLlama_34b | Pretrain |    [Link](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/faq/func_related.html)     |
| code-alpaca | CodeLlama_34b | Finetune | [Link](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) |
| HumanEval   | CodeLlama_34b | Evaluate |                      [Link](https://github.com/openai/human-eval)                       |

数据预处理中所用的`tokenizer.model`可以点击[链接](https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/tokenizer.model)进行下载。

- **Wikitext2 数据预处理**

  使用`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
  python llama_preprocess.py \
    --dataset_type wiki \
    --input_glob /{path}/wiki.train.tokens \
    --model_file /{path}/tokenizer.model \
    --seq_length 4096 \
    --output_file /{path}/wiki4096.mindrecord

  # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.train.tokens的文件路径
  model_file:   模型tokenizer.model文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

- **code-alpaca 数据预处理**

1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，将原始数据集转换为多轮对话格式。

   ```shell
   python alpaca_converter.py \
    --data_path /{path}/code_alpaca_data.json \
    --output_path /{path}/code-alpaca-data-conversation.json

   # 参数说明
   data_path:   下载的alpaca数据路径
   output_path: 输出转换后对话格式的数据路径
   ```

2. 执行`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

   ```shell
   python llama_preprocess.py \
    --dataset_type qa \
    --input_glob /{path}/code-alpaca-data-conversation.json \
    --model_file /{path}/tokenizer.model \
    --seq_length 4096 \
    --output_file /{path}/code-alpaca-fastchat4096.mindrecord
   ```

#### 模型权重下载

MindFormers提供下载HuggingFace官方权重的下载链接，用户可通过链接下载权重并经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/tokenizer.model)

| 模型名称                        |                           HuggingFace权重                            |
|:----------------------------|:------------------------------------------------------------------:|
| CodeLlama-34b               |     [Link](https://huggingface.co/codellama/CodeLlama-34b-hf)      |
| CodeLlama-34b-Python        |  [Link](https://huggingface.co/codellama/CodeLlama-34b-Python-hf)  |
| CodeLlama_34b-Instruct      | [Link](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

#### 模型权重转换

下载完成后，运行转换脚本`mindformers/convert_weight.py`，将huggingface的权重转换为完整的ckpt权重。

```shell
# 使用transformers = 4.34.0，torch>=2.0进行转换
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 预训练

MindFormers提供了`Code Llama 34b`多机预训练示例，使用`Wikitext2`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

`Code Llama 34b`由于模型规模较大，仅支持多机预训练，至少使用2机16卡进行训练。

1. 修改配置文件`config/codellama/pretrain_codellama_34b.yaml`

   根据服务器节点数等信息，修改相应的并行配置。

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 2
     use_seq_parallel: True
     micro_batch_num: 128
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

2. 在分布式节点上执行脚本

   多机多卡训练需要不同节点上执行启动命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)。

   ```shell
   # 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/codellama/pretrain_codellama_34b.yaml \
    --train_dataset_dir /path/wiki4096.mindrecord \
    --run_mode train" \
   16 8 {ip_addr} 8118 0 output/msrun_log False 300

   # 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/codellama/pretrain_codellama_34b.yaml \
    --train_dataset_dir /path/wiki4096.mindrecord \
    --run_mode train" \
   16 8 {ip_addr} 8118 1 output/msrun_log False 300

   # 参数说明
   config:            配置文件路径
   train_dataset_dir: 训练数据集路径
   run_mode:          运行模式, 预训练时设置为train
   ```

## 微调

MindFormers提供`Code Llama 34b`的微调示例，使用`code-alpaca`数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

`Code Llama 34b`由于模型规模较大，仅支持多机微调，至少使用2机16卡进行训练。

1. 生成多机分布式权重

   如果使用共享存储，可以将模型完整权重放在共享存储内，同时设置配置文件或脚本参数`auto_trans_ckpt=True`，使用权重自动转换功能。

   如果不使用共享存储，可以参考[多卡权重之间的转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/transform_weight.html#%E5%9C%BA%E6%99%AF%E4%BA%8C-%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B9%8B%E9%97%B4%E6%97%A0%E5%85%B1%E4%BA%AB%E7%9B%98)完成分布式权重转换后拉起预训练任务。

2. 修改配置文件`config/codellama/finetune_codellama_34b_16p.yaml`

   根据服务器节点数等信息，修改相应的并行配置。

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 2
     use_seq_parallel: True
     micro_batch_num: 128
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

3. 在分布式节点上执行脚本，进行2机16卡微调

   多机多卡训练需要不同节点上执行启动命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)。

   示例使用共享存储并开启`auto_trans_ckpt`进行权重自动转换。

   ```shell
   # 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/codellama/finetune_codellama_34b_16p.yaml \
    --load_checkpoint /path/codellama_34b.ckpt \
    --auto_trans_ckpt True \
    --train_dataset_dir /path/code-alpaca-fastchat4096.mindrecord \
    --run_mode finetune" \
   16 8 {ip_addr} 8118 0 output/msrun_log False 300

   # 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/codellama/finetune_codellama_34b_16p.yaml \
    --load_checkpoint /path/codellama_34b.ckpt \
    --auto_trans_ckpt True \
    --train_dataset_dir /path/code-alpaca-fastchat4096.mindrecord \
    --run_mode finetune" \
   16 8 {ip_addr} 8118 1 output/msrun_log False 300

   # 参数说明
   config:            配置文件路径
   load_checkpoint:   模型权重文件路径
   auto_trans_ckpt:   是否开启自动权重转换
   train_dataset_dir: 训练数据集路径
   run_mode:          运行模式, 微调时设置为finetune
   ```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/transform_weight.html#%E8%87%AA%E5%8A%A8%E8%BD%AC%E6%8D%A2)和[离线权重转换](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/function/transform_weight.html#%E7%A6%BB%E7%BA%BF%E8%BD%AC%E6%8D%A2)进行分布式模型权重转换。

## 推理

MindFormers提供`CodeLlama_34b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/codellama/run_codellama_predict.sh CONFIG_PATH CKPT_PATH DEVICE_NUM

# 参数说明
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
DEVICE_NUM:  使用卡数
```

`CodeLlama_34b`仅支持多卡推理，以`CodeLlama_34b`4卡推理为例。

需要去配置文件`predict_codellama_34b.yaml`中修改配置，指定`tokenizer.model`的实际路径：

```yaml
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "{path}/tokenizer.model"
```

执行脚本如下：

```shell
bash scripts/examples/codellama/run_codellama_predict.sh \
 configs/codellama/predict_codellama_34b.yaml \
 path/to/codellama_34b.ckpt 4

# 推理结果
# <s>def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n):
# ...
# def selection_sort(arr):
#     n = len(arr)
#     for i in range(n):
# ...
```

## 评测

`Code Llama`当前支持的评测任务如下：

| 任务类型 |  评测指标  |    数据集     |
|:----:|:------:|:----------:|
| 代码生成 | Pass@1 | HumanEeval |

### 代码生成

评测使用`HumanEval`数据集可通过[数据集下载](#数据集下载)获得，使用`git`下载代码仓。

1. 构建如下`preprocess.py`脚本放入数据集代码仓中的`human-eval`文件夹中，进行数据集预处理。

   ```python
   # preprocess.py
   import argparse

   from data import stream_jsonl


   def process_data(tasks):
       prompt_input = [task["prompt"] for task in tasks]
       user_ids = [task["task_id"] for task in tasks]
       entry_inputs = [task["entry_point"] for task in tasks]
       return prompt_input, user_ids, entry_inputs


   if __name__ == "__main__":
       parser = argparse.ArgumentParser("copy prompt")
       parser.add_argument("--data_path", default="", type=str)
       args = parser.parse_args()

       data_list = []
       for data in stream_jsonl(args.data_path):
           data_list.append(data)
       prompt_input, task_ids, entry_inputs = process_data(data_list)

       print(prompt_input)
       print(task_ids)
       print(entry_inputs)
   # ['from typing import List\n\n\ndef has_close_e...
   # ['HumanEval/0', 'HumanEval/1', 'HumanEval/2',...
   # ['has_close_elements', 'separate_paren_groups',...
   ```

   执行`preprocess.py`脚本，提取出`data/HumanEval.jsonl.gz`中的输入`prompt_input`，任务id`task_ids`和执行函数`entry_inputs`字符串列表。

   ```shell
    python preprocess.py --data_path path/to/HumanEval.jsonl.gz
    ```

2. 使用`prompt_input`数据进行推理并将推理结果用`completion`字段保存，整理为`samples.jsonl`。

   ```text
   # 例如`task_id`为`HumanEval/0`,对应`prompt_input`为`from typing import List\n\n\ndef has_close_e...`；
   # 使用`prompt_input`进行推理，得到推理结果为`"from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\nfor i in range(len(numbers)):\n...`；
   # 将推理结果去除`prompt_input`数据后保存至对应任务的`completion`字段；
   # 最后将`{'task_id': "HumanEval/0","completion": "for i in range(len(numbers)):\n..."}`保存至`samples.jsonl`文件。
   {'task_id': "HumanEval/0","completion": "inference result"}
   ```

3. 安装`HumanEval`依赖

   ```shell
   pip install -e human-eval
   ```

4. 生成测试分数

   ```shell
   evaluate_functional_correctness samples.jsonl
    # {'pass@1': 测试分数}
   ```

