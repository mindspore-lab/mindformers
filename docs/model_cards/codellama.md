# Code Llama

## 模型描述

Code Llama是基于Llama 2的一系列大型代码语言模型，它在开源模型中提供了最先进的性能、填充能力、对大型输入上下文的支持以及zero-shot指令跟随能力，用于编程任务。现有多种不同版本来覆盖广泛的应用领域：基础模型（Code Llama）、Python专业化模型（Code Llama - Python）和指令跟随模型（Code Llama - Instruct），每个模型分别具有7B、13B和34B个参数。所有模型都是在16k标记序列上进行训练，并对高达100k标记的输入显示出改进效果。7B和13B版本的Code Llama以及Code Llama - Instruct变体支持基于周围内容的填充功能。Code Llama是通过对Llama 2进行更高比例的代码取样进行微调而开发的。

[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)

``` text
@article{roziere2023code,
  title={Code llama: Open foundation models for code},
  author={Roziere, Baptiste and Gehring, Jonas and Gloeckle, Fabian and Sootla, Sten and Gat, Itai and Tan, Xiaoqing Ellen and Adi, Yossi and Liu, Jingyu and Remez, Tal and Rapin, J{\'e}r{\'e}my and others},
  journal={arXiv preprint arXiv:2308.12950},
  year={2023}
}
```

## 模型文件

`Code Llama` 基于 `mindformers` 实现，本仓库当前支持34b模型配置，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/llama`

   ```bash
   llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：`configs/codellama`

   ```bash
   llama
       ├── pretrain_codellama_34b.yaml             # 34b模型预训练启动配置
       ├── finetune_codellama_34b_16p.yaml         # 34b模型2机16p微调启动配置
       ├── finetune_codellama_34b_32p.yaml         # 34b模型4机32p微调启动配置
       └── predict_codellama_34b.yaml              # 34b模型推理配置
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama/
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       └── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：34b推理使用Atlas 800T A2 至少使用2卡，全量微调至少需要2机16卡，建议4机32卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext2**作为[预训练](#预训练)数据集，**code-alpaca**作为[微调](#微调)数据集。

| 数据集名称     |                    适用模型                     |          适用阶段           |                                                         下载链接                                                          |
|:----------|:-------------------------------------------:|:-----------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2 | CodeLlama_34b | Pretrain | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| code-alpaca  | CodeLlama_34b |    Finetune   |    [Link](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json)          |
| HumanEval | CodeLlama_34b |        Evaluate         |       [Link](https://github.com/openai/human-eval)                    |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

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

  1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

    ``` bash
    # 执行转换脚本
    python alpaca_converter.py \
    --data_path /{path}/code_alpaca_data.json \
    --output_path /{path}/code-alpaca-data-conversation.json
    ```

    ```text
    # 参数说明
    data_path: 存放alpaca数据的路径
    output_path: 输出转换后对话格式的数据路径
    ```

  2. 执行`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

    ```bash
    # 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13 python = 3.9
    python llama_preprocess.py \
    --dataset_type qa \
    --input_glob /{path}/code-alpaca-data-conversation.json \
    --model_file /{path}/tokenizer.model \
    --seq_length 4096 \
    --output_file /{path}/code-alpaca-fastchat4096.mindrecord
  ```

#### 模型权重下载

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重。

从huggingface下载预训练权重（权重来源于`Code Llama`），模型总计有三大类，每大类都有三种权重代码：

1. Code Llama

- [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf)
- [CodeLlama-13b](https://huggingface.co/codellama/CodeLlama-13b-hf)
- [CodeLlama-34b](https://huggingface.co/codellama/CodeLlama-34b-hf)

2. Code Llama-Python

- [CodeLlama-7b-Python](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)
- [CodeLlama-13b-Python](https://huggingface.co/codellama/CodeLlama-13b-Python-hf)
- [CodeLlama-34b-Python](https://huggingface.co/codellama/CodeLlama-34b-Python-hf)

3. Code Llama-Instruct

- [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [CodeLlama_13b-Instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
- [CodeLlama_34b-Instruct](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

CodeLlama_34b的tokenizer.model文件下载参考：

- [tokenizer.model](https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/tokenizer.model)

#### 模型权重转换

下载完成后，运行转换脚本`mindformers/convert_weight.py`，将huggingface的权重转换为完整的ckpt权重。

```shell
# 使用transformers = 4.34.0，torch>=2.0进行转换
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME
# 参数说明
input_path: huggingface权重保存目录路径
output_path: 权重保存文件名，可以指定自定义保存路径
```

## 预训练

MindFormers提供了`Code Llama 34b`多机训练示例，过程中使用**Wikitext2**数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 多机训练

CodeLlama_34b至少使用2机16卡进行训练。

使用msrun快速启动命令启动训练，具体参见[msrun快速启动](../../README.md#方式一使用已有脚本启动)。

1. 将`config/codellama/pretrain_codellama_34b.yaml`中训练数据集路径为`wiki`数据集路径。

    ```yaml
    train_dataset: &train_dataset
      data_loader:
        type: MindDataset
        dataset_dir: "/{path}/wiki4096.mindrecord"
        shuffle: True
      input_columns: ["input_ids"]
    ```

2. 根据服务器节点数等信息，修改相应的配置。

    ```yaml
    # 以codellama_34b模型四机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。
    # 配置文件路径：../configs/codellama/pretrain_codellama_34b.yaml
    parallel_config:
      data_parallel: 1
      model_parallel: 8
      pipeline_stage: 2
      use_seq_parallel: True
      micro_batch_num: 128
      vocab_emb_dp: True
      gradient_aggregation_group: 4
    ```

3. 执行运行脚本。

    多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。各个参数位置含义参见[msrun快速启动](../../README.md#方式一使用已有脚本启动)。

      ```shell
      # 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
      bash scripts/msrun_launcher.sh "run_mindformer.py \
      --config configs/codellama/pretrain_codellama_34b.yaml \
      --run_mode train" \
      16 8 192.168.1.1 8118 0 output/msrun_log False 300

      # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
      bash scripts/msrun_launcher.sh "run_mindformer.py \
      --config configs/codellama/pretrain_codellama_34b.yaml \
      --run_mode train" \
      16 8 192.168.1.1 8118 1 output/msrun_log False 300
      ```

## 微调

MindFormers提供`CodeLlama_34b`的微调示例，
过程中使用**code-alpaca**数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 多机训练

以codellama 34b为例。

1. 将`config/codellama/finetune_codellama_34b_16p.yaml`中训练数据集路径为微调数据集路径，并在`input_columns`中添加`labels`。

    ```python
    train_dataset: &train_dataset
      data_loader:
        type: MindDataset
        dataset_dir: "/{path}/code-alpaca-fastchat4096.mindrecord"
        shuffle: True
      input_columns: ["input_ids", "labels"]
    ```

2. 修改微调时学习率, 优化器参数，微调配置如下：

    ```python
    # optimizer
    optimizer:
      type: FP32StateAdamWeightDecay
      beta1: 0.9
      beta2: 0.999
      eps: 1.e-8
      learning_rate: 5.e-6

    # lr sechdule
    lr_schedule:
      type: CosineWithWarmUpLR
      learning_rate: 5.e-6
      lr_end: 0
      warmup_ratio: 0.03
      total_steps: -1 # -1 means it will load the total steps of the dataset
    ```

3. 在需要进行训练的机器中**都导入权重**，添加预训练权重路径，修改配置文件中的`load_checkpoint`，配置预训练权重路径。参考[权重切分与合并](../feature_cards/Transform_Ckpt.md)的物理机训练案例，修改权重配置如下：

    1). 有共享盘

      ```python
      auto_trans_ckpt: True
      load_checkpoint: path/to/checkpoint_dir
      ```

      > 注：权重需要按照path/to/checkpoint_dir/rank_0/xxx.ckpt存放，load_checkpoint只需要填写到checkpoint_dir即可

    ​2). 无共享盘

      ```python
      auto_trans_ckpt: False
      load_checkpoint: path/to/transformed_checkpoint
      ```

      > 注：权重按照[权重切分与合并](../feature_cards/Transform_Ckpt.md)的教程先切成对应的份数，load_checkpoint填写到transformed_checkpoint，该文件夹下存放有rank_X的权重文件夹。

4. 启动微调任务，codellama-34b模型以2机16卡进行微调，多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

    ```shell
    # 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
    bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/codellama/finetune_codellama_34b_16p.yaml \
    --run_mode finetune" \
    16 8 192.168.1.1 8118 0 output/msrun_log False 300

    # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
    bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config finetune_codellama_34b_16p.yaml \
    --run_mode finetune" \
    16 8 192.168.1.1 8118 1 output/msrun_log False 300
    ```

### 分布式训练权重合并

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[[权重切分与合并](../feature_cards/Transform_Ckpt.md)]

1. 获取模型切分策略文件：

   在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

    ```shell
    python transform_ckpt.py \
    --src_ckpt_strategy {path}/output/strategy/ \
    --src_ckpt_dir {path}/output/checkpoint/ \
    --dst_ckpt_dir {path}/target_checkpoint/ \
    --prefix llama2_7b
    ```

    ```text
    # 参数说明
    src_ckpt_strategy: 步骤1中的切分策略文件路径
    src_ckpt_dir: 原切分权重文件夹
    dst_ckpt_dir: 目标路径
    prefix: ckpt文件前缀名
    ```

    > 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 推理

MindFormers提供`CodeLlama_34b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持多卡以及多batch推理。

  ```shell
  # 脚本使用
  bash scripts/examples/codellama/run_codellama_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM

  # 参数说明
  PARALLEL:    是否使用多卡推理, 用parallel表示, CodeLlama_34b目前只支持多卡推理,
  CONFIG_PATH: 模型配置文件路径
  CKPT_PATH:   模型权重文件路径
  DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
  ```

  > 注: CodeLlama_34b默认使用四卡进行推理

### 多卡推理

以`CodeLlama_34b`4卡推理为例。

  ```shell
  bash scripts/examples/codellama/run_codellama_predict.sh parallel \
  configs/codellama/predict_codellama_34b.yaml \
  path/to/codellama_34b.ckpt 4
  ```

## 评测

`Code Llama`当前支持的评测任务如下：

| 任务类型 | 评测指标 |   数据集   |
| :------: | :------: | :--------: |
| 代码生成 |  Pass@1  | HumanEeval |

### 代码生成

1. 获取数据集

   [HumanEval数据集](https://github.com/openai/human-eval)是一组164个手写的编程问题数据集，被称为HumanEval数据集。每个问题包括函数签名、文档字符串、主体和几个单元测试，平均每个问题有7.7个测试。运行下面的命令，数据集在`data`文件夹中，名为`HumanEval.jsonl.gz`。

    ```shell
    git clone https://github.com/openai/human-eval.git
    ```

2. 将下面的preprocess.py放入代码仓中的`human-eval`文件夹中，提取出`data/HumanEval.jsonl.gz`中的`prompt`字符串列表，为prompt_input，按照[推理章节](#推理)进行推理。

    ```python
    # preprocess.py 文件
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

    ```shell
    # 运行以下命令可以获得数据集中的输入(prompt_input)，任务id(task_ids)和执行函数(entry_points)。比如"HumanEval/0"的输入时from typing import List..., 而该代码的执行函数入口名称为has_close_elements.
    python preprocess.py --data_path path/to/HumanEval.jsonl.gz
    ```

3. 提出代码生成函数主干函数，由于生成代码会生成多余函数，评测时只需要评测函数即可，函数名为`data/HumanEval.jsonl.gz`中的`entry_point`，组成如下结构保存为`samples.jsonl`：

    ```bash
    {'task_id': "HumanEval/0","completion": "inference result"}
    ```

    4. 安装HumanEval

    ```python
    pip install -e human-eval
    ```

    > 注：
    >
    > 1. 解除`human-eval/human_eval/execution.py`的第58行注释;
    > 2. 由于代码生成时会自带prompt，因此将`human-eval/human_eval/execution.py`第39行的`problem["prompt"] + completion` 改为 `completion`即可。

5. 生成测试分数

    ```python
    evaluate_functional_correctness samples.jsonl
    # {'pass@1': 0.4695}
    ```

