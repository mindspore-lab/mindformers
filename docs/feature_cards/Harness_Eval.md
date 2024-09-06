# Harness评测

## 基本介绍

[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
是一个开源语言模型评测框架，提供60多种标准学术数据集的评测，支持HuggingFace模型评测、PEFT适配器评测、vLLM推理评测等多种评测方式，支持自定义prompt和评测指标，包含loglikelihood、generate_until、loglikelihood_rolling三种类型的评测任务。基于Harness评测框架对MindFormers进行适配后，支持加载MindFormers模型进行评测。

## 安装

```shell
pip install lm_eval==0.4.3
```

## 使用方式

### 查看数据集评测任务

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --tasks list
```

### 启动单卡评测脚本

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=MODEL_DIR,device_id=0" --tasks TASKS
```

### 启动多卡并行评测脚本

```shell
#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

bash  mindformers/scripts/msrun_launcher.sh "toolkit/benchmarks/eval_with_harness.py \
    --model mf \
    --model_args pretrained=MODEL_DIR,use_parallel=True,tp=1,dp=4 \
    --tasks TASKS \
    --batch_size 4" 4
```

可通过环境变量ASCEND_RT_VISIBLE_DEVICES设置多卡卡号。

### 评测参数

#### Harness主要参数

| 参数            | 类型  | 参数介绍                      | 是否必须 |
|---------------|-----|---------------------------|------|
| --model       | str | 须设置为mf，对应为MindFormers评估策略 | 是    |
| --model_args  | str | 模型及评估相关参数，见下方模型参数介绍       | 是    |
| --tasks       | str | 数据集名称，可传入多个数据集，逗号分割       | 是    |
| --batch_size  | int | 批处理样本数                    | 否    |
| --num_fewshot | int | Few_shot的样本数              | 否    |
| --limit       | int | 每个任务的样本数，多用于功能测试          | 否    |

#### MindFormers模型参数

| 参数           | 类型   | 参数介绍                              | 是否必须 |
|--------------|------|-----------------------------------|------|
| pretrained   | str  | 模型目录路径                            | 是    |
| use_past     | bool | 是否开启增量推理，generate_until类型的评测任务须开启 | 否    |
| device_id    | int  | 设备id                              | 否    |
| use_parallel | bool | 开启并行策略                            | 否    |
| dp           | int  | 数据并行                              | 否    |
| tp           | int  | 模型并行                              | 否    |

### 评测前准备

1. 创建模型目录MODEL_DIR；
2. 模型目录下须放置MindFormers权重、yaml配置文件、分词器文件，获取方式参考MindFormers模型README文档；
3. 配置yaml配置文件。

yaml配置参考：

```yaml
run_mode: 'predict'
model:
  model_config:
    use_past: True
    checkpoint_name_or_path: "model.ckpt"
processor:
  tokenizer:
    vocab_file: "tokenizer.model"
```

## 评测样例

```shell
#!/bin/bash

python toolkit/benchmarks/eval_with_harness.py --model mf --model_args "pretrained=./llama3-8b,use_past=True" --tasks gsm8k

# 评估结果如下，其中Filter对应匹配模型输出结果的方式，Metric对应评估指标，Value对应评估分数，stderr对应分数误差。
# mf (pretrained=./llama3-8b), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.5034|±  |0.0138|
# |     |       |strict-match    |     5|exact_match|↑  |0.5011|±  |0.0138|
```

## 支持特性说明

Harness全量评测任务见[查看数据集评测任务](#查看数据集评测任务)，loglikelihood_rolling类型的评测任务暂未支持：

- french_bench_perplexity
- french_bench_opus_perplexity
- french_bench_wikitext_fr
- paloma
- pile
- pile_10k
- wikitext
