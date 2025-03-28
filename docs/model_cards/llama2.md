# Llama 2

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.5.0* 之后的版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

Llama 2，是Meta基于LLaMA 1的更新版本，基于新的公开可用数据混合进行训练，同时将预训练语料库的大小增加了40%，最后将模型的上下文长度翻倍（由2048提高到4096），并采用了分组查询注意力机制。Llama 2模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。Llama 2按照参数量，目前有三个版本：Llama 2-7B（7B）、Llama 2-13B（13B）、Llama 2-70B（70B），本仓库已全部支持三版权重，权重文件来源于MetaLLama2。Llama 2 的7B和13B 模型结构与LLaMA 1一致，70B 则加入分组查询注意力（GQA）。

[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

``` text
@article{touvron2023llama,
  title={Llama 2: Open foundation and fine-tuned chat models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                                   |      Task       | Datasets | SeqLength | DataType |  Phase   |   Performance   |
|:-------------------------------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:--------:|:---------------:|
| [llama2_7b](../../configs/llama2/pretrain_llama2_7b_bf16.yaml)           | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 4160 tokens/s/p |
| [llama2_7b](../../configs/llama2/finetune_llama2_7b.yaml)                | text_generation |  alpaca  |   4096    | float16  | Finetune | 3484 tokens/s/p |
| [llama2_13b](../../configs/llama2/finetune_llama2_13b_bf16.yaml)         | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 1691 tokens/s/p |
| [llama2_13b_lora](../../configs/llama2/lora_llama2_13b.yaml)             | text_generation |  alpaca  |   4096    | float16  |   LoRA   | 2193 tokens/s/p |
| [llama2_70b_32p](../../configs/llama2/finetune_llama2_70b_bf16_32p.yaml) | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 337 tokens/s/p  |
| [llama2_7b](../../configs/llama2/predict_llama2_7b.yaml)                 | text_generation |    -     |   4096    |    -     | Predict  |  332 tokens/s   |
| [llama2_13b](../../configs/llama2/predict_llama2_13b.yaml)               | text_generation |    -     |   4096    |    -     | Predict  |  420 tokens/s   |
| [llama2_70b](../../configs/llama2/predict_llama2_70b.yaml)               | text_generation |    -     |   4096    |    -     | Predict  |  522 tokens/s   |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                                   |      Task       | Datasets | SeqLength | DataType |  Phase   |   Performance   |
|:-------------------------------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:--------:|:---------------:|
| [llama2_13b](../../configs/llama2/finetune_llama2_13b_bf16.yaml)         | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 1945 tokens/s/p |
| [llama2_13b](../../configs/llama2/finetune_llama2_13b.yaml)              | text_generation |  alpaca  |   4096    | float16  | Finetune | 1911 tokens/s/p |
| [llama2_70b_32p](../../configs/llama2/finetune_llama2_70b_bf16_32p.yaml) | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 404 tokens/s/p  |
| [llama2_70b_64p](../../configs/llama2/finetune_llama2_70b_bf16_64p.yaml) | text_generation |  alpaca  |   4096    | bfloat16 | Finetune | 405 tokens/s/p  |

## 模型文件

`Llama 2`基于`mindformers`实现，主要涉及的文件有：

1. 模型具体实现：

   ```bash
   mindformers/models/llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       ├── llama_tokenizer.py        # tokenizer
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：

   ```bash
   configs/llama2
       ├── predict_llama2_7b.yaml          # 7b模型推理启动配置
       ├── predict_llama2_13b.yaml         # 13b模型推理启动配置
       ├── predict_llama2_70b.yaml         # 70b模型推理启动配置
       ├── pretrain_llama2_7b.yaml         # 7b模型预训练启动配置
       ├── pretrain_llama2_13b.yaml        # 13b模型预训练启动配置
       ├── pretrain_llama2_70b.yaml        # 70b模型预训练启动配置
       ├── finetune_llama2_7b.yaml         # 7b模型全量微调启动配置
       ├── finetune_llama2_13b.yaml        # 13b模型全量微调启动配置
       └── finetune_llama2_70b.yaml        # 70b模型全量微调启动配置
   ```

3. 数据预处理脚本：

   ```bash
   mindformers/tools/dataset_preprocess/llama
       ├── alpaca_converter.py     # 基于fschat的alpaca数据集格式转换脚本
       ├── llama_preprocess.py     # llama模型的mindrecord数据处理脚本
       └── squad_data_process.py   # squad数据集格式转换脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持7b,13b单机单卡推理，70b推理至少使用8卡，全参微调至少需要4机32卡，推荐使用8机64卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext2**作为[预训练](#预训练)数据集和PPL评测数据集，**alpaca**作为[微调](#微调)数据集，**SQuAD1.1**为阅读理解评测数据集。

| 数据集名称     |                    适用模型                     |          适用阶段           |                                        下载链接                                        |
|:----------|:-------------------------------------------:|:-----------------------:|:----------------------------------------------------------------------------------:|
| Wikitext2 | llama2-7b <br/> llama2-13b <br/> llama2-70b | Pretrain <br/> Evaluate | [Link](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/faq/func_related.html)   |
| alpaca    | llama2-7b <br/> llama2-13b <br/> llama2-70b |        Finetune         |  [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)   |
| SQuAD 1.1 | llama2-7b <br/> llama2-13b <br/> llama2-70b |        Evaluate         |                    [Link](https://data.deepai.org/squad1.1.zip)                    |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **Wikitext2 数据预处理—预训练**

  使用`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
  python mindformers/tools/dataset_preprocess/llama/llama_preprocess.py \
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

  > 注：`bos`, `eos`, `pad`等特殊`ids`要和`yaml`配置文件中`model_config`部分保持一致，默认`bos_token_id=1`, `eos_token_id=2`, `pad_token_id=0`。
如果有所修改，配置文件中对应设置也需要修改，通常预训练数据不包含`pad_token`，因此建议设置`pad_token_id=-1`。

- **Wikitext2 数据预处理—评测**

  使用`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`对下载后的数据进行预处理，并生成Mindrecord数据。

  ```shell
   python mindformers/tools/dataset_preprocess/llama/llama_preprocess.py \
    --dataset_type wiki \
    --input_glob  /{path}/wiki.valid.tokens \
    --model_file /{path}/tokenizer.model \
    --seq_length 4095 \
    --output_file /{path}/wiki4096.mindrecord

    # 参数说明
  dataset_type: 预处理数据类型
  input_glob:   输入下载后wiki.valid.tokens的文件路径
  model_file:   模型tokenizer.model文件路径
  seq_length:   输出数据的序列长度
  output_file:  输出文件的保存路径
  ```

- **alpaca 数据预处理**

  1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

     ```shell
     python mindformers/tools/dataset_preprocess/llama/alpaca_converter.py \
       --data_path /{path}/alpaca_data.json \
       --output_path /{path}/alpaca-data-conversation.json

     # 参数说明
     data_path:   输入下载的文件路径
     output_path: 输出文件的保存路径
     ```

  2. 执行`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`，生成Mindrecord数据，将带有prompt模板的数据转换为mindrecord格式。

     ```shell
     # 此工具依赖fschat工具包解析prompt模板, 请提前安装fschat >= 0.2.13 python = 3.9
     python mindformers/tools/dataset_preprocess/llama/llama_preprocess.py \
       --dataset_type qa \
       --input_glob /{path}/alpaca-data-conversation.json \
       --model_file /{path}/tokenizer.model \
       --seq_length 4096 \
       --output_file /{path}/alpaca-fastchat4096.mindrecord

     # 参数说明
     dataset_type: 预处理数据类型
     input_glob:   转换后的alpaca的文件路径
     model_file:   模型tokenizer.model文件路径
     seq_length:   输出数据的序列长度
     output_file:  输出文件的保存路径
     ```

- **SQuAD 1.1 数据预处理**

  执行`mindformers/tools/dataset_preprocess/llama/squad_data_process.py`生成Mindrecord数据

  ```shell
  python mindformers/tools/dataset_preprocess/llama/squad_data_process.py \
    --input_file /{path}/squad/dev-v1.1.json \
    --output_file /{path}/squad2048.mindrecord \
    --mode eval \
    --max_length 2048 \
    --tokenizer_type "llama2_7b"
  ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

| 模型名称            |                                                 MindSpore权重                                                  |                      HuggingFace权重                       |
|:----------------|:------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| llama2-7b       |    [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)    | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| llama2-13b      | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt) | [Link](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| llama2-70b      |                                                      /                                                       | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) |

> 注：Llama2的所有权重都需要通过向Meta[提交申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)来获取，如有需要请自行申请。

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 预训练

MindFormers提供`llama2-7b`单机多卡以及`llama2_13b`多机多卡的预训练示例，过程中使用`Wikitext2`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机训练

以Llama2-7b为例，执行msrun启动脚本，进行8卡分布式训练。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/pretrain_llama2_7b.yaml \
 --train_dataset_dir /{path}/wiki4096.mindrecord \
 --use_parallel True \
 --run_mode train" 8
```

在`llama2_70b`预训练中，可以通过如下方式提升模型性能：

1. 修改配置文件中`qkv_concat=True`, `micro_batch_num=256`
2. 创建`parallel_speed_up.json文件`，文件内容如下

   ```json
   {
     "recompute_comm_overlap": false,
     "matmul_grad_comm_overlap": true,
     "enable_task_opt": false,
     "enable_grad_comm_opt": false,
     "enable_opt_shard_comm_opt": false,
     "enable_concat_eliminate_opt": false,
     "enable_begin_end_inline_opt": false,
     "compute_communicate_fusion_level": 0
   }
   ```

   同时在配置文件`context`部分添加`ascend_config`

   ```yaml
   context:
     ascend_config:
       parallel_speed_up_json_path: "/{path}/parallel_speed_up.json"
   ```

> 如果报错提示显存不足，可以通过`export HCCL_BUFFSIZE=100`将对应环境变量下调至100。

`ymal`配置文件中各参数含义详见[Config配置说明](../../configs/README.md)，`parallel_speed_up`各参数含义详见[parallel_speed_up说明](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.set_context.html#mindspore.set_context)。

### 多机训练

以Llama2-13b为例，执行2机16卡预训练。

1. 根据使用节点数等信息，修改相应的配置文件`configs/llama2/pretrain_llama2_13b.yaml`

   ```yaml
   parallel_config:
     data_parallel: 2
     model_parallel: 4
     pipeline_stage: 2
     micro_batch_num: 16
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

   > 注：如使用节点数和卡数改变需要修改`data_parallel`, `model_parallel`, `pipeline_stage`满足实际运行的卡数 `device_num=data_parallel×model_parallel×pipeline_stage`，
同时满足`micro_batch_num >= pipeline_stage`。

2. 执行msrun启动脚本

   多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数`MASTER_ADDR`设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，各个参数位置含义参见msrun快速启动。

   ```shell
   # 节点0作为主节点, {ip_addr}处填写节点0实际ip, 总共16卡且每个节点8卡
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config {CONFIG_PATH} \
     --train_dataset_dir /{path}/wiki4096.mindrecord \
     --use_parallel True \
     --run_mode {train}" \
     16 8 {ip_addr} 8118 0 output/msrun_log False 300

   # 节点1，{ip_addr}处填写节点0实际ip，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config {CONFIG_PATH} \
     --train_dataset_dir /{path}/wiki4096.mindrecord \
     --use_parallel True \
     --run_mode {train}" \
     16 8 {ip_addr} 8118 1 output/msrun_log False 300
   ```

3. 对于llama2-70b模型，再训练之前请定义以下环境变量。

   ```shell
   export MS_DEV_SIDE_EFFECT_LOAD_ELIM=3 # 优化显存
   ```

## 微调

MindFormers提供`Llama2-7b`的微调示例，过程中使用`alpaca`数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

以Llama2-7b为例，执行msrun启动脚本，进行8卡分布式训练。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b.yaml \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --use_parallel True \
 --run_mode finetune" 8
```

#### 多机训练

多机多卡微调任务启动预训练类似，可参考[预训练章节](#预训练)并对启动命令进行如下修改：

1. 增加脚本入参`--load_checkpoint /{path}/llama2_7b.ckpt`加载预训练权重
2. 设置启动脚本中的`--train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord`加载微调数据集
3. 设置启动脚本中的`--run_mode finetune`

### LoRA微调

LoRA低参微调算法，可以冻结原模型权重，仅在小规模参数量上进行训练，使大模型在少量资源的情况下也能训练。

MindFormers提供`Llama2-7b`的LoRA微调示例，微调过程中使用的数据集可以参考[数据集下载](#数据集下载)获得。

以Llama2-7b为例，执行msrun启动脚本，进行8卡分布式微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" 8
```

如果加载分布式权重，加载权重路径应设置为rank_0的上一层路径，同时开启权重自动转换功能`--auto_trans_ckpt True`：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/rank_0/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

### PrefixTuning微调

PrefixTuning低参微调算法，可以冻结原模型权重，仅在kv向量前添加可训练前缀向量进行训练，使大模型在少量资源的情况下也能训练。

MindFormers提供`Llama2-7b`的PrefixTuning微调示例，微调过程中使用的数据集可以参考[数据集下载](#数据集下载)获得。

以Llama2-7b为例，执行msrun启动脚本，进行8卡分布式微调。

> 注：PrefixTuning微调使用数据集`seq_length=512`，数据预处理时应按该序列长度对数据进行处理。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b_prefixtuning.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat512.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" 8
```

如果加载分布式权重，加载权重路径应设置为rank_0的上一层路径，同时开启权重自动转换功能`--auto_trans_ckpt True`：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/finetune_llama2_7b_prefixtuning.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat512.mindrecord \
 --load_checkpoint /{path}/rank_0/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 推理

MindFormers提供`Llama2-7b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡、多卡以及多batch推理。

```shell
# 脚本使用
bash scripts/examples/llama2/run_llama2_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

以`Llama2-7b`单卡推理为例。

```shell
bash scripts/examples/llama2/run_llama2_predict.sh single \
 configs/llama2/predict_llama2_7b.yaml \
 path/to/llama2_7b.ckpt

# 多batch输出
# <s>I love Beijing, because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained ...
# <s>Huawei is a company that has been around for a long time. ...
```

### 多卡推理

以`Llama2-7b`2卡推理为例。

```shell
bash scripts/examples/llama2/run_llama2_predict.sh parallel \
 configs/llama2/predict_llama2_7b.yaml \
 path/to/llama2_7b.ckpt 2

# 多batch输出
# <s>I love Beijing, because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained ...
# <s>Huawei is a company that has been around for a long time. ...
```

## 评测

以Llama2_7b为例，Llama 2当前支持使用based model（初始权重）进行评测任务如下：

| 任务类型 |    评测指标    |    数据集    |
|:----:|:----------:|:---------:|
| 文本生成 | Perplexity | WikiText2 |
| 阅读理解 |   Em/F1    | SQuAD 1.1 |

评测时在`vocab_file`配置中加入相应`tokenizer.model`的路径，若使用Atlas 800T A2进行评测，则还需在配置中加入`ascend_config`配置。

```yaml
# context config
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"

# tokenizer
processor:
  return_tensors: ms
  tokenizer:
    vocab_file: "path/to/tokenizer.model"
```

### 文本生成

1. 获取数据集

   文本生成任务评测使用**WikiText2**数据集，可通过[数据集下载](#数据集下载)得到，并进行相应的预处理。

2. 修改模型配置文件`configs/llama2/pretrain_llama2_7b_bf16.yaml`

   ```yaml
   metric:
     type: PerplexityMetric
   ```

3. 执行评测命令，指标为PPL

   ```shell
   python run_mindformer.py \
     --config configs/llama2/pretrain_llama2_7b_bf16.yaml \
     --eval_dataset_dir /{path}/wiki4096.mindrecord \
     --run_mode eval \
     --load_checkpoint /{path}/llama2_7b.ckpt \
     --epochs 1 \
     --use_parallel False \
     --device_id 0

   # PerplexityMetric = {'PerplexityMetric': {'loss': 2.1142693907022476, 'PPL': 6.58}}
   ```

### 阅读理解

1. 获取数据集

   阅读理解任务评测使用**SQuAD 1.1**数据集，可通过[数据集下载](#数据集下载)得到，并进行相应的预处理。**SQuAD 1.1**中包含针对500+文章的10万+问答对，是一个阅读理解数据集，由维基百科文章上提出的问题组成，其中每个问题的答案都是相应文章中的一段文本。

2. 修改模型配置文件`configs/llama2/predict_llama2_7b.yaml`

   ```yaml
   # eval dataset
   eval_dataset:
     data_loader:
       type: MindDataset
       dataset_dir: "/{path}/squad2048.mindrecord"  # 处理后的评测数据集路径
       shuffle: False
     input_columns: ["input_ids", "labels"]

   # metric
   metric:
     type: EmF1Metric

   # model config
   model:
     model_config:
       type: LlamaConfig
       batch_size: 1
       seq_length: 2048
       max_decode_length: 700
       max_new_tokens: 20
   ```

3. 执行评测命令，指标为`Em/F1`

   ```shell
   python run_mindformer.py \
     --config configs/llama2/predict_llama2_7b.yaml \
     --eval_dataset_dir /{path}/squad2048.mindrecord \
     --run_mode eval \
     --load_checkpoint /{path}/llama2_7b.ckpt \
     --epochs 1 \
     --batch_size 1 \
     --use_parallel False \
     --device_id 0

   # F1 score: 60.5, Em score: 39.6, total_count: 2067
   ```

### 分布式评测

对于较大模型比如`llama2_70b`，模型无法完全导入到单卡中进行评测，就需要进行分布式评测。

以`llama2_70b`在**SQuAD 1.1**数据集上进行测评为例。

1. 切分模型权重

   可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md#离线转换案例一完整权重转换为分布式权重)中的推理案例三进行完整权重切分以用于分布式评测。

   修改权重文件夹目录结构如下，将模型权重放入`rank_0`的文件夹中。

   ```text
   path/to/checkpoint_dir
       ├──rank_0
       │  ├──model.ckpt
   ```

2. 修改模型配置文件

   ```yaml
   load_checkpoint: 'path/to/checkpoint_dir'
   auto_trans_ckpt: True
   use_parallel: True
   parallel_config:
     data_parallel: 1
     model_parallel: 8  # 修改为使用卡数， 70b推荐设置为8卡推理
     pipeline_stage: 1
     use_seq_parallel: False

   # metric
   metric:
     type: EmF1Metric

   eval_dataset:
     data_loader:
       type: MindDataset
       dataset_dir: "{path}/squad2048.mindrecord"
   ```

3. 执行评测命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/predict_llama2_70b.yaml \
     --run_mode eval \
     --use_parallel True" 8
   ```
