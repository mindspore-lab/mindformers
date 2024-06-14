# Llama 2

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

llama2_7b:

| Config                                                     |         Task          | Datasets  | SeqLength | Metric |  Phase   |   Score   | Performance  |
|:-----------------------------------------------------------|:---------------------:|:---------:|:---------:|:------:|:--------:|:---------:|:------------:|
| [llama2_7b](../../configs/llama2/pretrain_llama2_7b.yaml)  |    text_generation    |   wiki    |   4096    |   -    | Pretrain |     -     | 4820 tks/s/p |
| [llama2_7b](../../configs/llama2/finetune_llama2_7b.yaml)  |    text_generation    |  alpaca   |   4096    |   -    | Finetune |     -     | 4820 tks/s/p |
| [llama2_7b_lora](../../configs/llama2/lora_llama2_7b.yaml) |    text_generation    |  alpaca   |   4096    |   -    | Finetune |     -     | 5217 tks/s/p |
| [llama2_7b](../../configs/llama2/predict_llama2_7b.yaml)   |    text_generation    | WikiText2 |     -     |  PPL   |   Eval   |   6.58    |      -       |
| [llama2_7b](../../configs/llama2/predict_llama2_7b.yaml)   | reading comprehension | SQuAD 1.1 |     -     | EM/F1  |   Eval   | 39.6/60.5 |      -       |

llama2_13b:

| Config                                                        |         Task          | Datasets  | SeqLength | Metric |  Phase   |    Score    | Performance  |
|:--------------------------------------------------------------|:---------------------:|:---------:|:---------:|:------:|:--------:|:-----------:|:------------:|
| [llama2_13b](../../configs/llama2/pretrain_llama2_13b.yaml)   |    text_generation    |   wiki    |   4096    |   -    | Pretrain |      -      | 1883 tks/s/p |
| [llama2_13b](../../configs/llama2/finetune_llama2_13b.yaml)   |    text_generation    |  alpaca   |   4096    |   -    | Finetune |      -      | 1883 tks/s/p |
| [llama2_13b_lora](../../configs/llama2/lora_llama2_13b.yaml)  |    text_generation    |  alpaca   |   4096    |   -    | Finetune |      -      | 2322 tks/s/p |
| [llama2_13b](../../configs/llama2/predict_llama2_13b.yaml)    |    text_generation    | WikiText2 |     -     |  PPL   |   Eval   |    6.14     |      -       |
| [llama2_13b](../../configs/llama2/predict_llama2_13b.yaml)    | reading comprehension | SQuAD 1.1 |     -     | EM/F1  |   Eval   | 27.91/44.23 |      -       |

llama2_70b：

| Config                                                      |         Task          |  Datasets   |  SeqLength  |  Metric  |   Phase    |     Score     | Performance |
|:------------------------------------------------------------|:---------------------:|:-----------:|:-----------:|:--------:|:----------:|:-------------:|:-----------:|
| [llama2_70b](../../configs/llama2/pretrain_llama2_70b.yaml) |    text_generation    |    wiki     |    4096     |    -     |  Pretrain  |       -       | 407 tks/s/p |
| [llama2_70b](../../configs/llama2/finetune_llama2_70b.yaml) |    text_generation    |   alpaca    |    4096     |    -     |  Finetune  |       -       | 414 tks/s/p |
| [llama2_70b](../../configs/llama2/predict_llama2_70b.yaml)  |    text_generation    |  WikiText2  |      -      |   PPL    |    Eval    |     4.92      |      -      |
| [llama2_70b](../../configs/llama2/predict_llama2_70b.yaml)  | reading comprehension |  SQuAD 1.1  |      -      |  EM/F1   |    Eval    |  41.94/63.86  |      -      |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                      |      Task       | Datasets | SeqLength |  Phase   | Performance  |
|:------------------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:------------:|
| [llama2_7b](../../configs/llama2/pretrain_llama2_7b.yaml)   | text_generation |   wiki   |   4096    | Pretrain | 4100 tks/s/p |
| [llama2_13b](../../configs/llama2/pretrain_llama2_13b.yaml) | text_generation |   wiki   |   4096    | Pretrain | 1658 tks/s/p |
| [llama2_70b](../../configs/llama2/pretrain_llama2_70b.yaml) | text_generation |   wiki   |   4096    | Pretrain | 406 tks/s/p  |

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

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：Atlas 800T A2芯片支持7b,13b单机单卡推理，70b推理至少使用8卡，全参微调至少需要4机32卡，推荐使用8机64卡。

### 数据及权重准备

#### 数据集下载

MindFormers提供**Wikitext2**作为[预训练](#预训练)数据集，**alpaca**作为[微调](#微调)数据集。

| 数据集名称     |                    适用模型                     |          适用阶段           |                                                         下载链接                                                          |
|:----------|:-------------------------------------------:|:-----------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2 | llama2-7b <br/> llama2-13b <br/> llama2-70b | Pretrain <br/> Evaluate | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip) |
| alpaca    | llama2-7b <br/> llama2-13b <br/> llama2-70b |        Finetune         |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |
| SQuAD 1.1 | llama2-7b <br/> llama2-13b <br/> llama2-70b |        Evaluate         |                                     [Link](https://data.deepai.org/squad1.1.zip)                                      |

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

  > 注：`bos`, `eos`, `pad`等特殊`ids`要和`yaml`配置文件中`model_config`部分保持一致，默认`bos_token_id=1`, `eos_token_id=2`, `pad_token_id=0`。
如果有所修改，配置文件中对应设置也需要修改，通常预训练数据不包含`pad_token`，因此建议设置`pad_token_id=-1`。

- **alpaca 数据预处理**

  1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

     ```shell
     python alpaca_converter.py \
       --data_path /{path}/alpaca_data.json \
       --output_path /{path}/alpaca-data-conversation.json

     # 参数说明
     data_path:   输入下载的文件路径
     output_path: 输出文件的保存路径
     ```

  2. 执行`mindformers/tools/dataset_preprocess/llama/llama_preprocess.py`，生成Mindrecord数据，将带有prompt模板的数据转换为mindrecord格式。

     ```shell
     # 此工具依赖fschat工具包解析prompt模板, 请提前安装fschat >= 0.2.13 python = 3.9
     python llama_preprocess.py \
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

  执行`tools/dataset_preprocess/llama/squad_data_process.py`生成Mindrecord数据

  ```shell
  python squad_data_process.py \
    --input_file /{path}/squad/dev-v1.1.json \
    --output_file /{path}/squad2048.mindrecord \
    --mode eval \
    --max_length 2048 \
    --tokenizer_type "llama2_7b"
  ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

| 模型名称            |                                                 MindSpore权重                                                  |                      HuggingFace权重                       |
|:----------------|:------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| llama2-7b       |    [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)    | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| llama2-13b      | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2-13b-fp16.ckpt) | [Link](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| llama2-70b      |                                                      /                                                       | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) |
| tokenizer.model |   [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)    |                            /                             |

> 注：Llama2的所有权重都需要通过向Meta[提交申请](https://ai.meta.com/resources/models-and-libraries/llama-downloads)来获取，如有需要请自行申请。

#### 模型权重转换

执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/llama2`

```python
import mindspore
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('llama2_7b')

# model的实例化有以下两种方式，选择其中一种进行实例化即可
# 1. 直接根据默认配置实例化
model = AutoModel.from_pretrained('llama2_7b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('llama2_7b')
# config.xxx = xxx                      # 根据需求自定义修改其余模型配置
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

inputs = tokenizer("I love Beijing, because")["input_ids"]
# 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
outputs = model.generate(inputs, max_new_tokens=30, do_sample=False)
response = tokenizer.decode(outputs)
print(response)
# ['<s>I love Beijing, because it’s a city that is constantly changing. I have been living here for 10 years and I have seen the city change so much.I']
```

### 基于Trainer的快速推理

> 注：下面仅显示接口使用方式，模型启动训练需求多卡分布式训练，训练脚本需配合分布式脚本启动

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='llama2_7b',
                  train_dataset='path/to/train_dataset')

# 开启推理
predict_result = trainer.predict(input_data="I love Beijing, because")
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='llama2_7b', max_length=30)
pipeline_result = pipeline_task("I love Beijing, because", do_sample=False)
print(pipeline_result)
# [{'text_generation_text': ['<s>I love Beijing, because it’s a a city that is constantly changing. I have been living here for 10 years and I have']}]
```

## 预训练

MindFormers提供`llama2-7b`单机多卡以及`llama2_13b`多机多卡的预训练示例，
过程中使用**Wikitext2**数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 单机训练

以Llama2-7b为例。

1. 修改模型配置文件`configs/llama2/pretrain_llama2_7b.yaml`

   ```yaml
   train_dataset:
     data_loader:
       dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"  # 预训练数据集的文件路径

   model:
     model_config:
       use_flash_attention: True                              # 可加速训练
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

   `ymal`配置文件中各参数含义详见[Config配置说明](../../configs/README.md)，
   `parallel_speed_up`各参数含义详见[parallel_speed_up说明](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore/mindspore.set_context.html#mindspore.set_context)。

2. 执行msrun启动脚本，进行8卡分布式训练

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/pretrain_llama2_7b.yaml \
     --run_mode train" 8
   ```

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
     --run_mode {train}" \
     16 8 {ip_addr} 8118 0 output/msrun_log False 300

   # 节点1，{ip_addr}处填写节点0实际ip，节点0与节点1启动命令仅参数NODE_RANK不同
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config {CONFIG_PATH} \
     --run_mode {train}" \
     16 8 {ip_addr} 8118 1 output/msrun_log False 300
   ```

## 微调

MindFormers提供`Llama2-7b`的微调示例，
过程中使用**alpaca**数据集对模型进行预训练，数据集可以参考[数据集下载](#数据集下载)获得。

### 全参微调

#### 单机训练

1. 修改模型配置文件`config/llama2/finetune_llama2_7b.yaml`

   ```yaml
   load_checkpoint: '{path}/llama2_7b.ckpt'

   train_dataset:
     data_loader:
       dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"
     input_columns: ["input_ids", "labels"]

   # optimizer
   optimizer:
     type: FP32StateAdamWeightDecay
     beta1: 0.9
     beta2: 0.999
     eps: 1.e-8
     learning_rate: 1.e-5

   # lr sechdule
   lr_schedule:
     type: CosineWithWarmUpLR
     learning_rate: 1.e-5
     lr_end: 0
     warmup_ratio: 0.03
     total_steps: -1  # -1 means it will load the total steps of the dataset

   # model config
   model:
     model_config:
       type: LlamaConfig
       seq_length: 4096
       use_flash_attention: True

   # context
   context:
     runtime_num_threads: 1
   ```

2. 执行msrun启动脚本，进行8卡分布式微调

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/finetune_llama2_7b.yaml \
     --run_mode finetune" 8
   ```

#### 多机训练

多机多卡微调任务启动参考[预训练章节](#预训练)，添加预训练权重，修改启动脚本中的`RUN_MODE`为`finetune`即可。

### LoRA微调

使用LoRA低参微调算法，冻结原模型权重，仅在小规模参数量上进行训练，使大模型在少量资源的情况下也能训练。

MindFormers提供Llama2-7b的LoRA微调示例，微调过程中使用的数据集可以参考[数据集下载](#数据集下载)获得。

1. 修改模型配置文件`configs/llama2/lora_llama2_7b.yaml`

   ```yaml
   train_dataset:
     data_loader:
       dataset_dir: "/{path}/alpaca-fastchat4096.mindrecord"  # 预训练数据集的文件路径

   model:
     model_config:
       use_flash_attention: True                              # 可加速训练
   ```

   如果加载完整权重，进行如下修改：

   ```yaml
   load_checkpoint: {path}/llama2_7b.ckpt
   auto_trans_ckpt: False
   ```

   如果加载分布式权重，加载权重路径需要设置为rank_0的上一层路径：

   ```yaml
   load_checkpoint: {path}/rank_0/
   anto_trans_ckpt: True
   ```

2. 执行msrun启动脚本，进行8卡分布式微调

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/lora_llama2_7b.yaml \
     --run_mode finetune" 8
   ```

### PrefixTuning微调

使用PrefixTuning低参微调算法，冻结原模型权重，仅在kv向量前添加可训练前缀向量进行训练，使大模型在少量资源的情况下也能训练。

MindFormers提供Llama2-7b的PrefixTuning微调示例，微调过程中使用的数据集可以参考[数据集下载](#数据集下载)获得。

1. 修改模型配置文件`configs/llama2/finetune_llama2_7b_prefixtuning.yaml`

   ```yaml
   train_dataset:
     data_loader:
       dataset_dir: "/{path}/alpaca-fastchat512.mindrecord"  # 预训练数据集的文件路径

   model:
     model_config:
       use_flash_attention: True                              # 可加速训练
       ...
       pet_config:
       pet_type: prefixtuning
       prefix_token_num: 16 # depend on dataset scale
       mid_dim: 512
       dropout_rate: 0.01
   ```

   如果加载完整权重，进行如下修改：

   ```yaml
   load_checkpoint: {path}/llama2_7b.ckpt
   auto_trans_ckpt: False
   ```

   如果加载分布式权重，加载权重路径需要设置为rank_0的上一层路径：

   ```yaml
   load_checkpoint: {path}/rank_0/
   anto_trans_ckpt: True
   ```

2. 执行msrun启动脚本，进行8卡分布式微调

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/finetune_llama2_7b_prefixtuning.yaml \
     --run_mode finetune" 8
   ```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

涉及到模型权重的单卡或多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)。

1. 获取模型切分策略文件：

   在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

   ```shell
   python transform_ckpt.py \
     --src_ckpt_strategy {path}/output/strategy/ \
     --src_ckpt_dir {path}/output/checkpoint/ \
     --dst_ckpt_dir {path}/target_checkpoint/ \
     --prefix llama2_7b

   # 参数说明
   src_ckpt_strategy: 切分策略文件路径
   src_ckpt_dir:      原切分权重文件夹
   dst_ckpt_dir:      目标路径
   prefix:            ckpt文件前缀
   ```

   > 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以通过mindspore 2.0的cpu版本以执行该脚本。

## 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造了全新的训推一体高性能推理引擎，保证训练与推理使用同一套脚本，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　MindSpore 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。在Atlas 800T A2硬件环境下推理，可加入如下配置。

```yaml
context:
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
```

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

```python
# run_llama2_predict.py
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # multi batch inputs
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # init environment
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = len(inputs)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path # 如果本地已有ckpt，可加绝对路径：/path/to/model.ckpt
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_type) # 如果本地已有tokenizer.model，可加绝对路径：/path/to/tokenizer_directory/
    # build model from config
    model = LlamaForCausalLM(model_config)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        input_ids = ms.Tensor(np.ones(shape=(model_config.batch_size, model_config.seq_length)), ms.int32)
        if model_config.use_past:
            infer_data = model.prepare_inputs_for_predict_layout(input_ids)
            warm_up_model.infer_predict_layout(*infer_data)
        else:
            warm_up_model.infer_predict_layout(input_ids)
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids,
                             max_length=model_config.max_decode_length,
                             do_sample=model_config.do_sample,
                             top_k=model_config.top_k,
                             top_p=model_config.top_p)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 多batch输出
# <s>I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# <s>LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model....
# <s>Huawei is a company that has been around for a long time. ...
```

#### 单卡推理

1. 修改模型配置文件，并创建上述推理脚本`run_llama2_predict.py`

   ```yaml
   use_parallel: False
   ```

2. 执行推理命令

   ```shell
   python run_llama2_predict.py --yaml_file path/to/predict_llama2_7b.yaml --checkpoint_path path/to/checkpoint.ckpt --model_type llama2_7b

   # 参数说明
   yaml_file:       配置文件路径
   checkpoint_path: 加载权重路径
   model_type:      模型类型
   ```

#### 多卡推理

以2卡推理为例。

1. 修改模型配置文件，并创建上述推理脚本`run_llama2_predict.py`

   ```yaml
   use_parallel: True

   parallel_config:
     data_parallel: 1
     model_parallel: 2  # 修改为使用卡数
     pipeline_stage: 1
     use_seq_parallel: False
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   micro_batch_interleave_num: 1
   ```

2. 切分权重

   切分权重可以参考[权重切分与合并](../feature_cards/Transform_Ckpt.md#离线转换案例一完整权重转换为分布式权重)中的推理案例三，使用自动转换权重得到的分布式权重在`output/transformed_checkpoint`文件夹中。

3. 执行推理命令

   ```shell
   bash scripts/msrun_launcher.sh "run_llama2_predict.py \
     --yaml_file path/to/predict_llama2_7b.yaml \
     --checkpoint_path path/to/shard_checkpoint_dir \
     --model_type llama2_7b" 2
   ```

   > 注：多卡推理时, 使用的checkpoint必须是经过切分的, shard_checkpoint_dir文件夹中包含rank_{}文件夹。

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡推理。

```python
# run_llama2_predict.py
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint


def main(args):
    """main function."""
    # 多输入
    inputs = ["I love Beijing, because",
              "LLaMA is a",
              "Huawei is a company that"]

    # set model config
    config = MindFormerConfig(args.yaml_file)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.use_past = args.use_past
    model_config.seq_length = args.seq_length
    if args.checkpoint_path and not config.use_parallel:
        model_config.checkpoint_name_or_path = args.checkpoint_path # 如果本地已有ckpt，可加绝对路径：/path/to/model.ckpt
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_type) # 如果本地已有tokenizer.model，可加绝对路径：/path/to/tokenizer_directory/

    model = LlamaForCausalLM(model_config)
    model.set_train(False)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(model)
        input_ids = ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32)
        if model_config.use_past:
            infer_data = model.prepare_inputs_for_predict_layout(input_ids)
            warm_up_model.infer_predict_layout(*infer_data)
        else:
            warm_up_model.infer_predict_layout(input_ids)
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs,
                                       max_length=model_config.max_decode_length,
                                       do_sample=model_config.do_sample,
                                       top_k=model_config.top_k,
                                       top_p=model_config.top_p)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str,
                        help='which model to use.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--seq_length', default=512, type=int,
                        help='predict max length')
    args = parser.parse_args()
    main(args)

# 推理输出
# 'text_generation_text':['I love Beijing,because it is a city that is constantly changing. I have been living here for 10 years ...
# 'text_generation_text':['LlaMa is a large-scale, open-source, multimodal, multilingual, multitask, and multimodal pretrained language model....
# 'text_generation_text':['Huawei is a company that has been around for a long time. ...
```

#### 单卡推理

推理过程与[**基于generate推理的单卡推理**](#单卡推理)一致，只需要修改`run_llama2_predict.py`内容和配置文件之后运行即可。

#### 多卡推理

推理过程与[**基于generate推理的多卡推理**](#多卡推理)一致，只需要修改`run_llama2_predict.py`内容和配置文件之后运行即可。

### 基于run_mindformer脚本的推理

使用`run_mindformer.py`进行推理，在`tokenizer`配置下添加`vocab_file`及其`tokenizer.model`的路径，`tokenizer.model`可通过[模型权重下载](#模型权重下载)得到。

#### 单卡推理

1. 修改模型配置文件`configs/llama2/predict_llama2_7b.yaml`

   ```yaml
   processor:
     return_tensors: ms
     tokenizer:
       vocab_file: "path/to/tokenizer.model"  # 增加tokenizer.model文件路径
   ```

2. 执行推理命令

   ```shell
   python run_mindformer.py --config configs/llama2/predict_llama2_7b.yaml --run_mode predict --predict_data 'I love Beijing, because' --use_parallel False

   # 推理输出
   # I love Beijing, because it is a city that is constantly changing. I have been living here for 10 years and I...
   ```

#### 多卡推理

以2卡推理为例

1. 模型权重切分

   可参考[权重切分与合并](../feature_cards/Transform_Ckpt.md#离线转换案例一完整权重转换为分布式权重)中的推理案例三进行完整权重切分以用于多卡推理。

2. 修改模型配置文件`configs/llama2/predict_llama2_7b.yaml`

   ```yaml
   use_parallel: True
   parallel_config:
     data_parallel: 1
     model_parallel: 2  # 修改为使用卡数
     pipeline_stage: 1
     use_seq_parallel: False
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   micro_batch_interleave_num: 1

   processor:
     return_tensors: ms
     tokenizer:
       vocab_file: "path/to/tokenizer.model"  # 增加tokenizer.model文件路径
   ```

3. 执行推理命令

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
     --config configs/llama2/predict_llama2_7b.yaml \
     --run_mode predict \
     --use_parallel True \
     --predict_data \"I love Beijing, because\"" 2
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

2. 修改模型配置文件`configs/llama2/pretrain_llama2_7b.yaml`

   ```yaml
   metric:
     type: PerplexityMetric

   model:
     model_config:
       use_flash_attention: True
   ```

3. 执行评测命令，指标为PPL

   ```shell
   python run_mindformer.py \
     --config configs/llama2/pretrain_llama2_7b.yaml \
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

2. 修改模型配置文件`configs/llama2/pretrain_llama2_7b.yaml`

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
