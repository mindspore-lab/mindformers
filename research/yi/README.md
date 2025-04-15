# Yi大模型

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 之后的版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

Yi系列是由零一万物研究的大规模语言预训练模型，目前开源的有Yi-6B/34B-Base/Chat，Yi-VL-6B/34B，MindFormers已支持Yi-6B-Base,Yi-34B-Base/Chat。当前训练使用Base权重，推理使用Base/Chat权重

[Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652v1)

``` text
@article{ai2024yiopenfoundationmodels,
      title={Yi: Open Foundation Models by 01.AI},
      author={01. AI and : and Alex Young and Bei Chen and Chao Li and Chengen Huang and Ge Zhang and Guanwei Zhang and Heng Li and Jiangcheng Zhu and Jianqun Chen and Jing Chang and Kaidong Yu and Peng Liu and Qiang Liu and Shawn Yue and Senbin Yang and Shiming Yang and Tao Yu and Wen Xie and Wenhao Huang and Xiaohui Hu and Xiaoyi Ren and Xinyao Niu and Pengcheng Nie and Yuchi Xu and Yudong Liu and Yue Wang and Yuxuan Cai and Zhenyu Gu and Zhiyuan Liu and Zonghong Dai},
      year={2024},
      eprint={2403.04652},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.04652},
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                               |      Task       |      Datasets       | SeqLength |   Performance   |  Phase   |
|:-------------------------------------|:---------------:|:-------------------:|:---------:|:---------------:|:--------:|
| [yi_6b](yi_6b/finetune_yi_6b.yaml)       | text_generation | alpaca_gpt4_data_zh |   2048    | 3324 tokens/s/p | Finetune |
| [yi_34b](yi_34b/finetune_yi_34b.yaml)     | text_generation |       alpaca        |   4096    | 660 tokens/s/p  | Finetune |
| [yi_6b](yi_6b/predict_yi_6b.yaml)        | text_generation |          -          |    512    |   31 tokens/s   | Predict  |
| [yi_34b](yi_34b/predict_yi_34b_chat.yaml) | text_generation |          -          |   16384   |   41 tokens/s   | Predict  |

## 模型文件

1. 模型实现：

   ```text
    research/yi
     └── yi_model.py           # 模型实例
   ```

2. 模型配置：

   ```text
    research/yi
     ├── yi_6b
     |    ├── finetune_yi_6b.yaml                   # 6B 全参微调启动配置
     |    └── predict_yi_6b.yaml                    # 6B base在线推理启动配置  
     └── yi_34b
          ├── pretrain_yi_34b.yaml                  # 34B 预训练启动配置
          ├── finetune_yi_34b.yaml                  # 34B 全参微调启动配置
          ├── predict_yi_34b.yaml                   # 34B base在线推理启动配置
          └── predict_yi_34b_chat.yaml              # 34B chat在线推理启动配置
   ```

3. 模型相关脚本：

   ```text
    research/yi
     ├── alpaca_converter.py           # alpaca数据集格式转换脚本
     ├── conversation.py               # 微调数据集处理，将原始alpaca转换为对话形式alpaca
     ├── yi_preprocess.py              # 数据集预处理脚本
     ├── convert_ckpt_bf16.py          # 权重转换脚本
     ├── convert_reversed.py           # 权重转换脚本
     └── convert_weight.py             # 权重转换脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

> 注：Atlas 800T A2芯片支持6b单卡推理，全参微调至少需要4卡，建议8卡；34b推理需要4卡，全参微调需要双机32卡。

### 数据及权重准备

#### 数据集下载

| 数据集名称               |        适用模型        |   适用阶段   |                                                         下载链接                                                          |
|:--------------------|:------------------:|:--------:|:---------------------------------------------------------------------------------------------------------------------:|
| Wikitext2           |       yi-34b       | Pretrain | [Link](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/faq/func_related.html) |
| alpaca              | yi-6b <br/> yi-34b | Finetune |                    [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)                    |
| alpaca_gpt4_data_zh | yi-6b <br/> yi-34b | Finetune |       [Link](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh/resolve/main/alpaca_gpt4_data_zh.json?download=true)       |

数据集处理过程中使用的`tokenizer.model`可以通过[链接](https://huggingface.co/01-ai/Yi-6B/blob/main/tokenizer.model)下载。

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

- **alpaca_gpt4_data_zh 数据预处理**

  1. 执行`research/yi/alpaca_converter.py`，将原始数据集转换为对话格式。

     ```shell
     python research/yi/alpaca_converter.py \
      --data_path /{path}/alpaca_gpt4_data_zh.json \
      --output_path /{path}/alpaca_gpt4_data_zh-conversation.json

     # 参数说明
     data_path:   输入下载的数据集路径
     output_path: 输出转换后数据集保存路径
     ```

  2. 执行`research/yi/yi_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。执行此脚本需要添加PYTHONPATH指定到research/yi目录。

     ```shell
     python research/yi/yi_preprocess.py \
      --dataset_type qa \
      --input_glob /{path}/alpaca_gpt4_data_zh-conversation.json \
      --model_file /{path}/tokenizer.model \
      --seq_length 2048 \
      --output_file /{path}/alpaca_gpt4_data_zh.mindrecord

     # 参数说明
     input_file_path: 输入数据集文件路径
     output_file:     输出文件的保存路径
     dataset_type:    数据集类型, 目前仅支持'text'和'qa'
     model_file:      模型词表文件路径
     seq_length:      数据序列长度
     ```

#### 模型权重下载

MindFormers提供下载HuggingFace官方权重的下载链接，用户可通过链接下载权重并经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/01-ai/Yi-6B/blob/main/tokenizer.model)

| 模型名称        | MindSpore权重 |                  HuggingFace权重                   |
|:------------|:-----------:|:------------------------------------------------:|
| Yi-6B-Base  |      -      |    [Link](https://huggingface.co/01-ai/Yi-6B)    |
| Yi-34B-Base |      -      |   [Link](https://huggingface.co/01-ai/Yi-34B)    |
| Yi-34B-Chat |      -      | [Link](https://huggingface.co/01-ai/Yi-34B-Chat) |

#### 模型权重转换

执行`mindformers/convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model yi --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 预训练

MindFormers提供`Yi-34b`多机多卡预训练示例，目前`Yi-34b`模型不支持进行单机预训练任务，预训练数据集可通过[数据集下载](#数据集下载)获得。

多机多卡拉起任务需要多机同时执行命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)。

以下为`Yi-34b`2机16卡执行命令：

```shell
# 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_34b/pretrain_yi_34b.yaml \
 --use_parallel True \
 --run_mode train \
 --auto_trans_ckpt False \
 --train_dataset /{path}/wiki4096.mindrecord" \
 16 8 {ip_addr} 8118 0 output/msrun_log False 300

# 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_34b/pretrain_yi_34b.yaml \
 --use_parallel True \
 --run_mode train \
 --auto_trans_ckpt False \
 --train_dataset /{path}/wiki4096.mindrecord" \
 16 8 {ip_addr} 8118 1 output/msrun_log False 300

# 参数说明
register_path:   注册路径
config:          配置文件路径
use_parallel:    是否开启并行训练
run_mode:        运行模式, 预训练时设置为train
auto_trans_ckpt: 是否开启自动权重转换
train_dataset:   训练数据集路径
```

## 微调

### 全参微调

MindFormers提供`Yi-6b`单机微调以及`Yi-34b`多机微调示例，目前`Yi-34b`模型不支持进行单机微调任务，微调数据集可通过[数据集下载](#数据集下载)获得。

#### 单机训练

以`Yi-6b`全参微调为例，使用配置文件`research/yi/yi_6b/finetune_yi_6b.yaml`，执行如下命令拉起单机8卡微调任务。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_6b/finetune_yi_6b.yaml \
 --run_mode finetune \
 --load_checkpoint /{path}/yi_6b.ckpt \
 --train_dataset /{path}/alpaca_gpt4_data_zh.mindrecord \
 --auto_trans_ckpt False \
 --use_parallel True" 8


# 参数说明
register_path:      注册路径
config:             配置文件路径
run_mode:           运行模式, 微调时设置为finetune
load_checkpoint:    预训练权重路径
train_dataset:      训练数据集路径
auto_trans_ckpt:    是否开启自动权重转换
use_parallel:       是否开启并行训练
```

#### 多机训练

以`Yi-34b`全参微调为例，使用配置文件`research/yi/yi_34b/finetune_yi_34b.yaml`，执行如下命令拉起2机16卡微调任务。

多机多卡拉起任务需要多机同时执行命令，将参数`MASTER_ADDR`设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数`NODE_RANK`不同，具体可参考[使用指南](../../README_CN.md#三使用指南)。

```shell
# 节点0，节点ip为{ip_addr}，作为主节点，总共16卡且每个节点8卡
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_34b/finetune_yi_34b.yaml \
 --load_checkpoint /path/ckpt_dir \
 --use_parallel True \
 --run_mode finetune \
 --auto_trans_ckpt False \
 --train_dataset /{path}/alpaca.mindrecord" \
 16 8 {ip_addr} 8118 0 output/msrun_log False 300

# 节点1，节点0与节点1启动命令仅参数NODE_RANK不同
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_34b/finetune_yi_34b.yaml \
 --load_checkpoint /path/ckpt_dir \
 --use_parallel True \
 --run_mode finetune \
 --auto_trans_ckpt False \
 --train_dataset /{path}/alpaca.mindrecord" \
 16 8 {ip_addr} 8118 1 output/msrun_log False 300

# 参数说明
register_path:      注册路径
config:             配置文件路径
load_checkpoint:    权重文件夹路径, 权重按照'model_dir/rank_0/xxx.ckpt'格式存放
auto_trans_ckpt:    自动权重转换开关
run_mode:           运行模式, 微调时设置为finetune
train_dataset:      训练数据集路径
```

### 分布式训练权重合并

分布式训练（微调）后所得到的权重文件为根据策略切分后的权重，可以手动将切分权重合一，以用于评估和推理。

MindFormers提供自动权重转换和离线权重转换功能，可参考[自动转换案例](../../../../docs/feature_cards/Transform_Ckpt.md#自动转换案例)和[离线权重转换](../../../../docs/feature_cards/Transform_Ckpt.md#离线权重转换)进行分布式模型权重转换。

## 推理

`Yi-6b-Base`支持单卡推理，`Yi-34b`模型规模较大，仅支持多卡推理。

### 单卡推理

以`Yi-6b`单卡推理为例，执行如下命令进行推理。

1. 修改模型配置文件 `research/yi/yi_6b/predict_yi_6b.yaml`

```yaml
 processor:
     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
```

2. 启动推理脚本

```shell
python run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_6b/predict_yi_6b.yaml \
 --run_mode predict \
 --load_checkpoint /path/ckpt_dir \
 --predict_data '以雷霆之力'

# 推理结果
# 以雷霆之力，将这股力量化为一道道剑气。“噗！”一柄长枪被斩断成两截后，...
```

### 多卡推理

以`Yi-34b-Chat`4卡推理为例，执行如下命令进行推理。

1. 修改模型配置文件 `research/yi/yi_34b/predict_yi_34b_chat.yaml`

```yaml
processor:
  tokenizer:
    vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
```

2. 启动推理脚本

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/yi \
 --config research/yi/yi_34b/predict_yi_34b_chat.yaml \
 --load_checkpoint /path/ckpt_dir \
 --predict_data '以雷霆之力' \
 --use_parallel True \
 --run_mode predict \
 --auto_trans_ckpt True" 4

# 推理结果
# 以雷霆之力，将这股力量化为一道道剑气。“噗！”一柄长枪被斩断成两截后，...
```
