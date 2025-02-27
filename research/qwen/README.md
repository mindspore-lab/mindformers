# 通义千问

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

```text
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                             |      Task       | Datasets | SeqLength |  Phase   |                           Performance                           | DataType |
|:---------------------------------------------------|:---------------:|:--------:|:---------:|:--------:|:---------------------------------------------------------------:|:--------:|
| [Qwen-7B](./qwen_7b/finetune_qwen_7b.yaml)         | text_generation |  alpaca  |   8192    | Finetune |                         1571 tokens/s/p                         | float16  |
| [Qwen-7B](./qwen_7b/finetune_qwen_7b_bf16.yaml)    | text_generation |  alpaca  |   2048    | Finetune |                         2955 tokens/s/p                         | bfloat16 |
| [Qwen-14B](./qwen_14b/finetune_qwen_14b.yaml)      | text_generation |  alpaca  |   8192    | Finetune |                         911 tokens/s/p                          | float16  |
| [Qwen-14B](./qwen_14b/finetune_qwen_14b_bf16.yaml) | text_generation |  alpaca  |   2048    | Finetune |                         1106 tokens/s/p                         | bfloat16 |
| [Qwen-7B](./qwen_7b/predict_qwen_7b.yaml)          | text_generation |    -     |   2048    | Predict  | 23 tokens/s (bastch_size=1) <br/> 196 tokens/s (bastch_size=16) |    -     |
| [Qwen-14B](./qwen_14b/predict_qwen_14b.yaml)       | text_generation |    -     |   2048    | Predict  |                           35 tokens/s                           |    -     |

## 模型文件

`Qwen` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型实现：

   ```text
   research/qwen
     ├── qwen_config.py         # 模型配置
     ├── qwen_model.py          # 模型实例
     └── qwen_tokenizer.py      # 模型tokenizer
   ```

2. 模型配置：

   ```text
   research/qwen
     ├── qwen_7b                                    # qwen 7B 配置文件
     │   ├── finetune_qwen_7b.yaml                  # 7B 全参微调启动配置(8K)
     │   ├── finetune_qwen_7b_auto_parallel.yaml    # 7B 全参微调启动配置(自动并行)
     │   ├── finetune_qwen_7b_bf16.yaml             # 7B 全参微调启动配置(bf16, 2K)
     │   ├── finetune_qwen_7b_lora.yaml             # 7B lora微调启动配置
     │   └── predict_qwen_7b.yaml                   # 7B 推理启动配置
     └── qwen_14b                                   # qwen 14B 配置文件
         ├── finetune_qwen_14b.yaml                 # 14B 全参微调启动配置(8K)
         ├── finetune_qwen_14b_auto_parallel.yaml   # 14B 全参微调启动配置(自动并行)
         ├── finetune_qwen_14b_bf16.yaml            # 14B 全参微调启动配置(bf16, 2K)
         ├── finetune_qwen_14b_lora.yaml            # 14B lora微调启动配置
         └── predict_qwen_14b.yaml                  # 14B 推理启动配置
   ```

3. 模型相关脚本：

   ```text
   research/qwen
     ├── alpaca_converter.py      # alpaca数据集格式转换脚本
     ├── convert_weight.py        # hf->ms权重转换脚本
     ├── convert_reversed.py      # ms->hf权重转换脚本
     ├── qwen_chat.py             # Qwen Chat功能函数
     ├── qwen_preprocess.py       # 数据集预处理脚本
     └── run_qwen_chat.py         # Qwen Chat功能启动脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供 `alpaca` 作为[微调](#微调)数据集。

| 数据集名称  |          适用模型          |   适用阶段   |                                      下载链接                                       |
|:-------|:----------------------:|:--------:|:-------------------------------------------------------------------------------:|
| alpaca | Qwen-7B <br/> Qwen-14B | Finetune | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) |

数据预处理中所用的qwen.tiktoken可以参考[模型权重下载](#模型权重下载)进行下载。

- **alpaca 数据预处理**

  目前提供alpaca数据集的预处理脚本用于全参微调任务。执行 `research/qwen/alpaca_converter.py` ，将原始数据集转换为指定格式。

  ```shell
  python research/qwen/alpaca_converter.py \
   --data_path /path/alpaca_data.json \
   --output_path /path/alpaca-data-conversation.json
  # 参数说明
  data_path:   输入下载的文件路径
  output_path: 输出文件的保存路径
  ```

  执行 `research/qwen/qwen_preprocess.py` ，进行数据预处理和Mindrecord数据生成。

  ```shell
  python research/qwen/qwen_preprocess.py \
   --input_glob /path/alpaca-data-conversation.json \
   --model_file /path/qwen.tiktoken \
   --seq_length 8192 \
   --output_file /path/alpaca-8192.mindrecord
  ```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于预训练、微调和推理，用户也可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/Qwen/Qwen-7B/resolve/main/qwen.tiktoken)

| 模型名称          | MindSpore权重 |                        HuggingFace权重                        |
|:--------------|:-----------:|:-----------------------------------------------------------:|
| Qwen-7B-Base  |      -      |    [Link](https://huggingface.co/Qwen/Qwen-7B/tree/main)    |
| Qwen-7B-Chat  |      -      | [Link](https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main)  |
| Qwen-14B-Base |      -      |   [Link](https://huggingface.co/Qwen/Qwen-14B/tree/main)    |
| Qwen-14B-Chat |      -      | [Link](https://huggingface.co/Qwen/Qwen-14B-Chat/tree/main) |

#### 模型权重转换

首先，请安装官方Qwen模型所需的依赖软件包:

```shell
pip install torch==2.0.1 transformers==4.32.0 transformers_stream_generator einops accelerate tiktoken tokenizers==0.13.0
```

然后运行 [Mindformers 的权重转换工具](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/weight_conversion.html), 将huggingface的权重转换为 Mindspore 的ckpt格式。

以 `Qwen-7B` 为例，将下载完的safetensor权重存储到 `/path/models/qwen_7b` 文件夹中后，使用如下的转换脚本进行权重转换：

```shell
python convert_weight.py \
 --model qwen \
 --input_path /path/models/qwen_7b \
 --output_path /path/qwen_7b.ckpt
```

## 微调

### 全参微调

全参微调所用到的 `alpaca` 数据集和预训练权重可参考[数据集下载](#数据集下载)和[模型权重下载](#模型权重下载)获得。

#### 单机训练

以 `Qwen-7B` 单机8卡为例，即使用 `research/qwen/qwen_7b/finetune_qwen_7b.yaml` 配置文件。

执行分布式启动脚本，进行微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen \
 --config research/qwen/qwen_7b/finetune_qwen_7b.yaml \
 --load_checkpoint /path/qwen_7b.ckpt \
 --train_dataset /path/alpaca-8192.mindrecord \
 --use_parallel True \
 --auto_trans_ckpt True \
 --run_mode finetune" 8

# 参数说明
config:          配置文件路径
load_checkpoint: 权重路径或分布式权重文件夹路径, 若使用分布式权重则按照 `model_dir/rank_0/xxx.ckpt` 格式存放
train_dataset:   训练数据集路径
use_parallel:    是否开启分布式并行
auto_trans_ckpt: 是否开启自动权重转换
run_mode:        运行模式, 微调时设置为finetune
```

补充说明：

1. 训练的log日志路径： `./output/msrun_log` ；
2. checkpoint(含优化器参数)存储路径： `./output/checkpoint` ；
3. checkpoint(不含优化器参数)存储路径： `./output/checkpoint_network` ；
4. 若想合并ckpt用于后续评估，选择不含优化器参数的权重即可。

多卡微调后，如果想单卡运行推理或者评估，需要合并权重文件，可参考[分布式训练权重合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html)。

#### 多机训练

多机多卡训练可以参考[多机多卡启动方式](../../README_CN.md#多机多卡)。

### LoRA微调

LoRA微调所用到的 `alpaca` 数据集和预训练权重可参考[数据集下载](#数据集下载)和[模型权重下载](#模型权重下载)获得。

> LoRA微调配置文件中默认使用 `seq_length=1024` ，因此需要在数据处理时将数据 `seq_length` 设置为 `1024` 。

以 `Qwen-7B` 单机8卡为例，即使用 `research/qwen/qwen_7b/finetune_qwen_7b_lora.yaml` 配置文件。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen
 --config research/qwen/qwen_7b/finetune_qwen_7b_lora.yaml \
 --load_checkpoint /path/qwen_7b.ckpt \
 --train_dataset /path/alpaca-1024.mindrecord \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8

# 参数说明
config:          配置文件路径
load_checkpoint: 权重路径或分布式权重文件夹路径, 若使用分布式权重则按照 `model_dir/rank_0/xxx.ckpt` 格式存放
train_dataset:   训练数据集路径
use_parallel:    是否开启分布式并行
auto_trans_ckpt: 是否开启自动权重转换
run_mode:        运行模式, 微调时设置为finetune
```

## 推理

以 `Qwen-7B` 为例，即使用 `research/qwen/qwen_7b/predict_qwen_7b.yaml` 配置文件。

### 单卡推理

修改配置文件 `research/qwen/qwen_7b/predict_qwen_7b.yaml` ：

```yaml
processor:
  tokenizer:
    vocab_file: "/path/qwen.tiktoken"    # 指定tiktoken文件路径
```

执行如下推理命令。

```shell
python run_mindformer.py \
 --register_path research/qwen \
 --config research/qwen/qwen_7b/predict_qwen_7b.yaml \
 --load_checkpoint /path/qwen_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel False \
 --run_mode predict \
 --predict_data '比较适合深度学习入门的书籍有'
# 比较适合深度学习入门的书籍有：
# 1. 《Python深度学习》（Francois Chollet）：这本书是深度学习领域非常受欢迎的入门书籍，作者Francois Chollet是Keras库的创建者，...
```

### 多卡推理

以 `Qwen-7B` 2卡推理为例。

1. 修改配置文件 `research/qwen/qwen_7b/predict_qwen_7b.yaml` ：

   ```yaml
   processor:
    tokenizer:
      vocab_file: "/path/qwen.tiktoken"    # 指定tiktoken文件路径

   # parallel of device num = 2
   parallel_config:
     data_parallel: 1
     model_parallel: 2
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

2. 执行如下推理命令，进行2卡推理

   ```shell
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --register_path research/qwen
    --config research/qwen/qwen_7b/predict_qwen_7b.yaml \
    --load_checkpoint /path/qwen_7b.ckpt \
    --auto_trans_ckpt True \
    --use_parallel True \
    --run_mode predict \
    --predict_data 比较适合深度学习入门的书籍有" 2
   # 比较适合深度学习入门的书籍有：
   # 1. 《Python深度学习》（Francois Chollet）：这本书是深度学习领域非常受欢迎的入门书籍，作者Francois Chollet是Keras库的创建者，...
   ```

## 常见问题

### BF16 支持

1. 当前版本仅支持 `bf16` 数据类型的训练，暂不支持推理。

   `convert_weight.py` 脚本默认的数据类型已经改为与原始权重一致（对于通义千问而言，即 `bfloat16` ）;

   推理时可将YAML配置中的 `compute_dtype` 和 `param_init_type` 改为 `float16` ;

   如果打算基于 `bf16` 进行训练，建议加载 `bf16` 格式的权重，以减少数据类型转换带来的消耗和精度损失;

2. 权重转换完成之后，如果运行模型时出现部分第三方库版本不匹配问题，需要根据本项目[requirements.txt](../../requirements.txt)重新安装对应第三方库版本。

   `pip install -r requirements.txt`
