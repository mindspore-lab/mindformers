# 星辰语义大模型 Telechat

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

星辰语义大模型Telechat是由中电信人工智能科技有限公司研发训练的大语言模型，采用3万亿Tokens中英文高质量语料进行训练。目前开源模型：Telechat-7B，Telechat-12B, Telechat-52B模型，本仓库已支持7B、12B和52B模型的微调权重，权重文件来源于中电信人工智能科技有限公司。

基于GPU，Torch版本的Telechat链接：

[Telechat](https://github.com/Tele-AI/Telechat)

[TeleChat Technical Report](https://arxiv.org/abs/2401.03804)

``` text
@article{wang2024telechat,
      title={TeleChat Technical Report},
      author={Zihan Wang and Xinzhang Liu and Shixuan Liu and Yitong Yao and Yuyao Huang and Zhongjiang He and Xuelong Li and Yongxiang Li and Zhonghao Che and Zhaoxi Zhang and Yan Wang and Xin Wang and Luwen Pu and Huihan Xu and Ruiyu Fang and Yu Zhao and Jie Zhang and Xiaomeng Huang and Zhilong Lu and Jiaxin Peng and Wenjun Zheng and Shiquan Wang and Bingkai Yang and Xuewei he and Zhuoru Jiang and Qiyi Xie and Yanhan Zhang and Zhongqiu Li and Lingling Shi and Weiwei Fu and Yin Zhang and Zilu Huang and Sishi Xiong and Yuxiang Zhang and Chao Wang and Shuangyong Song},
      journal={arXiv preprint arXiv:2401.03804},
      year={2024}
}
```

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| config                                                | task                  | Datasets   | SeqLength | phase           | performance  |
|-------------------------------------------------------| --------------------- |------------|-----------|-----------------|--------------|
| [telechat_7b](./finetune_telechat_7b.yaml)       | text_generation       | example_dataset | 2048      | [finetune](#微调) | 3400 tks/s/p |
| [telechat_12b](./finetune_telechat_12b.yaml) | text_generation | example_dataset | 1024 | [finetune](#微调) | 1996 tks/s/p |
| [telechat_52b](./finetune_telechat_52b.yaml) | text_generation       | example_dataset     | 4096   | [finetune](#微调) | 364 tks/s/p |
| [telechat_7b](./predict_telechat_7b.yaml) | text_generation | / | / | [predict](#推理) | 67 tokens/s (单卡) |
| [telechat_12b](./predict_telechat_12b.yaml) | text_generation | / | / | [predict](#推理) | 40 tokens/s (单卡) |
| [telechat_52b](./predict_telechat_52b.yaml) | text_generation | / | / | [predict](#推理) | 32 tokens/s (四卡) |

## 仓库介绍

`Telechat` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/research/telechat`

   ```bash
   telechat
       ├── convert_weight_ms_to_torch.py         # ms->torch权重转换脚本
       ├── convert_weight_torch_to_ms.py         # torch->ms权重转换脚本
       ├── telechat_preprocess.py                # telechat模型的mindrecord数据处理脚本
       ├── telechat.py                           # 模型实现
       ├── telechat_config.py                    # 模型配置项
       ├── telechat_layer.py                     # telechat网络层定义
       ├── telechat_predict_utils.py             # telechat推理模块
       ├── telechat_tokenizer.py                 # telechat tokenizer
       └── telechat_transformer.py               # transformer层实现
   ```

2. 模型配置：`mindformers/research/telechat`

   ```bash
   telechat
       ├── finetune_telechat_7b.yaml   # 7b模型微调启动配置
       ├── finetune_telechat_12b.yaml  # 12b模型微调启动配置
       ├── finetune_telechat_52b.yaml  # 52b模型微调启动配置
       ├── predict_telechat_7b.yaml    # 7b模型推理启动配置
       ├── predict_telechat_12b.yaml   # 12b模型推理启动配置
       └── predict_telechat_52b.yaml   # 52b模型推理启动配置
   ```

3. 任务启动脚本：`mindformers/research/telechat`

   ```bash
   telechat
       ├── run_telechat_predict.py     # telechat推理脚本
       └── run_telechat.py             # telechat训练脚本
   ```

## 前期准备

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.4.0
- CANN: 8.0.rc3
- MindFormers版本：dev

注：Atlas 800T A2芯片：7b, 12b推理可在单机单卡上完成部署；52b推理可在单机四卡上完成部署。

### [mindformers安装](../../README_CN.md#二mindformers安装)

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1.torch模型权重及词模型下载链接：

- [telechat-7b](https://huggingface.co/Tele-AI/Telechat-7B/)
- [telechat-12b](https://huggingface.co/Tele-AI/TeleChat-12B)
- [telechat-52b](https://huggingface.co/Tele-AI/TeleChat-52B)

7b/12b模型权重转换，运行如下转换脚本，将全量微调的权重转换为完整的ckpt权重。

```bash
python mindformers/research/telechat/convert_weight_torch_to_ms.py \
--torch_path TORCH_CKPT_DIR \
--mindspore_path {path}/MS_CKPT_NAME \
--model_name 'telechat_7b'
```

52b模型权重转换，需要额外指定`mp`参数，该参数指定为8时，得到的完整权重适用于`mp=8`的权重自动转换。

```bash
python mindformers/research/telechat/convert_weight_torch_to_ms.py \
--torch_path TORCH_CKPT_DIR \
--mindspore_path {path}/MS_CKPT_NAME \
--model_name 'telechat_52b' \
--mp 8
```

```yaml
# 参数说明
torch_path: torch版本权重保存目录路径
mindspore_path: 权重保存文件名，可以指定自定义保存路径
model_name: 模型的名称
mp: 目标切分个数，比如指定为8时，得到的完整权重适用于mp=8的权重自动转换；
```

2.获取MindFormers提供的已转换权重，可直接从下面的链接获取。

- [telechat-7b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/mindspore.ckpt)
- [telechat-12b](https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/model_weight/mindspore_12B.ckpt)

### [分布式训练/微调权重合并](../../docs/feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)。

## 微调

### 数据集准备

中电信人工智能科技有限公司提供了[多轮对话微调数据集](https://gitee.com/Tele-AI/tele-chat/blob/master/example_datas/multi_turn_example.jsonl)样例，下载好微调数据集后，需要将json格式数据集转为训练支持的mindrecord格式。

- 7b模型微调数据集转换命令

```bash
python telechat_preprocess.py \
--input_dataset_file multi_turn_example.jsonl \
--output_path path/to/dataset_dir \
--vocab_file_path path/to/tokenizer.json \
--max_length 2048 \
--pad_token '<pad>'
```

- 12b模型微调数据集转换命令

```bash
python telechat_preprocess.py \
--input_dataset_file multi_turn_example.jsonl \
--output_path path/to/dataset_dir \
--vocab_file_path path/to/tokenizer.model \
--max_length 1024
```

- 52b模型微调数据集转换命令

```bash
python telechat_preprocess.py \
--input_dataset_file multi_turn_example.jsonl \
--output_path path/to/dataset_dir \
--vocab_file_path path/to/tokenizer.model \
--max_length 4096 \
--start_token '<reserve3>' \
--user_token '<reserve1>' \
--bot_token '<reserve2>' \
--pad_token '<pad>'
```

### 全参微调

当前模型已支持使用**Flash Attention算法**进行全参微调，默认开启flash_attention，可加速训练。详请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)。

通过`run_mindforemr.py`启动微调，多卡任务均使用[msrun启动方式](https://gitee.com/mindspore/mindformers#%E5%8D%95%E6%9C%BA%E5%A4%9A%E5%8D%A1)。

- 7b/12b模型支持单机8卡微调，权重为[模型权重下载与转换](#模型权重下载与转换)章节得到的完整权重，需要开启`auto_tans_ckpt`自动转为分布式权重，若直接加载8卡分布式权重，则无需开启。

```bash
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config ./research/telechat/finetune_telechat_7b.yaml \
--train_dataset path/to/dataset_dir \
--load_checkpoint path/to/ckpt_path \
--auto_trans_ckpt True \
--use_parallel True \
--register_path ./research/telechat"
```

- 52b模型支持双机16卡微调，权重为[模型权重下载与转换](#模型权重下载与转换)章节得到的完整权重，需要开启`auto_tans_ckpt`自动转为分布式权重，若直接加载16卡分布式权重，则无需开启。

```bash
# 1号机器
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config telechat/finetune_telechat_52b.yaml \
--train_dataset path/to/dataset_dir \
--load_checkpoint path/to/ckpt_path \
--auto_trans_ckpt True \
--use_parallel True \
--register_path ./research/telechat" 16 8 机器IP 8118 0 output/msrun_log False 300

# 2号机器
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config telechat/finetune_telechat_52b.yaml \
--train_dataset path/to/dataset_dir \
--load_checkpoint path/to/ckpt_path \
--auto_trans_ckpt True \
--use_parallel True \
--register_path ./research/telechat" 16 8 机器IP 8118 1 output/msrun_log False 300
```

```yaml
# 参数说明
config: 配置文件路径
train_dataset: 数据集路径
load_checkpoint: 权重路径
auto_tans_ckpt: 权重自动转换开关
use_parallel: 并行模式开关
register_path: 外部模型注册路径
```

## 推理

推理需要用户自定义`input.json`文件，格式如下：

```json
{"input": "生抽与老抽的区别？"}
```

telechat提供了基础多轮对话推理脚本`run_telechat_predict.py`，可直接加载完整权重，支持权重自动转换。

7b模型词表使用[模型权重下载与转换](#模型权重下载与转换)章节得到的`tokenizer.json`，12b和52b模型词表使用`tokenizer.model`。

- 7b模型支持**单卡推理**

在`predict_telechat_7b.yaml`中填写`tokenizer_file`字段

```yaml
processor:
  tokenizer:
    tokenizer_file: 'path/to/tokenizer.json'
```

运行`run_mindformer.py`启动推理

```bash
python run_mindformer.py \
--config ./research/telechat/predict_telechat_7b.yaml \
--load_checkpoint path/to/ckpt_path \
--predict_data "<_user>生抽与老抽的区别？<_bot>" \
--register_path ./research/telechat
```

- 12b模型支持**单卡推理**

在`predict_telechat_12b.yaml`中填写`vocab_file`字段

```yaml
processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

运行`run_mindformer.py`启动推理

```bash
python run_mindformer.py \
--config ./research/telechat/predict_telechat_12b.yaml \
--load_checkpoint path/to/ckpt_path \
--predict_data "<_user>生抽与老抽的区别？<_bot>" \
--register_path ./research/telechat
```

- 52b模型支持单机**4卡推理**

在`predict_telechat_52b.yaml`中填写`vocab_file`字段

```yaml
processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'
```

- 运行`run_mindformer.py`启动推理，需要开启`auto_trans_ckpt`将[模型权重下载与转换](#模型权重下载与转换)章节得到的完整权重转为分布式权重，若直接加载4卡分布式权重，则无需开启。

```bash
bash scripts/msrun_launcher.sh "python run_mindformer.py \
--config ./research/telechat/predict_telechat_52b.yaml \
--load_checkpoint path/to/ckpt_path \
--predict_data '<reserve3><reserve1>生抽与老抽的区别？<reserve2>' \
--auto_trans_ckpt True \
--register_path ./research/telechat" 4
```

> 注：52b模型进行4卡推理，使用的完整权重通过convert_weight_torch_to_ms.py转换时mp须为4。

```yaml
# 参数说明
input_file: 多轮对话文件路径
train_dataset: 数据集路径
load_checkpoint: 权重路径
auto_tans_ckpt: 权重自动转换开关
use_parallel: 并行模式开关
```

52b模型推理结果如下：

```text
生抽与老抽的区别？
答：生抽和老抽是两种不同的酱油，它们的主要区别在于颜色、味道和用途。
1.颜色：生抽的颜色较浅，呈红褐色;老抽的颜色较深，呈棕褐色。
2.味道：生抽的味道较咸，鲜味较浓;老抽的味道较甜，香味较浓。
3.用途：生抽主要用于调味，可以增加菜肴的鲜味和咸味;老抽主要用于上色，可以使菜肴的颜色更加鲜艳。
总的来说，生抽和老抽在颜色、味道和用途上都有所不同，可以根据个人口味和烹饪需求选择使用。
```
