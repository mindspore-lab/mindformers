# Llava1.5

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 之后的版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

LLaVA 1.5是一个端到端训练的大型多模态模型，连接视觉编码器和大语言模型，以实现通用视觉和语言理解，通过在 GPT 生成的多模式指令跟踪数据上微调 LLaMA/Vicuna 进行训练。它是一种基于 Transformer 架构的自回归语言模型。

```text
@inproceedings{liu2023llava,
    author      = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    title       = {Visual Instruction Tuning},
    booktitle   = {NeurIPS},
    year        = {2023}
  }
```

## 模型文件

`Llava1.5` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/llava/
       ├── llava_model.py            # 模型实现
       └── llava_config.py           # 模型配置
   ```

2. 模型配置：

   ```text
   research/llava/
       └── llava1_5_7B
                └── predict_llava1_5_7b.yaml     # 7B推理配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

|     模型     |     硬件      | 推理 |
| :----------: | :-----------: | :--: |
| Llava-1.5-7b | Atlas 800T A2 | 单卡 |

### 权重准备

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/tokenizer.model)

| 模型名称    | MindSpore权重 |                       HuggingFace权重                        |
| :---------- | :-----------: | :----------------------------------------------------------: |
| Llava1.5-7B |       \       | [Link](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llava --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
```

## 推理

进行推理前，模型权重以及tokenizer文件可参考[模型权重下载](#模型权重下载)进行准备，并修改`predict_llava1_5_7b.yaml`中相关配置，补充词表路径。

   ```yaml
   processor:
     tokenizer:
       add_bos_token: True
       add_eos_token: False
       vocab_file: "/path/to/tokenizer.model"
       type: LlavaTokenizer
       auto_register: llava_tokenizer.LlavaTokenizer
   ```

### 单卡推理

以`llava1.5-7b`单卡推理为例，以下提供推理样例。

- 首先下载示例图片[demo.jpeg](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg)，并在如下示例脚本中的`--predict_data`处指定该图片路径。
- 运行如下脚本

```shell
python run_mindformer.py \
--config research/llava/llava1_5_7B/predict_llava1_5_7b.yaml \
--register_path research/llava \
--run_mode predict \
--predict_data 'path of demo.jpeg' 'Describe the image in English:' \ # 依次传入图片路径或链接、提词
--modal_type image text \ # 对应模态为image和text
--load_checkpoint /path/to/ckpt \
--use_parallel False \
--auto_trans_ckpt False
# load_checkpoint: 单卡推理需传入完整权重的ckpt路径
# auto_trans_ckpt: 单卡推理不进行权重转换，传入False
```

### 多卡推理

以`Llava1.5-7b`2卡推理为例，进行推理前，还需修改并行配置

   ```yaml
    parallel_config:
      data_parallel: 1
      model_parallel: 2 # 对于2卡并行设置mp=2
      pipeline_stage: 1
      use_seq_parallel: False
      micro_batch_num: 1
      vocab_emb_dp: True
      gradient_aggregation_group: 4
   micro_batch_interleave_num: 1
   ```

此后运行并行脚本msrun_launcher.sh拉起并行推理进程

- 首先下载示例图片[demo.jpeg](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwenvl/demo.jpeg)，并在如下示例脚本中的`--predict_data`处指定该图片路径。
- 运行如下脚本

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config research/llava/llava1_5_7B/predict_llava1_5_7b.yaml \
--register_path research/llava \
--run_mode predict \
--predict_data 'path of demo.jpeg' 'Describe the image in English:' \ # 依次传入图片路径或链接、提词
--modal_type image text \ # 对应模态为image和text
--load_checkpoint /path/to/ckpt \
--use_parallel True \
--auto_trans_ckpt True" 2
# load_checkpoint: 当使用完整权重时传入ckpt路径；当使用分布式权重时传入权重文件夹路径model_dir，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
# auto_trans_ckpt: 自动权重转换开关，当传入完整权重时打开
```
