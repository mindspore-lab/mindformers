# CogVLM2-Image

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 之后的版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

CogVLM2 是智谱开发的多模态理解系列大模型，该系列中包含了图文理解以及视频理解大模型。**cogvlm2-llama3-chat-19B** 作为图片理解大模型，在诸如 TextVQA、DocVQA 等多个基准测试中取得了显著的提升。目前该模型支持**8K序列长度**、**支持最高 1344 * 1344 的图像分辨率**以及**提供支持中英文的开源模型版本**等功能。

```text
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models},
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 模型文件

`CogVLM2-Image`基于`mindformers`实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/cogvlm2
       ├── __init__.py
       ├── cogvlm2.py                # 模型实现
       ├── cogvlm2_config.py         # 模型配置项
       ├── cogvlm2image_llm.py       # cogvlm2 语言模型实现
       ├── cogvlm2image_processor.py # cogvlm2 数据预处理
       └── cogvlm2_tokenizer.py      # cogvlm2 tokenizer
   ```

2. 模型配置：

   ```text
   configs/cogvlm2
       └── predict_cogvlm2_image_llama3_chat_19b.yaml  # 模型推理启动配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 模型权重下载

MindFormers提供HuggingFace官方权重下载链接，用户可下载权重并经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

> 该tokenizer与llama3模型相同，请自行申请huggingface上llama3使用权限进行下载。

| 模型名称                    | MindSpore权重 |                        HuggingFace权重                         |
|:------------------------|:-----------:|:------------------------------------------------------------:|
| cogvlm2-llama3-chat-19B |      -      | [Link](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) |

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
pip install transformers torch
python convert_weight.py --modal image --model cogvlm2 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype 'fp16'

# 参数说明
modal:       模型模态, 该模型输入'image'
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换后的MindSpore权重参数类型
```

## 推理

MindFormers提供`cogvlm2-llama3-chat-19B`的推理示例，支持单卡推理、多卡推理。

### 单卡推理

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml`

   ```yaml
   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: False                      # 关闭动态shape

     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动推理脚本

   ```shell
   python run_mindformer.py \
    --config configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml \
    --run_mode predict \
    --predict_data "/path/image.jpg" "Please describe this image." \
    --modal_type image text \
    --load_checkpoint /{path}/cogvlm2-image-llama3-chat.ckpt

   # 参数说明
   config:          模型配置文件路径
   run_mode:        模型执行模式, 'predict'表示推理
   predict_data:    模型推理输入, 第一个输入是图片路径, 第二个输入是文本
   modal_type:      模型推理输入对应模态, 图片路径对应'image', 文本对应'text'
   load_checkpoint: 模型权重文件路径
   ```

### 多卡推理

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml`

   ```yaml
   auto_trans_ckpt: True                      # 开启权重自动转换
   use_parallel: True
   parallel_config:
     model_parallel: 2                        # 可根据使用device数进行修改

   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: False                      # 关闭动态shape

     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动推理脚本

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml \
    --run_mode predict \
    --predict_data \"/path/image.jpg\" \"Please describe this image.\" \
    --modal_type image text \
    --load_checkpoint /{path}/cogvlm2-image-llama3-chat.ckpt" 2
   ```

****