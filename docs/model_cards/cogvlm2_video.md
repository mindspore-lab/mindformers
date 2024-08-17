# CogVLM2-Video

## 模型描述

CogVLM2 是智谱开发的多模态理解系列大模型，该系列中包含了图文理解以及视频理解大模型。**CogVLM2-Video-13B** 作为视频理解大模型，在多个视频问答任务中达到了最优的性能，它可以快速完成对输入视频的理解并根据输入文本作出回答。目前该模型支持**2K序列长度**、**224×224分辨率的视频理解**以及**中英文回答**等功能。

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

`CogVLM2-Video`基于`mindformers`实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/cogvlm2
       ├── __init__.py
       ├── cogvlm2.py                # 模型实现
       ├── cogvlm2_config.py         # 模型配置项
       ├── cogvlm2_llm.py            # cogvlm2 语言模型实现
       ├── cogvlm2_processor.py      # cogvlm2 数据预处理
       └── cogvlm2_tokenizer.py      # cogvlm2 tokenizer
   ```

2. 模型配置：

   ```text
   configs/cogvlm2
       └── predict_cogvlm2_video_llama3_chat_13b.yaml  # cogvlm2-video-13b模型推理启动配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 模型权重下载

MindFormers提供HuggingFace官方权重下载链接，用户可下载权重并经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

> 该tokenizer与llama3模型相同，请自行申请huggingface上llama3使用权限进行下载。

| 模型名称                   | MindSpore权重 |                              HuggingFace权重                               |
|:-----------------------|:-----------:|:------------------------------------------------------------------------:|
| CogVLM2-Video-Chat-13B |      -      | [Link](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat/tree/main) |

#### 模型权重转换

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
pip install transformers torch
python convert_weight.py --model cogvlm2 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype 'fp32'

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换后的MindSpore权重参数类型
```

## 推理

MindFormers提供`CogVLM2-Video-Chat-13B`的推理示例，目前支持单卡推理。

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_video_llama3_chat_13b.yaml`

   ```yaml
   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: True                       # 开启动态shape

     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动推理脚本

   ```shell
   export USE_ROPE_SELF_DEFINE=True
   python run_mindformer.py \
    --config configs/cogvlm2/predict_cogvlm2_video_llama3_chat_13b.yaml \
    --run_mode predict \
    --predict_data "/path/video.mp4" "Please describe this video." \
    --modal_type "video" "text" \
    --load_checkpoint /{path}/cogvlm2-video-llama3-chat.ckpt

   # 参数说明
   config:          模型配置文件路径
   run_mode:        模型执行模式, 'predict'表示推理
   predict_data:    模型推理输入, 第一个输入是视频文件路径, 第二个输入是prompt
   modal_type:      模型推理输入的模态类型, 内容顺序对应predict_data中输入的模态类型，支持 "video"，"text"
   load_checkpoint: 模型权重文件路径
   ```

   推理结果示例：

   ```text
   inputs: "run.mp4" "Please describe this video."
   outputs: "The video features a series of close-up shots of a person's feet running on a sidewalk.
   The footage is captured in a slow-motion style, with each frame highlighting the person's feet and the details of their running shoes.
   The background is a green screen, which remains consistent throughout the video..."
   ```
