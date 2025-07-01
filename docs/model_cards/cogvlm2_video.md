# CogVLM2-Video

> ## 🚨 弃用说明
>
> 本模型已过时，不再进行维护，并将在 *1.6.0* 版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/start/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

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
       ├── cogvlm2_tokenizer.py      # cogvlm2 tokenizer
       └── convert_weight.py         # 权重转换脚本
   ```

2. 模型配置：

   ```text
   configs/cogvlm2
       ├── finetune_cogvlm2_video_llama3_chat_13b_lora.yaml  # cogvlm2-video-13b模型LoRA微调启动配置
       └── predict_cogvlm2_video_llama3_chat_13b.yaml        # cogvlm2-video-13b模型推理启动配置
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据及权重准备

#### 数据集下载

MindFormers提供`RWF2000`作为[微调](#微调)数据集，用户可通过如下链接进行下载。

| 数据集名称   |       适用模型        |   适用阶段   |                            下载链接                             |
|:--------|:-----------------:|:--------:|:-----------------------------------------------------------:|
| RWF2000 | CogVLM2-Video-13B | Finetune | [Link](https://www.kaggle.com/datasets/vulamnguyen/rwf2000) |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **RWF2000 数据预处理**

  执行数据预处理脚本`mindformers/tools/dataset_preprocess/cogvlm2/rwf2000_process.py`制作数据集。

  ```shell
  cd mindformers/tools/dataset_preprocess/cogvlm2
  python rwf2000_process.py \
   --data_dir /path/RWF-2000/ \
   --output_file /path/RWF-2000/train.json

  # 参数说明
  data_dir:   下载后保存数据的文件夹路径, 文件夹内包含'train'和'val'文件夹
  output_dir: 生成数据集标签文件路径
  ```

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

## 微调

### LoRA微调

MindFormers支持对`CogVLM2-Video-Chat-13B`进行LoRA微调，微调数据集可参考[数据集下载](#数据集下载)部分获取。

1. 将HuggingFace权重转换为可加载的LoRA权重

   ```shell
   pip install transformers torch
   cd mindformers/models/cogvlm2
   python convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path {path}/MS_CKPT_NAME --sft 'lora'

   # 参数说明
   input_path:  下载HuggingFace权重的文件夹路径
   output_path: 转换后的MindSpore权重文件保存路径
   sft:         转换微调权重类型, 'lora'表示将原始权重转换为可加载的LoRA权重
   ```

2. 修改模型配置文件`configs/cogvlm2/finetune_cogvlm2_video_llama3_chat_13b_lora.yaml`

   ```yaml
   train_dataset:
     data_loader:
       annotation_file: "/{path}/RWF-2000/train.json"  # 预处理后的数据集文件路径
       shuffle: True                                   # 开启数据集随机采样
     tokenizer:
       vocab_file: "/{path}/tokenizer.model"           # 指定tokenizer文件路径
   ```

3. 启动微调脚本

   ```shell
   python run_mindformer.py \
    --config configs/cogvlm2/finetune_cogvlm2_video_llama3_chat_13b_lora.yaml \
    --run_mode finetune \
    --load_checkpoint /{path}/cogvlm2-video-llama3-chat_lora.ckpt

   # 参数说明
   config:          模型配置文件路径
   run_mode:        模型执行模式, 'finetune'表示微调
   load_checkpoint: 模型权重文件路径
   ```

## 推理

MindFormers提供`CogVLM2-Video-Chat-13B`的推理示例，支持单卡推理、多卡推理。

### 单卡推理

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_video_llama3_chat_13b.yaml`

   ```yaml
   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: True                       # 开启动态shape

   processor:
     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动推理脚本

   ```shell
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
   The footage is captured in a slow-motion style, with each frame highlighting the feet' movement and the texture of the shoes..."
   ```

### 多卡推理

以`CogVLM2-Video-Chat-13B`2卡推理为例。

1. 修改模型配置文件`configs/cogvlm2/predict_cogvlm2_video_llama3_chat_13b.yaml`

   ```yaml
   auto_trans_ckpt: True                      # 开启权重自动转换
   use_parallel: True
   parallel_config:
     model_parallel: 2                        # 可根据使用device数进行修改

   model:
     model_config:
       use_past: True                         # 开启增量推理
       is_dynamic: True                       # 开启动态shape

   processor:
     tokenizer:
       vocab_file: "/{path}/tokenizer.model"  # 指定tokenizer文件路径
   ```

2. 启动推理脚本

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config configs/cogvlm2/predict_cogvlm2_video_llama3_chat_13b.yaml \
    --run_mode predict \
    --predict_data \"/path/video.mp4\" \"Please describe this video.\" \
    --modal_type video text \
    --load_checkpoint /{path}/cogvlm2-video-llama3-chat.ckpt" 2
   ```

   推理结果示例：

   ```text
   inputs: "run.mp4" "Please describe this video."
   outputs: "The video features a series of close-up shots of a person's feet running on a sidewalk.
   The footage is captured in a slow-motion style, with each frame highlighting the feet' movement and the texture of the shoes..."
   ```
