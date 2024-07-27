# 通义千问

## 模型描述

Qwen2是Qwen系列的新的大型语言模型。Qwen2发布了许多基本语言模型和指令调整的语言模型，参数范围从5亿到720亿，包括专家混合模型。
与最先进的开源语言模型（包括之前发布的Qwen1.5）相比，Qwen2总体上超越了大多数开源模型，并在一系列针对语言理解，语言生成，多语言能力，编码，数学，推理等的基准测试中表现出对专有模型的竞争力。

```text
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```

## 模型文件

`Qwen2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/qwen2
     └── qwen2_tokenizer.py                        # tokenizer
   ```

2. 模型配置：

   ```text
   research/qwen2
     └── predict_qwen2_72b_instruct.yaml           # 72B 在线推理启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   research/qwen2
     ├── convert_weight.py                         # 权重转换脚本
     └── run_qwen2.py                              # Qwen2多轮对话脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#源码编译安装)和[版本匹配关系](../../README.md#版本匹配关系)。

### 数据及权重准备

#### 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，`vocab.json`和`merges.txt`文件也在链接中下载。

词表下载链接：[vocab.json](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/vocab.json)和[merges.txt](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/merges.txt)

| 模型名称        |                     Base权重（建议训练和微调使用）                     |                         Instruct权重（建议推理使用）                         |
|:------------|:---------------------------------------------------------:|:--------------------------------------------------------------:|
| qwen2-72b-Instruct | [Link](https://huggingface.co/Qwen/Qwen2-72B/tree/main) | [Link](https://huggingface.co/Qwen/Qwen2-72B-Instruct/tree/main) |

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model qwen2 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

- **[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)**

  从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

  通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

  以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 推理

大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

注意事项：

1. 当前支持模型已提供推理相关配置文件，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen2`目录下，或者先将`research/qwen2`目录所在路径加入到`PYTHONPATH`环境变量中。

### 基于高阶接口的推理

#### 多卡推理

以`qwen2_72b`4卡推理为例，执行如下命令进行推理, 推理前先参考[权重转换](../../docs/feature_cards/Transform_Ckpt.md)切分权重。

1. 主要参数配置参考：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: False
     gradient_aggregation_group: 4
   ```

2. 启动多卡推理：

   ```shell
   cd research/qwen2
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash ../../scripts/msrun_launcher.sh "run_qwen2.py \
    --config predict_qwen2_72b_instruct.yaml \
    --load_checkpoint /path/model_dir \
    --vocab_file /path/vocab.json \
    --merges_file /path/merges.txt \
    --run_mode predict \
    --use_parallel True \
    --auto_trans_ckpt False \
    --predict_data 帮助我制定一份去上海的旅游攻略" 4

   # 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
   ```