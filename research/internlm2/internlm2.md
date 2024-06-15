# InternLM2

## 模型描述

第二代浦语模型，InternLM2 的基础模型具备以下的技术特点

有效支持20万字超长上下文：模型在20万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 和 L-Eval 等长文任务中的表现也达到开源模型中的领先水平。
综合性能全面提升：各能力维度相比上一代模型全面进步，在推理、数学、代码等方面的能力提升显著。

本仓库支持InternLM2-7B的推理。由于InternLM2与LLaMA结构相似，模型实现中的Embedding、FeedForward、RMSNorm等模块复用仓上LLaMA的代码。

注: 由于InternLM2基于高阶接口的形式开发，存放于research文件夹下，使用时需要将MindFormers[安装](https://gitee.com/mindspore/mindformers#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)为python包，才能直接进入research/internlm2目录下执行相关命令。

## 模型性能

|                                       config                                       |      task       | train performance |       [predict performance](###快速推理)        |
|:----------------------------------------------------------------------------------:|:---------------:|:-----------------:|:-------------------------------------------:|
| [InternLM2_7B (Atlas 800T A2)](../../research/internlm2/predict_internlm2_7b.yaml) | text_generation |         /         | 38.3 tokens/s (batch_size=1, use_past=True) |

## 代码结构介绍

`InternLM2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/internlm2`

    ```bash
    internlm
        ├── internlm2_tokenizer.py       # tokenizer
        ├── internlm2_transformer.py     # transformer层实现
        ├── internlm2_config.py          # 模型config
        └── internlm2.py                 # 模型实现
    ```

2. 模型配置：`research/internlm2`

    ```bash
    internlm
        └── predict_internlm2_7b.yaml             # InternLM2-7B推理Atlas 800T A2启动配置
    ```

3. 预处理脚本和任务启动脚本：`research/internlm2`

    ```bash
    internlm
        ├── convert_weight.py             # hf->mf权重转换
        ├── convert_reversed.py           # mf->hf权重转换
        └── run_internlm2.py               # 高阶接口使用脚本
    ```

## 权重转换

也可选择从Hugging Face下载预训练权重后根据以下步骤进行权重转换，包含对应的分词模型，需要下载整个工程，Hugging Face权重的链接如下：

- [InternLM2-chat-7B](https://huggingface.co/internlm/internlm2-chat-7b)

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model internlm2 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --qkv_concat True
# 参数说明
input_path: huggingface权重保存目录路径
output_path: 权重保存文件名，可以指定自定义保存路径
qkv_concat: 是否qkv融合
```

## InternLM2-7B

### 快速推理

#### 基于高阶接口的推理

1. 配置文件设置，添加tokenizer路径`vocab_file`，`batch_size`的值

在使用Trainer接口进行推理时，由于InternLM2-7b的tokenizer需要用户自行下载，因此在启动前，请先在配置文件中将tokenizer.model的路径自行配置，配置项为vocab_file。

```python
# research/internlm2/predict_internlm2_7b.yaml
...
# model config
model:
  model_config:
    type: InternLM2Config
    ...
    batch_size: 1        # batch size设为1
...
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<unk>'
   bos_token: '<s>'
   eos_token: '</s>'
   pad_token: '</s>'
   vocab_file: './tokenizer.model'        # 添加tokenizer路径
   type: InternLM2Tokenizer
```

2. Trainer接口启动推理

InternLM2-7B的高阶接口使用脚本已集成在run_internlm2.py脚本中，运行此脚本命令示例：

```shell
python run_internlm2.py \
--config "predict_internlm2_7b.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint ckpt_path_or_dir \
--predict_data '你是谁？' \
--predict_length 256 \
--device_id 0

# [{'text_generation_text': ['<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<|im_end|>\n<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我的设计理念是有用、诚实并且无害，我能够使用汉语和英语与您进行交流。<|im_end|>']}]
```

#### Pipeline推理

```python
from mindspore import context
from mindformers.pipeline import pipeline

from internlm2 import InternLM2ForCausalLM
from internlm2_tokenizer import InternLM2Tokenizer
from internlm2_config import InternLM2Config

context.set_context(device_id=0, mode=0)
# init model
internlm_model_path = "/path/InternLM2-7B/internlm2-chat-7b.ckpt" # InternLM2 ckpt path
internlm_config = InternLM2Config(
    checkpoint_name_or_path=internlm_model_path,
    use_past=True,
    eos_token_id=[2, 92542]
)
internlm_model = InternLM2ForCausalLM(
    config=internlm_config
)
# init tokenizer
tokenizer_path = "/path/InternLM2-7B/tokenizer.model" # InternLM2-7B tokenizer.model path
tokenizer = InternLM2Tokenizer(
    vocab_file=tokenizer_path
)
pipeline_task = pipeline(task="text_generation", model=internlm_model, tokenizer=tokenizer)
pipeline_result = pipeline_task("<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n"
                                "- InternLM (书生·浦语) is a conversational language model that is developed by "
                                "Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, "
                                "and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in "
                                "the language chosen by the user such as English and 中文.<|im_end|>\n<|im_start|>"
                                "user\n你是谁？<|im_end|>\n<|im_start|>assistant\n",
                                do_sample=False,
                                repetition_penalty=1.0,
                                max_length=256)

print(pipeline_result)

# [{'text_generation_text': ['<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<|im_end|>\n<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n我是一个名叫书生·浦语的人工智能助手，由上海人工智能实验室开发。我的设计理念是有用、诚实并且无害，我能够使用汉语和英语与您进行交流。<|im_end|>']}]
```
