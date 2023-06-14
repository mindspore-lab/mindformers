# ChatGLM6B

## 模型描述

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考清华的[博客](https://chatglm.cn/blog)。在此仓中，提供ChatGLM6B的推理和微调能力。

## 仓库介绍

`chatGLM6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/glm`

    ```bash
    glm
        ├── __init__.py
        ├── attention.py            # 自注意力
        ├── chatglm_6b_tokenizer.py # tokenizer
        ├── glm_config.py           # 模型配置项
        ├── glm.py                  # 模型实现
        └── layers.py               # glm 层定义
    ```

2. 模型配置：`configs/glm`

    ```bash
    glm
        ├── run_glm_6b_fintune.yaml     # 全量微调启动配置
        ├── run_glm_6b_lora.yaml        # 低参微调启动配置
        └── run_glm_6b_infer.yaml       # 推理启动配置
    ```

## 环境要求

- 硬件：Ascend 910A
- MindSpore：2.0.0rc1 / 1.10.1
- MindFormers版本：dev

推理可在单机单卡上完成部署

训练需要最少单机8卡

## ChatGLM6B推理

> 需开发者提前pip安装。具体接口说明请参[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

### AutoClass推理

可以使用AutoClass接口，通过模型名称获取相应的模型/tokenizer实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/glm`

首次运行pipeline推理时需要进行模型编译，需等待一段时间

```python
>>> from mindformers import AutoModel, AutoTokenizer, TextGenerationPipeline
>>> model = AutoModel.from_pretrained("glm_6b_chat")
>>> tokenizer = AutoTokenizer.from_pretrained("glm_6b")
>>> pipeline = TextGenerationPipeline(model, tokenizer, max_length=2048)
>>> pipeline("你好", top_p=0.7)
[{'text_generation_text': ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。']}]
```

> 注：`AutoModel.from_pretrained()` 接口当前支持 `glm_6b` 和 `glm_6b_chat` 两类模型，前者为通用模型，后者具备推理加速特性，两者共享权重，在推理场景下建议使用后者

### pipeline推理

也可以不实例化构造模型，直接通过指定任务模型与模型名的方式进行pipeline的构造

pipeline中，`glm_6b` 默认使用推理加速模型

```python
>>> from mindformers import pipeline
>>> task_pipeline = pipeline(task='text_generation', model='glm_6b', max_length=2048)
>>> task_pipeline('你好', top_p=0.7)
[{'text_generation_text': ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。']}]
```

### 基于API接口的推理

可使用如下`chat_glm.py`脚本：

```python
import time
import mindspore as ms
import numpy as np
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response

config = GLMConfig(
    position_encoding_2d=True,
    phase="predict",
    use_past=True,
    is_npu_acceleration=True,
)

def chat_glm():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=7)
    model = GLMChatModel(config)
    ms.load_checkpoint("./checkpoint_download/glm/glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('./checkpoint_download/glm/ice_text.model')

    prompts = ["你好", "请介绍一下华为"]
    history = []
    for query in prompts:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer(prompt)

        start_time = time.time()
        outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                                    max_length=config.max_decode_length, do_sample=False, top_p=0.7, top_k=1)
        end_time = time.time()
        print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')
        response = tokenizer.decode(outputs)
        response = process_response(response[0])
        history = history + [(query, response)]
        print(response)

if __name__ == "__main__":
    chat_glm()
```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据处理

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 任意目录下

使用 `mindformers/mindformers/dataset/glm_data_process/adgen_dataset.py` 脚本将数据集处理成mindrecord格式

执行命令生成训练数据集：

```bash
python adgen_dataset.py \
--input_file /data3/l00806781/dataset/AdvertiseGen/train.json \
--vocab_file /root/l00806781/mindformers_glm/ice_text.model\
--output_file /data3/lzd/dataset/AdvertiseGen/train_0604_128.mindrecord \
--max_source_length 64 \
--max_target_length 64 \
--mode train
```

执行命令生成评估数据集：

```bash
python adgen_dataset.py \
--input_file /data3/l00806781/dataset/AdvertiseGen/dev.json \
--vocab_file /root/l00806781/mindformers_glm/ice_text.model \
--output_file /data3/lzd/dataset/AdvertiseGen/eval_0604_256.mindrecord \
--max_source_length 256 \
--max_target_length 256 \
--mode eval
```

### 生成HCCL文件

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

```shell
# step1：机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

> 注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 训练启动命令说明

#### 修改配置文件

- 数据集：修改 `mindformers/configs/glm/glm/run_glm_6b_finetune.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `mindformers/configs/glm/glm/run_glm_6b_finetune.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

#### 启动全参微调脚本

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/glm/run_glm_6b_finetune.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的glm/run_glm_6b.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

> 注：由于GLM6B的模型较大，无法在单卡上运行，此处仅提供分布式启动脚本

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

## 评估

### 模型文件合一

finetune所得到的权重文件为根据模型切分策略切分后的权重，我们需要手动将切分权重合一，以用于评估和推理

1. 获取模型切分策略文件：
   在执行全参微调脚本时，模型完成编译后，将会在 `mindformers/scripts/mf_parallelx` 文件夹下，生成名为 `ckpt_strategy.ckpt` 的切分策略文件，将其保存

2. MindSpore提供了根据切分策略转换模型权重切分的接口，[mindspore.transform_checkpoints](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.transform_checkpoints.html)，执行以下python脚本，将8份模型文件合成一份

    ```python
    from mindspore import transform_checkpoints
    transform_checkpoints(
        src_checkpoints_dir="./output/checkpoint/", # 原切分权重文件夹
        dst_checkpoints_dir="./target_checkpoint/", # 目标路径
        ckpt_prefix="glm-6b_rank_xx", # .ckpt文件前缀名
        src_strategy_file="ckpt_stragery.ckpt" # 步骤1中的切分策略文件路径
        dst_strategy_file=None # None表示不切分，权重合一
    )
    ```

### 启动 eval 脚本

启动如下shell脚本，执行单卡评估

```bash
python run_mindformer.py --config configs/glm/run_glm_6b_finetune.yaml --run_mode eval --load_checkpoint checkpoint_download/glm/glm_6b.ckpt --eval_dataset_dir /./data/AdvertiseGen/adgen_dev.mindrecord --device_id 7
```

各项参数：

- `--config`: 指定用于评估的配置文件名称，此处为`configs/glm/run_glm_6b_finetune.yaml`
- `run_mode`: 指定执行模式，此为`eval`，表示为评估模式
- `load_checkpoint`: 指定要加载的checkpoint路径，此处为`checkpoint_download/glm/glm_6b.ckpt`，可替换为需加载的权重的真实路径
- `eval_dataset_dir`: 评估数据集的路径
- `device_id`: 指定要使用的设备编号（从0开始）

评估完成后会打印评估指标 `bleu-4`、`rouge-1`、`rouge-2`、`rouge-l`

> 注：由于默认评估指标的获取方式为生成完整文本后与预期文本做比较，评估速度将受限于文本生成速度，比较缓慢

## 模型权重转化

本仓库中的`glm`来自于HuggingFace的[chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)，基于下述的步骤获取：

1. 克隆chatglm-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm-6b
   ```

2. 执行下列python脚本将8份分布式的pytorch模型文件保存成1份。

   ```python
   from transformers import AutoModel
   import torch as pt

   pt_ckpt_path="Your chatglm-6b path"
   model = AutoModel.from_pretrained(pt_ckpt_path, trust_remote_code=True).half()
   pt_pth_path = "pt_glm_6b.pth"
   pt.save(model.state_dict(), pt_pth_path)
   ```

3. 执行转换脚本，得到转换后的输出文件`ms_glm_6b.ckpt`。

   ```shell
   python mindformers/models/glm/convert_weight.py --pt_ckpt_path "replace your ptroch pth path" --ms_ckpt_path ./ms_glm_6b.ckpt
   ```
