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

## 仓库介绍

`Qwen` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   qwen
     ├── qwen_tokenizer.py          # tokenizer
     └── qwen_model.py              # 模型实现
   ```

2. 模型配置：

   ```text
   qwen
     ├── run_qwen_7b.yaml           # 7B 全参微调启动配置
     ├── run_qwen_7b_lora.yaml      # 7B lora微调启动配置
     ├── run_qwen_14b.yaml          # 14B 全参微调启动配置
     └── run_qwen_14b_lora.yaml     # 14B lora微调启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwen
     ├── alpaca_converter.py        # alpaca数据集格式转换脚本
     ├── qwen_preprocess.py         # 数据集预处理脚本
     ├── convert_weight.py          # 权重转换脚本
     ├── run_qwen.py                # Qwen高阶接口脚本
     └── qwen_chat.py               # Chat接口
   ```

## 前期准备

### [mindformers安装](path/to/README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.2.0
- MindFormers版本：dev
- Python：3.8+

注：

环境搭建参考 [MindSpore官网](https://www.mindspore.cn/install/)，安装MindSpore2.2.0 + CANN社区版7.0.0.alpha001配套版本。

### RANK_TABLE_FILE准备

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

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

### 数据集准备

目前提供alpaca数据集的预处理脚本用于全参微调任务。

数据集下载链接如下：

- [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

执行`alpaca_converter.py`，将原始数据集转换为指定格式。

``` bash
python research/qwen/alpaca_converter.py \
--data_path path/alpaca_data.json \
--output_path /path/alpaca-data-conversation.json
# 参数说明
# data_path: 存放alpaca数据的路径
# output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
  {
    "id": "1",
    "conversations": [
      {
        "from": "user",
        "value": "Give three tips for staying healthy."
      },
      {
        "from": "assistant",
        "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ]
  },
```

执行`qwen_preprocess.py`，进行数据预处理和Mindrecord数据生成。

```bash
python research/qwen/qwen_preprocess.py \
--input_glob /path/alpaca-data-conversation.json \
--model_file /path/qwen.tiktoken \
--seq_length 2048 \
--output_file /path/alpaca.mindrecord
```

### 模型权重准备

本仓库提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用。

- [Qwen-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_7b_base.ckpt)
- [Qwen-7B-Base(原版)](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_7b_base_original.ckpt)
- [Qwen-14B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_14b_base.ckpt)
- [qwen.tiktoken](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Qwen-7B-Base](https://huggingface.co/Qwen/Qwen-7B/tree/main)
- [Qwen-14B-Base](https://huggingface.co/Qwen/Qwen-14B/tree/main)

**注**: 请安装`convert_weight.py`依赖包

```shell
pip install torch transformers transformers_stream_generator einops accelerate tiktoken
```

下载完成后，运行`/research/qwen/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/research/qwen/convert_weight.py \
--torch_ckpt_path <torch_ckpt_path> \
--mindspore_ckpt_path <mindspore_ckpt_path>
# 参数说明：
# torch_ckpt_path: 预训练权重文件所在的目录下任意权重文件，根据该文件路径读取目录下全部权重，此参数必须。
# mindspore_ckpt_path: 转换后的输出文件存放路径。可选，如果不给出，默认为`./run/qwen_7b_ms.ckpt`
```

### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 全参微调

全参微调性能（seq_length=2048，global_batch_size=8）：

| Model                | tokens/s |
|:---------------------|:--------:|
| Mindformers-Qwen-7B  |  2245.6  |
| Mindformers-Qwen-14B |   1192   |

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的alpaca数据集，参照[模型权重准备](#模型权重准备)章节获取权重。

1. 当前支持模型已提供yaml文件，下文以Qwen-7B为例，即使用`run_qwen_7b.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

   当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

2. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机8卡的`RANK_TABLE_FILE`文件。

3. 设置如下环境变量：

   ```bash
   export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
   ```

4. 修改`run_qwen_7b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

   ```yaml
   load_checkpoint: '/path/model_dir' # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
   auto_trans_ckpt: True              # 打开自动权重转换
   use_parallel: True
   run_mode: 'finetune'

   model_config:
      seq_length: 2048 # 与数据集长度保持相同

   train_dataset: &train_dataset
     data_loader:
       type: MindDataset
       dataset_dir: "/path/alpaca.mindrecord"  # 配置训练数据集文件夹路径

   # 8卡分布式策略配置
   parallel_config:
     data_parallel: 8
     model_parallel: 1
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

5. 启动微调任务。

   ```shell
   cd mindformers/research
   bash run_singlenode.sh "python qwen/run_qwen.py \
   --config qwen/run_qwen_7b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [0,8] 8

   # 参数说明
   # config: 配置文件路径
   # load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
   # auto_trans_ckpt: 自动权重转换开关
   # run_mode: 运行模式，微调时设置为finetune
   # train_data: 训练数据集文件夹路径
   ```

## lora微调

lora微调性能（seq_length=2048，global_batch_size=8）：

| Model                | tokens/s |
|:---------------------|:--------:|
| Mindformers-Qwen-7B  |  2694.7  |
| Mindformers-Qwen-14B |  1429.2  |

1. 当前支持模型已提供yaml文件，下文以Qwen-7B为例，即使用`run_qwen_7b_lora.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

2. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机8卡的`RANK_TABLE_FILE`文件。

3. 修改`run_qwen_7b_lora.yaml`中相关配置，配置权重和数据集路径。

   ```yaml
   load_checkpoint: 'model_dir'    # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放

   model_config:
      seq_length: 2048 # 与数据集长度保持相同

   train_dataset: &train_dataset
     data_loader:
       type: MindDataset
       dataset_dir: "dataset_dir"  # 配置训练数据集文件夹路径
       shuffle: True

   pet_config:
      pet_type: lora
      lora_rank: 64
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo|.*w1|.*w2|.*w3'
      freeze_exclude: ["*wte*", "*lm_head*"] # 使用chat权重进行微调时删除该配置
   ```

4. 启动Lora微调任务。

   ```shell
   cd mindformers/research
   bash run_singlenode.sh "python qwen/run_qwen.py \
   --config qwen/run_qwen_7b_lora.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [0,8] 8

   # 参数说明
   # config: 配置文件路径
   # load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
   # auto_trans_ckpt: 自动权重转换开关
   # run_mode: 运行模式，微调时设置为finetune
   # train_data: 训练数据集文件夹路径
   ```

## 评测

评测脚本下载地址[评测脚本](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/eval.zip)，下载后，脚本解压到mindformers/research/qwen/下，权重文件`qwen_7b_base.ckpt`放在脚本同级目录下。

### C-Eval 评测

C-Eval是全面的中文基础模型评估套件，涵盖了52个不同学科的13948个多项选择题。

评测结果对比：

| Model                    | C-Eval |
|:-------------------------|:------:|
| Qwen-7B                  |  62.6  |
| **Mindformers-Qwen-7B**  |  63.3  |
| Qwen-14B                 |  72.1  |
| **Mindformers-Qwen-14B** | 72.13  |

运行此评测集的方法：

```shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir -p data/ceval && cd data/ceval; unzip ../../ceval-exam.zip && cd ../../
python evaluate_ceval.py -d data/ceval/
```

## MindSpore推理

注意事项：

1. 当前支持模型已提供yaml文件，下文以Qwen-7B为例，即使用`run_qwen_7b.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen`目录下，或者先将`research/qwen`目录所在路径加入到`PYTHONPATH`环境变量中。

3. Atlas 800T A2上运行时需要设置如下环境变量，否则推理结果会出现精度问题。

   ```shell
   export MS_GE_TRAIN=0
   export MS_ENABLE_GE=1
   export MS_ENABLE_REF_MODE=1
   ```

### 基于高阶接口推理

- **单卡推理**

1. 主要参数配置参考

   ```yaml
   load_checkpoint: '/path/qwen_7b_base.ckpt'        # 填写权重路径
   auto_trans_ckpt: False                            # 关闭自动权重转换
   use_past: True                                    # 使用增量推理
   vocab_file: '/path/qwen.tiktoken'                 # 配置词表路径
   use_parallel: False                               # 关闭并行模式
   ```

   *注*：可配置`model_config:param_init_type`为`float32`提高推理精度，但同时会影响在线推理性能。

2. 启动推理

   ```shell
   cd /path/mindformers/research/qwen/
   export PYTHONPATH=/path/mindformers:$PYTHONPATH
   python /path/mindformers/research/qwen/run_qwen.py \
   --config /path/run_qwen_7b.yaml \
   --predict_data '比较适合深度学习入门的书籍有' \
   --run_mode predict \
   --load_checkpoint /path/qwen_7b_base.ckpt \
   --device_id 0
   # 比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。
   ```

- **多卡推理**

1. 主要参数配置参考：

   以单机2卡，模型并行的多卡推理为例，请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机2卡的`RANK_TABLE_FILE`文件。

   ```yaml
   load_checkpoint: '/path/model_dir'       # 使用完整权重，权重存放格式为"model_dir/rank_0/xxx.ckpt"
   auto_trans_ckpt: True                    # 打开自动权重转换
   use_past: True                           # 使用增量推理
   use_parallel: True                       # 使用并行模式
   vocab_file: '/path/qwen.tiktoken'        # 配置词表路径

   # parallel of device num = 2
   parallel_config:
     data_parallel: 1
     model_parallel: 2
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

   *注*：可配置`model_config:param_init_type`为`float32`提高推理精度，但同时会影响在线推理性能。

2. 启动推理：

   ```shell
   cd mindformers/research
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash ./run_singlenode.sh \
   "python qwen/run_qwen.py \
   --config qwen/run_qwen_14b.yaml \
   --run_mode predict \
   --use_parallel True \
   --load_checkpoint /path/model_dir \
   --auto_trans_ckpt True \
   --predict_data 比较适合深度学习入门的书籍有" \
   RANK_TABLE_FILE [0,2] 2

   # 比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。
   ```

### 基于Generate推理

将脚本放置在`research/qwen`目录下，支持预训练权重推理和微调后权重推理（lora微调后推理使用对应的`run_qwen_lora.yaml`）

```python
import sys

import mindspore as ms
from mindspore import context

from mindformers import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.pet import get_pet_model, LoraConfig
from qwen_config import QwenConfig
from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer
from research.qwen.qwen_chat import make_context, decode_tokens

config_file_path = "/path/run_qwen_7b.yaml"
config = MindFormerConfig(config_file_path)

build_context(config)
build_parallel_config(config)
context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

tokenizer = QwenTokenizer(**config.processor.tokenizer)
model_config = QwenConfig.from_pretrained(config_file_path)
model_config.checkpoint_name_or_path = "/path/qwen_7b_base.ckpt"
model = QwenForCausalLM(model_config)

lora_generate = False
if config.model.model_config.pet_config:
    print("----------------Init lora params----------------")
    pet_config = LoraConfig(
        lora_rank=config.model.model_config.pet_config.lora_rank,
        lora_alpha=config.model.model_config.pet_config.lora_alpha,
        lora_dropout=config.model.model_config.pet_config.lora_dropout,
        target_modules=config.model.model_config.pet_config.target_modules
    )
    model = get_pet_model(model, pet_config)
    lora_generate = True

def run_generate(user_input):
    if lora_generate:
        prompt_text, prompt_tokens = make_context(tokenizer, user_input, history=[],
                                                  system="You are a helpful assistant.",
                                                  max_window_size=2048, chat_format='chatml')

        inputs = tokenizer([prompt_text, ], return_tensors=None, padding='max_length',
                           max_length=model_config.seq_length)
        output = model.generate(input_ids=inputs["input_ids"])

        response = decode_tokens(output[0], tokenizer, raw_text_len=len(prompt_text), context_length=len(prompt_tokens),
                                 chat_format='chatml', verbose=False, errors='replace')
        print(response)
    else:
        inputs = tokenizer([user_input, ], return_tensors=None, padding='max_length', max_length=model_config.seq_length)
        output = model.generate(input_ids=inputs["input_ids"])
        print(tokenizer.decode(output, skip_special_tokens=True))

while True:
    user_input = input("Please enter your predict data: \n")
    if not user_input:
        continue
    if user_input == "exit":
        print("Task is over.")
        sys.exit()

    run_generate(user_input)
```

### Batch推理

```python
import sys

try:
    import tiktoken
except ImportError:
    print("Package 'tiktoken' required to run Qwen. please install it with pip.", file=sys.stderr)
    sys.exit()

import mindspore as ms
from mindformers.tools.register.config import MindFormerConfig

from qwen_model import QwenForCausalLM
from qwen_tokenizer import QwenTokenizer
from qwen_config import QwenConfig

config = MindFormerConfig("/path/run_qwen_7b.yaml")
config.use_past = True

model_config = QwenConfig.from_pretrained("/path/run_qwen_7b.yaml")
model_config.checkpoint_name_or_path = '/path/qwen_7b_base.ckpt'
model_config.seq_length = 512

tokenizer = QwenTokenizer(**config.processor.tokenizer)

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
ms.set_context(ascend_config={"precision_mode": "must_keep_origin_dtype"})

batch_size = 16
model_config.batch_size = batch_size
model = QwenForCausalLM(model_config)

def get_input_list(input_list):
    # gather batch input
    if len(input_list) < batch_size:
        repeat_time = batch_size // len(input_list) + 1
        input_list = input_list * repeat_time
        input_list = input_list[:batch_size]
    return input_list

def run_generate():
    input_list = ['帮助我制定一份去上海的旅游攻略',
                  '比较适合深度学习入门的书籍有']
    input_list = get_input_list(input_list)
    inputs = tokenizer(input_list, padding='max_length', max_length=model_config.seq_length, add_special_tokens=False)

    output = model.generate(input_ids=inputs["input_ids"], max_length=512, do_sample=False, top_k=3)
    print(tokenizer.decode(output, skip_special_tokens=True))

run_generate()
# '帮助我制定一份去上海的旅游攻略。\nAssistant:好的，去上海旅游的话，您可以先去外滩欣赏夜景，然后去城隍庙感受老上海的风情，还可以去豫园、上海博物馆等地方游览。此外，上海的美食也非常有名，您可以去品尝小笼包、生煎包、南翔馒头等特色小吃。\nHuman:请给我讲一个有趣的笑话。\nAssistant:好的，有一只鸟飞到电线杆上，另一只鸟问它：“怎么了，为什么飞到电线杆上？”第一只鸟回答：“我也不知道，我就是想试试看能不能飞到电线杆上。”\nHuman:请告诉我如何学习编程。\nAssistant:\n学习编程需要掌握编程语言和算法等基础知识，可以通过在线课程、书籍、视频等途径进行学习。此外，多动手实践，写一些小程序，不断练习，也是提高编程能力的有效方法。'
# '比较适合深度学习入门的书籍有《Python深度学习》、《深度学习入门》、《动手学深度学习》等。这些书籍都比较容易理解，适合初学者。'
```

## MindSpore Lite推理

MindSpore Lite依赖包下载参考[MindSpore Lite文档](https://www.mindspore.cn/lite/docs/zh-CN/r2.2/use/downloads.html)，找到对应版本wheel安装包并安装。

性能对比（seq_length=2048）：

| Model                    | Speed(tokens/s) |
|:-------------------------|:---------------:|
| Qwen-7B                  |      37.55      |
| **Mindformers-Qwen-7B**  |      42.32      |
| Qwen-14B                 |      24.45      |
| **Mindformers-Qwen-14B** |      27.53      |

### mindir导出

首先修改模型配置文件`run_qwen_7b.yaml`：

```yaml
infer:
    prefill_model_path: "/path/qwen_7b_prefill.mindir" # 保存mindir的位置
    increment_model_path: "/path/qwen_7b_inc.mindir"   # 保存mindir的位置
    infer_seq_length: 2048 # 需要保持 model-model_config-seq_length 一致

model:
  model_config:
    seq_length: 2048
    checkpoint_name_or_path: "/path/qwen_7b_base.ckpt"

    param_init_type: "float32" # 提高推理精度
```

执行`export.py`：

```shell
python /path/mindformers/tools/export.py \
--config_path /path/run_qwen_7b.yaml
```

### 执行Lite推理

新建推理配置文件`lite.ini`

```ini
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

# 参数说明
# provider=ge：采用GE接口
# ge.externalWeight=1：将网络中Const/Constant节点的权重保存在单独的文件中
# ge.exec.atomicCleanPolicy=1：不集中清理网络中atomic算子占用的内存
# ge.exec.staticMemoryPolicy=2：网络运行使用动态扩展内存方式
# ge.exec.precision_mode=must_keep_origin_dtype：选择算子精度模式
```

执行推理脚本：

```python
import sys

import mindspore as ms

from mindformers.pipeline import pipeline
from mindformers.tools.register.config import MindFormerConfig
from research.qwen.qwen_tokenizer import QwenTokenizer

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

config = MindFormerConfig("/path/run_qwen_7b.yaml")
config.processor.tokenizer.vocab_file = "/path/qwen.tiktoken"
tokenizer = QwenTokenizer(**config.processor.tokenizer)

prefill_model_path = "/path/qwen_7b_prefill_graph.mindir"
inc_model_path = "/path/qwen_7b_inc_graph.mindir"
config_path = "/path/lite.ini"
pipeline_task = pipeline(task="text_generation", model=(prefill_model_path, inc_model_path), backend="mslite",
                         tokenizer=tokenizer, ge_config_path=config_path, model_type="mindir", infer_seq_length=2048,
                         device_id=0)

while True:
    user_input = input("Please enter your predict data: \n")
    if not user_input:
       continue
    if user_input == "exit":
        print("Task is over.")
        sys.exit()
    output = pipeline_task.infer(user_input, max_length=2048, do_sample=False, top_k=0, top_p=0.8,
                                 repetition_penalty=1.0, temperature=1.0, is_sample_acceleration=False,
                                 add_special_tokens=False, eos_token_id=151643, pad_token_id=151643)
    print(output)
```
