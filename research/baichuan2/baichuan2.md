# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。
目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## 模型性能

| Config                                                                                   |      Task       | Datasets |   Performance   |  Phase   |
|:-----------------------------------------------------------------------------------------|:---------------:|:--------:|:---------------:|:--------:|
| [finetune_baichuan2_7b](../../research/baichuan2/finetune_baichuan2_7b.yaml)             | text_generation |  belle   | 3010 tokens/s/p | Finetune |
| [finetune_baichuan2_13b](../../research/baichuan2/finetune_baichuan2_13b.yaml)           | text_generation |  belle   | 1359 tokens/s/p | Finetune |
| [finetune_baichuan2_7b_lora](../../research/baichuan2/finetune_baichuan2_7b_lora.yaml)   | text_generation |  belle   | 3375 tokens/s/p |   LoRA   |
| [finetune_baichuan2_13b_lora](../../research/baichuan2/finetune_baichuan2_13b_lora.yaml) | text_generation |  belle   | 1880 tokens/s/p |   LoRA   |
| [predict_baichuan2_7b](../../research/baichuan2/predict_baichuan2_7b.yaml)               | text_generation |    /     |   42tokens/s    | Predict  |
| [predict_baichuan2_13b](../../research/baichuan2/predict_baichuan2_13b.yaml)             | text_generation |    /     |   23tokens/s    | Predict  |

## 模型文件

`Baichuan2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   research/baichuan2
       ├── baichuan2_tokenizer.py    # tokenizer
       ├── baichuan2_7b.py           # 7B模型实现
       └── baichuan2_13b.py          # 13B模型实现
   ```

2. 模型配置：

   ```text
   research/baichuan2
       ├── finetune_baichuan2_7b.yaml               # 7B全参微调配置
       ├── finetune_baichuan2_13b.yaml              # 13B全参微调配置
       ├── finetune_baichuan2_7b_lora.yaml          # 7B-lora微调配置
       ├── finetune_baichuan2_13b_lora.yaml         # 13B-lora微调配置
       ├── predict_baichuan2_7b.yaml                # 7B单卡推理配置
       └── predict_baichuan2_13b.yaml               # 13B单卡推理配置
   ```

3. 数据预处理和任务启动脚本：

   ```text
   research/baichuan2
       ├── belle_preprocess.py     # belle数据集预处理脚本
       └── run_baichuan2.py        # baichuan2高阶接口使用脚本
       └── run_baichuan2_chat.py   # baichuan2 chat推理使用脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README.md#二mindformers安装)和[版本匹配关系](../../README.md#三版本匹配关系)。

> 注：Atlas 800T A2芯片支持Baichuan2-7b和Baichuan2-13b进行单机8卡全参微调和LoRA微调，以及单机单卡推理。

### 数据及权重准备

#### 数据集下载

当前提供belle_chat_ramdon数据集的预处理和微调样例，用于对Baichuan2-7B-Base，Baichuan2-13B-Base模型进行微调。

| 数据集名称             |               适用模型               |      适用阶段       |                                                 下载链接                                                  |
|:------------------|:--------------------------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|
| belle_chat_ramdon | Baichuan2-7B <br/> Baichuan2-13B | finetune / lora | [Link](https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/data/belle_chat_ramdon_10k.json) |

下载数据集后，需要执行`belle_preprocess.py`脚本进行数据预处理，将原始数据转换为mindrecord格式。

```shell
cd research/baichuan2
python belle_preprocess.py \
 --input_glob /path/to/belle_chat_ramdon_10k.json \
 --model_file /path/to/tokenizer.model \
 --output_file /path/to/belle_chat_ramdon_10k_4096.mindrecord \
 --seq_length 4096

# 参数说明
input_glob: 原始数据集文件路径
model_file: 词表文件路径
output_file: 输出数据集文件路径
seq_length: 生成数据集的序列长度
```

#### 模型权重下载

MindFormers提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

也可选择从HuggingFace下载所有工程文件后进行[模型权重转换](#模型权重转换)使用。

| 模型名称               |                                                    MindSpore权重                                                     |                         HuggingFace权重                          |
|:-------------------|:------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------:|
| Baichuan2-7B-Base  | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Base.ckpt)  | [Link](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)  |
| Baichuan2-7B-Chat  | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt)  | [Link](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |
| Baichuan2-13B-Base | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_13B_Base.ckpt) | [Link](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) |
| Baichuan2-13B-Chat | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt) | [Link](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) |
| tokenizer.model    |     [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)     |                               /                                |

#### 模型权重转换

进行权重转换需要安装以下依赖包。

```shell
pip install torch transformers protobuf wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
```

执行`convert_weight.py`转换脚本，将HuggingFace的权重转换为完整的ckpt权重。

```shell
cd research/baichuan2
python convert_weight.py \
 --torch_ckpt_path path/to/*.bin \
 --mindspore_ckpt_path path/to/baichuan2.ckpt

# 参数说明
torch_ckpt_path: HuggingFace权重保存目录路径下任意权重bin文件, 根据该文件路径读取目录下全部权重
mindspore_ckpt_path: 转换后的MindSpore权重文件保存路径
```

## 微调

MindFormers提供Baichuan2全参微调和LoRA微调的示例。

### 全参微调

MindFormers提供了默认微调配置`finetune_baichuan2_7b.yaml`和`finetune_baichuan2_13b.yaml`，
默认使用数据集[belle_chat_ramdon](#数据集下载)，并在配置文件中设置`seq_length=4096`。

#### 单机训练

以Baichuan2-7B为例，Baichuan2-13B只需要修改配置文件和权重即可。

使用msrun方式拉起分布式训练，默认使用8卡进行训练，如果需要修改使用卡数参考[msrun方式启动](../../README.md#方式一使用已有脚本启动)进行配置。

```shell
cd research/baichuan2
bash ../../script/msrun_launcher.sh \
 "run_baichuan2.py \
 --config finetune_baichuan2_7b.yaml \
 --load_checkpoint path/to/baichuan2_7b_base.ckpt \
 --auto_trans_ckpt True \
 --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord"

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
train_data: 训练数据集文件夹路径或mindrecord文件路径
```

#### 多机训练

以Baichuan2-7B进行2机8卡训练为例，Baichuan2-13B只需要修改配置文件和权重即可。

1. 修改`finetune_baichuan2_7b.yaml`

    ```yaml
    context:
      runtime_num_threads: 1  # 新增配置

    parallel_config:
      data_parallel: 8
      model_parallel: 1
      pipeline_stage: 2
      use_seq_parallel: False
      micro_batch_num: 8
    ```

2. 设置环境变量

    ```shell
    export MS_MEMORY_POOL_RECYCLE=1
    export GE_NOT_CUT=1
    ```

3. 使用msrun脚本训练

   多机训练需要分别在不同节点执行命令，以下为2机8卡训练过程，参数说明以及使用更多节点参考[msrun方式启动](../../README.md#方式一使用已有脚本启动)多机多卡部分进行配置。

   > 注：如果各节点间使用共享存储存放工程文件，则可以使用**支持权重自动转换**功能，否则需要修改配置文件`auto_trans_ckpt=False`或在运行命令时设置`--auto_trans_ckpt False`，
   此时，预训练权重可以使用[权重转换工具](../../docs/feature_cards/Transform_Ckpt.md)转换得到。

- 在节点0执行如下命令，其中192.168.1.1需要改为节点0的实际ip，将节点0作为主节点，2机共16卡且每个节点8卡。

  ```shell
  # 以使用共享盘为例
  bash ../../script/msrun_launcher.sh \
   "run_baichuan2.py \
   --config finetune_baichuan2_7b.yaml \
   --load_checkpoint path/to/baichuan2_7b_base.ckpt \
   --auto_trans_ckpt True \
   --use_parallel True \
   --run_mode finetune \
   --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
   16 8 192.168.1.1 8118 0 output/msrun_log False 300
  ```

- 在节点1执行如下命令，其中192.168.1.1需要改为节点0的实际ip。

  ```shell
  bash ../../script/msrun_launcher.sh \
   "run_baichuan2.py \
   --config finetune_baichuan2_7b.yaml \
   --load_checkpoint path/to/baichuan2_7b_base.ckpt \
   --auto_trans_ckpt True \
   --use_parallel True \
   --run_mode finetune \
   --train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
   16 8 192.168.1.1 8118 1 output/msrun_log False 300
  ```

  > 注：训练过程中出现问题，可以参考[**常见问题**](#常见问题)进行解决。

### LoRA微调

#### 单机训练

LoRA微调执行命令与[全参微调](#全参微调)相同，修改`--config`为`finetune_baichuan2_7b_lora.yaml`或`finetune_baichuan2_13b_lora.yaml`即可。

如果仅需要单独保存Lora权重，修改配置文件中相关参数：

```yaml
callbacks:
  - type: CheckpointMonitor
    save_trainable_params: True
```

> 注：微调完成后，可使用lora权重合并工具将Lora权重和Base权重合并，具体说明可参考[lora权重合并教程](../../docs/feature_cards/Transform_Lorackpt.md)。

## 推理

推理使用`Baichuan2-7B-Chat`和`Baichuan2-13B-Chat`权重，默认设置`seq_length=4096`，支持单卡推理。

### 多轮对话推理

```shell
# 以Baichuan2-7B为例, 多轮对话目前仅支持单卡推理
cd research/baichuan2
python run_baichuan2_chat.py \
 --config predict_baichuan2_7b.yaml \
 --load_checkpoint path/to/baichuan2_7b_chat.ckpt

# 请输入：你是谁？
# 我是我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问
```

### 基于pipeline的推理

pipeline推理支持**自动权重转换**，支持加载**lora权重**。

根据是否加载lora权重进行推理，可以分为以下几种情况：

1. 使用预训练权重或微调权重

   使用`predict_baichuan2_*.yaml`，修改`load_checkpoint`为权重路径即可。

2. 使用训练过的lora权重
   1. **lora权重已与base权重合并，参考[lora权重合并教程](../../docs/feature_cards/Transform_Lorackpt.md)**

      使用`predict_baichuan2_*.yaml`，修改`load_checkpoint`为合并权重路径即可。

   2. **lora权重未合并，权重包含base层和lora层**

      使用`predict_baichuan2_*_lora.yaml`，修改`load_checkpoint`为lora权重路径，若使用分布式权重则修改为权重文件夹路径。

   3. **仅包含lora权重**

      使用`predict_baichuan2_*_lora.yaml`，修改`load_chjeckpoint`为文件夹路径，文件夹包含lora权重和base权重。
      若使用的lora权重是分布式权重，则需要先使用权重转换工具将分布式权重合并为完整权重，使用方法参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

以下为基于pipeline接口的自定义推理代码示例。

```python
# run_baichuan2_pipeline.py
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig
from mindformers.pipeline import pipeline
from mindformers.models import LlamaConfig
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

model_dict = {
    "baichuan2_7b": Baichuan7BV2ForCausalLM,
    "baichuan2_7b_lora": Baichuan7BV2ForCausalLM,
    "baichuan2_13b": Baichuan13BV2ForCausalLM,
    "baichuan2_13b_lora": Baichuan13BV2ForCausalLM
}

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>"]

# init config, 以Baichuan2-7B为例
config_path = "path/to/predict_baichuan2_7b.yaml"
config = MindFormerConfig(config_path)

# init context
build_context(config)
build_parallel_config(config)

# init model
batch_size = 1
config.model.model_config.parallel_config = config.parallel_config
config.model.model_config.batch_size = batch_size
model_config = LlamaConfig(**config.model.model_config)
model_config.checkpoint_name_or_path = None
model_name = config.trainer.model_name
network = model_dict[model_name](
    config=model_config
)

if config.model.model_config.pet_config:
    print("----------------Init lora params----------------")
    pet_config = config.model.model_config.pet_config
    pet_config = LoraConfig(
        lora_rank=pet_config.lora_rank,
        lora_alpha=pet_config.lora_alpha,
        lora_dropout=pet_config.lora_dropout,
        target_modules=pet_config.target_modules
    )
    network = get_pet_model(network, pet_config)

model = Model(network)

# load checkpoint
if config.load_checkpoint:
    print("----------------Transform and load checkpoint----------------")
    seq_length = config.model.model_config.seq_length
    input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
    infer_data = network.prepare_inputs_for_predict_layout(input_ids)
    transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=config.processor.tokenizer.vocab_file
)

# init and run pipeline
pipeline_task = pipeline(task="text_generation", model=model, tokenizer=tokenizer)
outputs = pipeline_task(inputs,
                        do_sample=False,
                        top_k=1,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        temperature=1.0,
                        max_length=64)
for output in outputs:
    print(output)
```

#### 单卡推理

以Baichuan2-7B推理为例。

1. 修改`predict_baichuan2_7b.yaml`

   ```yaml
   load_checkpoint: 'path/to/baichuan2_7b_chat.ckpt'  # 权重路径
   use_parallel: False                                # 关闭并行模式
   auto_trans_ckpt: False                             # 关闭自动权重转换
   is_dynamic: True                                   # 开启动态推理
   model:
     model_config:
       use_past: True                                 # 开启增量推理

   processor:
     tokenizer:
       vocab_file: 'path/to/tokenizer.model'          # 词表路径
   ```

   若加载**分布式权重进行单卡推理**，则涉及将分布式权重转换为完整权重，参考以下配置修改相关参数。

   ```yaml
   # 需要将分布式权重转换为完整权重
   load_checkpoint: 'path/to/dist_model_dir'          # 分布式权重文件夹路径
   src_strategy_path_or_dir: 'strategy_path'          # 分布式策略文件路径
   auto_trans_ckpt: True                              # 开启自动权重转换
   ```

   分布式权重应按如下结构存放，以8卡分布式权重为例。

   ```text
   dist_model_dir
       ├── rank_0
       |   └── checkpoint_0.ckpt  # 权重名称可不同
       ├── rank_1
       |   └── checkpoint_1.ckpt
       ...
       └── rank_7
           └── checkpoint_7.ckpt
   ```

2. 执行命令

   ```shell
   # 运行示例代码
   python run_baichuan2_pipeline.py

   # 输出推理结果
   # {'text_generation_text': [<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>]}
   # {'text_generation_text': [<reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>]}
   # {'text_generation_text': [<reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>]}
   ```

#### 多卡推理

暂不支持

### 基于generate的推理

generate推理支持**自动权重转换**，支持加载**lora权重**。关于推理中使用lora权重的说明可以参考[基于pipeline的推理](#基于pipeline的推理)。

以下为基于generate接口的自定义推理代码示例。

```python
# run_baichuan2_generate.py
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.common import initializer as init

from mindformers import MindFormerConfig
from mindformers.models import LlamaConfig
from mindformers.pet import get_pet_model, LoraConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from baichuan2_7b import Baichuan7BV2ForCausalLM
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer

model_dict = {
    "baichuan2_7b": Baichuan7BV2ForCausalLM,
    "baichuan2_7b_lora": Baichuan7BV2ForCausalLM,
    "baichuan2_13b": Baichuan13BV2ForCausalLM,
    "baichuan2_13b_lora": Baichuan13BV2ForCausalLM
}

inputs = ["<reserved_106>你是谁？<reserved_107>",
          "<reserved_106>《静夜思》作者是？<reserved_107>",
          "<reserved_106>白日依山尽，下一句是？<reserved_107>",
          "<reserved_106>推荐一些杭州的美食？<reserved_107>"]
batch_size = len(inputs)

# init config with yaml
config_path = "path/to/predict_baichuan2_7b.yaml"
config = MindFormerConfig(config_path)

# init context
build_context(config)
build_parallel_config(config)

# init model
config.model.model_config.parallel_config = config.parallel_config
config.model.model_config.batch_size = batch_size
model_config = LlamaConfig(**config.model.model_config)
model_config.checkpoint_name_or_path = None
model_name = config.trainer.model_name
network = model_dict[model_name](
    config=model_config
)

if config.model.model_config.pet_config:
    print("----------------Init lora params----------------")
    pet_config = config.model.model_config.pet_config
    pet_config = LoraConfig(
        lora_rank=pet_config.lora_rank,
        lora_alpha=pet_config.lora_alpha,
        lora_dropout=pet_config.lora_dropout,
        target_modules=pet_config.target_modules
    )
    baichuan2_network = get_pet_model(network, pet_config)

model = Model(network)

# load checkpoint
if config.load_checkpoint:
    print("----------------Transform and load checkpoint----------------")
    seq_length = config.model.model_config.seq_length
    input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
    infer_data = network.prepare_inputs_for_predict_layout(input_ids)
    transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

# init tokenizer
tokenizer = Baichuan2Tokenizer(
    vocab_file=config.processor.tokenizer.vocab_file
)

# predict using generate
inputs_ids = tokenizer(inputs, max_length=64, padding="max_length")["input_ids"]
outputs = network.generate(inputs_ids,
                           do_sample=False,
                           top_k=1,
                           top_p=1.0,
                           repetition_penalty=1.0,
                           temperature=1.0,
                           max_length=64)
for output in outputs:
    print(tokenizer.decode(output))
```

#### 单卡推理

同[基于pipeline的单卡推理](#单卡推理)，执行代码替换为`run_baichuan2_generate.py`即可。

#### 多卡推理

同[基于pipeline的多卡推理](#多卡推理)

## 常见问题

### 显存不足

如出现显存不足问题，可适当减少`data_paralell`并增大`model_parallel`，如`data_parallel=4`、`model_paralllel=2`。
