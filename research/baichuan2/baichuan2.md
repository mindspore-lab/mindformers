# Baichuan2

## 模型描述

Baichuan2 是由百川智能开发的开源可商用的大规模预训练语言模型，基于 Transformer 结构，支持中英双语，上下文窗口长度为 4096。目前支持Baichuan2-7B和Baichuan2-13B模型，参数量分别为70亿和130亿。 本仓库提供了Baichuan2-7B和Baichuan2-13B预训练模型。

## 模型性能

|                                          config                                          |      task       | Datasets | [train performance](#全参微调) | [predict performance](#推理) |
|:----------------------------------------------------------------------------------------:|:---------------:|:--------:|:--------------------------:|:--------------------------:|
|       [finetune_baichuan2_7b](../../research/baichuan2/finetune_baichuan2_7b.yaml)       | text_generation |  belle   |      3010 tokens/s/p       |             /              |
|      [finetune_baichuan2_13b](../../research/baichuan2/finetune_baichuan2_13b.yaml)      | text_generation |  belle   |      1359 tokens/s/p       |             /              |
|  [finetune_baichuan2_7b_lora](../../research/baichuan2/finetune_baichuan2_7b_lora.yaml)  | text_generation |  belle   |      3375 tokens/s/p       |             /              |
| [finetune_baichuan2_13b_lora](../../research/baichuan2/finetune_baichuan2_13b_lora.yaml) | text_generation |  belle   |      1880 tokens/s/p       |             /              |
|        [predict_baichuan2_7b](../../research/baichuan2/predict_baichuan2_7b.yaml)        | text_generation |    /     |             /              |         42tokens/s         |
|       [predict_baichuan2_13b](../../research/baichuan2/predict_baichuan2_13b.yaml)       | text_generation |    /     |             /              |        23tokens/s-         |

## 仓库介绍

`Baichuan2` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：`research/baichuan2`

   ```text
   baichuan2
       ├── baichuan2_tokenizer.py    # tokenizer
       ├── baichuan2_7b.py           # 7B模型实现
       └── baichuan2_13b.py          # 13B模型实现
   ```

2. 模型配置：`research/baichuan2`

   ```text
   baichuan2
       ├── finetune_baichuan2_7b.yaml               # 7B全量微调配置
       ├── finetune_baichuan2_13b.yaml              # 13B全量微调配置
       ├── finetune_baichuan2_7b_lora.yaml          # 7B-lora微调配置
       ├── finetune_baichuan2_13b_lora.yaml         # 13B-lora微调配置
       ├── predict_baichuan2_7b.yaml                # 7B单卡推理配置
       └── predict_baichuan2_13b.yaml               # 13B单卡推理配置
   ```

3. 数据处理脚本和任务启动脚本：`research/baichuan2`

   ```text
   baichuan2
       ├── belle_preprocess.py     # belle数据集预处理脚本
       └── run_baichuan2.py        # baichuan2高阶接口使用脚本
       └── run_baichuan2_chat.py   # baichuan2 chat推理使用脚本
   ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.3.0
- MindFormers版本：r1.1.0
- 硬件支持矩阵

|      模型       |      硬件       | 全量微调 | lora微调 | 推理 |
|:-------------:|:-------------:|:----:|:------:|:--:|
| Baichuan2-7b  | Atlas 800T A2 | 单节点  |  单节点   | 单卡 |
| Baichuan2-13b | Atlas 800T A2 | 单节点  |  单节点   | 单卡 |

### 数据集准备

当前提供belle_chat_ramdon数据集的预处理和微调样例，用于对Baichuan2-7B-Base，Baichuan2-13B-Base模型进行微调。数据集下载链接如下：

- [belle_chat_ramdon_10k](https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/data/belle_chat_ramdon_10k.json)

执行`belle_preprocess.py`，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。

```shell
# 脚本路径：research/baichuan2/belle_preprocess.py
python research/baichuan2/belle_preprocess.py \
--input_glob /{path}/belle_chat_ramdon_10k.json \
--model_file /{path}/tokenizer.model \
--output_file /{path}/belle_chat_ramdon_10k_4096.mindrecord \
--seq_length 4096

# 参数说明
input_glob: 输入数据集路径
model_file: 词表文件路径
output_file: 输出数据集的路径和名称
seq_length: 生成数据集的序列长度
```

### 模型权重准备

本仓库提供已经转换完成的预训练权重、词表文件用于训练/微调/推理，用户可自行从下方链接拉取后直接使用，Base用于微调，Chat用于推理。

- [Baichuan2-7B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Base.ckpt)
- [Baichuan2-7B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt)
- [Baichuan2-13B-Base](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_13B_Base.ckpt)
- [Baichuan2-13B-Chat](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt)
- [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

也可选择从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

转换需要安装以下包

```shell
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf==3.20 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wandb==0.16.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载完成后，运行`/research/baichuan/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python ./research/baichuan/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME

# 参数说明
torch_ckpt_path: huggingface权重保存目录路径下任意权重bin文件，根据该文件路径读取目录下全部权重
mindspore_ckpt_path: mindspore权重文件保存路径
```

### [模型权重转换](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是完整权重。

- 基于完整权重进行多卡分布式训练，需要将完整权重转换为分布式权重。

- 基于训完的分布式权重进行单卡推理，需要将分布式权重转换为完整权重。
- 修改分布式策略训练，需要将权重转换为对应分布式权重。

Mindformer支持权重自动转换，详细教程请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

## 全参微调

使用Baichuan2-7B/13B-Base权重用于微调，默认`seq_length=4096`，分布式配置为`data_parallel=8`、`model_parallel=1`、`pipeline_stage=1`。以`belle_chat_ramdon_10k.json`数据集为例，给出了默认配置文件`finetune_baichuan2_7/13b.yaml`。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

### 数据集

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的belle数据集

### 预训练权重

请参照[模型权重准备](#模型权重准备)章节获取Baichuan2-7B/13B-Base权重。

### 单机训练

- 以Baichuan2-7B为例，Baichuan2-13B只需要修改配置文件和权重即可。

```shell
# 拷贝msrun运行脚本至baichuan2目录
cp scripts/msrun_launcher.sh research/baichuan2
# 使用msrun启动方式拉起分布式训练
cd research/baichuan2
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord"

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径或mindrecord文件路径
```

### 多机训练

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址， 所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

```bash
# 多机多卡自定义启动方式
bash msrun_launcher.sh "run_baichuan2.py \
--config {CONFIG_PATH} \
--run_mode {train/finetune/eval/predict} \
--xxx ..." \
WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK LOG_DIR JOIN CLUSTER_TIME_OUT

# 参数说明
WORKER_NUM: 参与分布式任务的Worker进程总数，一般等于总卡数
LOCAL_WORKER: 当前节点上拉起的Worker进程数，一般等于单节点卡数
MASTER_ADDR: 指定Scheduler的IP地址
MASTER_PORT: 指定Scheduler绑定端口号
NODE_RANK: 当前节点的索引
LOG_DIR：Worker以及Scheduler日志输出路径
JOIN：msrun是否等待Worker以及Scheduler退出
CLUSTER_TIME_OUT：集群组网超时时间，单位为秒
```

- 以双机训练Baichuan2-7B为例，13B只需要修改配置文件和权重即可。

**步骤**：

① 修改`finetune_baichuan2_7/13b.yaml`的分布式配置为双机配置

```yaml
context:
  runtime_num_threads: 1 # 新增配置

parallel_config:
  data_parallel: 8
  model_parallel: 1
  pipeline_stage: 2
  use_seq_parallel: False
  micro_batch_num: 8
```

② 设置以下环境变量

```bash
export MS_MEMORY_POOL_RECYCLE=1
export GE_NOT_CUT=1
```

③ msrun拉起双机训练

以下为双机存在共享盘，且工程代码在共享盘下的启动示例，**支持权重自动转换**。

```shell
# 拷贝msrun运行脚本至baichuan2目录
cp scripts/msrun_launcher.sh research/baichuan2
# 使用msrun启动方式拉起分布式训练
cd research/baichuan2

# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点0，节点ip为192.168.1.2，以192.168.1.1作为主节点，总共16卡且每个节点8卡，启动命令仅参数NODE_RANK不同
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

以下为双机不存在共享盘，工程代码在个子节点目录下的启动示例，预训练权重为使用权重转换工具离线转换得到，请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

```bash
# 拷贝msrun运行脚本至baichuan2目录
cp scripts/msrun_launcher.sh research/baichuan2
# 使用msrun启动方式拉起分布式训练
cd research/baichuan2

# 节点0，节点ip为192.168.1.1，作为主节点，总共16卡且每个节点8卡
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
16 8 192.168.1.1 8118 0 output/msrun_log False 300

# 节点0，节点ip为192.168.1.2，以192.168.1.1作为主节点，总共16卡且每个节点8卡，启动命令仅参数NODE_RANK不同
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base_dir \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_chat_ramdon_10k_4096.mindrecord" \
16 8 192.168.1.1 8118 1 output/msrun_log False 300
```

**注**：如出现显存不足问题，可适当减少`data_paralell`并增大`model_parallel`，如`data_parallel=4`、`model_paralllel=2`。

## Lora微调

LoRA（Low-Rank Adaptation）微调方法是一种针对预训练模型的高效微调技术，其核心思想是在预训练模型的基础上，通过注入可训练层（秩-分解矩阵）来实现模型性能的提升。

[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.09685.pdf)

### 数据集

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的belle数据集

### 预训练权重

请参照[模型权重准备](#模型权重准备)章节获取Baichuan2-7B/13B-Base权重。

### 单机训练

- 以Baichuan2-7B为例，Baichuan2-13B只需要修改配置文件和权重即可。

```shell
# 拷贝msrun运行脚本至baichuan2目录
cp scripts/msrun_launcher.sh research/baichuan2
# 使用msrun启动方式拉起分布式训练
cd research/baichuan2
bash msrun_launcher.sh \
"run_baichuan2.py \
--config finetune_baichuan2_7b_lora.yaml \
--load_checkpoint path/to/baichuan2_7b_base.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset path/to/belle_dataset_dir"

# 参数说明
config: 配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode: 运行模式，微调时设置为finetune
train_data: 训练数据集文件夹路径
```

如只需单独保存Lora权重，可在`finetune_baichuan2_7/13b_lora.yaml`中修改如下配置：

```yaml
callbacks:
  - type: CheckpointMonitor
    save_trainable_params: True
```

**Tips**：微调完成后，可使用lora权重合并工具将Lora权重和Base权重合并，请参考[Lora合并教程](https://gitee.com/mindspore/mindformers/blob/r1.1.0/docs/feature_cards/Transform_Lorackpt.md)。

## 推理

使用Baichuan2-7B/13B-Chat权重用于推理，默认`seq_length=4096`，支持单卡推理。

以Baichuan2-7B推理为例，Baichuan2-13B只需要修改配置文件和权重即可。

### 多轮对话推理

```bash
# 多轮对话方式只支持单卡推理
cd research/baichuan2
python run_baichuan2_chat.py \
--config predict_baichuan2_7b.yaml \
--load_checkpoint path/to/baichuan2_7b_base.ckpt

# 请输入：你是谁？
# 我是我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问
```

### 基于Pipeline推理

- 构建run_baichuan2_pipeline.py，支持**自动权重转换**，支持加载**Lora权重**。

如果加载lora权重进行推理，分以下情况讨论：

1、**已根据[Lora合并教程](https://gitee.com/mindspore/mindformers/blob/r1.1.0/docs/feature_cards/Transform_Lorackpt.md)将Lora权重合并**。此时权重和原始Chat权重参数层名一致，可使用`predict_baichuan2_7b.yaml`作为参数配置，`load_chjeckpoint`配置为合并权重路径。

2、**Lora权重未合并，权重里同时包含Base层和Lora层**，使用`predict_baichuan2_7b_lora.yaml`作为参数配置，`load_chjeckpoint`配置为Lora权重路径，权重为分布式则填写文件夹路径。

3、**只带Lora层的Lora权重**，使用`predict_baichuan2_7b_lora.yaml`作为参数配置，`load_chjeckpoint`配置为文件夹路径，文件夹下同时存放Baichuan2_7B_Lora.ckpt和Baichuan2_7B_Base.ckpt。**注**：若只带Lora层的Lora权重为分布式权重，需要首先使用权重转换工具进行合并为完整权重，请参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)。

```shell
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

# init config
yaml_path = "path/to/predict_baichuan2_7b.yaml"
config = MindFormerConfig(yaml_path)

# init context
build_context(config)
build_parallel_config(config)

# init model
config.model.model_config.parallel_config = config.parallel_config
config.model.model_config.batch_size = 1
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

#### **单卡推理**

- `predict_baichuan2_7b.yaml`主要参数配置参考

```yaml
# 使用完整权重
load_checkpoint: 'path/to/baichuan2_7b_chat.ckpt'            # 填写权重绝对路径
use_parallel: False                                          # 关闭并行模式
auto_trans_ckpt: False                                       # 关闭自动权重转换
use_past: True                                               # 使用增量推理
is_dynamic: True                                             # 使用动态推理

processor:
  tokenizer:
    vocab_file: 'path/to/tokenizer.model'                    # 配置词表路径
```

如果加载**分布式权重进行单卡推理**，则涉及将分布式权重转换为完整权重，参考以下配置修改相关参数。

```yaml
# 需要将分布式权重转换为完整权重
load_checkpoint: 'model_dir'              # 分布式权重文件夹路径
src_strategy_path_or_dir: 'strategy_path' # 填写分布式策略文件路径
auto_trans_ckpt: True                     # 打开自动权重转换
```

分布式权重存放格式，以8卡权重为例：

```text
model_dir
    ├── rank_0
      └── checkpoint_0.ckpt # 权重名称可任意
    ├── rank_1
      └── checkpoint_1.ckpt
    ...
    ├── rank_7
      └── checkpoint_7.ckpt
```

2. 运行run_baichuan2_pipeline.py

```bash
cd research/baichuan2
python run_baichuan2_pipeline.py

# 推理输出
# {'text_generation_text': [<reserved_106>你是谁？<reserved_107>我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问</s>]}
# {'text_generation_text': [<reserved_106>《静夜思》作者是？<reserved_107>《静夜思》的作者是唐代著名诗人李白。这是一首描绘夜晚思乡之情的诗篇，表达了作者对故乡的思念和对亲人的牵挂之情。</s>]}
# {'text_generation_text': [<reserved_106>白日依山尽，下一句是？<reserved_107>黄河入海流。</s>]}
```

#### 多卡推理

暂不支持

### 基于Generate推理

基于Generate推理使用方式可完全参考[基于Pipeline推理](#基于Pipeline推理)，只需要修改脚本文件。

- Generate推理脚本参考，支持**自动权重转换**，支持加载**Lora权重**。

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

# init config，默认使用Atlas 800配置文件
yaml_path = "path/to/predict_baichuan2_7b.yaml"
config = MindFormerConfig(yaml_path)

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
