# MindFormers大模型使用教程

## 配套版本

| 版本对应关系 | MindFormers | MindPet | MindSpore |  Python   |        芯片        |
| :----------: | :---------: | :-----: | :-------: | :-------: | :----------------: |
|    版本号    |     dev     |  1.0.0  | 2.0/1.10  | 3.7.5/3.9 | Ascend910A NPU/CPU |

## 支持镜像

### 裸金属镜像

* docker下载命令

```shell
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_2_0:mindformers_0.6.0dev_20230616_py39_37
```

* 创建容器

```shell
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_2_0:mindformers_0.6.0dev_20230616_py39_37 \
/bin/bash
```

### AICC镜像

**详情请参考[MindFormers AICC使用教程](https://gitee.com/mindspore/mindformers/blob/dev/docs/aicc_cards/aicc_tutorial.md)**

我们在[镜像仓库网 (hqcases.com)](https://gitee.com/link?target=http%3A%2F%2Fai.hqcases.com%2Fmirrors.html)上发布了一些经过验证的**标准镜像版本**，可以通过几行简单的docker命令的形式，直接使用验证过的标准镜像拉起MindFormers套件的训练任务，而无需进行较为繁琐的自定义镜像并上传的步骤。

- 镜像列表

```
1. swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_1_10_1:mindformers_0.6.0dev_20230615_py39
```

- 在一台准备好docker引擎的计算机上，root用户执行docker pull命令拉取该镜像

```
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_1_10_1:mindformers_0.6.0dev_20230615_py39
```

## 模型矩阵

**此处给出了MindFormers套件中支持的任务名称和模型名称，用于高阶开发时的索引名**

|                             模型                             |                      任务（task name）                       | 模型（model name）                                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
| [BERT](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bert.md) | masked_language_modeling [text_classification](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/text_classification.md) [token_classification](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/token_classification.md) [question_answering](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/question_answering.md) | bert_base_uncased <br>txtcls_bert_base_uncased<br>txtcls_bert_base_uncased_mnli <br>tokcls_bert_base_chinese<br>tokcls_bert_base_chinese_cluener <br>qa_bert_base_uncased<br>qa_bert_base_chinese_uncased |
| [T5](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/t5.md) |                         translation                          | t5_small                                                     |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |                       text_generation                        | gpt2_small <br>gpt2_13b <br>gpt2_52b                         |
| [PanGuAlpha](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/pangualpha.md) |                       text_generation                        | pangualpha_2_6_b<br>pangualpha_13b                           |
| [GLM](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md) |                       text_generation                        | glm_6b<br>glm_6b_lora                                        |
| [LLama](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |                       text_generation                        | llama_7b <br>llama_13b <br>llama_65b <br>llama_7b_lora       |
|                            [Bloom](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md)                             |                       text_generation                        | bloom_560m<br>bloom_7.1b <br>bloom_65b<br>bloom_176b         |
| [MAE](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/mae.md) |                    masked_image_modeling                     | mae_vit_base_p16                                             |
| [VIT](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/vit.md) | [image_classification](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/image_classification.md) | vit_base_p16                                                 |
| [Swin](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/swin.md) | [image_classification](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/image_classification.md) | swin_base_p4w7                                               |
| [CLIP](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/clip.md) | [contrastive_language_image_pretrain](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/contrastive_language_image_pretrain.md), [zero_shot_image_classification](https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/zero_shot_image_classification.md) | clip_vit_b_32<br>clip_vit_b_16 <br>clip_vit_l_14<br>clip_vit_l_14@336 |

**核心关键模型能力一览表：**

| 关键模型 |             并行模式             | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 | 预训练 |        微调        |      评估      |     推理 | 是否上库 |
| :------: | :------------------------------: | :------: | :--------: | :------: | :------: | :--------: | ------ | :----------------: | :------------: | -------: | :------: |
|   GPT    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 单卡推理 |    是    |
|  PanGu   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |    PPL评估     | 单卡推理 |    是    |
|  Bloom   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     |      全参微调      |     不支持     | 单卡推理 |    是    |
|  LLaMa   | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 |    PPL评估     | 单卡推理 |    是    |
|   GLM    | data_parallel\semi_auto_parallel |    是    |     是     |    是    |    是    |     是     | 是     | 全参微调，Lora微调 | Blue/Rouge评估 | 单卡推理 |    是    |

**其余库上模型分布式支持情况一览表：**

| 模型 |   并行模式    | 数据并行 | 优化器并行 | 模型并行 | 流水并行 | 多副本并行 | 是否上库 |
| :--: | :-----------: | :------: | :--------: | :------: | :------: | :--------: | :------: |
| MAE  | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |
|  T5  | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |
| Bert | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |
| Swin | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |
| VIT  | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |
| CLIP | data_parallel |    是    |     是     |    否    |    否    |     否     |    是    |

## AutoClass

MindFormers大模型套件提供了AutoClass类，包含AutoConfig、AutoModel、AutoTokenizer、AutoProcessor4个便捷高阶接口，方便用户调用套件中已封装的API接口。上述4类提供了相应领域模型的ModelConfig、Model、Tokenzier、Processor的实例化功能。

![输入图片说明](https://foruda.gitee.com/images/1686128219487920380/00f18fec_9324149.png)

| AutoClass类   | from_pretrained属性（实例化功能） | from_config属性（实例化功能） | save_pretrained（保存配置功能） |
| ------------- | --------------------------------- | ----------------------------- | :-----------------------------: |
| AutoConfig    | √                                 | ×                             |                √                |
| AutoModel     | √                                 | √                             |                √                |
| AutoProcessor | √                                 | ×                             |                √                |
| AutoTokenizer | √                                 | ×                             |                √                |

### AutoConfig&&AutoModel

使用AutoConfig和AutoModel自动实例化MindFormers中支持的模型配置或模型架构：

```python
from mindformers import AutoConfig, AutoModel

gpt_config = AutoConfig.from_pretrained("gpt2")
gpt_model = AutoModel.from_pretrained("gpt2") # 自动加载预置权重到网络中
```

使用已有模型配置或网络架构实例化相应的模型配置或网络架构实例：

```python
from mindformers import GPT2LMHeadModel, GPT2Config

gpt_13b_config = GPT2Config.from_pretrained("gpt2_13b")
gpt_13b_model = GPT2LMHeadModel.from_pretrained("gpt2_13b") # 自动加载预置权重到网络中
```

使用已有模型配置或模型架构进行二次开发：

```python
from mindformers import GPT2LMHeadModel, GPT2Config

gpt_config = GPT2Config(hidden_size=768, num_layers=12)
gpt_model = GPT2LMHeadModel(gpt_config)
```

### AutoTokenizer

使用AutoTokenizer自动获取文本分词器：

```python
from mindformers import AutoTokenizer

gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_tokenizer("hello!")
```

```text
{'input_ids': [50256, 31373, 0, 50256], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

使用已有Tokenizer函数对文本进行分词：

```python
from mindformers import GPT2Tokenizer

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer("hello!")
```

```text
{'input_ids': [50256, 31373, 0, 50256], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
```

### AutoProcessor

使用AutoProcessor自动获取模型数据预处理过程：

```python
from mindformers import AutoProcessor

gpt_processor = AutoProcessor.from_pretrained("gpt2")
gpt_processor("hello!")
```

```text
{'text': Tensor(shape=[1, 128], dtype=Int32, value=
[[50256, 31373,     0 ... 50256, 50256, 50256]])}
```

使用已有模型数据预处理过程对相应数据做模型输入前的预处理：

```python
from mindformers import GPT2Processor

gpt_processor = GPT2Processor.from_pretrained('gpt2')
gpt_processor("hello!")
```

```text
{'text': Tensor(shape=[1, 128], dtype=Int32, value=
[[50256, 31373,     0 ... 50256, 50256, 50256]])}
```

使用已有数据预处理过程进行二次开发：

```python
from mindformers import GPT2Processor, GPT2Tokenizer

# 自定义 tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt_processor = GPT2Processor(tokenizer=tok, max_length=256, return_tensors='ms')
gpt_processor("hello!")
```

```text
{'text': Tensor(shape=[1, 256], dtype=Int32, value=
[[50256, 31373,     0 ... 50256, 50256, 50256]])}
```

## pipeline

MindFormers大模型套件面向任务设计pipeline推理接口，旨在让用户可以便捷的体验不同AI领域的大模型在线推理服务，当前已集成10+任务的推理流程；

![输入图片说明](https://foruda.gitee.com/images/1673432339378334189/fb24c2fe_9324149.png)

### pipeline使用样例

使用MindFormers预置任务和模型开发一个推理流：

```python
from mindformers import pipeline

text_generation = pipeline(task='text_generation', model='gpt2', max_length=50)
text_generation("I love Beijing, because", top_k=3)
```

```text
[{'text_generation_text': ['I love Beijing, because of all its beautiful scenery and beautiful people," said the mayor of Beijing, who was visiting the capital for the first time since he took office.\n\n"The Chinese government is not interested in any of this," he added']}]
```

使用自定义的模型、tokenizer等进行任务推理：

```python
from mindformers import pipeline
from mindformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

text_generation = pipeline(task='text_generation', model=gpt_model, tokenizer=tok, max_length=50)
text_generation("I love Beijing, because", top_k=3)
```

```text
[{'text_generation_text': ['I love Beijing, because of all its beautiful scenery and beautiful people," said the mayor of Beijing, who was visiting the capital for the first time since he took office.\n\n"The Chinese government is not interested in any of this," he added']}]
```

## Trainer

MindFormers大模型套件面向任务设计Trainer接口，旨在让用户可以快速使用我们预置任务和模型的训练、微调、评估、推理能力，当前已集成10+任务和10+模型的全流程开发能力；

![输入图片说明](https://foruda.gitee.com/images/1673431864815390341/da621a72_9324149.png)

### init_context

mindspore相关环境的初始化，MindFormers中提供了init_context标准接口帮助用户完成单卡或多卡并行环境的初始化：

[init_context]()  [ContextConfig]() [ParallelContextConfig]()

单卡初始化：

```python
from mindformers import init_context, ContextConfig

def context_init():
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    rank_id, device_num = init_context(use_parallel=False, context_config=context_config)
```

多卡数据并行模式初始化：

```python
from mindformers import init_context, ContextConfig, ParallelContextConfig
def context_init():
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = ParallelContextConfig(parallel_mode='DATA_PARALLEL', gradients_mean=True, enable_parallel_optimizer=False)
    rank_id, device_num = init_context(use_parallel=True, context_config=context_config, parallel_config=parallel_config)
```

多卡半自动并行模式初始化：

```python
from mindformers import init_context, ContextConfig, ParallelContextConfig

def context_init():
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL', gradients_mean=False, enable_parallel_optimizer=False, full_batch=True)
    rank_id, device_num = init_context(use_parallel=True, context_config=context_config, parallel_config=parallel_config)
```

### TrainingArguments&&Trainer

MindFormers套件对用户提供了`TrainingArguments`类，用于自定义大模型训练过程中的各类参数，支持参数详见：[TrainingArguments]()

同时，MindFormers也提供了`Trainer`高阶接口，用于大模型任务的开发、训练、微调、评估、推理等流程；

使用TrainingArguments自定义大模型训练过程参数：

```python
from mindformers import TrainingArguments

# 环境初始化，参考上述`init_context`章节实现
context_init()
# 训练超参数定义
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000, sink_mode=True)
```

使用Trainer接口创建内部预置任务：数据集按照官方教程准备[GPT预训练数据集准备]()，自定义训练参数

```python
from mindformers import Trainer, TrainingArguments

# 环境初始化，参考上述`init_context`章节实现
context_init()
# 训练超参数定义
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000, sink_mode=True)

text_generation = Trainer(task='text_generation', model='gpt2', args=training_args, train_dataset='./train', eval_dataset='./eval')
```

使用Trainer接口创建内部预置任务：自定义数据集，模型，训练参数

```python
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments, AutoModel

def generator():
    """text dataset generator."""
    seq_len = 1025
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(512):
        yield input_ids


# 环境初始化，参考上述`init_context`章节实现
context_init()
# 自定义训练超参数
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000, sink_mode=True)
# 自定义模型
pangu_model = AutoModel.from_pretrained("pangualpha_2_6b")
# 自定义数据集
dataset = GeneratorDataset(generator, column_names=["input_ids"])
train_dataset = dataset.batch(batch_size=4)
eval_dataset = dataset.batch(batch_size=4)
# 定义文本生成任务，传入自定义模型、数据集、超参数
text_generation = Trainer(task='text_generation', model=pangu_model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
```

### 并行&&重计算配置

MindFormers的Trainer接口提供了并行的配置接口`set_parallel_config`和重计算配置接口`set_recompute_config`，其中`set_parallel_config`接口仅在**半自动并行**或**全自动并行模式**下生效，同时需要模型本身已支持或已配置[并行策略]();

[set_parallel_config]()  [set_recompute_config]()

使用Trainer高阶接口，自定义并行和重计算配置：

```python
import numpy as np
from mindspore.dataset import GeneratorDataset

from mindformers import Trainer, TrainingArguments
from mindformers import PanguAlphaHeadModel, PanguAlphaConfig

def generator():
    """text dataset generator."""
    seq_len = 1025
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(512):
        yield input_ids


# 环境初始化，参考上述`init_context`章节实现
context_init()
# 自定义训练超参数
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000, sink_mode=True)
# 自定义模型
pangu_config = PanguAlphaConfig(hidden_size=768, ffn_hidden_size=768 * 4, num_layers=12, num_heads=12,
                                checkpoint_name_or_path='')
pangu_model = PanguAlphaHeadModel(pangu_config)
# 自定义数据集
dataset = GeneratorDataset(generator, column_names=["input_ids"])
train_dataset = dataset.batch(batch_size=4)
eval_dataset = dataset.batch(batch_size=4)
# 定义文本生成任务，传入自定义模型、数据集、超参数
text_generation = Trainer(task='text_generation', model=pangu_model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)

# 设定并行策略，比如2机16卡,设定数据并行4 模型并行2 流水并行2 微批次大小为2 打开优化器并行
text_generation.set_parallel_config(data_parallel=4, model_parallel=2, pipeline_stage=2, micro_batch_num=2, optimizer_shard=True)

# 设置重计算配置，打开重计算
text_generation.set_recompute_config(recompute=True)
```

### 训练&&微调&&评估&&推理

MindFormers套件的Trainer高阶接口提供了`train`、`finetune`、`evaluate`、`predict`4个关键属性函数，帮助用户快速拉起任务的训练、微调、评估、推理流程：[Trainer.train]() [Trainer.finetune]() [Trainer.evaluate]() [Trainer.predict]()  

使用`Trainer.train` `Trainer.finetune` `Trainer.evaluate` `Trainer.predict` 拉起任务的训练、微调、评估、推理流程，以下为使用`Trainer`高阶接口进行全流程开发的使用样例（多卡分布式并行），命名为`task.py`：

```python
import argparse

from mindformers import Trainer, TrainingArguments
from mindformers import init_context, ContextConfig, ParallelContextConfig

def context_init(use_parallel=False, optimizer_parallel=False):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    rank_id, device_num = init_context(use_parallel=use_parallel,
                                       context_config=context_config,
                                       parallel_config=parallel_config)

def main(use_parallel=False,
         run_mode='train',
         task='text_generation',
         model_type='gpt2',
         pet_method='',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='hello!',
         dp=1, mp=1, pp=1, micro_size=1, op=False):
    # 环境初始化
    context_init(use_parallel, op)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=2, learning_rate=0.001, warmup_steps=100, sink_mode=True, sink_size=2)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task,
                   model=model_type,
                   pet_method=pet_method,
                   args=training_args,
                   train_dataset=train_dataset,
                   eval_dataset=eval_dataset)
    task.set_parallel_config(data_parallel=dp,
                             model_parallel=mp,
                             pipeline_stage=pp,
                             optimizer_shard=op,
                             micro_batch_num=micro_size)
    if run_mode == 'train':
        task.train()
    elif run_mode == 'finetune':
        task.finetune()
    elif run_mode == 'eval':
        task.evaluate()
    elif run_mode == 'predict':
        # 推理，仅支持单卡推理
        assert use_parallel == False, "only support predict under stand_alone mode."
        result = task.predict(input_data=predict_data)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='train', required=True, help='set run mode for model.')
    parser.add_argument('--use_parallel', default=False, help='open parallel for model.')
    parser.add_argument('--task', default='text_generation', required=True, help='set task type.')
    parser.add_argument('--model_type', default='gpt2', required=True, help='set model type.')
    parser.add_argument('--train_dataset', default=None, help='set train dataset.')
    parser.add_argument('--eval_dataset', default=None, help='set eval dataset.')
    parser.add_argument('--pet_method', default='', help="set finetune method, now support type: ['', 'lora']")
    parser.add_argument('--data_parallel', default=1, type=int,help='set data parallel number. Default: None')
    parser.add_argument('--model_parallel', default=1, type=int, help='set model parallel number. Default: None')
    parser.add_argument('--pipeline_parallel', default=1, type=int, help='set pipeline parallel number. Default: None')
    parser.add_argument('--micro_size', default=1, type=int, help='set micro batch number. Default: None')
    parser.add_argument('--optimizer_parallel', default=False, type=bool, help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()
    main(run_mode=args.run_mode,
         task=args.task,
         use_parallel=args.use_parallel,
         model_type=args.model_type,
         pet_method=args.pet_method,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         dp=args.data_parallel,
         mp=args.model_parallel,
         pp=args.pipeline_parallel,
         micro_size=args.micro_size,
         op=args.optimizer_parallel)
```

* 单卡使用样例：

```shell
# 训练
python task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode train

# 评估
python task.py --task text_generation --model_type gpt2 --eval_dataset ./eval --run_mode eval

# 微调，支持
python task.py --task text_generation --model_type gpt2 --train_dataset ./finetune --pet_method lora --run_mode finetune

# 推理
python task.py --task text_generation --model_type gpt2 --predict_data 'hello!' --run_mode predict
```

* 多卡分布式使用样例：

  * 单机多卡标准启动脚本：`run_distribute_single_node.sh`

    ```bash
    #!/bin/bash
    # Copyright 2023 Huawei Technologies Co., Ltd
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ============================================================================
    
    if [ $# != 4 ]
    then
      echo "Usage Help: bash run_distribute_single_node.sh [EXECUTE_ORDER] [RANK_TABLE_PATH]  [DEVICE_RANGE] [RANK_SIZE] For Multiple Devices In Single Machine"
      exit 1
    fi
    
    check_real_path(){
      if [ "${1:0:1}" == "/" ]; then
        echo "$1"
      else
        echo "$(realpath -m $PWD/$1)"
      fi
    }
    
    EXECUTE_ORDER=$1
    RANK_TABLE_PATH=$(check_real_path $2)
    DEVICE_RANGE=$3
    
    DEVICE_RANGE_LEN=${#DEVICE_RANGE}
    DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
    PREFIX=${DEVICE_RANGE%%","*}
    INDEX=${#PREFIX}
    START_DEVICE=${DEVICE_RANGE:0:INDEX}
    END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}
    
    if [ ! -f $RANK_TABLE_PATH ]
    then
        echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
    exit 1
    fi
    
    
    if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: start_device=$START_DEVICE is not a number"
    exit 1
    fi
    
    if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: end_device=$END_DEVICE is not a number"
    exit 1
    fi
    
    ulimit -u unlimited
    
    export RANK_SIZE=$4
    export RANK_TABLE_FILE=$RANK_TABLE_PATH
    
    shopt -s extglob
    
    for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
    do
        export DEVICE_ID=${i}
        export RANK_ID=$((i-START_DEVICE))
        mkdir -p ./output/log/rank_$RANK_ID
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
    done
    
    shopt -u extglob
    ```

  * 多机多卡标准启动脚本：`run_distribute_multi_node.sh`

    ```bash
    #!/bin/bash
    # Copyright 2023 Huawei Technologies Co., Ltd
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ============================================================================
    
    if [ $# != 4 ]
    then
      echo "Usage Help: bash run_distribute_multi_node.sh [EXECUTE_ORDER] [RANK_TABLE_FILE] [DEVICE_RANGE] [RANK_SIZE]"
      exit 1
    fi
    
    check_real_path(){
      if [ "${1:0:1}" == "/" ]; then
        echo "$1"
      else
        echo "$(realpath -m $PWD/$1)"
      fi
    }
    
    EXECUTE_ORDER=$1
    RANK_TABLE_PATH=$(check_real_path $2)
    DEVICE_RANGE=$3
    
    DEVICE_RANGE_LEN=${#DEVICE_RANGE}
    DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
    PREFIX=${DEVICE_RANGE%%","*}
    INDEX=${#PREFIX}
    START_DEVICE=${DEVICE_RANGE:0:INDEX}
    END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}
    
    if [ ! -f $RANK_TABLE_PATH ]
    then
        echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
    exit 1
    fi
    
    if [[ ! $START_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: start_device=$START_DEVICE is not a number"
    exit 1
    fi
    
    if [[ ! $END_DEVICE =~ ^[0-9]+$ ]]; then
        echo "error: end_device=$END_DEVICE is not a number"
    exit 1
    fi
    
    ulimit -u unlimited
    
    export RANK_SIZE=$4
    export RANK_TABLE_FILE=$RANK_TABLE_PATH
    
    shopt -s extglob
    for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
    do
        export RANK_ID=${i}
        export DEVICE_ID=$((i-START_DEVICE))
        echo "start training for rank $RANK_ID, device $DEVICE_ID"
        mkdir -p ./output/log/rank_$RANK_ID
        $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
    done
    
    shopt -u extglob
    ```

  * 分布式并行执行`task.py`样例：需提前生成`RANK_TABLE_FILE`，同时`task.py`中默认使用**半自动并行模式**。

    **注意单机时使用{single}，多机时使用{multi}**

    ```shell
    # 分布式训练
    bash run_distribute_{single/multi}_node.sh "python task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode train --use_parallel True --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_192.168.89.35.json [0,4] 4
    
    # 分布式评估
    bash run_distribute_{single/multi}_node.sh "python task.py --task text_generation --model_type gpt2 --eval_dataset ./eval --run_mode eval --use_parallel True --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_192.168.89.35.json [0,4] 4
    
    # 分布式微调
    bash run_distribute_{single/multi}_node.sh "python task.py --task text_generation --model_type gpt2 --train_dataset ./train --run_mode finetune --pet_method lora --use_parallel True --data_parallel 1 --model_parallel 2 --pipeline_parallel 2 --micro_size 2" hccl_4p_0123_192.168.89.35.json [0,4] 4
    
    # 分布式推理，暂不支持, 630支持特性
    ```