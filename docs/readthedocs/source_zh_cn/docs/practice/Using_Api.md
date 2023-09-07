# 高阶接口使用样例

## AutoClass

MindFormers大模型套件提供了AutoClass类，包含AutoConfig、AutoModel、AutoTokenizer、AutoProcessor4个便捷高阶接口，方便用户调用套件中已封装的API接口。上述4类提供了相应领域模型的ModelConfig、Model、Tokenzier、Processor的实例化功能。

### AutoConfig&&AutoModel

使用AutoConfig和AutoModel自动实例化MindFormers中支持的模型配置或模型架构：

```python
from mindformers import AutoConfig, AutoModel

gpt_config = AutoConfig.from_pretrained("gpt2")
gpt_model = AutoModel.from_pretrained("gpt2")  # 自动加载预置权重到网络中
```

使用已有模型配置或网络架构实例化相应的模型配置或网络架构实例：

```python
from mindformers import GPT2LMHeadModel, GPT2Config

gpt_13b_config = GPT2Config.from_pretrained("gpt2_13b")
gpt_13b_model = GPT2LMHeadModel.from_pretrained("gpt2_13b")  # 自动加载预置权重到网络中
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

### init_context

mindspore相关环境的初始化，MindFormers中提供了init_context标准接口帮助用户完成单卡或多卡并行环境的初始化：

[init_context](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/core/context/build_context.py#L53)  [ContextConfig](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/config_args.py#L26) [ParallelContextConfig](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/config_args.py#L26)

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
    parallel_config = ParallelContextConfig(parallel_mode='DATA_PARALLEL', gradients_mean=True,
                                            enable_parallel_optimizer=False)
    rank_id, device_num = init_context(use_parallel=True, context_config=context_config,
                                       parallel_config=parallel_config)
```

多卡半自动并行模式初始化：

```python
from mindformers import init_context, ContextConfig, ParallelContextConfig


def context_init():
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL', gradients_mean=False,
                                            enable_parallel_optimizer=False, full_batch=True)
    rank_id, device_num = init_context(use_parallel=True, context_config=context_config,
                                       parallel_config=parallel_config)
```

### TrainingArguments&&Trainer

MindFormers套件对用户提供了`TrainingArguments`类，用于自定义大模型训练过程中的各类参数，支持参数详见：[TrainingArguments](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/training_args.py)

同时，MindFormers也提供了`Trainer`高阶接口，用于大模型任务的开发、训练、微调、评估、推理等流程；

使用TrainingArguments自定义大模型训练过程参数：

```python
from mindformers import TrainingArguments

# 环境初始化，参考上述`init_context`章节实现
context_init()
# 训练超参数定义
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000,
                                  sink_mode=True)
```

使用Trainer接口创建内部预置任务：数据集按照官方教程准备[GPT预训练数据集准备](../model_cards/gpt2.md)，自定义训练参数

```python
from mindformers import Trainer, TrainingArguments

# 环境初始化，参考上述`init_context`章节实现
context_init()
# 训练超参数定义
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000,
                                  sink_mode=True)

text_generation = Trainer(task='text_generation', model='gpt2', args=training_args, train_dataset='./train',
                          eval_dataset='./eval')
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
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000,
                                  sink_mode=True)
# 自定义模型
pangu_model = AutoModel.from_pretrained("pangualpha_2_6b")
# 自定义数据集
dataset = GeneratorDataset(generator, column_names=["input_ids"])
train_dataset = dataset.batch(batch_size=4)
eval_dataset = dataset.batch(batch_size=4)
# 定义文本生成任务，传入自定义模型、数据集、超参数
text_generation = Trainer(task='text_generation', model=pangu_model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)
```

### 并行&&重计算配置

MindFormers的Trainer接口提供了并行的配置接口`set_parallel_config`和重计算配置接口`set_recompute_config`，其中`set_parallel_config`接口仅在**半自动并行**
或**全自动并行模式**下生效，同时需要模型本身已支持或已配置[并行策略](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0/parallel/introduction.html);

[set_parallel_config](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L690)  [set_recompute_config](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/trainer/trainer.py#L731)

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
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001, warmup_steps=1000,
                                  sink_mode=True)
# 自定义模型
pangu_config = PanguAlphaConfig(hidden_size=768, ffn_hidden_size=768 * 4, num_layers=12, num_heads=12,
                                checkpoint_name_or_path='')
pangu_model = PanguAlphaHeadModel(pangu_config)
# 自定义数据集
dataset = GeneratorDataset(generator, column_names=["input_ids"])
train_dataset = dataset.batch(batch_size=4)
eval_dataset = dataset.batch(batch_size=4)
# 定义文本生成任务，传入自定义模型、数据集、超参数
text_generation = Trainer(task='text_generation', model=pangu_model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)

# 设定并行策略，比如2机16卡,设定数据并行4 模型并行2 流水并行2 微批次大小为2 打开优化器并行
text_generation.set_parallel_config(data_parallel=4, model_parallel=2, pipeline_stage=2, micro_batch_num=2,
                                    optimizer_shard=True)

# 设置重计算配置，打开重计算
text_generation.set_recompute_config(recompute=True)
```
