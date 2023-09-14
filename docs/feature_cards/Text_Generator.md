# Text Generator

Mindformers大模型套件提供了text generator方法，旨在让用户能够便捷地使用生成类语言模型进行文本生成任务，包括但不限于解答问题、填充不完整文本或翻译源语言到目标语言等。

当前该方法支持Minformers大模型套件中6个生成类语言模型

## [Text Generator支持度表](../model_support_list.md#text-generator支持度表)

## 增量推理

Mindformers大模型套件的`text generator`方法支持增量推理逻辑，该逻辑旨在加快用户在调用`text generator`方法进行文本生成时的文本生成速度。

在此提供使用高阶接口进行各模型增量推理的**测试样例脚本**：

```python
# mindspore设置图模式和环境
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers import AutoConfig, AutoModel, AutoTokenizer

# 按需设置模型类型名，高阶接口将根据类型名实例化相应模型
model_type = "glm_6b"
# 按需设置测试的输入文本
input_text = "中国的首都是哪个城市？"

# 获取模型默认配置项并按需修改
config = AutoConfig.from_pretrained(model_type)
# use_past设置为True时为增量推理，反之为自回归推理
config.use_past = True
# 修改batch_size和模型seq_length
config.batch_size = 1; config.seq_length=512

# 根据配置项实例化模型
model = AutoModel.from_config(config)
# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type)
# 对输入进行tokenizer编码
input_ids = tokenizer(input_text)["input_ids"]
# 调用model.generate接口执行增量推理
output = model.generate(input_ids, max_length=128, do_sample=False)
# 解码并打印输出
print(tokenizer.decode(output))
```

> 注：
>
> 1. 首次调用generate时需要进行mindspore图编译，耗时较长；在统计在线推理的文本生成速度时，可以多次重复调用并排除首次调用的执行时间
> 2. 使用增量推理(use_past=True)时的生成速度预期快于自回归推理(use_past=False)

## Batch推理

`text generator`方法也支持同时对多个输入样本进行batch推理；在单batch推理算力不足的情况下，多batch推理能够提升推理时的吞吐率

以下给出测试batch推理能力的**标准测试脚本**，仅上述增量推理测试脚本仅有少数区别

```python
import mindspore;mindspore.set_context(mode=0, device_id=0)
from mindformers import AutoConfig, AutoModel, AutoTokenizer

model_type = "glm_6b"
# 多batch输入文本
input_text = [
    "中国的首都是哪个城市？",
    "你好",
    "请介绍一下华为",
    "I love Beijing, because"
]
# 是否使用增量推理
use_past = True
# 预设模型seq_length
seq_len = 512

config = AutoConfig.from_pretrained(model_type)
# 将batch size修改为输入的样本数
config.batch_size = len(input_text)
config.use_past = use_past
config.seq_length = seq_len

model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_type)

# 对输入batch进行pad
input_ids = tokenizer(input_text, max_length=config.seq_length, padding="max_length")["input_ids"]
output = model.generate(input_ids, max_length=128, do_sample=False)
print(tokenizer.decode(output))
```

> 注：
> batch推理的推理吞吐率提升表现与设备计算负荷相关；在seq_len较短并开启增量推理的情况下，计算负荷较小，使用batch推理通常会获得较好的提升

## 流式推理

Mindformers大模型套件提供Streamer类，旨在用户在调用text generator方法进行文本生成时能够实时看到生成的每一个词，而不必等待所有结果均生成结束。

实例化streamer并向text generator方法传入该实例：

```python
from mindformers import AutoModel, AutoTokenizer, TextStreamer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

streamer = TextStreamer(tok)

_ = model.generate(inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
# 'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```

上述脚本不再对输出文本进行统一解码打印，而是每生成一个中间结果就由streamer实时打印

## 分布式推理

**说明:** 由于MindSpore版本问题，分布式推理仅支持MindSpore 2.0及以上版本，且暂不支持流水并行推理模式。

[分布式推理参考用例](../model_cards/bloom.md#a1-模型并行推理以1机8卡推理bloom_71b为例)
