# Text Generator

Mindformers大模型套件提供了text generator方法，旨在让用户能够便捷地使用生成类语言模型进行文本生成任务，包括但不限于解答问题、填充不完整文本或翻译源语言到目标语言等。

当前该方法支持Minformers大模型套件中6个生成类语言模型

|    model    |                         模型文档链接                         | 增量推理 | 流式推理 |
| :---------: | :----------------------------------------------------------: | :------: | :------: |
|    bloom    | [link](../model_cards/bloom.md) |    √     |    √     |
|     GLM     | [link](../model_cards/glm.md) |    √     |    √     |
|     GPT     | [link](../model_cards/gpt2.md) |    ×     |    √     |
|    llama    | [link](../model_cards/llama.md) |    √     |    √     |
| pangu-alpha | [link](../model_cards/pangualpha.md) |    ×     |    √     |
|     T5      | [link](../model_cards/t5.md) |    ×     |    √     |

## 增量推理

Mindformers大模型套件的`text generator`方法支持增量推理逻辑，该逻辑旨在加快用户在调用`text generator`方法进行文本生成时的文本生成速度。

通过实例化的模型调用：

```python
from mindspore import set_context
from mindformers import GLMChatModel, ChatGLMTokenizer, GLMConfig
set_context(mode=0)
# use_past设置成True时为增量推理，反之为自回归推理
glm_config = GLMConfig(use_past=True, checkpoint_name_or_path="glm_6b")
glm_model = GLMChatModel(glm_config)
tokenizer = ChatGLMTokenizer.from_pretrained("glm_6b")
words = "中国的首都是哪个城市？"
words = tokenizer(words)['input_ids']
output = glm_model.generate(words, max_length=20, top_k=1)
output = tokenizer.decode(output[0], skip_special_tokens=True)
print(output)
# 中国的首都是哪个城市? 中国的首都是北京。
```

## 流式推理

Mindformers大模型套件提供Streamer类，旨在用户在调用text generator方法进行文本生成时能够实时看到生成的每一个词，而不必等待所有结果均生成结束。

实例化streamer并向text generator方法传入该实例：

```python
from mindformers import GPT2LMHeadModel, GPT2Tokenizer, TextStreamer

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
inputs = tok(["An increasing sequence: one,"], return_tensors=None, add_special_tokens=False)

streamer = TextStreamer(tok)

_ = model.generate(inputs["input_ids"], streamer=streamer, max_length=20, top_k=1)
# 'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```

## 分布式推理

**说明:** 由于MindSpore版本问题，分布式推理仅支持MindSpore 2.0及以上版本，且暂不支持流水并行推理模式。

[分布式推理参考用例](../model_cards/bloom.md#a1-模型并行推理以1机8卡推理bloom_71b为例)
