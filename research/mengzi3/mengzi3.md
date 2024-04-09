# 孟子

孟子是澜舟科技自主开发的自然语言处理大模型。孟子可控大模型的能力平台已经推出了机器翻译平台、金融NLP、AIGC智能创作等多个企业级解决方案及对外开放服务，并与同花顺、华夏基金、传神语联网、数说故事、中文阅读集团等分别在金融舆情分析、多语言机器翻译、AIGC营销文案写作、网络文学AI辅助创作上进行了深度合作。MindFormers已支持MengZi3-13B-Base推理任务。

## 前期准备

### 安装mindformers

参考[README](https://gitee.com/mindspore/mindformers/blob/r1.0/README.md#二、mindformers安装)安装mindformers。 本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件：Atlas 800T A2  64G
- MindSpore：2.2.12
- CANN: 7.0.0
- MindFormers版本：1.0

### MengZi3权重下载和转换

1.下载权重

https://huggingface.co/Langboat/Mengzi3-13B-Base

2.下载Mengzi3-13B权重完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```bash
python mindformers/models/llama/convert_weight.py \
--torch_ckpt_dir TORCH_CKPT_DIR \
--mindspore_ckpt_path /path/MengZi3-13B/MengZi3_13b.ckpt
```

```text
# 参数说明
torch_ckpt_dir: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，可以指定自定义保存路径（输出文件格式要为*.ckpt）
```

## 推理

在启动前，请先行在配置文件run_mengzi3_13b_910b.yaml中将processor.tokenizer.vocab_file的路径配置为实际路径

```yaml
processor:
  return_tensors: ms
  tokenizer:
    ...
    vocab_file: '/path/MengZi3-13B/tokenizer.model'  # 修改为实际路径
    ...
```

- generate接口推理：

```python
from mindspore import context
from mindformers.generation import GenerationConfig
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

# init MengZi3-13b-Base model
mengzi_model_path = "/path/MengZi3-13B/MengZi3_13b.ckpt"  # 填写实际路径
config_path = 'mengzi3/run_mengzi3_13b_910b.yaml'  # 填写实际路径

config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = mengzi_model_path
mengzi_config = LlamaConfig(**config.model.model_config)

mengzi_model = LlamaForCausalLM(config=mengzi_config)

# init MengZi3-13b-Base tokenizer
tokenizer_path = "/path/MengZi3-13B/tokenizer.model"  # 填写实际路径
tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)

inputs = tokenizer("背诵李绅的悯农，答案是")["input_ids"]
outputs = mengzi_model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs))

# 运行结果
# [{'text_generation_text': ['背诵李绅的悯农，答案是：锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。\n锄禾日当午，汗滴禾下土。\n
# 谁知盘中餐，粒粒皆辛苦。\n译文：\n农民在正午烈日的暴晒下锄禾，汗水从身上滴在禾苗生长的土地上。又有谁知道盘中的饭食，每颗每粒都是农民用
# 辛勤的劳动换来的呢？\n赏析：\n这首诗描绘了在烈日当空的正午农民田里劳作的景象，概括地表现了农民终年辛勤劳动的生活，最后以“谁知盘中餐']}]
```

### 基于pipeline的推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

# init MengZi3-13b-Base model
mengzi_model_path = "/path/MengZi3-13B/MengZi3_13b.ckpt"  # 填写实际路径
config_path = 'mengzi3/run_mengzi3_13b_910b.yaml'  # 填写实际路径

config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = mengzi_model_path
mengzi_config = LlamaConfig(**config.model.model_config)

mengzi_model = LlamaForCausalLM(config=mengzi_config)

# init MengZi3-13b-Base tokenizer
tokenizer_path = "/path/MengZi3-13B/tokenizer.model"  # 填写实际路径
tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)

pipeline_task = pipeline("text_generation", model=mengzi_model, tokenizer=tokenizer)
peline_result = pipeline_task("背诵李绅的悯农，答案是", max_length=128, add_special_tokens=False)

print(peline_result)

# 运行结果
# ['背诵李绅的悯农，答案是：锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。\n锄禾日当午，汗滴禾下土。\n
# 谁知盘中餐，粒粒皆辛苦。\n译文：\n农民在正午烈日的暴晒下锄禾，汗水从身上滴在禾苗生长的土地上。又有谁知道盘中的饭食，每颗每粒都是农民用
# 辛勤的劳动换来的呢？\n赏析：\n这首诗描绘了在烈日当空的正午农民田里劳作的景象，概括地表现了农民终年辛勤劳动的生活，最后以“谁知盘中餐']
```