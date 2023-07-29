# 百川

百川大模型系列是由百川智能研究的大规模语言预训练模型，目前有Baichuan-7B、Baichuan-13B-base和Baichuan-13B-Chat三个系列。`目前仅支持Baichuan-7B预训练模型。`

## Baichuan-7B

Baichuan-7B 是由百川智能开发的一个开源可商用的大规模预训练语言模型。基于 Transformer 结构，在大约 1.2 万亿 tokens 上训练的 70 亿参数模型，支持中英双语，上下文窗口长度为 4096。在标准的中文和英文 benchmark（C-Eval/MMLU）上均取得同尺寸最好的效果。

Baichuan-7B 是采用llama的模型结构设计，模型实现我们复用llama的代码。

``` text
Model Description
Developed by: 百川智能(Baichuan Intelligent Technology)
Email: opensource@baichuan-inc.com
Language(s) (NLP): Chinese/English
License: Baichuan-7B License
```

### 快速使用

#### Baichuan-7B 预训练权重转换

从huggingface下载[Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main);需要将整个工程下载下来。

执行权重转换脚本

```shell
python research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

#### API方式调用

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
> `遵从Baichuan-7B的license，本模型需要用户自行下载权重进行处理，故使用时和llama存在一定区别，具体如下：`

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer

# 在使用Trainer接口进行训练推理时，由于百川模型的tokenizer需要用户自行下载，因此在启动前，请先行在配置文件中将tokenizer.model的路径配置完成，具体修改如下
# 增加 vocab_file: '/path/Baichuan-7B/tokenizer.model'，这样一个配置即可
#processor:
#  return_tensors: ms
#  tokenizer:
#    unk_token: '<unk>'
#    bos_token: '<s>'
#    eos_token: '</s>'
#    pad_token: '<pad>'
#    vocab_file: '/path/Baichuan-7B/tokenizer.model'
#    type: LlamaTokenizer

# 初始化预训练任务
trainer = Trainer(task='text_generation', model='baichuan_7b', train_dataset="your data file path")
# 方式1: 开启训练，并使用训练好的权重进行推理
trainer.train()
res = trainer.predict(predict_checkpoint=True, input_data="I love Beijing, because")

# 方式2： 使用自行下载的Baichuan-7B权重并进行推理
baichuan_model_path = "/path/Baichuan-7B/transform.ckpt" # Baichuan-7B ckpt path
res = trainer.predict(predict_checkpoint=baichuan_model_path, input_data="I love Beijing, because")
```

- pipeline接口开启快速推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=3)
# init baichuan-7b model
baichuan_model_path = "/path/Baichuan-7B/transform.ckpt" # Baichuan-7B ckpt path
baichuan_config = LlamaConfig(
    vocab_size=64000,
    pad_token_id=0,
    checkpoint_name_or_path=baichuan_model_path,
    use_past=True
)
baichuan_model = LlamaForCausalLM(
    config=baichuan_config
)
# init baichuan-7b tokenizer
tokenizer_path = "/path/Baichuan-7B/tokenizer.model" # Baichuan-7B tokenizer.model path
tokenizer = LlamaTokenizer(
    vocab_file=tokenizer_path
)
pipeline_task = pipeline("text_generation", model=baichuan_model, tokenizer=tokenizer, max_length=32)
peline_result = pipeline_task("登鹳雀楼->王之涣\n夜雨寄北->", top_k=3, do_sample=True, top_p=0.95, repetition_penalty=1.1, max_length=256)

print(peline_result)
```

#### 训练与微调

基于Baichuan-7B，目前提供了模型的基础配置文件`configs/baichuan/run_baichuan_7b.yaml`。可参考[llama](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md)的训练与微调章节进行数据准备，而后启动微调，不在此赘述。

`注：使用Baichuan-7B进行训练或者微调时，需要使用Baichuan-7B配套的tokenizer.model处理数据集，以及选用Baichuan-7B的yaml配置文件进行任务启动。`