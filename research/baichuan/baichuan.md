# 百川

百川大模型系列是由百川智能研究的大规模语言预训练模型，目前有Baichuan-7B、Baichuan-13B-base和Baichuan-13B-Chat三个系列。目前支持`Baichuan-7B`和`Baichuan-13B-base`预训练模型。

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

## Baichuan-13B-base

Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。

Baichuan-13B 有如下几个特点：

- **更大尺寸、更多数据**：Baichuan-13B 在 Baichuan-7B 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模- 型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。
- **同时开源预训练和对齐模型**：预训练模型是适用开发者的『 基座 』，而广大普通用户对有对话功能的对齐模型具有更强的需求。
- **更高效的推理**：为了支持更广大用户的使用，百川智能同时开源了 int8 和 int4 的量化版本，相对非量化版本在几乎没有效果损失的情况下大大降低了部署的机器资源门槛。
- **开源免费可商用**：Baichuan-13B 不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。

``` text
Model Description
Developed by: 百川智能(Baichuan Intelligent Technology)
Email: opensource@baichuan-inc.com
Language(s) (NLP): Chinese/English
License: Baichuan-13B-base License
```

### 快速使用

#### Baichuan-13B 预训练权重转换

从huggingface下载[Baichuan-13B-base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main);需要将整个工程下载下来。

执行权重转换脚本

```shell
python research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径(即刚刚从hugging face下载的工程目录)
mindspore_ckpt_path: 权重保存文件名，默认保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

#### API方式调用

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/mindformer/wikis/API/)
> `遵从Baichuan-13B-base的license，本模型需要用户自行下载权重进行处理`

`Baichuan-13B-base`的高阶接口使用脚本已集成在`run_baichuan_13b.py`脚本中

**注1**：由于模型较大，910A不支持单卡推理，不支持8卡训练，910B支持单卡推理，单机8卡训练。如果使用910A，并且推理对于seq_length不硬性要求4096，可以在yaml中修改seq_length为1024，910A也可以单卡运行推理。

**注2**: 由于baichuan-13B-base基于高阶接口的形式开发，存放于research文件夹下，使用时需要将mindformers[安装](../../README.md#二mindformers安装)为python的包，才能直接进入research目录下执行相关命令。

**注3**: 当前`run_baichuan_13b.yaml`文件默认为train配置，用于eval和predict时需要修改并行策略。910B请使用`run_baichuan_13b_910b.yaml`

**注4**: 加载权重和并行策略强相关，指定并行策略（数据并行data_parallel, 模型并行model_parallel)后, 需要根据相应的strategy文件，将单卡权重切分为对应并行的权重，之后才能加载进行微调或者评估推理！！！ **[多卡权重的切分与合并](../../docs/feature_cards/Transform_Ckpt.md)**，由于使用自定义脚本启动，不能使用`权重自动转换`，请使用`权重离线切分转换`！！！

**注5**：使用predict前需要下载baichuan13b的tokenizer文件，并且在`baichuan/run_baichuan_13b.yaml`该文件中修改tokenzier路径到hugging face`Baichuan-13B-Base/tokenizer.model`文件

- 910B单卡eval示例

```shell
cd mindformers/research
python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b_910b.yaml --load_checkpoint path/to/baichuan_13b.ckpt --run_mode=eval --eval_data path/to/mindrecord_dir
```

- 910B单卡predict示例

```shell
cd mindformers/research
python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b_910b.yaml --load_checkpoint path/to/baichuan_13b.ckpt --run_mode=predict --predict_data TLS1.2协议的基本流程 --predict_length 100 --use_parallel False
#运行结果：[{'text_generation_text': ['TLS1.2协议的基本流程如下: 1.客户端向服务器发送一个ClientHello消息,其中包含客户端支持的加密算法、压缩算法、随机数、客户端支持的扩展等信息。 2.服务器收到ClientHello消息后,向客户端发送一个ServerHello消息,其中包含服务器支持的加密算法、压缩算法、随机数、服务器支持的扩展等信息。 3.客户端收到ServerHello消息后,向服务']}]
```

- 单机多卡运行eval示例

```shell
cd mindformers/research
bash run_singlenode.sh "python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=eval --eval_data path/to/mindrecord_dir" path/to/rank_table_file [0,2] 2
```

**注意看，这里load checkpoint后的路径为多卡切分权重**

- 单机多卡运行predict示例

```shell
cd mindformers/research
bash run_singlenode.sh "python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=predict --predict_data TLS1.2协议的基本流程 --predict_length 100" path/to/rank_table_file [0,2] 2
#运行结果：[{'text_generation_text': ['TLS1.2协议的基本流程如下: 1.客户端向服务器发送一个ClientHello消息,其中包含客户端支持的加密算法、压缩算法、随机数、客户端支持的扩展等信息。 2.服务器收到ClientHello消息后,向客户端发送一个ServerHello消息,其中包含服务器支持的加密算法、压缩算法、随机数、服务器支持的扩展等信息。 3.客户端收到ServerHello消息后,向服务']}]
```

**注意看，这里load checkpoint后的路径为多卡切分权重**

- 多机多卡运行train示例

```shell
# node 1
cd mindformers/research
bash run_multinode.sh "python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=train --train_data path/to/mindrecord_dir" path/to/rank_table_file [0,8] 16
# node 2
cd mindformers/research
bash run_multinode.sh "python baichuan/run_baichuan_13b.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=train --train_data path/to/mindrecord_dir" .path/to/rank_table_file [8,16] 16
```

**参数说明**
  `config`: huggingface权重保存目录路径(即刚刚从hugging face下载的工程目录)
  `load_checkpoint`: 推理所使用的的权重，需从huggingface获取，通过conver_weight转换为mindspore单卡权重，参考[权重切分](../../docs/feature_cards/Transform_Ckpt.md)转换为多卡权重
  `run_mode`：运行模式，包括train，finetune，eval，predict
  `train_data`：eval数据，训练时需要填入，数据获取方法参考[llama数据准备](../../docs/model_cards/llama.md#数据集准备)，注意tokenzier需使用baichuan的。
  `eval_data`：eval数据，eval是需要填入，同train。
  `predict_data`：predict数据，predict时需要填入

  更多输入可参考`run_baichuan_13b.py`脚本内入参
