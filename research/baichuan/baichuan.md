# Baichuan

百川大模型系列是由百川智能研究的大规模语言预训练模型，目前有Baichuan-7B、Baichuan-13B-base和Baichuan-13B-Chat三个系列。目前MindFormers已全部支持。

**注: 7B与13B实现方式不同，请参考对应参数的文档进行使用**

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
python research/baichuan/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_path: huggingface权重保存目录下任意权重bin文件，根据该文件路径读取目录下所有权重
mindspore_ckpt_path: mindspore权重文件保存路径
```

#### [多卡权重切分](../../docs/feature_cards/Transform_Ckpt.md#方案1源码执行)

#### 脚本启动

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

#### Baichuan-13B-Base/Chat 权重转换

从huggingface下载[Baichuan-13B-base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main)或者[Baichuan-13B-chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/tree/main)，需要将整个工程下载下来。

执行权重转换脚本

```shell
python research/baichuan/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_path: huggingface权重保存目录下任意权重bin文件,根据该文件路径读取目录下全部权重
mindspore_ckpt_path: mindspore权重文件保存路径
```

#### [多卡权重切分](../../docs/feature_cards/Transform_Ckpt.md#方案1源码执行)

非单卡运行，无论是train, finetune, eval, predict均需要把权重按照并行配置进行切分！

#### 脚本启动Baichuan-13B-Base

> 需开发者提前pip安装。具体接口说明请参考[API接口](../../README.md#二mindformers安装)
> `遵从Baichuan-13B-base的license，本模型需要用户自行下载权重进行处理`

`Baichuan-13B-base`的高阶接口使用脚本已集成在`run_baichuan_13b_base.py`脚本中

**注1**：由于模型较大，Atlas 800不支持单卡推理，不支持单机8卡训练。如果使用Atlas 800进行单卡推理，需要修改`run_baichuan_13b.yaml`中`seq_length`为1024。

**注2**：增量推理需要修改`run_baichuan_13b.yaml`中`use_past`为True。

**注3**：使用predict前需要下载baichuan13b的tokenizer文件，并且在`baichuan/run_baichuan_13b.yaml`该文件中修改tokenzier路径到hugging face`Baichuan-13B-Base/tokenizer.model`文件

- Atlas 800T A2单卡eval示例

```shell
cd mindformers/research
python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b_910b.yaml --load_checkpoint path/to/baichuan_13b.ckpt --run_mode=eval --eval_data path/to/mindrecord_dir --use_parallel False
```

- Atlas 800T A2单卡predict示例

```shell
cd mindformers/research
python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b_910b.yaml --load_checkpoint path/to/baichuan_13b.ckpt --run_mode=predict --predict_data TLS1.2协议的基本流程 --predict_length 100 --use_parallel False
#运行结果：[{'text_generation_text': ['TLS1.2协议的基本流程如下: 1.客户端向服务器发送一个ClientHello消息,其中包含客户端支持的加密算法、压缩算法、随机数、客户端支持的扩展等信息。 2.服务器收到ClientHello消息后,向客户端发送一个ServerHello消息,其中包含服务器支持的加密算法、压缩算法、随机数、服务器支持的扩展等信息。 3.客户端收到ServerHello消息后,向服务']}]
```

- 单机多卡运行eval示例

```shell
cd mindformers/research
bash run_singlenode.sh "python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=eval --eval_data path/to/mindrecord_dir" path/to/rank_table_file [0,2] 2
```

**注意，此处load checkpoint后的路径为多卡切分权重**

- 单机多卡运行predict示例

```shell
cd mindformers/research
bash run_singlenode.sh "python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=predict --predict_data TLS1.2协议的基本流程 --predict_length 100" path/to/rank_table_file [0,2] 2
#运行结果：[{'text_generation_text': ['TLS1.2协议的基本流程如下: 1.客户端向服务器发送一个ClientHello消息,其中包含客户端支持的加密算法、压缩算法、随机数、客户端支持的扩展等信息。 2.服务器收到ClientHello消息后,向客户端发送一个ServerHello消息,其中包含服务器支持的加密算法、压缩算法、随机数、服务器支持的扩展等信息。 3.客户端收到ServerHello消息后,向服务']}]
```

**注意，此处load checkpoint后的路径为多卡切分权重**

- 多机多卡运行train示例

```shell
# node 1
cd mindformers/research
bash run_multinode.sh "python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=train --train_data path/to/mindrecord_dir" path/to/rank_table_file [0,8] 16
# node 2
cd mindformers/research
bash run_multinode.sh "python baichuan/run_baichuan_13b_base.py --config baichuan/run_baichuan_13b.yaml --load_checkpoint path/to/baichuan_13b_ckpt_dp1mp2 --run_mode=train --train_data path/to/mindrecord_dir" .path/to/rank_table_file [8,16] 16
```

**参数说明**
  `config`: huggingface权重保存目录路径(即刚刚从hugging face下载的工程目录)
  `load_checkpoint`: 推理所使用的的权重，需从huggingface获取，通过conver_weight转换为mindspore单卡权重，参考[权重切分](../../docs/feature_cards/Transform_Ckpt.md)转换为多卡权重
  `run_mode`：运行模式，包括train，finetune，eval，predict
  `train_data`：train数据，训练时需要填入，数据获取方法参考[llama数据准备](../../docs/model_cards/llama.md#数据集准备)，注意tokenzier需使用baichuan的。
  `eval_data`：eval数据，eval是需要填入，同train。
  `predict_data`：predict数据，predict时需要填入

  更多输入可参考`run_baichuan_13b_base.py`脚本内入参

#### 脚本启动Baichuan-13B-Chat

> 需开发者提前pip安装。具体接口说明请参考[API接口](../../README.md#二mindformers安装)
> `遵从Baichuan-13B-chat的license，本模型需要用户自行下载权重进行处理`

`Baichuan-13B-chat`的高阶接口使用脚本已集成在`run_baichuan_13b_chat.py`脚本中

```shell
cd mindformers/research
python baichuan/run_baichuan_13b_chat.py --config baichuan --load_checkpoint path/to/baichuan_13b.ckpt --max_new_tokens 512
#请输入：世界上第二高的山峰是哪座？
#世界上第二高的山峰是喀喇昆仑山脉的乔戈里峰(K2)，海拔8,611米(28,251英尺)。它位于巴基斯坦和中国边境附近，是喀喇昆仑山脉的最高峰峰。</s>
#请输入：那第三高的山峰呢？
#世界第三高的山峰是喜马拉雅山脉的康峰(Kangchenjunga)，海拔8,586米(28,169英尺)。它位于尼泊尔和印度边境附近，是世界上最高的14座山峰之一一。</s>
#请输入：我想攀爬高峰，在前面说的两座高峰里，你推荐我先爬哪一座？
#在选择攀爬的顺序时，需要考虑多种因素，如个人体能、技能水平、时间限制等。以下是一些建议供您参考：...(省略更多输出)
```

**参数说明**
  `config`: 用于生成tokenizer的配置文件，路径指定到文件夹，需把yaml文件单独放置于一个文件夹内
  `load_checkpoint`: 推理所使用的的权重，需从huggingface获取，通过conver_weight转换为mindspore单卡权重，参考[权重切分](../../docs/feature_cards/Transform_Ckpt.md)转换为多卡权重
  `max_new_tokens`: 最大生成tokens数，多轮对话时，如果记忆的总tokens大于`seq_length-max_new_tokens`会遗忘以前的对话。
