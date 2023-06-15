# LLaMA

## 模型描述

LLaMA是由Meta于2023年发布。LLaMA模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。LLaMA目前按照参数量，目前有四个版本：LLaMA-7B（7B）、LLaMA-13B（13B）、LLaMA-33B（33B）以及LLaMA-65B（65B），目前在本仓库中，支持了7B，13B和65B三个规格的模型，权重文件来源于OpenLLaMA。

[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

``` text
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## 快速使用

### 数据集准备

以Wikitext2数据集为例:

- 数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

- 分词模型下载：[例如下载huggingface的tokenizer.model](https://huggingface.co/huggyllama/llama-13b/blob/main/tokenizer.model)

- 使用预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理
# 数据预处理+Mindrecord数据生成
python llama_preprocess.py --input_glob  'data/wiki.txt' --model_file /{path}/tokenizer.model --seq_length 2048 --output_file /{path}/wiki2048.mindrecord
```

### 脚本启动（LLaMA-7B为例）

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

#### 单机多卡启动

- 运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

```shell

# step1：机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：# 执行运行脚本：8卡分布式运行， DEVICE_RANGE = [0, 8]， 不包含8本身。
cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_STATUS

```

```python
# RANK_TABLE_FILE 参考样例
# 单机8卡
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

```text
# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的llama/run_llama_7b.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\predict
```

其中，模型和训练等相关配置可在`configs/llama`目录下的yaml文件中配置，如数据集路径，可在`configs/llama/run_llama_{7/13/65}b.yaml`中配置中`train_dataset`的`dataset_dir`参数。

#### 多机多卡启动

- 首先参考单机多卡启动方式，在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

- 将每台机器上生成的RANK_TABLE_FILE拷贝到一起，执行merge_hccl.py脚本将不同机器上生成的RANK_TABLE_FILE文件中的hccl*.json进行合并，包括server_list合并，server_count设为机器数，rank_id顺序增加，生成合并后的RANK_TABLE_FILE文件，再拷贝到所有机器中，保证不同机器上的RANK_TABLE_FILE相同；

- 在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式，需注意的是，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# step1：在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json

# step3：将step2得到的合并后的RANK_TABLE_FILE文件分别复制到所有的机器上。

# step4：根据服务器节点数等信息，修改相应的配置
'''
以llama-13b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。配置文件在../configs/llama/run_llama_13b.yaml

parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  optimizer_shard: True
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
'''

# step5：执行运行脚本
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama/run_llama_13b.yaml [0,8] train 16
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/llama/run_llama_13b.yaml [8,16] train 16
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained('llama_7b')
model.set_train(False)
tokenizer = LlamaTokenizer.from_pretrained('llama_7b')
inputs = tokenizer(["hello world"],
                 padding='max_length',
                 max_length=model.config.seq_length,
                 return_tensors='ms')
output = model(input_ids=inputs["input_ids"])
print(output)  # 计算输出的logits

model.set_train(True)
inputs = tokenizer(["hello world"],
                   padding='max_length',
                   max_length=model.config.seq_length+1,
                   return_tensors='ms')
output = model(input_ids=inputs["input_ids"])
print(output)  # 计算loss
```

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer
# 初始化预训练任务
trainer = Trainer(task='text_generation', model='llama_7b', train_dataset="your data file path")
# 方式1: 开启训练，并使用训练好的权重进行推理
trainer.train()
res = trainer.predict(predict_checkpoint=True, input_data="I love Beijing, because")

# 方式2： 从obs下载训练好的权重并进行推理
res = trainer.predict(input_data="I love Beijing, because")
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
pipeline_task = pipeline("text_generation", model='llama_7b', max_length=20)
pipeline_result = pipeline_task("I love Beijing, because", top_k=3)
print(pipeline_result)
```

### 预训练权重准备

从huggingface下载英文预训练权重（权重来源于OpenLLaMA）：

- [llama-3b](https://huggingface.co/openlm-research/open_llama_3b)

- [llama-7b](https://huggingface.co/openlm-research/open_llama_7b)

- [llama-13b](https://huggingface.co/openlm-research/open_llama_13b_600bt)

注意：65B权重OpenLLaMA未提供，如有需要，请开发者自行解决。

下载完成后，运行如下转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python mindformers/models/llama/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

### 指令微调

#### 数据集准备

目前支持增加alpaca数据的预处理：

- 数据集下载[alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

``` json
# alpaca examples:
    {
        "instruction": "Describe a time when you had to make a difficult decision.",
        "input": "",
        "output": "I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client\u2019s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team\u2019s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client\u2019s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities."
    },
    {
        "instruction": "Identify the odd one out.",
        "input": "Twitter, Instagram, Telegram",
        "output": "Telegram"
    },
```

- 使用fastchat工具添加prompts模板，转换为多轮对话格式

``` bash
# 使用tools/dataset_preprocess/llama/alpaca_converter.py添加prompt模板
# 执行转换脚本
python alpaca_converter.py --data_path /{path}/alpaca_data.json --output_path /{path}/alpaca-data-conversation.json
```

```text
# 参数说明
data_path: 存放alpaca数据的路径
output_path: 输出转换后对话格式的数据路径
```

- 执行llama预处理脚本，将带有prompt模板的数据转换为mindrecord格式

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理
# 数据预处理+Mindrecord数据生成
# 由于此工具依赖fschat工具包解析prompt模板，请提前安装fschat >= 0.2.13
python llama_preprocess.py --input_glob  '{path}/alpaca-data-conversation.json' --dataset_type qa --model_file /{path}/tokenizer.model --seq_length 2048 --output_file /{path}alpaca-fastchat2048.mindrecord
```

#### 全参微调

以llama7b为例

- step 1. 修改`config/llama/run_llama_7b.yaml`中训练数据集路径为微调数据集路径；

``` json
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/{path}alpaca-fastchat2048.mindrecord"
    shuffle: True
```

- step 2. 修改训练时学习率和优化器参数，和预训练不同，微调学习率配置如下：

``` json
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.999
  eps: 1.e-8 # 1e-8
  learning_rate: 1.e-5

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset
```

- step 3. 添加预训练权重路径，修改配置文件中的`load_checkpoint`后配置预训练权重路径

- step 4. 启动微调任务，7b可以使用单机八卡进行全参微调，命令如下：

``` shell
cd scripts
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama/run_llama_7b.yaml [0,8] finetune
```

#### lora微调

目前lora微调适配了llama_7b模型，并给出了默认配置文件`config/llama/run_llama_7b_lora.yaml`。

- step 1. 参考全参微调修改训练数据集路径与预训练权重路径。

- step 2. 启动lora微调任务，7b模型可以使用单机八卡进行训练，命令如下：

``` shell
cd scripts
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/llama/run_llama_7b_lora.yaml [0,8] finetune
```