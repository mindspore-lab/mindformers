# 天工

天工Skywork-13B 系列是由昆仑万维研究的大规模语言预训练模型，目前开源的有Skywork-13B-Base，Skywork-13B-Chat，Skywork-13B-Math，Skywork-13B-MM，MindFormers已支持Skywork-13B-Base。

## 前期准备

### 安装mindformers

参考[README](../../README.md#二、mindformers安装)安装mindformers。
本文操作的相对路径均为安装mindformers后的代码仓根路径。

### 环境要求

- 硬件: Ascend 910 64GB
- MindSpore: 2.2.0
- MindSpore Lite: 2.2.0
- MindFormers: dev
- Mindpet: 1.0.2

**注** skywork-13b推理可以在单卡上完成部署，全量微调至少需要8卡。

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，在当前路径生成该机器的RANK_TABLE_FILE的json文件，生成的文件名形如hccl_8p_01234567_127.0.0.1.json
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注** 若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

### Skywork-13B-Base 预训练权重下载和转换

- 下载已转换的ckpt文件

本仓库提供已经转换完成的预训练权重用于训练/微调/推理，用户可自行从下方链接拉取后直接使用。

下载链接：

权重：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/sky_work/skywork_13b.ckpt

词表：https://huggingface.co/Skywork/Skywork-13B-base/blob/main/tokenizer.model

linux可用如下命令下载。

```shell
mkdir -p ckpt/rank_0
cd ./ckpt/rank_0
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/sky_work/skywork_13b.ckpt
wget https://huggingface.co/Skywork/Skywork-13B-base/blob/main/tokenizer.model
cd ../..
```

- 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[Skywork-13B-Base](https://huggingface.co/Skywork/Skywork-13B-base)

如果使用git命令下载，下载前请先确保已安装git-lfs。

```shell
git lfs install
git clone https://huggingface.co/Skywork/Skywork-13B-base
```

执行权重转换脚本

```shell
cd research
python skywork/convert_weight.py --torch_ckpt_path TORCH_CKPT_PATH --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
torch_ckpt_path: huggingface Skywork-13B-Base权重保存目录路径下任意权重bin文件，根据该文件路径读取目录下全部权重
mindspore_ckpt_path: mindspore权重文件保存路径
```

**注**: 请安装torch=1.13.1和transformers=4.30.2版本。如果执行报错，请检查并安装requests、decorator、pandas、sympy。

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

# Skywork-13B-Base

Skywork-13B-Base 是在经过高质量清洗过滤的3.2万亿个多语言（主要是中文和英文）和代码数据上进行训练的，它在多种评测和各种基准测试上都展现了同等规模模型的最佳效果。网络结构与llama相同。

## 训练与微调

目前提供了模型的基础配置文件`research/skywork/run_skywork_13b.yaml`。使用前请将配置文件中路径相关参数修改为实际路径。

## 模型性能

| config                                                       | task                  | Datasets  | SeqLength | metric | phase             | score     | performance(tokens/s/p)  |
| ------------------------------------------------------------ | --------------------- | --------- | --------- | ------ | ----------------- | --------- | ------------ |
| [skywork_13b](./run_skywork_13b.yaml)    | text_generation       | ADGEN      | 4096      | -      | [train](#预训练)  | -         | 1105.92  |
| [skywork_13b](./run_skywork_13b.yaml)    | text_generation       | ADGEN      | 4096      | -      | [finetune](#微调)  | -         | 1105.92  |

### 数据集准备

使用Skywork-13B-Base进行训练或者微调时，需要使用Skywork-13B-Base配套的tokenizer.model处理数据集，以及选用Skywork-13B-Base的yaml配置文件进行任务启动。

目前提供[ADGEN](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) （广告生成）数据集的预处理脚本用于全参微调任务。

ADGEN数据集样式

```text
{
    "content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
    "summary": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。"
}
{
    "content": "类型#裙*风格#简约*图案#条纹*图案#线条*图案#撞色*裙型#鱼尾裙*裙袖长#无袖",
    "summary": "圆形领口修饰脖颈线条，适合各种脸型，耐看有气质。无袖设计，尤显清凉，简约横条纹装饰，使得整身人鱼造型更为生动立体。加之撞色的鱼尾下摆，深邃富有诗意。收腰包臀,修饰女性身体曲线，结合别出心裁的鱼尾裙摆设计，勾勒出自然流畅的身体轮廓，展现了婀娜多姿的迷人姿态。"
}
```

- 转换成mindrecord格式

```shell
cd research
python skywork/skywork_dataprocess.py --input_file_path /{path}/AdvertiseGenTrain_text.jsonl --output_file /{path}/AdvertiseGenTrain_text.mindrecord --model_file /{path}/tokenizer.model --seq_length 4096
```

参数说明

```text
input_file_path：ADGEN数据集输入文件路径
output_file：生成的mindrecord目标文件路径
dataset_type：数据集类型，目前仅支持"text"
model_file：tokenizer.model文件路径
seq_length：数据长度
```

### 预训练

- 单机多卡预训练示例

```shell
cd research
# Usage Help: bash run_singlenode.sh [START_CMD] [RANK_TABLE_FILE] [DEVICE_RANGE] [DEVICE_NUM]
bash run_singlenode.sh \
"python skywork/run_skywork.py \
--config skywork/run_skywork_13b.yaml \
--run_mode finetune \
--train_dataset /{path}/AdvertiseGenTrain_text.mindrecord \
--auto_trans_ckpt True \
--use_parallel True" \
../hccl_8p_01234567_127.0.0.1.json [0,8] 8
```

**参数说明**

```text
START_CMD：Python启动命令，其中
 config：为research/skywork文件夹下面的run_skywork_13b.yaml配置文件，配置文件参数请按需修改
 run_mode：任务运行状态，支持关键字train/finetune/eval/predict/export
 train_dataset：训练数据集路径
 auto_trans_ckpt：是否自动转换ckpt
 use_parallel：是否使用并行模式
RANK_TABLE_FILE：由 mindformers/tools/hccl_tools.py 生成的分布式json文件
DEVICE_RANGE：为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
DEVICE_NUM：使用的卡的个数
```

**注**：由于模型较大，未切分的模型当seq_length为4096时，仅能进行batch_size为1的单机8卡训练。如果要使用其他并行策略训练，请参考 [多卡权重切分](../../docs/feature_cards/Transform_Ckpt.md)

### 微调

- 单机多卡微调示例

```shell
cd research
# Usage Help: bash run_singlenode.sh [START_CMD] [RANK_TABLE_FILE] [DEVICE_RANGE] [DEVICE_NUM]
bash run_singlenode.sh \
"python skywork/run_skywork.py \
--config skywork/run_skywork_13b.yaml \
--run_mode finetune \
--load_checkpoint  /{path}/ \
--train_dataset /{path}/AdvertiseGenTrain_text.mindrecord \
--auto_trans_ckpt True \
--use_parallel True" \
../hccl_8p_01234567_127.0.0.1.json [0,8] 8
```

**参数说明**

```text
START_CMD：Python启动命令，其中
 config：为research/skywork文件夹下面的run_skywork_13b.yaml配置文件，配置文件参数请按需修改
 run_mode：任务运行状态，支持关键字train/finetune/eval/predict/export
 load_checkpoint：权重路径。例如路径形式为/path/ckpt/rank_0/skywork_13b.ckpt，则参数填写为/path/ckpt
 train_dataset：训练数据集路径
 auto_trans_ckpt：是否自动转换ckpt
 use_parallel：是否使用并行模式
RANK_TABLE_FILE：由 mindformers/tools/hccl_tools.py 生成的分布式json文件
DEVICE_RANGE：为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
DEVICE_NUM：使用的卡的个数
```

**注**：由于模型较大，未切分的模型当seq_length为4096时，仅能进行batch_size为1的单机8卡训练。如果要使用其他并行策略训练，请参考 [多卡权重切分](../../docs/feature_cards/Transform_Ckpt.md)

## MindSpore推理

> 接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
>
> 遵从Skywork-13B的license，本模型需要用户自行下载权重进行处理，故使用时和llama存在一定区别，具体如下：

在启动前，请先行在配置文件run_skywork_13b.yaml中将processor.tokenizer.vocab_file的路径配置为实际路径；如果使用增量推理，需要在配置文件中将model.model_config.use_past值设置为True。例如：

```yaml
processor:
  return_tensors: ms
  tokenizer:
    ...
    vocab_file: '/path/Skywork-13B/tokenizer.model'  # 修改为实际路径
    ...
model:
  model_config:
    ...
    use_past: True
    ...
```

- generate接口推理：

```python
from mindspore import context
from mindformers.generation import GenerationConfig
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

# init skywork-13b-Base model
skywork_model_path = "/path/Skywork-13B/skywork_13b.ckpt"  # 填写实际路径
config_path = 'skywork/run_skywork_13b.yaml'  # 填写实际路径

config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = skywork_model_path
skywork_config = LlamaConfig(**config.model.model_config)

skywork_model = LlamaForCausalLM(config=skywork_config)

# init skywork-13b-Base tokenizer
tokenizer_path = "/path/Skywork-13B/tokenizer.model"  # 填写实际路径
tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
generation_config = GenerationConfig(
    temperature=1,
    top_p=1.0,
    top_k=1,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    max_length=128,
)

inputs = tokenizer("陕西的省会是西安")["input_ids"]
outputs = skywork_model.generate(inputs, generation_config=generation_config)
print(tokenizer.decode(outputs))

# 运行结果
# ['<s>陕西的省会是西安，西安是陕西的政治、经济、文化中心，也是陕西的交通枢纽。西安的交通非常发达，有很多的交通工具，可以方便的到达陕西的各个地方。\n西安的交通工具有：\n1、飞机：西安咸阳国际机场是中国重要的航空港，也是中国大陆第四大航空港。西安咸阳国际机场位于西安市西北方向，距市中心约30公里，有高速公路与市区相连。\n2、火车：西安火车站位于西安市解放路，是西安最大的火车站，也是中国西部地区最大']
```

- Trainer高阶接口推理

skywork的高阶接口使用脚本已集成在run_skywork.py脚本中，运行此脚本命令示例：

```shell
cd research
python skywork/run_skywork.py --config skywork/run_skywork_13b.yaml --load_checkpoint /path/Skywork-13B/skywork_13b.ckpt --run_mode=predict --predict_data "陕西的省会是西安" --predict_length 100 --use_parallel False --device_id 0
#运行结果：[{'text_generation_text': ['陕西的省会是西安，西安是陕西的政治、经济、文化中心，也是陕西的交通枢纽。西安的交通非常发达，有很多的交通工具，可以方便的到达陕西的各个地方。\n西安的交通工具有：\n1、飞机：西安咸阳国际机场是中国重要的航空港，也是中国大陆第四大航空港。西安咸阳国际机场位于西安市西北方向，距市中心约30公里，有高速公路与']}]
```

- pipeline接口推理

```python
from mindspore import context
from mindformers.pipeline import pipeline
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

skywork_model_path = "/path/Skywork-13B/skywork_13b.ckpt"  # 填写实际路径
config_path = 'skywork/run_skywork_13b.yaml'  # 填写实际路径
config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = skywork_model_path
use_past = True  # 按需设置
config.model.model_config.use_past = use_past
skywork_config = LlamaConfig(**config.model.model_config)

skywork_model = LlamaForCausalLM(skywork_config)

# init skywork-13b-Base tokenizer
tokenizer_path = "/path/Skywork-13B/tokenizer.model"  # 填写实际路径
tokenizer = LlamaTokenizer(tokenizer_path, add_bos_token=True, add_eos_token=False)
pipeline_task = pipeline("text_generation", model=skywork_model, tokenizer=tokenizer, max_length=32)
peline_result = pipeline_task("陕西的省会是西安",
                              top_k=1,
                              do_sample=False,
                              top_p=1.0,
                              repetition_penalty=1,
                              max_length=128,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              use_past=use_past)

print(peline_result)

# 运行结果
[{'text_generation_text': ['陕西的省会是西安，西安是陕西的政治、经济、文化中心，也是陕西的交通枢纽。西安的交通非常发达，有很多的交通工具，可以方便的到达陕西的各个地方。\n西安的交通工具有：\n1、飞机：西安咸阳国际机场是中国重要的航空港，也是中国大陆第四大航空港。西安咸阳国际机场位于西安市西北方向，距市中心约30公里，有高速公路与市区相连。\n2、火车：西安火车站位于西安市解放路，是西安最大的火车站，也是中国西部地区最大']}]
```

## MindSpore Lite推理

### ckpt转换为mindir

```shell
# 如果需要使用增量推理，use_past设置为True；如果
cd research
python skywork/run_skywork.py --config skywork/run_skywork_13b.yaml --load_checkpoint /path/Skywork-13B/skywork_13b.ckpt --run_mode=export --use_parallel False --use_past True --batch_size 1 --device_id 0
```

**注**

1. 如果需要使用增量推理，use_past设置为True。设置use_past=True后生成的mindir有两个，分别在output/mindir_full_checkpoint和output/mindir_inc_checkpoint目录中。如果不设置use_past或者use_past=False，则只生成mindir_full_checkpoint目录，后续无法使用增量推理。

2. 不同batch_size的推理需求需要对应生成不同的mindir，由参数--batch_size指定。

### lite推理

- step1. 新建context.cfg配置文件

```text
[ascend_context]
plugin_custom_ops=All
provider=ge
[ge_session_options]
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
```

- step2. 新建Python脚本

```python
# run_skywork_infer_lite.py
import argparse

from mindformers import LlamaTokenizer
from mindformers.inference import InferTask
from mindformers.inference.infer_config import InferConfig


def infer_main(args):
    lite_config = InferConfig(
        prefill_model_path=args.full_model_path,
        increment_model_path=args.inc_model_path,
        model_name='llama',
        model_type="mindir",
        infer_seq_length=args.seq_length,
        ge_config_path=args.config_path,
        device_id=args.device_id,
        add_special_tokens=False,
    )
    tokenizer_path = args.token_path
    tokenizer = LlamaTokenizer(tokenizer_path, add_bos_token=True, add_eos_token=False)

    batch_input = [
        ["陕西的省会是西安，甘肃的省会是兰州，河南的省会是郑州" for i in range(args.batch_size)]
    ]
    input_list = batch_input * args.loop

    infer_model = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)

    for user_input in input_list:
        output = infer_model.infer(user_input,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   max_length=args.max_length,
                                   add_special_tokens=True)
        for out in output:
            print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    parser.add_argument('--config_path', default=None, type=str, help="ge config path")
    parser.add_argument('--seq_length', default=4096, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--loop', default=2, type=int)
    parser.add_argument('--token_path', default=None, type=str)
    args = parser.parse_args()
    print(args)
    infer_main(args)
```

- step3. 使用shell命令启动推理

```shell
# 如果需要增量推理，使用inc_model_path指定路径，否则不需要。token_path参数需指定为tokenizer.model实际路径。
cd research
python skywork/run_skywork_infer_lite.py --full_model_path output/mindir_full_checkpoint/rank_0_graph.mindir \
--inc_model_path output/mindir_inc_checkpoint/rank_0_graph.mindir --config_path skywork/context.cfg \
--token_path {path}/tokenizer.model \
--seq_length 4096 --max_length 128 --batch_size 1 --loop 2 --device_id 0

# 运行结果：
# 陕西的省会是西安，甘肃的省会是兰州，河南的省会是郑州，湖北的省会是武汉，湖南的省会是长沙，江西的省会是南昌，安徽的省会是合肥，四川的省会是成都，贵州的省会是贵阳，云南的省会是昆明，西藏的省会是拉萨，青海的省会是西宁，宁夏的省会是银川，新疆的省会是乌鲁木齐。
```

## 推理性能评测

### 评测结果

|batch_size|seq_length|Atlas 800T A2（400T）tokens/s|A100（首次） tokens/s|对比
|----------|----------|----------|----------|----------|
|2|1024|45.16967126|36.73233689|1.229697729
|2|512|43.1641737|38.4874702|1.121512364
|2|256|39.14945113|38.0915182|1.027773452
|2|128|32.82671155|35.46970082|0.925486
|2|64|23.67107342|29.16003315|0.811764284
|2|32|10.86891748|16.52500627|0.657725468
|平均|-|32.47499976|32.41101092|1.001974293

### 评测流程

推理性能评测基于[MindSpore Lite推理](#mindspore-lite)进行。

- step1 生成增量推理的mindir文件

```shell
cd research
python skywork/run_skywork.py --config skywork/run_skywork_13b.yaml --load_checkpoint /path/Skywork-13B/skywork_13b.ckpt --run_mode=export --use_parallel False --use_past True --batch_size 1 --device_id 0
```

## ms方式开源数据集评测

### 评测结果

|  |ceval|mmlu|cmmlu|
|---|-----|----|-----|
|官网|60.6|62.1|61.8 |
|ms|60.63|62.14|61.83 |

### 评测流程

所用数据集为ceval、mmlu、cmmlu评测集。

评测代码参考目录[evaluate](../../scripts/examples/evaluate)

参数说明

```text
-d：数据集路径。
-c：模型文件路径，pytorch需指定到目录，mindformers需指定到ckpt文件。
-t：tokenizer.model文件路径。
--config：配置文件路径。
```

1. CEVAL

```shell
cd research/skywork
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir -p data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../

python ../../scripts/examples/evaluate/ceval/evaluate_ceval.py -d data/ceval/ -c /{path}/skywork_13b.ckpt -t /{path}/tokenizer.model --config run_skywork_13b.yaml --device_id 0
```

2. MMLU

```Shell
cd research/skywork
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

python ../../scripts/examples/evaluate/mmlu/evaluate_mmlu.py -d data/mmlu/data -c /{path}/skywork_13b.ckpt -t /{path}/tokenizer.model --config run_skywork_13b.yaml --device_id 1
```

3. CMMLU

```Shell
cd research/skywork
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../

python ../../scripts/examples/evaluate/cmmlu/evaluate_cmmlu.py -d data/cmmlu/ -c /{path}/skywork_13b.ckpt -t /{path}/tokenizer.model --config run_skywork_13b.yaml --device_id 7
```

## mslite方式开源数据集评测

### 评测结果

|  |ceval|mmlu|cmmlu|
|---|-----|----|-----|
|官网|60.6|62.1|61.8 |
|mslite|60.61|62.13|61.83 |

### 评测流程

所用数据集为ceval、mmlu、cmmlu评测集。

评测代码参考目录[evaluate](../../scripts/examples/evaluate)

参数说明

```text
-d：数据集路径。
-c：模型文件路径，pytorch需指定到目录，mindformers需指定到ckpt文件。
-t：tokenizer.model文件路径。
--config_path：GE配置文件context.cfg路径。
--full_model_path：导出的mindir路径。
```

**注** context.cfg文件生成和mindir的导出方式参考[MindSpore Lite推理](#MindSpore Lite推理)

1. CEVAL

```shell
cd research/skywork
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir -p data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../

python ../../scripts/examples/evaluate/ceval/evaluate_ceval_lite.py -d data/ceval --config_path context.cfg --token_path /{path}/tokenizer.model --full_model_path output/mindir_full_checkpoint/rank_0_graph.mindir --device_id 6
```

2. MMLU

```shell
cd research/skywork
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

python ../../scripts/examples/evaluate/mmlu/evaluate_mmlu_lite.py -d data/mmlu/data  --config_path context.cfg --token_path /{path}/tokenizer.model --full_model_path output/mindir_full_checkpoint/rank_0_graph.mindir --device_id 6
```

3. CMMLU

```shell
cd research/skywork
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir data/cmmlu
mv cmmlu_v1_0_1.zip data/cmmlu
cd data/cmmlu; unzip cmmlu_v1_0_1.zip
cd ../../

python ../../scripts/examples/evaluate/cmmlu/evaluate_cmmlu_lite.py -d data/cmmlu/  --config_path context.cfg --token_path /{path}/tokenizer.model --full_model_path output/mindir_full_checkpoint/rank_0_graph.mindir --device_id 6
```
