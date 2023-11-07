# 天工（Skywork）

Skywork-13B 系列是由昆仑万维研究的大规模语言预训练模型，目前有Skywork-13B-Base，Skywork-13B-Chat，Skywork-13B-Math，Skywork-13B-MM，MindFormers已支持Skywork-13B-Base。

## Skywork-13B-Base

Skywork-13B-Base 是在经过高质量清洗过滤的3.2万亿个多语言（主要是中文和英文）和代码数据上进行训练的，它在多种评测和各种基准测试上都展现了同等规模模型的最佳效果。

Skywork-13B-Base 是采用llama的模型结构设计，模型实现我们复用llama的代码。

## 前期准备

### 安装mindformers

参考[README](../../README.md) "mindformers安装" 安装mindformers。

### 环境要求

- 硬件: Ascend 910B
- MindSpore: 2.2.0
- MindSpore Lite: 2.2.0
- MindFormers: dev
- Mindpet: 1.0.2

**注** skywork-13b推理可以在单卡上完成部署，全量微调至少需要8卡。

### 生成RANK_TABLE_FILE(多卡运行必须环节)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注** 若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

### Skywork-13B-Base 预训练权重下载和转换

- 从huggingface下载原始权重后转换

需要将整个工程下载下来。

[Skywork-13B-Base](https://huggingface.co/Skywork/Skywork-13B-base)

执行权重转换脚本

```shell
cd research
python skywork/convert_ckpt.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface Skywork-13B-Base权重保存目录路径
MS_CKPT_NAME: 自定义mindspore权重文件保存路径和名称
```

**注**: 请安装torch=2.0.1和transformers=4.33.2版本，cuda版本11.6及以上

### 模型权重切分与合并

从huggingface或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 训练与微调

### 数据集准备

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

### 微调

目前提供了模型的基础配置文件`research/skywork/run_skywork_13b.yaml`。

`注：使用Skywork-13B-Base进行训练或者微调时，需要使用Skywork-13B-Base配套的tokenizer.model处理数据集，以及选用Skywork-13B-Base的yaml配置文件进行任务启动。`

- 单机多卡微调示例

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

## MindSpore推理

> 接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)
>
> 遵从Skywork-13B的license，本模型需要用户自行下载权重进行处理，故使用时和llama存在一定区别，具体如下：

由于天工模型的tokenizer需要用户自行下载，因此在启动前，请先行在配置文件中将tokenizer.model的路径配置完成，具体修改如下：

```yaml
# 增加 vocab_file: '/path/Skywork-13B/tokenizer.model' 这样一个配置即可
processor:
 return_tensors: ms
 tokenizer:
   unk_token: '<unk>'
   bos_token: '<s>'
   eos_token: '</s>'
   pad_token: '<pad>'
   vocab_file: '/path/Skywork-13B/tokenizer.model'
   type: LlamaTokenizer
```

如果使用增量推理，需要在配置文件中use_past值设置为True。

- generate接口推理：

```python
from mindspore import context
from mindformers.generation import GenerationConfig
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

context.set_context(device_id=0, mode=0)

# init skywork-13b-Base model
skywork_model_path = "/path/Skywork-13B/skywork_13b.ckpt"
config_path = 'skywork/run_skywork_13b.yaml'  # 填写实际路径

config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = skywork_model_path
skywork_config = LlamaConfig(**config.model.model_config)

skywork_model = LlamaForCausalLM(config=skywork_config)

# init skywork-13b-Base tokenizer
tokenizer_path = "/path/Skywork-13B/tokenizer.model"  # Skywork-13B tokenizer.model path
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

skywork_model_path = "/path/Skywork-13B/skywork_13b.ckpt"
config_path = 'skywork/run_skywork_13b.yaml'  # 填写实际路径
config = MindFormerConfig(config_path)
config.model.model_config.checkpoint_name_or_path = skywork_model_path
skywork_config = LlamaConfig(**config.model.model_config)

skywork_model = LlamaForCausalLM(skywork_config)

# init skywork-13b-Base tokenizer
tokenizer_path = "/path/Skywork-13B/tokenizer.model"  # Skywork-13B tokenizer.model path
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
                              use_past=True)

print(peline_result)

# 运行结果
[{'text_generation_text': ['陕西的省会是西安，西安是陕西的政治、经济、文化中心，也是陕西的交通枢纽。西安的交通非常发达，有很多的交通工具，可以方便的到达陕西的各个地方。\n西安的交通工具有：\n1、飞机：西安咸阳国际机场是中国重要的航空港，也是中国大陆第四大航空港。西安咸阳国际机场位于西安市西北方向，距市中心约30公里，有高速公路与市区相连。\n2、火车：西安火车站位于西安市解放路，是西安最大的火车站，也是中国西部地区最大']}]
```

## MindSpore Lite推理

### ckpt转换为mindir

```shell
# 如果需要使用增量推理，配置文件中use_past设置为True
cd research
python skywork/run_skywork.py --config skywork/run_skywork_13b.yaml --load_checkpoint /path/Skywork-13B/skywork_13b.ckpt --run_mode=export --train_dataset /path/Skywork-13B/eval4096.mindrecord --use_past True --device_id 0
```

设置use_past=True后生成的mindir有两个，分别在output/mindir_full_checkpoint和output/mindir_inc_checkpoint目录中。
如果不设置use_past或者use_past=False，则只生成mindir_full_checkpoint目录，后续无法使用增量推理。

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

    tokenizer_path = '/path/Skywork-13B/tokenizer.model'
    tokenizer = LlamaTokenizer(tokenizer_path, add_bos_token=True, add_eos_token=False)

    infer_model = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)
    if args.batch_size <= 1:
        batch_input = [
            "请帮我制定一下西安旅游计划",
            "陕西的省会是西安",
            "我喜欢看电影，因为",
            "请帮我制定一下北京旅游计划",
            "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"
        ]
    else:
        batch_input = [
            ["请帮我制定一下西安旅游计划" for i in range(args.batch_size)],
            ["陕西的省会是西安" for i in range(args.batch_size)],
            ["我喜欢看电影，因为" for i in range(args.batch_size)],
            ["请帮我制定一下北京旅游计划" for i in range(args.batch_size)],
            ["类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤" for i in range(args.batch_size)]
        ]

    for user_input in batch_input:
        output = infer_model.infer(user_input, pad_token_id=0, input_seq_length=args.input_seq_length, eos_token_id=2)
        for out in output:
            print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', default=0, type=int, help='ID of the target device')
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    parser.add_argument('--config_path', default=None, type=str, help="context config path")
    parser.add_argument('--seq_length', default=4096, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_seq_length', default=128, type=int)
    args = parser.parse_args()
    infer_main(args)
```

- step3. 使用shell命令启动推理

```shell
# 如果需要增量推理，使用inc_model_path指定路径，否则不需要
cd research
python skywork/run_skywork_infer_lite.py --full_model_path output/mindir_full_checkpoint/rank_0_graph.mindir --inc_model_path output/mindir_inc_checkpoint/rank_0_graph.mindir --config_path skywork/context.cfg --seq_length 4096 --input_seq_length 4096

# 运行结果：
# 请帮我制定一下西安旅游计划，谢谢！
# 西安的旅游景点很多，你可以根据自己的喜好来选择。
# 西安的旅游景点很多，你可以根据自己的喜好来选择。 1、兵马俑 秦始皇兵马俑博物馆位于西安临潼区秦始皇陵东1.5公里处，是秦始皇嬴政的陵墓。 2、大雁塔 大雁塔位于西安市的市中心，是西安的标志性建筑物之一。 3、华清池 华清池位于西安临潼区，是唐代皇家温泉园林和行宫。 4、秦始皇陵 秦始皇陵位于西安临潼区，是中国历史上第一位皇帝嬴政的陵墓。 5、大慈恩寺 大慈恩寺位于西安市的市中心，是西安的标志性建筑物之一。 6、西安城墙 西安城墙位于西安市中心区，是中国现存最完整的古城墙。 7、陕西省历史博物馆 陕西省历史博物馆位于西安市的市中心，是中国第一座大型现代化博物馆。 8、西安钟楼 西安钟楼位于西安市中心，是
```
