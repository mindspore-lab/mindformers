# ChatGLM3

## 模型描述

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：**更强大的基础模型**，**更完整的功能支持**，**更全面的开源序列**

```text
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```

## 仓库介绍

`chatGLM3-6B` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

    ```text
    mindformers/models/glm3
    ├── __init__.py
    └── glm3_tokenizer.py        # tokenizer
    ```

  glm3的模型结构和config同glm2

2. 模型配置：

    ```bash
    configs/glm3
    ├── predict_glm3_6b.yaml                              # 在线推理配置文件
    ├── run_glm3_6b_finetune_2k_800T_A2_64G.yaml          # Atlas 800T A2 最佳性能全量微调启动配置
    ├── run_glm3_6b_finetune_800T_A2_64G.yaml             # Atlas 800T A2 ADGEN 全量微调启动配置
    ├── run_glm3_6b_multiturn_finetune_800T_A2_64G.yaml   # Atlas 800T A2 多轮对话全量微调启动配置
    └── run_glm3_6b.yaml                                  # ChatGLM3配置模板
    ```

## 前期准备

### 生成RANK_TABLE_FILE

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
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

### 多机RANK_TABLE_FILE合并

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```bash
# 运行如下命令，合并每个机器上的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/merge_hccl.py hccl*.json
```

- step 3. 将合并后的`RANK_TABLE_FILE`文件拷贝到所有机器中，保证不同机器上的`RANK_TABLE_FILE`相同。

RANK_TABLE_FILE 双机16卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.0", "rank_id": "0"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.0", "rank_id": "1"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.0", "rank_id": "2"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.0", "rank_id": "3"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.1", "rank_id": "4"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.1", "rank_id": "5"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.1", "rank_id": "6"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.1", "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "xx.xx.xx.xx",
            "device": [
                {
                    "device_id": "0", "device_ip": "192.168.0.1", "rank_id": "8"
                },
                {
                    "device_id": "1", "device_ip": "192.168.1.1", "rank_id": "9"
                },
                {
                    "device_id": "2", "device_ip": "192.168.2.1", "rank_id": "10"
                },
                {
                    "device_id": "3", "device_ip": "192.168.3.1", "rank_id": "11"
                },
                {
                    "device_id": "4", "device_ip": "192.168.0.2", "rank_id": "12"
                },
                {
                    "device_id": "5", "device_ip": "192.168.1.2", "rank_id": "13"
                },
                {
                    "device_id": "6", "device_ip": "192.168.2.2", "rank_id": "14"
                },
                {
                    "device_id": "7", "device_ip": "192.168.3.2", "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

### 模型权重下载与转换

开发者可以下载获取官方权重后，通过下面提供的**权重转换脚本**，将官方权重转换为MindSpore权重；或直接使用MindFormers提供的**已转换权重**

1. 使用官方权重进行转换

   克隆glm3-6b代码仓，下载分布式的模型文件。

   ```shell
   git lfs install
   git clone https://huggingface.co/THUDM/chatglm3-6b
   ```

   执行 python 脚本，合并模型权重。

   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch

   tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
   model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

   with open("pt_model_arch.txt", "w") as fp:
       print(model, file=fp, flush=True)
   with open("pt_ckpt.txt", "w") as fp:
       for name, param in model.named_parameters():
           fp.write(f"{name} {param.shape} {param.dtype}\n")
   torch.save(model.state_dict(), "glm3_6b.pth")
   ```

   执行转换脚本，得到转换后的输出文件`glm3_6b.ckpt`。

   ```python
   import mindspore as ms
   import torch as pt
   from tqdm import tqdm

   pt_ckpt_path = "glm3_6b.pth"
   pt_param = pt.load(pt_ckpt_path)

   type_map = {"torch.float16": "ms.float16",
               "torch.float32": "ms.float32"}
   ms_param = []
   with open("check_pt_ckpt.txt", "w") as fp:
       for k, v in tqdm(pt_param.items()):
           if "word_embeddings.weight" in k:
               k = k.replace("word_embeddings.weight", "embedding_table")
           fp.write(f"{k} {v.shape} {v.dtype}\n")
           ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

   ms.save_checkpoint(ms_param, "glm3_6b.ckpt")
   ```

2. 获取MindFormers提供的已转换权重

   可通过from_pretrained接口下载，也可直接从下面的链接获取

   [glm3_6b权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/glm3_6b.ckpt)

   [tokenizer文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tokenizer.model)

### [分布式训练/微调权重合并](../feature_cards/Transform_Ckpt.md)

分布式训练/微调后所得到的权重文件为根据策略切分后的权重，需要手动将切分权重合一，以用于评估和推理。

涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

- step 1. 获取模型切分策略文件：

在执行微调脚本时，模型完成编译后，将会在`output/strategy`路径下生成各卡的切分策略文件，用于权重合并。

> 注：lora微调时需要确认配置文件`parallel context config`中`only_trainable_params`设为`False`，以获取所有参数完整策略。

- step 2. 运行`mindformers/tools/transform_ckpt.py`脚本进行多卡权重合并：

```shell
python transform_ckpt.py \
--src_ckpt_strategy {path}/output/strategy/ \
--src_ckpt_dir {path}/output/checkpoint/ \
--dst_ckpt_dir {path}/target_checkpoint/ \
--prefix glm3_6b
```

```text
# 参数说明
src_ckpt_strategy: 步骤1中的切分策略文件路径
src_ckpt_dir: 原切分权重文件夹
dst_ckpt_dir: 目标路径
prefix: ckpt文件前缀名
```

> 注：`transform_checkpoints` 接口当前仅mindspore 2.0以上版本支持，如当前硬件环境只支持2.0以下版本，可以新建conda环境安装mindspore 2.0的cpu版本以执行该脚本

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`./checkpoint_download/glm3`

```python
import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
# 2. 本地加载方式
# tokenizer = AutoProcessor.from_pretrained("/path/to/your.yaml").tokenizer

# 以下两种model的实例化方式选其一即可
# 1. 直接根据默认配置实例化
# model = AutoModel.from_pretrained('glm3_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('glm3_6b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 2048                 # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/path/to/your.ckpt"
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

role="user"

inputs_list=["你好", "请介绍一下华为", "晚上睡不着应该怎么办", "写一个快排算法"]

for input_item in inputs_list:
    history=[]
    inputs = tokenizer.build_chat_input(input_item, history=history, role=role)
    inputs = inputs['input_ids']
    # 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
    outputs = model.generate(inputs, do_sample=False, top_k=1, max_length=config.seq_length)
    for i, output in enumerate(outputs):
        output = output[len(inputs[i]):]
        response = tokenizer.decode(output)
        print(response)
# answer 1:
# 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。

# answer 2:
# 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。

# 华为的主要业务包括电信网络设备、智能手机、电脑和消费电子产品。公司在全球范围内有超过190,000名员工,其中约一半以上从事研发工作。华为以其高品质的产品和服务赢得了全球客户的信任和好评,也曾因其领先技术和创新精神而获得多项国际奖项和认可。

# 然而,华为也面临着来自一些国家政府的安全问题和政治压力,其中包括美国政府对其产品的禁令和限制。华为一直坚称自己的产品是安全的,并采取了一系列措施来确保其产品的安全性和透明度。

# answer 3:
#  晚上睡不着可以尝试以下方法:

# 1. 尝试放松身心,比如深呼吸、冥想、瑜伽等。

# 2. 避免饮用咖啡、茶、可乐等刺激性饮料。

# 3. 避免过度兴奋,比如看惊悚电影、玩刺激游戏等。

# 4. 保持规律的作息时间,尽量每天按时上床睡觉、按时起床。

# 5. 睡前适当运动,比如散步、慢跑等。

# 6. 睡前可以喝一杯温牛奶或者一些助眠的食品。

# 7. 如果长时间睡不着可以考虑咨询医生或心理咨询师。

# answer 4:
# 快速排序（Quick Sort）是一种常用的排序算法，其基本思想是通过一趟排序将待排序的数据分割成独立的两部分，其中一部分的所有数据都比另一部分的所有数据要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

# 下面是一个用Python实现的快速排序算法：

# ```python
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quick_sort(left) + middle + quick_sort(right)

# arr = [3,6,8,10,1,2,1]
# print(quick_sort(arr))
# ```

# 在这个实现中，我们首先判断输入数组的长度是否小于等于1，如果是，则直接返回数组，因为长度为1的数组本身就是有序的。否则，我们选择数组中间的元素作为基准值（pivot）。然后，我们将数组中的元素分成三部分：小于基准值的元素（left）、等于基准值的元素（middle）和大于基准值的元素（right）。接着，我们分别对left和right子数组进行递归调用quick_sort函数进行排序，并将排序后的结果与middle子数组连接起来，得到最终的排序结果。
```

如果需要加载本地词表，请修改配置文件中以下项：

  ```yaml
  processor:
    tokenizer:
      vocab_file: "/path/to/tokenizer.model"
  ```

## 微调

下面以 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集为例介绍代码的使用方法

### 数据集准备

#### 输入输出格式数据集

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{"content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳", "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，目录结构为

```text
AdvertiseGen
  ├── train.json
  └── dev.json
```

修改配置文件 `configs/glm3/run_glm3_6b_finetune*.yaml` 中的以下项：

```yaml
train_dataset: &train_dataset
    dataset_dir: "/path/to/AdvertiseGen/train.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 64
  max_target_length: 127

eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "/path/to/AdvertiseGen/dev.json"
    origin_columns: ["content", "summary"]
  tokenizer:
    vocab_file: "/path/to/tokenizer.model"
  max_source_length: 256
  max_target_length: 256
```

**注意**：微调时的模型`seq_length`需要等于微调数据集的`max_source_length + max_target_length + 1`。
yaml文件中默认的`seq_length: 192`以及`max_source_length: 64`和`max_target_length: 127`适用于ADGEN数据集，
其他数据集的`seq_length`设置，可以遍历并将数据集转换为token_id，取token_id最大长度，`seq_length`太大影响训练性能，
太小影响训练精度，需要做出权衡。

#### 多轮对话格式数据集

首先，克隆 [ToolAlpaca 数据集](https://github.com/tangqiaoyu/ToolAlpaca)，并下载处理脚本 [format_tool_alpaca.py](https://github.com/THUDM/ChatGLM3/blob/7cd5bc78bd6232d02764b60b33874bb2d63a0df0/finetune_chatmodel_demo/scripts/format_tool_alpaca.py)，然后执行脚本执行脚本：

```python
python mindformers/tools/format_tool_alpaca.py --path ToolAlpaca/data/train_data.json
```

脚本会在执行目录下生成 formatted_data/tool_alpaca.jsonl

也可以在这里下载处理好的数据集：

[tool_alpaca.jsonl](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm3/tool_alpaca.jsonl)

微调时选择配置文件：`configs/glm3/run_glm3_6b_multiturn_finetune*.yaml`

### 全参微调

全参微调使用 `configs/glm3/run_glm3_6b_finetune*.yaml` 配置文件，配置文件中定义了微调所需的各配置项

修改数据集/模型权重配置路径：

- 数据集：修改 `configs/glm3/run_glm3_6b_finetune*.yaml` 脚本中`train_dataset` 的 `dataset_dir` 为前文生成的数据集路径。
- 加载预训练模型权重：修改 `configs/glm3/run_glm3_6b_finetune*.yaml` 脚本中的 `load_checkpoint` 为预训练模型权重路径。

当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../feature_cards/Training_Algorithms.md#flash-attention)

#### 单卡微调

由于glm3_6b模型较大，全量微调不支持单卡运行

#### 多卡微调

- 单机多卡

多卡运行需要RANK_FILE_TABLE，请参考前期准备——[生成RANK_TABLE_FILE](#生成ranktablefile)

```shell
cd scripts
# Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [CONFIG_PATH] [DEVICE_RANGE] [RUN_STATUS]
bash run_distribute.sh /path/to/hccl_8p_01234567_127.0.1.1.json ../configs/glm3/run_glm3_6b_finetune*.yaml '[0,8]' finetune
# 将此处rank_table_file替换为实际路径
```

参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的glm3/run_glm3_6b_finetune*.yaml配置文件
DEVICE_RANGE: 为单机分布式卡的范围，如 '[0,8]' 为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

训练的log日志路径：mindformers/output/log

checkpoint存储路径：mindformers/output/checkpoint

- 多机多卡

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备——[多机RANK_TABLE_FILE合并](#多机ranktablefile合并)

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [0,8] finetune $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE path/to/config.yaml [$rank_start,$rank_end] finetune $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 推理

### 基本介绍

　　MindFormers 定位打造训练->微调->部署的端到端大模型工具套件，为了更好性能地部署已经微调训练好的大模型，我们利用MindSpore打造了全新的训推一体高性能推理引擎，保证训练与推理使用同一套脚本，为用户提供了开箱即用的推理部署方案，为用户提供端到端的大模型解决方案，帮助用户使能大模型业务。

　　MindSpore 大模型推理升级训推一体架构，实现脚本、分布式策略和运行时的统一，通过融合大算子降低推理时延，有效提升网络吞吐量。

### 基于generate的推理

下面提供一个模型推理样例脚本 `infer.py`

```python
import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
# 2. 本地加载方式
# tokenizer = AutoProcessor.from_pretrained("/path/to/your.yaml").tokenizer

# 以下两种model的实例化方式选其一即可
# 1. 直接根据默认配置实例化
# model = AutoModel.from_pretrained('glm3_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('glm3_6b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 2048                      # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/path/to/your.ckpt"
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

role="user"

inputs_list=["你好", "请介绍一下华为"]

for input_item in inputs_list:
    history=[]
    inputs = tokenizer.build_chat_input(input_item, history=history, role=role)
    inputs = inputs['input_ids']
    # 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
    outputs = model.generate(inputs, do_sample=False, top_k=1, max_length=config.seq_length)
    response = tokenizer.decode(outputs)
    for i, output in enumerate(outputs):
        output = output[len(inputs[i]):]
        response = tokenizer.decode(output)
        print(response)
    # answer 1:
    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。

    # answer 2:
    # 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。

    # 华为的主要业务包括电信网络设备、智能手机、电脑和消费电子产品。公司在全球范围内有超过190,000名员工,其中约一半以上从事研发工作。华为以其高品质的产品和服务赢得了全球客户的信任和好评,也曾因其领先技术和创新精神而获得多项国际奖项和认可。

    # 然而,华为也面临着来自一些国家政府的安全问题和政治压力,其中包括美国政府对其产品的禁令和限制。华为一直坚称自己的产品是安全的,并采取了一系列措施来确保其产品的安全性和透明度。

```

如果需要加载本地词表，请修改配置文件中以下项：

  ```yaml
  processor:
    tokenizer:
      vocab_file: "/path/to/tokenizer.model"
  ```

### 基于generate的多角色推理

下面提供一个模型推理样例。

```python
from copy import deepcopy

import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor


def process_response(output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content, history


# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
# 2. 本地加载方式
# tokenizer = AutoProcessor.from_pretrained("/path/to/your.yaml").tokenizer

# 以下两种model的实例化方式选其一即可
# 1. 直接根据默认配置实例化
# model = AutoModel.from_pretrained('glm3_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('glm3_6b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 8192                      # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/path/to/your.ckpt"
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

kwargs={}
gen_kwargs = {"max_length": config.seq_length,"num_beams": 1, "do_sample": False, "top_p": 1,"top_k": 1,
              "temperature": 1,**kwargs}

role="system"
text = "假设你现在是一个导游，请尽可能贴近这个角色回答问题。"
history = []
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第一个输入

outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
# 您好，我是您的人工智能助手，也可以是你的导游。请问有什么问题我可以帮您解答呢？
response, history = process_response(response, history)
print('history:', flush=True)
print(history, flush=True)

role="user"
text="我打算1月份去海南玩，可以介绍一下海南有哪些好玩的，好吃的么？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第二个输入
outputs = model.generate(inputs, **gen_kwargs) #, eos_token_id=eos_token_id)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
# 当然可以！海南是一个风景优美、气候宜人的热带海洋省份，拥有丰富的旅游资源和美食。以下是一些您可能会感兴趣的景点和美食：

# 1. 景点：
# - 海南岛：这是海南最著名的景点之一，拥有美丽的沙滩和热带雨林。
# - 亚龙湾：这是海南最著名的海滩之一，拥有柔软的沙滩和清澈的海水。
# - 南山寺：这是海南最著名的佛教寺庙之一，拥有精美的建筑和悠久的历史。
# - 博鳌亚洲论坛永久会址：这是中国最著名的国际会议中心，也是亚洲地区最重要的政治、经济、文化论坛之一。

# 2. 美食：
# - 海南鸡饭：这是海南最著名的美食之一，以鸡肉、米饭和椰汁为主要材料，味道鲜美。
# - 海鲜：海南的海鲜非常新鲜，您可以在当地的海鲜市场或餐厅品尝到各种海鲜美食，如清蒸海鲜、红烧海鲜等。
# - 椰子饭：这是海南最著名的传统美食之一，以椰子肉、糯米和椰子汁为主要材料，味道香甜。
# - 海南粉：这是海南最著名的传统小吃之一，以米粉、猪肉、花生、蔬菜等为主要材料，味道鲜美。

# 希望这些信息对您有所帮助，如果您还有其他问题，请随时问我。
response, history = process_response(response, history)

role="user"
text="哪里适合冲浪和潜水呢？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)

inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第三个输入

outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
# 在海南，冲浪和潜水的好去处有很多。以下是一些建议：

# 1. 冲浪：
# - 莺歌海：位于海南岛西海岸，是冲浪爱好者的天堂。这里的海浪适中，沙滩漂亮，非常适合冲浪。
# - 三亚：位于海南岛南端，是海南最著名的冲浪胜地之一。这里的沙滩细腻，海浪较大，非常适合冲浪。

# 2. 潜水：
# - 蜈支洲岛：位于海南岛东海岸，是海南最著名的潜水胜地之一。这里的潜水条件较好，能见度较高，水下生物丰富，非常适合潜水。
# - 西沙群岛：位于海南岛东南方向，是海南另一个著名的潜水胜地。这里的潜水条件非常好，水下生物丰富，适合各种级别的潜水爱好者。

# 当然，冲浪和潜水都需要一定的技能和经验，如果您是初学者，建议在专业人士的指导下进行。希望这些信息对您有所帮助，如果您还有其他问题，请随时问我。
response, history = process_response(response, history)

role="user"
text="可以帮我做一份旅游攻略吗？"
inputs = tokenizer.build_chat_input(text, history=history, role=role)
inputs = inputs['input_ids']
history.append({'role':role, 'content':text}) # 第四个输入
outputs = model.generate(inputs, **gen_kwargs)
outputs =outputs[0][len(inputs[0]):-1]
response = tokenizer.decode(outputs)
print(response, flush=True)
#  当然可以！以下是一份简要的海南旅游攻略，供您参考：

# 一、行程安排：
# 建议行程安排为7天6夜，具体如下：

# 第1天：抵达三亚，入住酒店，适应一下当地的气候和环境。

# 第2天：游览亚龙湾，享受阳光和沙滩，晚上可以品尝当地的美食。

# 第3天：游览南山寺，感受佛教文化的魅力，晚上可以前往三亚市区逛街购物。

# 第4天：前往蜈支洲岛，享受潜水和冲浪的乐趣，晚上可以在岛上住宿。

# 第5天：继续在蜈支洲岛游玩，探索更多的潜水点和冲浪场所，晚上可以在岛上住宿。

# 第6天：前往西沙群岛，进行一天一夜的潜水之旅，晚上返回三亚。

# 第7天：返回三亚，结束行程，离开海南。

# 二、注意事项：

# 1. 海南岛的气候比较热，建议您穿着轻便的衣物，并注意防晒。
# 2. 海南岛的海鲜美食丰富，但请注意食用安全，避免食物中毒。
# 3. 在海滩上要注意安全，避免在无人的海滩游泳，注意防晒和防水。
# 4. 潜水和冲浪需要一定的技能和经验，建议在专业人士的指导下进行。

# 希望这份攻略对您有所帮助，如果您还有其他问题，请随时问我。
response, history = process_response(response, history)

```

如果需要加载本地词表，请修改配置文件中以下项：

  ```yaml
  processor:
    tokenizer:
      vocab_file: "/path/to/tokenizer.model"
  ```

### 基于run_mindformer推理

#### 单卡推理

执行命令

```bash
python run_mindformer.py --use_parallel=False --config configs/glm3/predict_glm3_6b.yaml --run_mode predict --predict_data "[gMASK]sop<|user|> \n 你好<|assistant|> \n"
```

　　输出：

```bash
['[gMASK]sop<|user|> \n 你好<|assistant|> \n 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。']
```

## Q & A

### Q1: 网络训练 loss 不下降、网络训练溢出、`overflow_cond=True` 怎么办？

A1: 执行训练前设置环境变量：

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

重新启动训练。

### Q2: 推理速度非常慢、Mindspore只能跑在CPU上、报错中含有 `te`、`tbe`、`tvm`等字样？

A2: 一般是 Mindspore + Ascend 环境安装问题，确认环境安装过程参照
[安装指南](https://www.mindspore.cn/install/#%E6%89%8B%E5%8A%A8%E5%AE%89%E8%A3%85)并且成功设置了环境变量。执行：

```python
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```

假如执行输出：

```bash
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

并且没有报错，则说明成功安装了环境。

或许你想问，有没有更方便的环境安装方式？恭喜你，有的，我们还提供现成的
[docker镜像](http://mirrors.cn-central-221.ovaijisuan.com/mirrors.html)，可以依据需求自行取用。

### Q3: Sync stream Failed、exec graph xxx failed？

A3:这类报错较为宽泛，可以打开昇腾host日志进一步定位。

```bash
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

打开昇腾host日志后模型性能将明显下降，定位问题结束后需要取消昇腾日志：

```bash
unset ASCEND_GLOBAL_EVENT_ENABLE ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT
```

### Q4: the strategy is xxxxxx, shape xxxx cannot be divisible by value x

A4: 检查模型句长是否满足 `max_source_length + max_target_length + 1 = seq_length` 的要求。

### 仍然有疑问？欢迎向我们提出issue，我们将尽快为您解决

提问时麻烦提供以下信息：

1. 执行命令
2. 运行环境，包括硬件版本、CANN版本、Mindspore版本、Mindformers版本
3. 报错完整日志
