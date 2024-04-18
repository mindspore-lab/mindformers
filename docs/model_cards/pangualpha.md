# PanguAlpha

## 模型描述

「鹏程·盘古α」由以鹏城实验室为首的技术团队联合攻关，首次基于“鹏城云脑Ⅱ”和国产MindSpore框架的自动混合并行模式实现在2048卡算力集群上的大规模分布式训练，训练出业界首个2000亿参数以中文为核心的预训练生成语言模型。鹏程·盘古α预训练模型支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备很强的小样本学习能力。

[论文](https://arxiv.org/abs/2104.12369)J Wei Zeng, Xiaozhe Ren, Teng Su，et al., PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation, 2021

## 模型性能

- 基于Atlas 800

|                                              config                                              |      task       | Datasets  | metric |   score    | [train performance](#预训练) | [predict performance](#基于pipeline的推理) |
| :----------------------------------------------------------------------------------------------: | :-------------: | :-------: | :----: | :--------: | :--------------------------: | :----------------------------------------: |
|               [pangualpha_2_6b](../../configs/pangualpha/run_pangualpha_2_6b.yaml)               | text_generation | WikiText2 |   -    |     -      |       4075 tokens/s/p        |      19.5 tokens/s/p (use past True)       |
|                [pangualpha_13b](../../configs/pangualpha/run_pangualpha_13b.yaml)                | text_generation | WikiText2 |   -    |     -      |        575 tokens/s/p        |      12.5 tokens/s/p (use past True)       |
| [pangualpha_2_6b_prompt_txtcls](../../configs/pangualpha/run_pangualpha_2_6b_prompt_txtcls.yaml) | text_generation |   TNEWS   |  ACC   |   0.646    |              -               |                     -                      |
|         [pangualpha_2_6b_em_f1](../../configs/pangualpha/run_pangualpha_2_6b_em_f1.yaml)         | text_generation | CMRC2018  | Em/F1  | 2.10/21.12 |              -               |                     -                      |

## 仓库介绍

`PanguAlpha` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：`mindformers/models/pangualpha`

    ```bash
    pangualpha
        ├── __init__.py
        ├── convert_weight.py              # 权重转换脚本
        ├── pangualpha.py                  # 模型实现
        ├── pangualpha_config.py           # 模型配置项
        ├── pangualpha_processor.py        # Model预处理
        └── pangualpha_tokenizer.py        # tokenizer
    ```

2. 模型配置：`configs/pangualpha`

    ```bash
    pangualpha
        ├── run_pangualpha_2_6b.yaml                       # pangualpha_2_6b模型启动配置
        ├── run_pangualpha_13b.yaml                        # pangualpha_13b模型启动配置
        ├── run_pangualpha_2_6b_prompt_txtcls.yaml         # pangualpha_2_6b文本分类评测启动配置
        └── run_pangualpha_2_6b_em_f1.yaml             # run_pangualpha_2_6b阅读理解评测启动配置
    ```

3. 预处理脚本和任务启动脚本：`mindformers\tools\dataset_preprocess\pangualpha`

    ```bash
    pangualpha
        ├── pretrain_data_process.py     # wikitext-2等纯文本数据集预处理
        ├── cmrc2018_data_process.py     # cmrc2018数据集预处理
        └── tnews_data_process.py        # tnews数据集预处理
    ```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(多卡运行必须环节)

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

### 多机RANK_TABLE_FILE合并(多机多卡必备环节)

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
    [官方盘古Alpha权重下载](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)

    **下载清单：xxB_part0-4.tar，xxB_xxx_embedding.npy，pangu_alpha_xxB_ckpt_strategy.ckpt**
    需要全部下载xxB_part0-4.tar4个压缩包（解压后共有**512**个ckpt文件），**3**个不同的embedding.npy，以及对应参数的**strategy.ckpt**文件。

    下载完成后，首先解压4个压缩包到同一个文件夹`path/to/512ckpt`

    然后把3个不同的embedding.npy放置于同一个文件夹`path/to/embedding_dir`

    以上两个文件夹可以相同。

    然后运行如下转换脚本，将官方盘古Alpha的权重转换为完整的ckpt权重。

    ```shell
    python mindformers/models/pangualpha/convert_weight.py --config_path_or_name path/to/config --official_strategy_path path/to/pangu_alpha_13B_cktp_strategy.ckpt --official_ckpt_dir path/to/512ckpt --official_npy_dir path/to/embedding_dir --ckpt_save_path path/to/pangualpha.ckpt
    ```

    ```text
    # 参数说明
    config_path_or_name: 需要转换的模型配置文件，例如：'pangualpha_13b'或者 'path/to/run_pangualpha_13b.yaml'
    official_strategy_path: 官方权重的切分策略文件，例如pangu_alpha_13B_ckpt_strategy.ckpt
    official_ckpt_dir：官方权重文件夹，即path/to/512ckpt，存放了解压后的512个ckpt文件
    official_npy_dir：官方embedding文件夹，即path/to/embedding_dir，存放了3个不同的embedding.npy文件
    ckpt_save_path：你想存储最终转换完成的权重的路径以及权重名称
    ```

2. 获取MindFormers提供的已转换权重
   可通过from_pretrained接口下载，也可直接从下面的链接获取
    [MindFormers盘古Alpha2.6B权重下载](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/pangualpha/pangualpha_2_6b.ckpt)

    [MindFormers盘古Alpha13B权重下载](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/pangualpha/pangualpha_13b.ckpt)

### [模型权重切分与合并](../feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档模型[权重切分与合并](../feature_cards/Transform_Ckpt.md)

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/pangualpha`

```python
import mindspore
from mindformers import AutoModel, AutoTokenizer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

tokenizer = AutoTokenizer.from_pretrained('pangualpha_2_6b')
model = AutoModel.from_pretrained('pangualpha_2_6b')

inputs = tokenizer("上联：欢天喜地度佳节 下联：")
outputs = model.generate(inputs["input_ids"], max_length=100)
response = tokenizer.decode(outputs)[0]
print(response)
# 上联:欢天喜地度佳节 下联:笑逐颜开迎佳期 横批:幸福快乐<eot>'
```

**注：快速使用仅限单卡，该示例支持2.6B和13B模型。**
**注：多卡请参考[基于generate的推理](#基于generate的推理)。**

### 基于Trainer的快速训练，微调，评测，推理

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化预训练任务
trainer = Trainer(task='text_generation',
                  model='pangualpha_2_6b',
                  train_dataset='path/to/train_dataset',
                  eval_dataset='path/to/eval_dataset')

# 开启预训练
trainer.train()

# 开启全量微调
trainer.finetune()

# 开启评测
trainer.evaluate()

# 开启推理
predict_result = trainer.predict(input_data="上联：欢天喜地度佳节 下联：")
# output result is: [{'text_generation_text': ['上联:欢天喜地度佳节 下联:笑逐颜开迎佳期 横批:幸福快乐<eot>']}]
```

**注：快速使用仅限单卡，该示例在Atlas 800仅支持2.6B和13B的evaluate和predict，在Atlas 800T A2支持2.6Btrain和finetune及2.6B和13B的evaluate和predict。**
**注：多卡请参考[使用高阶接口开发教程](https://mindformers.readthedocs.io/zh_CN/latest/docs/practice/Develop_With_Api.html)。**

### 基于Pipeline的快速推理

```python
import mindspore
from mindformers.pipeline import pipeline

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline("text_generation", model='pangualpha_2_6b', max_length=50)
pipeline_result = pipeline_task("上联：欢天喜地度佳节 下联：", top_k=3)
print(pipeline_result)
# [{'text_generation_text': ['上联:欢天喜地度佳节 下联:笑逐颜开庆佳节 横批:欢度佳节<eot>']}]
```

**注：快速使用仅限单卡，该示例支持2.6B和13B模型。**
**注：多卡请参考[基于pipeline的推理](#基于pipeline的推理)。**

## 预训练

### 数据集准备-预训练

以Wikitext2数据集为例

- 数据集下载：[WikiText2数据集](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip)

- 词表下载：[model.vocab](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/pangualpha/vocab.model)

将数据处理成Mindrecord格式。注：训练数据处理时，长度应等于模型接收长度加一。

```bash
cd mindformers/tools/dataset_preprocess/pangualpha
# 生成Mindrecord数据，其中output_file需以字符串mindrecord结尾
# 训练
python pretrain_data_process.py --input_glob  'data/*.txt' --tokenizer jieba --eot 40000 --data_column_name input_ids --seq_length 1025
# 评测
python pretrain_data_process.py --input_glob  'data/*.txt' --tokenizer jieba --eot 40000 --data_column_name input_ids --seq_length 1024
```

### 脚本启动

#### 单卡训练

**注：在Atlas 800上无法单卡训练pangualpha模型。**
**注：在Atlas 800T A2上单卡训练需要修改`pangualpha_2_6b.yaml`配置文件中`max_device_memory`为`57GB`，`batch_size`减小为`2`。**

```yaml
# context
context:
  mode: 0 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  enable_graph_kernel: False
  graph_kernel_flags: "--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true"
  max_call_depth: 10000
  max_device_memory: "57GB"
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0
# runner
runner_config:
  epochs: 1
  batch_size: 2
  sink_mode: True
  sink_size: 2
```

- python启动

```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b.yaml --run_mode train --use_parallel False
```

- bash启动

```bash
cd scripts
bash run_standalone.sh ../configs/pangualpha/run_pangualpha_2_6b.yaml [DEVICE_ID] train
```

#### 多卡训练

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/pangualpha/run_pangualpha_2_6b.yaml [0,8] train 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

在每台机器上启动`bash run_distribute.sh`。

**注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE ../configs/pangualpha/run_pangualpha_2_6b.yaml [0,8] train $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE ../configs/pangualpha/run_pangualpha_2_6b.yaml [$rank_start,$rank_end] train $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

## 微调

### 全参微调

#### 单卡微调

**注：在Atlas 800上无法单卡全参微调pangualpha模型。**
**注：在Atlas 800T A2上单卡全参微调需要修改`pangualpha_2_6b.yaml`配置文件中`max_device_memory`为`57GB`，`batch_size`减小为`2`。**

```yaml
# context
context:
  mode: 0 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  enable_graph_kernel: False
  graph_kernel_flags: "--disable_expand_ops=Softmax,Dropout --enable_parallel_fusion=true --reduce_fuse_depth=8 --enable_auto_tensor_inplace=true"
  max_call_depth: 10000
  max_device_memory: "57GB"
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0
# runner
runner_config:
  epochs: 1
  batch_size: 2
  sink_mode: True
  sink_size: 2
```

- python启动

```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b.yaml --run_mode finetune
```

- bash启动

```bash
cd scripts
bash run_standalone.sh ../configs/pangualpha/run_pangualpha_2_6b.yaml [DEVICE_ID] finetune
```

#### 多卡微调

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡

```bash
cd scripts
bash run_distribute.sh RANK_TABLE_FILE ../configs/pangualpha/run_pangualpha_2_6b.yaml [0,8] finetune 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡

**注：需要保证执行的节点和RANK_TABLE_FIEL的节点顺序保持一致，即rank_id匹配。**

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

## 评测

### 文本分类

#### 数据集准备-文本分类

- 获取数据集: [TNEWS数据集](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

- 处理数据成mindrecord格式

```bash
# 注：生成的数据集文件需以.mindrecord结尾
cd mindformers/tools/dataset_preprocess/pangualpha
python tnews_data_process.py --input_file {your_path/dev.json} \
                             --label_file {your_path/labels.json} \
                             --output_file {your_path/tnews.mindrecord}
```

#### 单卡评测

```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b_prompt_txtcls.yaml \
                            --eval_dataset_dir {your_path/tnews.mindrecord} \
                            --run_mode eval
# ACC: 0.646, total_acc_num: 6458, total_num: 10000
```

### 阅读理解

#### 数据集准备-阅读理解

- 获取数据集: [CMRC2018数据集](https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip)是用于中文机器阅读理解的片段抽取任务(Span-Extraction)的数据，这个数据集由近20000个真实的问题组成，这些问题由人类专家在维基百科的段落中注释。

- 处理数据成mindrecord格式

```bash
# 注：生成的数据集文件需以.mindrecord结尾
cd mindformers/tools/dataset_preprocess/pangualpha
python cmrc2018_data_process.py --train_file {your_path/train.json} \
                                    --dev_file {your_path/dev.json} \
                                    --output_file {your_path/cmrc2018.mindrecord}
```

#### 单卡评测

```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b_prompt_txtcls.yaml \
                          --eval_dataset_dir {your_path/tnews.mindrecord} \
                          --run_mode eval
# ACC: 0.646, total_acc_num: 6458, total_num: 10000
```

## 推理

### 基于pipeline的推理

以下为基于pipeline接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["上联：欢天喜地度佳节 下联：",
              "四川的省会是哪里？",
              "李大钊如果在世，他会对今天的青年人说："]

    # set model config
    model_config = AutoConfig.from_pretrained("pangualpha_2_6b")
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("pangualpha_2_6b")
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard pangualpha and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
    outputs = text_generation_pipeline(inputs)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path CHECKPOINT_PATH &> minformers_$RANK_ID.log &
done
```

#### 单卡pipeline推理

```bash
python predict_custom.py
```

#### 多卡pipeline推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/pangualpha_2_6b_shard_checkpoint_dir
```

### 基于generate的推理

以下为基于model.generate接口的自定义推理脚本，支持多卡多batch推理。

```python
# predict_custom.py 文件
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindformers import AutoConfig, AutoTokenizer, AutoModel
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    # 初始化单卡/多卡环境
    context_init(use_parallel, device_id)

    # 多batch输入
    inputs = ["上联：欢天喜地度佳节 下联：",
              "四川的省会是哪里？",
              "李大钊如果在世，他会对今天的青年人说："]

    # set model config
    model_config = AutoConfig.from_pretrained("pangualpha_2_6b")
    model_config.batch_size = len(inputs)
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path
    print(f"config is: {model_config}")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained("pangualpha_2_6b")
    # build model from config
    model = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(get_real_rank()))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard pangualpha and load sharded ckpt
        model = Model(model)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs_ids = tokenizer(inputs, max_length=model_config.max_decode_length, padding="max_length")["input_ids"]
    outputs = model.generate(inputs_ids, max_length=model_config.max_decode_length)
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int,
                        help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help='whether use past.')
    args = parser.parse_args()

    main(args.use_parallel,
         args.device_id,
         args.checkpoint_path,
         args.use_past)
```

以下为多卡运行自定义多batch推理的脚本

```bash
# >>> `run_predict.sh`文件
CHECKPOINT_PATH=$2
export RANK_TABLE_FILE=$1

# define variable
export RANK_SIZE=8
export START_RANK=0 # this server start rank
export END_RANK=8 # this server end rank

# run
for((i=${START_RANK}; i<${END_RANK}; i++))
do
    export RANK_ID=$i
    export DEVICE_ID=$((i-START_RANK))
    echo "Start distribute running for rank $RANK_ID, device $DEVICE_ID"
    python3 ./predict_custom.py --use_parallel True --checkpoint_path CHECKPOINT_PATH &> minformers_$RANK_ID.log &
done
```

#### 单卡generate推理

```bash
python predict_custom.py
```

#### 多卡generate推理

```bash
bash run_predict.sh RANK_TABLE_FILE path/to/pangualpha_2_6b_shard_checkpoint_dir
```

### 脚本启动

#### 单卡推理

```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b.yaml --run_mode predict --predict_data 上联：欢天喜地度佳节 下联： --use_parallel False
# output result is: [{'text_generation_text': ['上联:欢天喜地度佳节 下联:笑逐颜开迎佳期 横批:幸福快乐<eot>']}]
```

**注**：要提高推理速度，可在对应模型配置文件中进行如下配置，设置增量推理`use_past`为True。

```yaml
# model config
use_past: True          # 开启增量推理
use_moe: False
expert_num: 1
per_token_num_experts_chosen: 1
checkpoint_name_or_path: "pangualpha_2_6b"
repetition_penalty: 1
max_decode_length: 1024
top_k: 3
top_p: 1
do_sample: False
```

## [mindspore-lite](../feature_cards/Inference.md)

如需导出模型，使用mindspore-lite进行离线推理请参考[推理特性使用文档](../feature_cards/Inference.md)
