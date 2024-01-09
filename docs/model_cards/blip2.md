# BLIP-2

## 模型描述

BLIP-2: 全名`Bootstrapping Language-Image Pre-training - 2`模型是2023 年 Salesforce提出的一种多模态模型，它从现成的冻结预训练图像编码器 (ViT)和冻结的大型语言模型 (LLM)中引导视觉语言预训练 (contrastive_language_image_pretrain), 中间添加一个轻量级的 Querying Transformer 弥补了模态 gap, 该 Transformer 分两个阶段进行预训练:

- 第一阶段从冻结图像编码器引导视觉-语言表示学习，强制 Q-Former 学习与文本最相关的视觉表示。
- 第二阶段基于冻结的语言模型引导从视觉到语言的生成学习，将Q-Former的输出连接到冻结的LLM，并对Q-Former进行训练，使其输出视觉表示能够被LLM解释。

[论文](https://arxiv.org/pdf/2301.12597.pdf) Junnan Li，et al., BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, 2023

``` text
@inproceedings{li2023blip2,
      title={{BLIP-2:} Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
      author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
      year={2023},
      booktitle={ICML},
}
```

## 模型性能（包括设备性能+评测指标）

- 基于Atlas 800

|                                                                              config                                                                               |                 task                 |                Datasets                 |  metric   |           score            | [train performance](#预训练) | [predict performance](#基于pipeline的推理) |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------:|:---------------------------------------:|:---------:|:--------------------------:|:-------------------------:|:-------------------------------------:|
|                 [blip2_stage1_vit_g](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml)                 |        BLIP-2 stage1 pretrain        | [train]: coco  [eval]: flickr30k (test) | itm_score | txt_r1: 89.8 img_r1: 77.08 |      49.97 samples/s      |                   -                   |
| [blip2_stage1_classification](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml) |         image_classification         |            [eval]: cifar100             | accuracy  |             -              |             -             |             5.61 iters/s              |
|                [blip2_stage2_vit_g_llama_7b](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml)                 | BLIP-2 stage2 pretrain with llama_7b |            [train]: coco2014            |     -     |             -              |       37 samples/s        |                   -                   |
|  [itt_blip2_stage2_vit_g_llama_7b](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_stage2_vit_g_llama_7b_image_to_text_generation.yaml)  |       image_to_text_generation       |                    -                    |     -     |             -              |             -             |      22 tokens/s(use past True)       |

## 仓库介绍

`BLIP-2` 基于 `mindformers` 实现，目前支持一阶段的训练，评估，推理以及二阶段的训练，推理（语言模型包含Llama7b及Baichuan7b）。 在二阶段使用`baichuan_7b`作为语言模型时进行训练/推理时，需要自行在配置文件`configs/blip2/run_blip2_stage2_vit_g_baichuan_7b_image_to_text_generation.yaml`中
配置对应的baichuan-7b权重文件及词表文件，如何配置可参考上文中关键配置项说明章节；权重文件和词表文件的获取及转换参考[baichuan_7b](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/baichuan.md).

本实现主要涉及的文件有：

1. 模型具体实现：`mindformers/models/blip2`

    ```bash
    blip2
        ├── __init__.py
        ├── convert_weight.py         # 权重转换脚本
        ├── blip2.py                  # 模型基础类实现
        ├── qformer.py                # QFormer实现
        ├── blip2_qformer.py          # BLIP-2一阶段QFormer实现
        ├── blip2_vit.py              # BLIP-2使用的vit模型实现
        ├── blip2_llama.py            # BLIP-2二阶段接入Llama模型实现
        ├── blip2_llm.py              # BLIP-2二阶段训练及推理类
        ├── qformer_config.py         # QFormer配置项
        ├── blip2_config.py           # BLIP-2配置项，包含QFormer配置项
        └── blip2_processor.py        # Model预处理
    ```

2. 模型配置：`configs/blip2`

    ```bash
    blip2
        ├── run_blip2_stage1_vit_g_qformer_pretrain.yaml                           # BLIP-2一阶段预训练启动配置
        ├── run_blip2_stage1_vit_g_retrieval_flickr30k.yaml                        # BLIP-2使用一阶段预训练模型做图像检索任务启动配置
        ├── run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml    # BLIP-2使用一阶段预训练模型做zero-shot图像分类任务启动配置
        └── run_blip2_stage2_vit_g_baichuan_7b.yaml                                # BLIP-2二阶段预训练启动配置（使用baichuan7b作为语言模型）
        └── run_blip2_stage2_vit_g_baichuan_7b_image_to_text_generation.yaml       # BLIP-2使用二阶段预训练做图生问任务启动配置（使用baichuan7b作为语言模型）
        └── run_blip2_stage2_vit_g_llama_7b.yaml                                   # BLIP-2二阶段预训练启动配置（使用llama7b作为语言模型）
        └── run_blip2_stage2_vit_g_llama_7b_image_to_text_generation.yaml          # BLIP-2使用二阶段预训练做图生问任务启动配置（使用llama7b作为语言模型）
    ```

3. 配置文件关键配置项

```yaml
train_dataset:
  text_transforms:
    type: CaptionTransform
    prompt: ""        # 二阶段中语言模型添加的prompt，目前仅支持固定prompt
    max_length: 32    # token ids的最大长度，在二阶段训练时，为了构造label，需要设置为33，即比预期最大长度加1

  tokenizer:
    type: LlamaTokenizer
    vocab_file: ""           # 词表文件路径

model:
  model_config:
    type: Blip2Config
    max_txt_len: 32   # token ids的最大长度，需要与train_dataset.text_transforms.max_length保持一致
    checkpoint_name_or_path: ""  # 模型的预训练权重，在二阶段训练时，可在此处配置一阶段训练得到的权重
    prompt: False            # 二阶段训练中，是否使用prompt
    prompt_length: 0         # 二阶段训练中，使用的固定prompt的token长度，具体值根据tokenizer及train_dataset.text_transforms.prompt得到
                             # 以上两项配置与train_dataset.text_transforms.prompt需要对应
    vision_config:           # ImageEncoder的相关配置
      type: ViTConfig
      image_size: 224        # 输入图像大小
      checkpoint_name_or_path: "vit_g_p16"  # vit的权重文件

    qformer_config:
      type: QformerConfig
      query_length: 32       # query数目

    text_config:             # 二阶段语言模型相关配置
      type: LlamaConfig
      seq_length: 64        # 语言模型的输入seq_length大小，在训练时该值要等于qformer的query数目+语言模型输入文字token id长度
                             # 即 seq_length=model.model_config.qformer_config.query_length + model.model_config.max_txt_len
      checkpoint_name_or_path: ""  # 语言模型权重文件
  arch:
    type: XXXX               # 根据BLIP-2的不同阶段填写不同模型结构，
                             # 一阶段：1）预训练或图像检索 Blip2Qformer 2) zero-shot图像分类  Blip2Classifier 3）
                             # 二阶段：推理/图生文任务 Blip2ImageToTextGeneration 使用llama7b或baichuan7b作为语言模型）

# 推理任务时需要配置
processor:
  type: Blip2Processor
  image_processor:
    type: Blip2ImageProcessor
    image_size: 224          # 输入图像大小
  tokenizer:
    type: LlamaTokenizer
    max_length: 32
    vocab_file: ""           # 词表文件路径
```

## 前期准备

### [mindformers安装](../../README.md#二mindformers安装)

### 生成RANK_TABLE_FILE(**多卡运行必须环节**)

运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

> 注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成

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

### 多机RANK_TABLE_FILE合并(**多机多卡必备环节**)

- step 1. 首先根据上章节内容，在每个机器上生成各自的`RANK_TABLE_FILE`文件，然后将不同机器上生成的`RANK_TABLE_FILE`文件全部拷贝到同一台机器上。

```bash
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)" --server_ip xx.xx.xx.xx
```

**注：需要根据机器的ip地址指定 --server_ip，避免由于不同机器server_ip不同，导致多节点间通信失败。**

- step 2. 运行mindformers/tools/merge_hccl.py将不同机器上生成的`RANK_TABLE_FILE`文件合并

```shell
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

本仓库中的`blip2_stage1_classification`来自于LAVIS的一阶段预训练权重`blip2_stage1_pretrained`, 基于下述的步骤获取：

1. 从[此链接](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth)中下载`blip2_stage1_pretrained`的pytorch权重，文件名为`blip2_pretrained.pth`

2. 执行转换脚本，得到转换后的输出文件`blip2_stage1_pretrained.ckpt`

权重转换步骤+权重转换命令

```bash
python mindformers/models/blip2/convert_weight.py --torch_path blip2_pretrained.pth --mindspore_path ./blip2_stage1_pretrained.ckpt
```

### 模型权重切分与合并

暂不涉及

## 基于API的快速使用

### 基于AutoClass的快速使用

可以使用AutoClass接口，通过模型名称获取相应的model/preprocess/tokenizer等实例，并自动下载并加载权重

`from_pretrained()` 接口会自动从云上下载预训练的模型，存储路径：`mindformers/checkpoint_download/model_name`

```python
import mindspore
from mindformers import AutoModel, AutoProcessor
from mindformers.tools.image_tools import load_image

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 通过AutoClass创建一阶段预训练任务
model = AutoModel.from_pretrained("blip2_stage1_vit_g")


# 通过AutoClass创建二阶段图生文任务
model = AutoModel.from_pretrained("itt_blip2_stage2_vit_g_llama_7b")
processor = AutoProcessor.from_pretrained("itt_blip2_stage2_vit_g_llama_7b")
tokenizer = processor.tokenizer
filepath = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009448.jpg"
input_images = processor.image_processor(load_image(filepath))
input_ids = tokenizer("it is a photo of", padding="max_length", return_tensors="ms")["input_ids"]
outputs = model.generate_text_for_image(input_images, input_ids)
response = tokenizer.decode(outputs, skip_special_tokens=True)
print(response)
# ['it is a photo of a girl holding an umbrella']
```

**注：快速使用仅限单卡**

### 基于Trainer的快速训练，微调，评测，推理

- `BLIP-2`一阶段训练

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化图像-文本数据集配置
data_loader = dict(
    type = 'MultiImgCapDataLoader',
    dataset_dir = "/data",
    annotation_files = [
      "vg/annotations/vg_caption.json",
      "coco2014/coco/annotations/coco_karpathy_train.json"
    ],
    image_dirs = [
     "vg/images",
      "coco2014/coco/images"
    ],
    stage = "train",
    column_names = ["image", "text"],
    shuffle = True,
)

# 通过修改args参数间接修改配置文件中的dataset设置，而且不改变transform过程
dataset_config = dict(data_loader=data_loader)
train_dataset_task = dict(dataset_config=dataset_config)
args = dict(train_dataset_task=train_dataset_task,
           train_dataset=dataset_config)

# BLIP-2一阶段初始化预训练任务
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='blip2_stage1_vit_g',
    args=args)

# 开启预训练
trainer.train()
```

- `BLIP-2`一阶段评估

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化图像-文本数据集配置
data_loader = dict(
    type = 'MultiImgCapDataLoader',
    dataset_dir = "/data",
    annotation_files = [
      "flickr30k/annotations/test.json"
    ],
    image_dirs = [
     "flickr30k/images"
    ],
    stage = "eval",
    column_names = ["image", "text"],
    shuffle = False,
)

# 通过修改args参数间接修改配置文件中的dataset设置，而且不改变transform过程
dataset_config = dict(data_loader=data_loader)
eval_dataset_task = dict(dataset_config=dataset_config)
args = dict(eval_dataset_task=eval_dataset_task,
           eval_dataset=dataset_config)

# 初始化评估任务
trainer = Trainer(task='image_to_text_retrieval',
    model='blip2_stage1_evaluator',
    args=args)

# 开启评估
trainer.evaluate()
```

- `BLIP-2`一阶段推理

```python
from mindformers.tools.image_tools import load_image
from mindformers import Trainer

cls_trainer = Trainer(task='zero_shot_image_classification',
                      model='blip2_stage1_classification',
                      candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
# 加载输入，一张太阳花图片
input_data = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 加载默认的权重以完成推理
predict_result = cls_trainer.predict(input_data=input_data)
print(predict_result)
# 输出
# [[{'score': 0.99999976, 'label': 'sunflower'}]]
```

- `BLIP-2`二阶段训练

```python
import mindspore
from mindformers.trainer import Trainer

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

# 初始化图像-文本数据集配置
data_loader = dict(
    type = 'MultiImgCapDataLoader',
    dataset_dir = "/data",
    annotation_files = [
      "coco2014/coco/annotations/coco_karpathy_train.json"
    ],
    image_dirs = [
      "coco2014/coco/images"
    ],
    stage = "train",
    column_names = ["image", "text"],
    shuffle = True,
)

# 通过修改args参数间接修改配置文件中的dataset设置，而且不改变transform过程
dataset_config = dict(data_loader=data_loader)
train_dataset_task = dict(dataset_config=dataset_config)
args = dict(train_dataset_task=train_dataset_task,
           train_dataset=dataset_config)

# BLIP-2二阶段初始化预训练任务(使用llama_7b作为语言模型)
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='blip2_stage2_vit_g_llama_7b',
    args=args)

# 开启预训练
trainer.train()
```

- `BLIP-2`二阶段推理

```python
import mindspore; mindspore.set_context(mode=0, device_id=0)
from mindformers.tools.image_tools import load_image
from mindformers import Trainer

generate_trainer = Trainer(task='image_to_text_generation',
                           model='itt_blip2_stage2_vit_g_llama_7b')
# 加载输入，一张太阳花图片
input_data = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 加载指定的权重以完成推理
predict_result = generate_trainer.predict(input_data=input_data,
                                          hypothesis_template="a picture of")
print(predict_result)
# 输出
# ['a picture of a yellow flower']
```

注：基于`Baichuan7b`的`BLIP-2`未提供预训练模型。

### 基于Pipeline的快速推理

- `BLIP-2`一阶段推理

```python
import mindspore
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image

# 指定图模式，指定使用训练卡id
mindspore.set_context(mode=0, device_id=0)

pipeline_task = pipeline(task="zero_shot_image_classification", model="blip2_stage1_classification")

input_data = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(input_data,
                                candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
                                hypothesis_template="This is a photo of {}.")
print(pipeline_result)
# 输出
# [[{'score': 0.99999714, 'label': 'sunflower'},
#   {'score': 1.315181e-06, 'label': 'tree'},
#   {'score': 7.0368844e-07, 'label': 'toy'},
#   {'score': 4.7594781e-07, 'label': 'dog'},
#   {'score': 3.93686e-07, 'label': 'cat'}]]

```

- `BLIP-2`二阶段推理

```python
import mindspore as ms

from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image

# 指定图模式，指定使用训练卡id
ms.set_context(mode=0, device_id=0)

pipeline_task = pipeline(task="image_to_text_generation", model="itt_blip2_stage2_vit_g_llama_7b")

# 加载输入，一张太阳花图片
input_data = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

predict_result = pipeline_task({
    "image": input_data,
    "prompt": "a picture of"})
print(predict_result)
# 输出
# ['a picture of a yellow flower']
```

**注：快速使用仅限单卡**

## 预训练

### 数据集准备-预训练

- coco2014数据集:

```shell
├── annotations
│   ├── coco_karpathy_test.json
│   ├── coco_karpathy_train.json
│   └── coco_karpathy_val.json
│
└── images
    ├── test2014
    ├── test2015
    ├── train2014
    └── val2014
```

可通过LAVIS github库提供的链接下载annotation_files ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml)) 和 images ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py))。

- Visual Genome数据集:

```shell
├── annotations
│   └── vg_caption.json
└── images
    └── VG_100K
```

同上，可通过LAVIS github库提供的链接下载annotation_files ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/vg/defaults_caption.yaml)) 和 images ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_vg.py))。

### 脚本启动

#### 单卡训练

- python启动训练`BLIP-2`

```bash
# 一阶段训练
python run_mindformer.py --config configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml --run_mode train

# 二阶段训练
python run_mindformer.py --config configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml --run_mode train
```

- bash启动训练`BLIP-2`

```bash
cd scripts

# 一阶段训练
bash run_standalone.sh --config configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml [DEVICE_ID] train

# 二阶段训练
bash run_standalone.sh --config configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml [DEVICE_ID] train
```

#### 多卡训练

多卡运行需要RANK_FILE_TABLE，请参考前期准备-[生成RANK_TABLE_FILE](#生成rank_table_file多卡运行必须环节)

- 单机多卡训练`BLIP-2`

```bash
cd scripts

# 一阶段训练
bash run_distribute.sh RANK_TABLE_FILE --config configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml [0,8] train 8

# 二阶段训练
bash run_distribute.sh RANK_TABLE_FILE --config configs/blip2/run_blip2_stage2_vit_g_llama_7b.yaml [0,8] train 8
```

多机多卡运行需要合并不同机器的RANK_FILE_TABLE，参考前期准备-[多机RANK_TABLE_FILE合并](#多机rank_table_file合并多机多卡必备环节)

- 多机多卡训练`BLIP-2`

在每台机器上启动`bash run_distribute.sh`。

```bash
server_count=12
device_num=8*$server_count
# launch ranks in the 0th server
cd scripts
bash run_distribute.sh $RANK_TABLE_FILE --config configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml [0,8] train $device_num

# launch ranks in the 1-11 server via ssh
for idx in {1..11}
do
    let rank_start=8*$idx
    let rank_end=$rank_start+8
    ssh ${IP_LIST[$idx]} "cd scripts; bash run_distribute.sh $RANK_TABLE_FILE --config configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml [$rank_start,$rank_end] train $device_num"
done
```

其中

- `RANK_TABLE_FILE`为上一步汇总并分发的总rank table文件；
- `IP_LIST`为12台服务器的IP地址。如192.168.0.[0-11]

```bash
IP_LIST=("192.168.0.0", "192.168.0.1", ..., "192.168.0.11")
```

### 全参微调与Lora微调

暂不支持

## 评测

### BLIP-2一阶段评测：图像检索

### 数据集准备-图像检索

- 获取数据集 (flickr30k):
    - images: [链接](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data)
    - annotations: [train](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json), [eval](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json), [test](https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json)

- 将下载的图像和字幕按以下目录规则放置:

```shell
flickr30k
├── annotations
│ ├── test.json
│ ├── train.json
│ └── val.json
└── images
  └── flickr30k-images
      ├── 1000092795.jpg
      ├── 10002456.jpg
      ├── 1000268201.jpg
      ├── 1000344755.jpg
      └── ...
```

其中flickr30k的父目录对应[run_blip2_stage1_vit_g_retrieval_flickr30k.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/blip2/run_blip2_stage1_vit_g_retrieval_flickr30k.yaml)配置中的`eval_dataset.dataloader.dataset_dir`配置, 其余目录摆放可以根据 `eval_dataset.dataloader.annotation_files` 和 `eval_dataset.dataloader.image_dirs` 的修改自行配置，详细逻辑可以参考图文对数据集[MultiImgCapDataLoader](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/dataset/dataloader/multi_image_cap_dataloader.py)的实现。

#### 单卡评测

默认配置文件只对 flickr30k - test部分数据 (1000 images, 5000 annotations) 进行评测。

```bash
python run_mindformer.py --config run_blip2_stage1_vit_g_retrieval_flickr30k.yaml --run_mode eval --eval_dataset_dir {parent_dir of flickr30k} --otherargs
# output
# {'txt_r1': 95.5, 'txt_r5': 99.9, 'txt_r10': 99.9, 'txt_r_mean': 98.43333333333334, 'img_r1': 86.6, 'img_r5': 97.16, 'img_r10': 98.52, 'img_r_mean': 94.09333333333332, 'r_mean': 96.26333333333332, 'agg_metrics': 98.43333333333334}
```

## 推理

### 基于pipeline的推理

`BLIP-2`二阶段推理中，支持通过语言模型生成图片的说明，本章节提供一个基于`pipeline`的推理脚本样例，该脚本样例执行以`Llama7b`为语言模型的`image_to_text_generation`任务，可将脚本保存成文件`blip2_stage2_pipeline_test.py`。

```python
import argparse

import mindspore as ms

from mindformers import AutoConfig, AutoModel
from mindformers import pipeline

def init_context(device_id):
    ms.set_context(mode=0, device_target="Ascend", device_id=device_id)


def build_text_input(prompts, templates):
    text_input = []
    for i in range(len(prompts)):
        text_input.append(templates[i].format(prompts[i]))
    return text_input


def str2bool(v):
    v_lower = v.lower()
    if v_lower in ["false", "0"]:
        output = False
    elif v_lower in ["true", "1"]:
        output = True
    else:
        raise ValueError("Invalid boolean value")
    return output


DEFAULT_IMAGE_TEXT_PAIR = [
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/titanic.jpg",
     "Question: What happened of this movie? Answer:"),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/elephant.jpg",
     "it is a photo of"),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009400.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009483.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009448.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000010363.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009769.jpg", "")
]


def main(args):
    if args.image_path is None:
        image_filepath = [pair[0] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        image_filepath = args.image_path.split(',')

    if args.prompt is None:
        if args.image_path is not None:
            prompts = [""] * len(image_filepath)
        else:
            prompts = [pair[1] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        prompts = args.prompt.split(',')

    if len(prompts) != len(image_filepath):
        raise ValueError("prompts length do not equal to image_path length, please check the args.")

    init_context(device_id=args.device_id)

    model_config = AutoConfig.from_pretrained(args.model_type)
    model_config.max_txt_len = args.seq_length

    model_config.batch_size = 1
    model_config.text_config.batch_size = model_config.batch_size
    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    model = AutoModel.from_config(model_config)
    pipeline_task = pipeline("image_to_text_generation", model=model)

    inputs = [{"image": image_filepath[index],
               "prompt": prompts[index]}
              for index in range(len(image_filepath))]
    for _ in range(args.generate_repeat_time):
        output = pipeline_task(inputs)
        print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        default="itt_blip2_stage2_vit_g_llama_7b",
        type=str,
        required=False,
        help='model type')

    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        required=False,
        help='device id')

    parser.add_argument(
        '--generate_repeat_time',
        type=int,
        default=5,
        required=False,
        help='generate repeat time')

    parser.add_argument(
        '--use_past',
        type=str2bool,
        default=True,
        required=False,
        help='whether use past')

    parser.add_argument(
        '--do_sample',
        type=str2bool,
        default=False,
        required=False,
        help='whether do sample')

    parser.add_argument(
        '--top_p',
        type=float,
        default=1,
        required=False,
        help='top p')

    parser.add_argument(
        '--top_k',
        type=int,
        default=0,
        required=False,
        help='top k')

    parser.add_argument(
        '--seq_length',
        type=int,
        default=32,
        required=False,
        help='seq length')

    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        required=False,
        help='image path')

    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        required=False,
        help='')

    args_ = parser.parse_args()
    print(args_)
    main(args_)
```

#### 单卡pipeline推理

```bash
# 增量推理 开采样
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past True --do_sample True --top_k 3

# 增量推理 不进行采样
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past True --do_sample False

# 自回归推理 开采样
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past False --do_sample True --top_k 3

# 自回归 不进行采样
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past False --do_sample False

# 增量推理 开采样 指定输入
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past True --do_sample True --top_k 3 --image_path /your/path/to/image --prompt your_promt

# 增量推理 开采样 指定输入
python blip2_stage2_pipeline_test.py --device_id 0 --generate_repeat_time 3 --use_past True --do_sample True --top_k 3 --image_path /your/path/to/image1,/your/path/to/image2,/your/path/to/image3 --prompt your_promt1,,your_promt3

```

**特殊参数说明：**

- `generate_repeat_time`: 重复执行次数，会对指定的图片输入进行多次重复推理，避免第一次编图时间影响速度；
- `image_path`: 指定推理的图片路径，支持输入多张图片路径，路径通过英文逗号`,`间隔，例如

  ```bash
  --image_path /your/path/to/image1,/your/path/to/image2,/your/path/to/image3
  ```

  当不指定路径时，会使用默认的图片进行推理；
- `prompt`: 指定推理的图片配对的文字提示，支持输入多条提示，路径通过英文逗号`,`间隔，每一条prompt与`image_path`中的路径相匹配，例如

   ```bash
   --prompt your_promt1,your_promt2,your_promt3
   ```

   `prompt`可传空值，但多条prompt仍需要以逗号隔开，当指定了`image_path`但未指定`prompt`时，会自动设置每张图像的prompt为""。

#### 多卡pipeline推理

暂不支持

### 基于generate的推理

`BLIP-2`二阶段推理中，支持通过语言模型生成图片的说明，本章节提供一个基于`generate`的推理脚本样例，该脚本样例执行以`Llama7b`为语言模型的`image_to_text_generation`任务，可将脚本保存成文件`blip2_stage2_generation_test.py`。

```python
import argparse

import mindspore as ms
from mindspore import ops

from mindformers import AutoConfig, AutoModel, AutoProcessor
from mindformers.tools.image_tools import load_image


def init_context(device_id):
    ms.set_context(mode=0, device_target="Ascend", device_id=device_id)


def build_text_input(prompts, templates):
    text_input = []
    for i in range(len(prompts)):
        text_input.append(templates[i].format(prompts[i]))
    return text_input


def str2bool(v):
    v_lower = v.lower()
    if v_lower in ["false", "0"]:
        output = False
    elif v_lower in ["true", "1"]:
        output = True
    else:
        raise ValueError("Invalid boolean value")
    return output


DEFAULT_IMAGE_TEXT_PAIR = [
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/titanic.jpg",
     "Question: What happened of this movie? Answer:"),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/elephant.jpg",
     "it is a photo of"),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009400.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009483.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009448.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000010363.jpg", ""),
    ("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/blip2/images/000000009769.jpg", "")
]


def main(args):
    if args.image_path is None:
        image_filepath = [pair[0] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        image_filepath = args.image_path.split(',')

    if args.prompt is None:
        if args.image_path is not None:
            prompts = [""] * len(image_filepath)
        else:
            prompts = [pair[1] for pair in DEFAULT_IMAGE_TEXT_PAIR]
    else:
        prompts = args.prompt.split(',')

    if len(prompts) != len(image_filepath):
        raise ValueError("prompts length do not equal to image_path length, please check the args.")

    init_context(device_id=args.device_id)

    model_config = AutoConfig.from_pretrained(args.model_type)

    model_config.max_txt_len = args.seq_length

    if args.batch_infer:
        model_config.batch_size = len(image_filepath)
    else:
        model_config.batch_size = 1

    model_config.text_config.batch_size = model_config.batch_size
    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    model = AutoModel.from_config(model_config)
    processor = AutoProcessor.from_pretrained(args.model_type)
    tokenizer = processor.tokenizer

    for _ in range(args.generate_repeat_time):
        if model_config.batch_size > 1:
            input_images = processor.image_processor([load_image(filepath) for filepath in image_filepath])
            input_ids = tokenizer(prompts,
                                  max_length=args.seq_length,
                                  padding="max_length",
                                  return_tensors="ms")["input_ids"]
            output = model.generate_text_for_image(input_images, input_ids)
            print(tokenizer.decode(output, skip_special_tokens=True))
        else:
            batch_size = len(image_filepath)
            for index in range(batch_size):
                input_image = processor.image_processor(load_image(image_filepath[index]))
                input_id = tokenizer(prompts[index],
                                     max_length=args.seq_length,
                                     padding="max_length",
                                     return_tensors="ms")["input_ids"]

                output = model.generate_text_for_image(input_image, input_id)
                print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type',
        default="itt_blip2_stage2_vit_g_llama_7b",
        type=str,
        required=False,
        help='model type')

    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        required=False,
        help='device id')

    parser.add_argument(
        '--batch_infer',
        type=str2bool,
        default=False,
        required=False,
        help='whether to batch infer')

    parser.add_argument(
        '--generate_repeat_time',
        type=int,
        default=5,
        required=False,
        help='generate repeat time')

    parser.add_argument(
        '--use_past',
        type=str2bool,
        default=True,
        required=False,
        help='whether use past')

    parser.add_argument(
        '--do_sample',
        type=str2bool,
        default=False,
        required=False,
        help='whether do sample')

    parser.add_argument(
        '--top_p',
        type=float,
        default=1,
        required=False,
        help='top p')

    parser.add_argument(
        '--top_k',
        type=int,
        default=0,
        required=False,
        help='top k')

    parser.add_argument(
        '--seq_length',
        type=int,
        default=32,
        required=False,
        help='seq length')

    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        required=False,
        help='image path')

    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        required=False,
        help='')

    args_ = parser.parse_args()
    print(args_)
    main(args_)
```

#### 单卡generate推理

```bash
# 单batch 增量推理 开采样
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past True --do_sample True --top_k 3

# 单batch 增量推理 不进行采样
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past True --do_sample False

# 单batch 自回归推理 开采样
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past False --do_sample True --top_k 3

# 单batch 自回归 不进行采样
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past False --do_sample False

# batch推理 增量推理 开采样
python blip2_stage2_generation_test.py --device_id 0 --batch_infer True --generate_repeat_time 3 --use_past True --do_sample True --top_k 3

# 单batch 增量推理 开采样 指定输入
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past True --do_sample True --top_k 3 --image_path /your/path/to/image --prompt your_promt

# 单batch 增量推理 开采样 指定输入
python blip2_stage2_generation_test.py --device_id 0 --batch_infer False --generate_repeat_time 3 --use_past True --do_sample True --top_k 3 --image_path /your/path/to/image1,/your/path/to/image2,/your/path/to/image3 --prompt your_promt1,,your_promt3

```

**特殊参数说明：**

- `batch_infer`: 是否开启batch推理；
- `generate_repeat_time`: 重复执行次数，会对指定的图片输入进行多次重复推理，避免第一次编图时间影响速度；
- `image_path`: 指定推理的图片路径，支持输入多张图片路径，路径通过英文逗号`,`间隔，例如

  ```bash
  --image_path /your/path/to/image1,/your/path/to/image2,/your/path/to/image3
  ```

  当不指定路径时，会使用默认的图片进行推理；
- `prompt`: 指定推理的图片配对的文字提示，支持输入多条提示，路径通过英文逗号`,`间隔，每一条prompt与`image_path`中的路径相匹配，例如

   ```bash
   --prompt your_promt1,your_promt2,your_promt3
   ```

   `prompt`可传空值，但多条prompt仍需要以逗号隔开，当指定了`image_path`但未指定`prompt`时，会自动设置每张图像的prompt为""。

#### 多卡generate推理

暂不支持

### 脚本启动

#### 单卡推理

- `BLIP-2`一阶段推理：zero shot图像分类任务

```bash
python run_mindformer.py --config configs/blip2/run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml --run_mode predict --device_target Ascend --device_id 0 --predict_data /path/to/image
```

- `BLIP-2`二阶段推理：图生文任务（ImageCaption）

```bash
python run_mindformer.py --config configs/blip2/run_blip2_stage2_vit_g_llama_7b_image_to_text_generation.yaml --run_mode predict --device_target Ascend --device_id 0 --predict_data /path/to/image
```

**注**：要提高推理速度，可在对应模型配置文件中`model.model_config.text_config.use_past`的值设为`True`。

## mindspore-lite

暂不支持
