# BLIP2

## 模型描述

BLIP-2: 全名`Bootstrapping Language-Image Pre-training - 2`模型是2023 年 Salesforce提出的一种多模态模型，它从现成的冻结预训练图像编码器 (ViT)和冻结的大型语言模型 (LLM)中引导视觉语言预训练 (contrastive_language_image_pretrain), 中间添加一个轻量级的 Querying Transformer 弥补了模态 gap, 该 Transformer 分两个阶段进行预训练:

- 第一阶段从冻结图像编码器引导视觉-语言表示学习，强制 Q-Former 学习与文本最相关的视觉表示。
- 第二阶段基于冻结的语言模型引导从视觉到语言的生成学习，将Q-Former的输出连接到冻结的LLM，并对Q-Former进行训练，使其输出视觉表示能够被LLM解释。

[论文](https://arxiv.org/pdf/2301.12597.pdf) Junnan Li，et al., BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, 2023

## 预训练数据集下载

1\. coco2014数据集:

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

2\. Visual Genome数据集:

```shell
├── annotations
│   └── vg_caption.json
└── images
    └── VG_100K
```

同上，可通过LAVIS github库提供的链接下载annotation_files ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/vg/defaults_caption.yaml)) 和 images ([链接](https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_vg.py))。

## 关键配置项说明

```yaml
train_dataset:
  text_transforms:
    type: CaptionTransform
    prompt: ""        # 二阶段中语言模型添加的prompt，目前仅支持固定prompt
    max_length: 32    # token ids的最大长度

  tokenizer:
    type: LlamaTokenizer
    max_length: 32           # token ids的最大长度
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
      seq_length: 160        # 语言模型的输入seq_length大小，在训练时该值要等于qformer的query数目+语言模型输入文字token id长度
                             # 即 seq_length=model.model_config.qformer_config.query_length + model.model_config.max_txt_len
      checkpoint_name_or_path: ""  # 语言模型权重文件
  arch:
    type: XXXX

# 推理任务时需要配置
processor:
  type: Blip2Processor
  image_processor:
    type: Blip2ImageProcessor
    image_size: 224          # 输入图像大小
  tokenizer:
    type: LlamaTokenizer
    max_length: 128
    vocab_file: ""           # 词表文件路径
```

## BLIP-2一阶段

### 快速使用

#### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

示例命令如下，将会执行一个39层ViT和18层Attention Layer的Qformer的一阶段预训练:

```shell
# pretrain
python run_mindformer.py --config configs/blip2/run_blip2_pretrain_stage_1.yaml --run_mode train  \
                         --device_target Ascend \
                         --train_dataset_dir [PATH_TO_DATASET]

# evaluate
python run_mindformer.py --config configs/blip2/run_blip2_stage1_vit_g_retrieval_flickr30k.yaml     --run_mode eval  \
                         --device_target Ascend \
                         --eval_dataset_dir [PATH_TO_DATASET]

# predict (单图片)
python run_mindformer.py --config configs/blip2/run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml
--run_mode predict  \
                         --device_target Ascend \
                         --predict_data [PATH_TO_IMAGE]

# predict (数据集)
python run_mindformer.py --config configs/blip2/run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml
--run_mode predict  \
                         --device_target Ascend \
                         --eval_dataset_dir [PATH_TO_DATASET]
```

#### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import Blip2Qformer, Blip2Config

Blip2Qformer.show_support_list()
# 输出：
# support list of Blip2Qformer is:
# ['blip2_stage1_vit_g', 'blip2_stage1_classification']
# -------------------------------------

# 模型标志加载模型
model = Blip2Qformer.from_pretrained('blip2_stage1_vit_g')

#模型配置加载模型
config = Blip2Config.from_pretrained("blip2_stage1_vit_g")
config
# {'img_size': 224,
#  'num_query_token': 32,
#  'cross_attention_freq': 2,
#  'drop_path_rate': 0,
#  'use_grad_checkpoint': False,
#  'vit_model': 'vit_g_p16',
#  'layernorm_dtype': 'float32',
#  'softmax_dtype': 'float32',
#  'model_type': 'blip2',
#  'batch_size': 8,
#  'embed_dim': 256,
#  'vocab_size': 21129,
#   ...
#  'is_training': True}
model = Blip2Qformer(config)
```

- Trainer接口开启训练：

```python
from mindformers.dataset.dataloader.multi_image_cap_dataloader import MultiImgCapDataLoader
from mindformers.trainer import Trainer

# 初始化图像-文本数据集
dataset_dir = "/data"
annotation_files = [
  "vg/annotations/vg_caption.json",
  "coco2014/coco/annotations/coco_karpathy_train.json"
]
image_dirs = [
  "vg/images",
  "coco2014/coco/images"
]
train_dataset = MultiImgCapDataLoader(dataset_dir=dataset_dir, annotation_files=annotation_files, image_dirs = image_dirs, stage="train")

# blip2一阶段初始化预训练任务
trainer = Trainer(task='contrastive_language_image_pretrain',
    model='blip2_stage1_vit_g',
    train_dataset=train_dataset)

# 开启预训练
trainer.train()
```

- Trainer接口开启评估:

```python
from mindformers.dataset.dataloader.multi_image_cap_dataloader import MultiImgCapDataLoader
from mindformers.trainer import Trainer

# 初始化图像-文本数据集
dataset_dir: "/data"
annotation_files: [
  "flickr30k/annotations/test.json"
]
image_dirs: [
  "flickr30k/images"
]
eval_dataset = MultiImgCapDataLoader(dataset_dir=dataset_dir, annotation_files=annotation_files, image_dirs = image_dirs, stage="eval")

# 初始化评估任务
trainer = Trainer(task='blip2_retireval',
    model='blip2_stage1_vit_g',
    eval_dataset=eval_dataset)

# 开启评估
trainer.eval()
```

- Trainer接口开启推理:

```python
from mindformers.tools.image_tools import load_image
from mindformers import Trainer

cls_trainer = Trainer(task='zero_shot_image_classification',
                      model='blip2_stage1_classification',
                      candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
# 加载输入，一张太阳花图片
input_data = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 加载指定的权重以完成推理
predict_result = cls_trainer.predict(input_data=input_data,
                                     predict_checkpoint='your_path_to/blip2_pretrained.ckpt')
print(predict_result)
# 输出
# output result is: [[{'score': 0.99999976, 'label': 'sunflower'}]]
# output result is saved at: zero_shot_image_classification_result.txt
# .........Predict Over!.............
```

- pipeline推理:

```python
from mindformers.tools.image_tools import load_image
from mindformers.pipeline import ZeroShotImageClassificationPipeline

# 初始化pipeline
classifier = ZeroShotImageClassificationPipeline(
    model='blip2_stage1_classification',
    candidate_labels=["sunflower", "tree", "dog", "cat", "toy"],
    hypothesis_template="This is a photo of {}."
    )

# 太阳花图片
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 可批次处理，输入图片数组
classifier(img)
# 输出
# [[{'score': 0.99999714, 'label': 'sunflower'},
#   {'score': 1.315181e-06, 'label': 'tree'},
#   {'score': 7.0368844e-07, 'label': 'toy'},
#   {'score': 4.7594781e-07, 'label': 'dog'},
#   {'score': 3.93686e-07, 'label': 'cat'}]]
```

#### 多卡训练

- 参考 [快速启动指导](https://gitee.com/mindspore/mindformers#%E6%96%B9%E5%BC%8F%E4%B8%80%E4%BD%BF%E7%94%A8%E5%B7%B2%E6%9C%89%E8%84%9A%E6%9C%AC%E5%90%AF%E5%8A%A8) 中的多卡启动部分，blip2的任务配置文件默认为单卡启动，如需多卡启动，注意将相关配置文件中的

```yaml
use_parallel: False
```

改为

```yaml
use_parallel: True
```

blip2现支持数据并行 (data parallel) 和模型并行 (model parallel)，需要注意根据卡数和并行模式正确配置`data_parallel` 和`model_parallel`的值 (dp * mp = 卡数)。

##### 评估一阶段Loss

```python
from mindformers import Blip2Qformer, Blip2Processor
from mindformers.tools.image_tools import load_image
from mindspore import Tensor

# 加载模型
model = Blip2Qformer.from_pretrained('blip2_stage1_vit_g')

# 加载处理器
processor = Blip2Processor.from_pretrained('blip2_stage1_vit_g')
image_processor = processor.image_processor
tokenizer = processor.tokenizer

# 加载输入，一张太阳花图片
image = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
                 "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
text = ["This is a picture of sunflower."]

# 输入预处理
image_input = image_processor(image)
text_ids = tokenizer(text, max_length=32, padding='max_length')
input_ids = Tensor(text_ids['input_ids'], ms.int32)

# 放入模型计算loss
output = model(image_input, input_ids, return_tuple=True)
print(output)
# (Tensor(shape=[], dtype=Float32, value= 4.45203), -- overall_loss = loss_itc + loss_itm + loss_lm
# Tensor(shape=[], dtype=Float32, value= 0), -- loss_itc
# Tensor(shape=[], dtype=Float32, value= 0.681192), -- loss_itm
# Tensor(shape=[], dtype=Float32, value= 3.77084)) -- loss_lm
```

### 模型权重

本仓库中的`blip2_stage1_classification`来自于LAVIS的一阶段预训练权重[`blip2_stage1_pretrained`](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth), 基于下述的步骤获取：

1. 从上述的链接中下载`blip2_stage1_pretrained`的pytorch权重，文件名为`blip2_pretrained.pth`

2. 执行转换脚本，得到转换后的输出文件`blip2_stage1_pretrained.ckpt`

```shell
python mindformers/models/blip2/convert_weight.py --torch_path blip2_pretrained.pth --mindspore_path ./blip2_stage1_pretrained.ckpt
```

## BLIP-2二阶段

BLIP-2第二阶段是通过冻结的语言模型引导从视觉到语言的生成学习，将Q-Former的输出连接到冻结的LLM，并对Q-Former进行训练，使其输出视觉表示能够被LLM解释。
当前BLIP-2二阶段支持使用接入Llama-7b及baichuan-7b的推理任务，并暂未提供预训练模型，以下推理任务中，未加载准确的预训练权重，不保证生成结果正确。

### 快速使用

#### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

示例命令如下，将会执行一个39层ViT, 18层Attention Layer的Qformer以及Llama-7b的二阶段推理，**Note:** 当前二阶段仅支持推理任务，并暂不提供预训练模型:

```shell
# predict (单图片)  当前二阶段未提供预训练模型，该预测代码仅展示使用方式，不保证预测结果
python run_mindformer.py --config configs/blip2/run_blip2_stage2_vit_g_llama_7b_image_to_text_generation.yaml
--run_mode predict  \
                         --device_target Ascend \
                         --predict_data [PATH_TO_IMAGE]
```

**注：** 当二阶段使用baichuan_7b作为LLM模型时进行推理时，需要自行在配置文件`configs/blip2/run_blip2_stage2_vit_g_baichuan_7b_image_to_text_generation.yaml`中
配置对应的baichuan-7b权重文件及词表文件，如何配置可参考上文中关键配置项说明章节；
权重文件和词表文件的获取及转换参考[baichuan_7b](https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan/baichuan.md).

#### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import Blip2ImageToTextGeneration, Blip2Config

Blip2ImageToTextGeneration.show_support_list()
# 输出：
# support list of Blip2ImageToTextGeneration is:
# ['itt_blip2_stage2_vit_g_baichuan_7b', 'itt_blip2_stage2_vit_g_llama_7b']
# -------------------------------------

# 初始化模型方式1：直接通过预训练加载
model = Blip2ImageToTextGeneration.from_pretrained('itt_blip2_stage2_vit_g_baichuan_7b')

# 初始化模型方式2：通过配置文件初始化模型
config = Blip2Config.from_pretrained("itt_blip2_stage2_vit_g_baichuan_7b")
# 可通过config配置更改相应的配置
config.model.model_config.text_config.checkpoint_name_or_path = "/path/to/the/llm_checkpoint"  # 修改语言模型权重路径

model = Blip2ImageToTextGeneration(config)
```

- Trainer接口开启推理:

```python
from mindformers.tools.image_tools import load_image
from mindformers import Blip2Config, Blip2ImageToTextGeneration, Trainer

config = Blip2Config.from_pretrained("itt_blip2_stage2_vit_g_baichuan_7b")
# 可通过config配置更改相应的配置
config.text_config.checkpoint_name_or_path = "/path/to/the/llm_checkpoint"  # 修改语言模型权重路径
config.checkpoint_name_or_path = "/path/to/the/1-st_pretrain_checkpoint"  # 修改语言模型权重路径

model = Blip2ImageToTextGeneration(config)

generate_trainer = Trainer(task='image_to_text_generation',
                           model=model,
                           model_name='itt_blip2_stage2_vit_g_baichuan_7b')
# 加载输入，一张太阳花图片
input_data = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 加载指定的权重以完成推理
predict_result = generate_trainer.predict(input_data=input_data,
                                          hypothesis_template="###问题：\n详细描述图片内容。\n\n###答案：")
print(predict_result)
# 输出
# output result is: ['###问题：\n详细描述图片内容。\n\n###答案：这是一张图片，上面有一朵黄色的花，花朵周围有许多小黄花，它们在阳光下闪闪发光。']
# output result is saved at: zero_shot_image_to_text_generation_result.txt
# .........Predict Over!.............
```

- pipeline推理:

```python
from mindformers.tools.image_tools import load_image
from mindformers import Blip2Config, Blip2ImageToTextGeneration, pipeline

config = Blip2Config.from_pretrained("itt_blip2_stage2_vit_g_baichuan_7b")
# 可通过config配置更改相应的配置
config.text_config.checkpoint_name_or_path = "/path/to/the/llm_checkpoint"  # 修改语言模型权重路径
config.checkpoint_name_or_path = "/path/to/the/1-st_pretrain_checkpoint"  # 修改语言模型权重路径

model = Blip2ImageToTextGeneration(config)

# 初始化pipeline
generator = pipeline("image_to_text_generation", model=model)

# 太阳花图片
img = load_image(
    "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 可批次处理，输入图片数组
pipeline_result = generator(img, hypothesis_template='###问题：\n详细描述图片内容。\n\n###答案：')
print(pipeline_result)
# ['###问题：\n详细描述图片内容。\n\n###答案：这是一张图片，上面有一朵黄色的花，花朵周围有许多小黄花，它们在阳光下闪闪发光。']
```