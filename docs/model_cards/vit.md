# ViT

## 模型描述

vit：全名vision transformer，不同于传统的基于CNN的网络结果，是基于transformer结构的cv网络，2021年谷歌研究发表网络，在大数据集上表现了非常强的泛化能力。大数据任务（如clip）基于该结构能有良好的效果。mindformers提供的Vit权重及精度均是是基于MAE预训练ImageNet-1K数据集进行微调得到。

[论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F2010.11929): Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.

## 数据集准备

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB

 ```text
数据集目录格式
└─imageNet-1k
    ├─train                # 训练数据集
    └─val                  # 评估数据集
 ```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

- 脚本运行测试

```shell
# pretrain
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode train

# evaluate
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode eval --eval_dataset_dir [DATASET_PATH]

# predict
python run_mindformer.py --config ./configs/vit/run_vit_base_p16_224_100ep.yaml --run_mode predict --predict_data [PATH_TO_IMAGE]
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import ViTForImageClassification, ViTConfig

ViTForImageClassification.show_support_list()
# 输出：
# - support list of ViTForImageClassification is:
# -    ['vit_base_p16']
# - -------------------------------------

# 模型标志加载模型
model = ViTForImageClassification.from_pretrained("vit_base_p16")

#模型配置加载模型
config = ViTConfig.from_pretrained("vit_base_p16")
# {'patch_size': 16, 'in_chans': 3, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4,
# ..., 'batch_size': 32, 'image_size': 224, 'num_classes': 1000}
model = ViTForImageClassification(config)
 ```

- Trainer接口开启训练/评估/推理：

```python
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 初始化任务
vit_trainer = Trainer(
    task='image_classification',
    model='vit_base_p16',
    train_dataset="imageNet-1k/train",
    eval_dataset="imageNet-1k/val")
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1：使用现有的预训练权重进行finetune， 并使用finetune获得的权重进行eval和推理
vit_trainer.train(resume_or_finetune_from_checkpoint="mae_vit_base_p16", do_finetune=True)
vit_trainer.evaluate(eval_checkpoint=True)
predict_result = vit_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式2: 重头开始训练，并使用训练好的权重进行eval和推理
vit_trainer.train()
vit_trainer.evaluate(eval_checkpoint=True)
predict_result = vit_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式3： 从obs下载训练好的权重并进行eval和推理
vit_trainer.evaluate()
predict_result = vit_trainer.predict(input_data=img, top_k=3)
print(predict_result)
 ```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image


pipeline_task = pipeline("image_classification", model='vit_base_p16')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img, top_k=3)
 ```

 Trainer和pipeline接口默认支持的task和model关键入参

|     task（string）     | model（string） |
|:--------------------:|:-------------:|
| image_classification | vit_base_p16  |

## 模型性能

| model |     type     |       pretrain       |  Datasets   | Top1-Accuracy | Log |                  pretrain_config                   |                    finetune_config                    |
|:-----:|:------------:|:--------------------:|:-----------:|:-------------:|:---:|:--------------------------------------------------:|:-----------------------------------------------------:|
|  vit  | vit_base_p16 | [mae_vit_base_p16]() | ImageNet-1K |    83.71%     |  \  | [link](../../configs/mae/run_mae_vit_base_p16_224_800ep.yaml) | [link](../../configs/vit/run_vit_base_p16_100ep.yaml) |

## 模型权重

本仓库中的`vit_base_p16`来自于facebookresearch/mae的[`ViT-Base`](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth), 基于下述的步骤获取：

1. 从上述的链接中下载`ViT-Base`的模型权重

2. 执行转换脚本，得到转换后的输出文件`vit_base_p16.ckpt`

```shell
python mindformers/models/vit/convert_weight.py --torch_path "PATH OF ViT-Base.pth" --mindspore_path "SAVE PATH OF vit_base_p16.ckpt"
```