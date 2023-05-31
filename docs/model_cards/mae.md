# MAE

## 模型描述

MAE是一种基于MIM（Masked Imange Modeling）的无监督学习方法。

MAE由何凯明团队提出，将NLP领域大获成功的自监督预训练模式用在了计算机视觉任务上，效果拔群，在NLP和CV两大领域间架起了一座更简便的桥梁。

[论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F2111.06377): He, Kaiming et al. “Masked Autoencoders Are Scalable Vision Learners.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 15979-15988.

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
python run_mindformer.py --config ./configs/mae/run_mae_vit_base_p16_224_800ep.yaml --run_mode train

# predict
python run_mindformer.py --config ./configs/mae/run_mae_vit_base_p16_224_800ep.yaml --run_mode predict --predict_data [PATH_TO_IMAGE]
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import ViTMAEForPreTraining, ViTMAEConfig

ViTMAEForPreTraining.show_support_list()
# 输出：
# - support list of ViTMAEForPreTraining is:
# -    ['mae_vit_base_p16']
# - -------------------------------------

# 模型标志加载模型
model = ViTMAEForPreTraining.from_pretrained("mae_vit_base_p16")

#模型配置加载模型
config = ViTMAEConfig.from_pretrained("mae_vit_base_p16")
# {'decoder_dim': 512, 'patch_size': 16, 'in_chans': 3, 'embed_dim': 768, 'depth': 12,
# ..., 'decoder_embed_dim': 512, 'norm_pixel_loss': True, 'window_size': None}
model = ViTMAEForPreTraining(config)
 ```

- Trainer接口开启训练/评估/推理：

```python
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 初始化任务
mae_trainer = Trainer(
    task='masked_image_modeling',
    model='mae_vit_base_p16',
    train_dataset="imageNet-1k/train")
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1: 重头开始训练，并使用训练好的权重进行推理
mae_trainer.train() # 开启训练
predict_result = mae_trainer.predict(predict_checkpoint=True, input_data=img)
print(predict_result)

# 方式3： 从obs下载训练好的权重并进行推理
predict_result = mae_trainer.predict(input_data=img)
print(predict_result)
 ```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image


pipeline_task = pipeline("masked_image_modeling", model='mae_vit_base_p16')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img)
 ```

 Trainer和pipeline接口默认支持的task和model关键入参

|     task（string）      |  model（string）   |
|:---------------------:|:----------------:|
| masked_image_modeling | mae_vit_base_p16 |

## 模型性能

| model |       type       |       pretrain       |  Datasets   | Top1-Accuracy | Log |                  pretrain_config                   |                    finetune_config                    |
|:-----:|:----------------:|:--------------------:|:-----------:|:-------------:|:---:|:--------------------------------------------------:|:-----------------------------------------------------:|
|  mae  | mae_vit_base_p16 | [mae_vit_base_p16]() | ImageNet-1K |    83.71%     |  \  | [link](../../configs/mae/run_mae_vit_base_p16_224_800ep.yaml) | [link](../../configs/vit/run_vit_base_p16_100ep.yaml) |

## 模型权重

本仓库中的`mae_vit_base_p16`来自于facebookresearch/mae的[`ViT-Base`](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), 基于下述的步骤获取：

1. 从上述的链接中下载`ViT-Base`的模型权重

2. 执行转换脚本，得到转换后的输出文件`mae_vit_base_p16.ckpt`

```python
python mindformers/models/mae/convert_weight.py --torch_path "PATH OF ViT-Base.pth" --mindspore_path "SAVE PATH OF mae_vit_base_p16.ckpt"
```