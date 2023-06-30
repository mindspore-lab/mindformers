# SWIN

## 模型描述

swin：全名swin transformer，是一个基于Transformer在视觉领域有着SOTA表现的深度学习模型。比起VIT拥有更好的性能和精度。

[论文](https://arxiv.org/abs/2103.14030) Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo, 2021

## 数据集准备

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB

 ```bash
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
python run_mindformer.py --config ./configs/swin/run_swin_base_p4w7_224_100ep.yaml --run_mode train --train_dataset_dir [DATASET_PATH]

# evaluate
python run_mindformer.py --config ./configs/swin/run_swin_base_p4w7_224_100ep.yaml --run_mode eval --eval_dataset_dir [DATASET_PATH]

# predict
python run_mindformer.py --config ./configs/swin/run_swin_base_p4w7_224_100ep.yaml --run_mode predict --predict_data [PATH_TO_IMAGE]
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import SwinForImageClassification, SwinConfig

SwinForImageClassification.show_support_list()
# 输出：
# - support list of SwinForImageClassification is:
# -    ['swin_base_p4w7']
# - -------------------------------------

# 模型标志加载模型
model = SwinForImageClassification.from_pretrained("swin_base_p4w7")

#模型配置加载模型
config = SwinConfig.from_pretrained("swin_base_p4w7")
# {'batch_size': 128, 'image_size': 224, 'patch_size': 4, 'num_labels': 1000, 'num_channels': 3,
# 'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32],
# 'checkpoint_name_or_path': 'swin_base_p4w7'}
model = SwinForImageClassification(config)
```

- Trainer接口开启训练/评估/推理：

```python
from mindformers.trainer import Trainer
from mindformers.tools.image_tools import load_image

# 初始化任务
swin_trainer = Trainer(
    task='image_classification',
    model='swin_base_p4w7',
    train_dataset="imageNet-1k/train",
    eval_dataset="imageNet-1k/val")
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
            "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")

# 方式1：开启训练，并使用训练好的权重进行eval和推理
swin_trainer.train()
swin_trainer.evaluate(eval_checkpoint=True)
predict_result = swin_trainer.predict(predict_checkpoint=True, input_data=img, top_k=3)
print(predict_result)

# 方式2： 从obs下载训练好的权重并进行eval和推理
swin_trainer.evaluate() # 下载权重进行评估
predict_result = swin_trainer.predict(input_data=img, top_k=3) # 下载权重进行推理
print(predict_result)

# 输出
# - mindformers - INFO - output result is: [[{'score': 0.89573187, 'label': 'daisy'},
# {'score': 0.005366202, 'label': 'bee'}, {'score': 0.0013296203, 'label': 'fly'}]]
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
from mindformers.tools.image_tools import load_image


pipeline_task = pipeline("image_classification", model='swin_base_p4w7')
img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
          "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
pipeline_result = pipeline_task(img, top_k=3)
print(pipeline_result)
# 输出
# [[{'score': 0.89573187, 'label': 'daisy'}, {'score': 0.005366202, 'label': 'bee'},
# {'score': 0.0013296203, 'label': 'fly'}]]
```

 Trainer和pipeline接口默认支持的task和model关键入参

  |    task（string）    | model（string）  |
  |:--------------:| :-------------: |
  | image_classification | swin_base_p4w7 |

## 模型性能

| model |      type      | pretrain | Datasets | Top1-Accuracy | Log | pretrain_config |                     finetune_config                      |
|:-----:|:--------------:|:--------:| :----: |:-------------:| :---: |:---------------:|:--------------------------------------------------------:|
| swin  | swin_base_p4w7 |    \     | ImageNet-1K |    83.44%     | \ |        \        | [link](../../configs/swin/run_swin_base_p4w7_100ep.yaml) |

## 模型权重

本仓库中的`swin_base_p4w7`来自于MicroSoft的[`Swin-Transformer`](https://github.com/microsoft/Swin-Transformer), 基于下述的步骤获取：

1. 从上述的链接中下载[`swin_base_p4w7`](https://pan.baidu.com/s/16bqCTEc70nC_isSsgBSaqQ?pwd=swin)的官方权重，文件名为`swin_base_patch4_window7_224.pth`

2. 执行转换脚本，得到转换后的输出文件`swin_base_p4w7.ckpt`

```shell
python mindformers/models/swin/convert_weight.py --torch_path swin_base_patch4_window7_224.pth --mindspore_path swin_base_p4w7.ckpt --is_pretrain False
```

如需转换官方simmim的预训练权重进行finetune，则执行如下步骤：

1. [从simmim官网](https://github.com/microsoft/SimMIM)提供的google网盘下载[`simmim_swin_192`](https://drive.google.com/file/d/1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d/view?usp=sharing)的官方权重，文件名为`simmim_pretrain__swin_base__img192_window6__100ep.pth`

2. 执行转换脚本，得到转换后的输出文件`simmim_swin_p4w6.ckpt`

```shell
python mindformers/models/swin/convert_weight.py --torch_path simmim_pretrain__swin_base__img192_window6__100ep.pth --mindspore_path simmim_swin_p4w6.ckpt --is_pretrain True
```