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

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

  ```python
  from mindformers import SwinModel, SwinConfig

  SwinModel.show_support_list()
  # 输出：
  # - support list of SwinModel is:
  # -    ['swin_base_p4w7']
  # - -------------------------------------

  # 模型标志加载模型
  model = SwinModel.from_pretrained("swin_base_p4w7")

  #模型配置加载模型
  config = SwinConfig.from_pretrained("swin_base_p4w7")
  # {'batch_size': 128, 'image_size': 224, 'patch_size': 4, 'num_labels': 1000, 'num_channels': 3,
  # 'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32],
  # 'checkpoint_name_or_path': 'swin_base_p4w7'}
  model = SwinModel(config)
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

  swin_trainer.train() # 开启训练
  swin_trainer.evaluate() # 开启评估

  img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
  predict_result = swin_trainer.predict(input_data=img, top_k=3) # 开启推理
  print(predict_result)
  ```

- pipeline接口开启快速推理

  ```python
  from mindformers.pipeline import pipeline
  from mindformers.tools.image_tools import load_image


  pipeline_task = pipeline("image_classification", model='swin_base_p4w7')
  img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
  pipeline_result = pipeline_task(img, top_k=3)
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
python mindformers/models/swin/convert_weight.py --torch_path pytorch_model.bin --mindspore_path ./swin_base_p4w7.ckpt
```