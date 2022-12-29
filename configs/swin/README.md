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
└─dataset
    ├─train                # 训练数据集
    └─val                  # 评估数据集
 ```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Trainer接口开启训练/评估/推理：

  ```python
  from mindformers.trainer import Trainer

  # 初始化任务
  swin_trainer = Trainer(
      task_name='image_classification',
      model='swin_base_p4w7',
      train_dataset="dataset/train",
      eval_dataset="dataset/eval")

  swin_trainer.train() # 开启训练
  swin_trainer.evaluate() # 开启评估
  input_data = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"
  predict_result = swin_trainer.predict(input_data) # 开启推理
  print(predict_result)
  ```

- pipeline接口开启快速推理

  ```python
  from mindformers.pipeline import pipeline

  pipeline_task = pipeline("image_classification", model='swin_base_p4w7')
  input_data = "https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png"
  pipeline_result = pipeline_task(input_data)
  ```

 Trainer和pipeline接口默认支持的task_name和model_name关键入参

  |    task（string）    | model（string）  |
  |:--------------:| :-------------: |
  | image_classification | swin_base_p4w7 |

## 模型性能

| model |      type      | pretrain | Datasets | Top1-Accuracy | Log | pretrain_config |            finetune_config            |
|:-----:|:--------------:|:--------:| :----: |:-------------:| :---: |:---------------:|:-------------------------------------:|
| swin  | swin_base_p4w7 |    \     | ImageNet-1K |    83.44%     | \ |        \        | [link](run_swin_base_p4w7_100ep.yaml) |

## 模型权重

本仓库中的`swin_base`来自于HuggingFace的[`swin_base`](https://huggingface.co/microsoft/swin-base-patch4-window7-224), 基于下述的步骤获取：

1. 从上述的链接中下载`swin_base`的HuggingFace权重，文件名为`pytorch_model.bin`

2. 执行转换脚本，得到转换后的输出文件`swin_base_p4w7.ckpt`

```python
python mindformers/models/swin/convert_weight.py --torch_path pytorch_model.bin --mindspore_path ./swin_base_p4w7.ckpt
```