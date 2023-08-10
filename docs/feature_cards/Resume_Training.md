# 断点续训

## 介绍

在长时间的训练当中，如果遇到意外情况导致训练中断，可以使用断点续训的方式恢复之前的状态继续训练。

恢复训练时，从之前训练的checkpoint文件或文件夹中，加载网络权重，并恢复训练时的epoch、step、loss_scale_value等信息。

> 局限：由于数据集暂不支持跳过已训练的数据，当前仅支持epoch级的断点续训，即仅支持恢复到某个epoch开始的状态。

## 使用

### 脚本启动场景

在run_xxx.yaml中配置load_checkpoint为checkpoint文件或文件夹路径，并将resume_training改为True。

```yaml
load_checkpoint: checkpoint_file_or_dir_path
resume_training: True
```

> 注意：如果load_checkpoint配置为文件路径，则认为是完整权重，如果配置为文件夹，则认为是分布式权重。下同。

### Trainer高阶接口启动场景

- 方式1：在Trainer.train()或Trainer.finetune()中配置

在Trainer.train()或Trainer.finetune()中，配置train_checkpoint或finetune_checkpoint参数为checkpoint文件或文件夹路径，并将resume_training参数设置为True。

```python
from mindformers import Trainer

cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                      model='vit_base_p16', # 已支持的模型名
                      train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                      eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式

cls_trainer.train(train_checkpoint="", resume_training=True) # 启动训练
cls_trainer.finetune(finetune_checkpoint="", resume_training=True) # 启动微调
```

- 方式2：在TrainingArguments中配置

在TrainingArguments中配置resume_from_checkpoint为checkpoint文件或文件夹路径，并将resume_training参数设置为True。

```python
from mindformers import Trainer, TrainingArguments

training_args = TrainingArguments(resume_from_checkpoint="", resume_training=True)

cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                      args=training_args,
                      model='vit_base_p16', # 已支持的模型名
                      train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                      eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式

cls_trainer.train() # 启动训练
cls_trainer.finetune() # 启动微调
```
