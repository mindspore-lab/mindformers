# 断点续训

## 介绍

Mindformer支持**step级断点续训**，在训练过程中如果遇到意外情况导致训练中断，可以使用断点续训的方式恢复之前的状态继续训练。

Mindformer在输出目录下会保存`checkpoint`和`checkpoint_network`两个权重输出文件夹：

- **checkpoint**：保存权重、优化器、step、epoch、loss_scale等参数信息，主要用于**断点恢复训练**，可完全恢复至中断处的训练状态；
- **checkpoint_network**：仅保存权重参数，可用作**预训练权重**或**推理评估**，不支持**断点恢复训练**。

> 注：分布式断点续训必须开启sink_mode。

## 使用

### 脚本启动场景

在run_xxx.yaml中配置`load_checkpoint`，并将`resume_training`改为**True**：

- 如果加载**分布式权重**，配置为checkpoint文件夹路径，权重按照`checkpoint_file_or_dir_path/rank_x/xxx.ckpt`格式存放。
- 如果加载**完整权重**，配置为checkpoint文件绝对路径。

```yaml
load_checkpoint: checkpoint_file_or_dir_path
resume_training: True
```

> 注：如果load_checkpoint配置为文件路径，则认为是完整权重，如果配置为文件夹，则认为是分布式权重。下同。

### Trainer高阶接口启动场景

- **方式1：在Trainer.train()或Trainer.finetune()中配置**

在Trainer.train()或Trainer.finetune()中，配置`train_checkpoint`或`finetune_checkpoint`参数为checkpoint文件或文件夹路径，并将`resume_training`参数设置为**True**。

```python
import mindspore as ms
from mindformers import Trainer

ms.set_context(mode=0) # 设定为图模式加速

cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                      model='vit_base_p16', # 已支持的模型名
                      train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                      eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式

cls_trainer.train(train_checkpoint="", resume_training=True) # 启动训练
cls_trainer.finetune(finetune_checkpoint="", resume_training=True) # 启动微调
```

- **方式2：在TrainingArguments中配置**

在TrainingArguments中配置`resume_from_checkpoint`为checkpoint文件或文件夹路径，并将`resume_training`参数设置为**True**。

```python
import mindspore as ms
from mindformers import Trainer, TrainingArguments

ms.set_context(mode=0) # 设定为图模式加速

training_args = TrainingArguments(resume_from_checkpoint="", resume_training=True)

cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                      args=training_args,
                      model='vit_base_p16', # 已支持的模型名
                      train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                      eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式

cls_trainer.train() # 启动训练
cls_trainer.finetune() # 启动微调
```
