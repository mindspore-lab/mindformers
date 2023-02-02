# Zero Shot Image Classification

## 任务描述

零样本图像分类：模型在基于图文对的预训练后，可以在给定任意图片与候选标签列表的情况下，完成对图像的分类，而无需任何微调。

[相关论文](https://arxiv.org/abs/2103.00020) Alec Radford, Jong Wook Kim, et al., Learning Transferable Visual Models From Natural Language Supervision, 2021.

## 已支持数据集性能

| model |                                   type                                   | datasets |                Top1-accuracy                 |      stage       |                                                                                          example                                                                                          |
|:-----:|:------------------------------------------------------------------------:|:--------:|:--------------------------------------------:|:----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| clip  | clip_vit_b_32 <br/>clip_vit_b_16<br/>clip_vit_l_14<br/>clip_vit_l_14@336 | Cifar100 | 57.24% <br/> 61.41% <br/> 69.67%<br/> 68.19% | eval<br/>predict | [link](../../examples/zero_shot_image_classification/clip_vit_b_32_eval_on_cifar100.sh) <br/> [link](../../examples/zero_shot_image_classification/clip_vit_b_32_predict_on_cifar100.sh) |

### [Cifar100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

- 数据集大小：161M，共60000张图片，100个类别
    - 训练集：50000张图片
    - 测试集：10000张图片
- 数据格式：二进制文件

 ```bash
数据集目录格式
└─cifar-100-python
    ├─meta
    ├─test  
    └─train  
 ```

## 快速任务接口

- Trainer接口开启评估/推理：

  ```python
  import os
  from mindformers import MindFormerBook
  from mindformers.trainer import Trainer
  from mindformers import build_dataset, MindFormerConfig

  # 构造数据集
  project_path = MindFormerBook.get_project_path()
  dataset_config = MindFormerConfig(os.path.join(project_path, "configs",
                                        "clip", "task_config", "clip_cifar100_dataset.yaml"))
  print(dataset_config.eval_dataset.data_loader.dataset_dir)
  # 将cifar-100-python数据集放置到路径：./cifar-100-python
  dataset = build_dataset(dataset_config.eval_dataset_task)

  # 显示Trainer的模型支持列表
  MindFormerBook.show_trainer_support_model_list("zero_shot_image_classification")
  # INFO - Trainer support model list for zero_shot_image_classification task is:
  # INFO -    ['clip_vit_b_32', 'clip_vit_b_16', 'clip_vit_l_14', 'clip_vit_l_14@336']
  # INFO - -------------------------------------

  # 初始化trainer
  trainer = Trainer(task='zero_shot_image_classification',
      model='clip_vit_b_32',
      eval_dataset=dataset
  )

  # trainer.train() #零样本分类无需微调训练
  trainer.evaluate()  #进行评估
  # INFO - Top1 Accuracy=57.24%
  trainer.predict(input_data=dataset)  #进行推理
  # INFO - output result is saved at ./results.txt
  ```

- pipeline接口开启快速推理

  ```python
  from mindformers import pipeline, MindFormerBook
  from mindformers.tools.image_tools import load_image

  # 显示pipeline支持的模型列表
  MindFormerBook.show_pipeline_support_model_list("zero_shot_image_classification")
  # INFO - Pipeline support model list for zero_shot_image_classification task is:
  # INFO -    ['clip_vit_b_32', 'clip_vit_b_16', 'clip_vit_l_14', 'clip_vit_l_14@336']
  # INFO - -------------------------------------

  # pipeline初始化
  classifier = pipeline("zero_shot_image_classification",
                        model="clip_vit_b_32"
                        candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
  img = load_image("https://ascend-repo-modelzoo.obs.cn-east-2."
            "myhuaweicloud.com/XFormer_for_mindspore/clip/sunflower.png")
  classifier(img)
  # 输出
  # [[{'score': 0.99995565, 'label': 'sunflower'}, {'score': 2.5318595e-05, 'label': 'toy'},
  # {'score': 9.903885e-06, 'label': 'dog'}, {'score': 6.75336e-06, 'label': 'tree'},
  # {'score': 2.396818e-06, 'label': 'cat'}]]
  ```
