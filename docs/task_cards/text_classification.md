# Text Classification

## 任务描述

文本分类：模型在基于文本对的微调后，可以在给定任意文本对与候选标签列表的情况下，完成对文本对的分类。

[相关论文](https://arxiv.org/pdf/1810.04805.pdf) Jacob Devlin, Ming-Wei Chang, et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019.

## 已支持数据集性能

| model  |                            type                            | datasets |  Top1-accuracy  |           stage            |                                                                                                                             example                                                                                                                              |
|:------:|:----------------------------------------------------------:|:--------:|:---------------:|:--------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  bert  | txtcls_bert_base_uncased |   Mnli   | 30.9% |          pretrain          |                                                                                                                                --                                                                                                                                |
| txtcls | txtcls_bert_case_uncased_mnli |   Mnli   | 84.8% | train<br/>eval<br/>predict | [link](../../txtcls_classification/txtcls_bert_base_uncased_train_on_mnli.sh) <br/> [link](../../txtcls_classification/txtcls_bert_base_uncased_mnli_eval_on_mnli.sh) <br/> [link](../../txtcls_classification/txtcls_bert_base_uncased_mnli_predict_on_mnli.sh) |

### [Mnli](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip)

- 数据集大小：298M，共431992个样本，3个类别
    - 训练集：392702个样本
    - 匹配测试集：9796个样本
    - 非匹配测试集：9847个样本
    - 匹配开发集：9815个样本
    - 非匹配开发集：9832个样本
- 数据格式：tsv文件

 ```bash
数据集目录格式
└─mnli
    ├─dev
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
                                        "txtcls", "task_config", "bert_mnli_dataset.yaml"))
  print(dataset_config.eval_dataset.data_loader.dataset_dir)
  # 将mnli数据集放置到路径：./mnli
  dataset = build_dataset(dataset_config.eval_dataset_task)

  # 显示Trainer的模型支持列表
  MindFormerBook.show_trainer_support_model_list("txt_classification")
  # INFO - Trainer support model list for txt_classification task is:
  # INFO -    ['txtcls_bert_base_uncased']
  # INFO - -------------------------------------

  # 初始化trainer
  trainer = Trainer(task='text_classification',
      model='txtcls_bert_base_uncased',
      eval_dataset=dataset
  )

  trainer.train() #微调训练
  trainer.evaluate()  #进行评估
  # INFO - Top1 Accuracy=84.8%
  trainer.predict(input_data=dataset, top_k=1)  #进行推理
  # INFO - output result is [[{'label': 'neutral', 'score': 0.9714198708534241}],
  #                         [{'label': 'contradiction', 'score': 0.9967639446258545}]]
  ```

- pipeline接口开启快速推理

  ```python
  from mindformers import pipeline, MindFormerBook

  # 显示pipeline支持的模型列表
  MindFormerBook.show_pipeline_support_model_list("text_classification")
  # INFO - Pipeline support model list for text_classification task is:
  # INFO -    ['txtcls_bert_base_uncased']
  # INFO - -------------------------------------

  # pipeline初始化
  classifier = pipeline("text_classification",
                        model="txtcls_bert_base_uncased_mnli")
  input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
                "i don't know um do you do a lot of camping-I know exactly."]
  classifier(input_data, top_k=1)
  # 输出
  # [[{'label': 'neutral', 'score': 0.9714198708534241}], [{'label': 'contradiction', 'score': 0.9967639446258545}]]
  ```
