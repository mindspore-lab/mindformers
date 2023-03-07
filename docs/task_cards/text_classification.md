# Text Classification

## 任务描述

文本分类：模型在基于文本对的微调后，可以在给定任意文本对与候选标签列表的情况下，完成对文本对关系的分类，文本对的两个文本之间以-分割。

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

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](https://gitee.com/mindspore/transformer/blob/master/README.md#%E6%96%B9%E5%BC%8F%E4%B8%80clone-%E5%B7%A5%E7%A8%8B%E4%BB%A3%E7%A0%81)

- 脚本运行测试

```shell
# finetune
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode train

# evaluate
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode eval --load_checkpoint txtcls_bert_base_uncased_mnli

# predict
python run_mindformer.py --config ./configs/txtcls/run_txtcls_bert_base_uncased.yaml --run_mode predict --load_checkpoint txtcls_bert_base_uncased_mnli --predict_data [TEXT]
```

### 调用API启动

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
  # 将mnli验证数据集放置到路径：./mnli/eval; 将mnli训练数据集放置到路径：./mnli/train
  dataset = build_dataset(dataset_config.eval_dataset_task)

  # 显示Trainer的模型支持列表
  MindFormerBook.show_trainer_support_model_list("text_classification")
  # INFO - Trainer support model list for txt_classification task is:
  # INFO -    ['txtcls_bert_base_uncased']
  # INFO - -------------------------------------

  # 初始化trainer
  trainer = Trainer(task='text_classification',
      model='txtcls_bert_base_uncased',
      eval_dataset=dataset)

  # 测试数据
  input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
                "i don't know um do you do a lot of camping-I know exactly."]

  trainer.train(resume_or_finetune_from_checkpoint="txtcls_bert_base_uncased",
                do_finetune=True) #微调训练
  trainer.evaluate(eval_checkpoint='./output/rank_0/checkpoint/mindformers_rank_0-3_6135.ckpt')  #进行评估
  # INFO - Top1 Accuracy=84.8%
  trainer.predict(predict_checkpoint='./output/rank_0/checkpoint/mindformers_rank_0-3_6135.ckpt', input_data=input_data, top_k=1)  #进行推理
  # INFO - output result is [[{'label': 'neutral', 'score': 0.9714198708534241}],
  #                         [{'label': 'contradiction', 'score': 0.9967639446258545}]]
  ```

- pipeline接口开启快速推理

  ```python
  from mindformers.pipeline import TextClassificationPipeline
  from mindformers import AutoTokenizer, BertForMultipleChoice, AutoConfig

  input_data = ["The new rights are nice enough-Everyone really likes the newest benefits ",
                  "i don't know um do you do a lot of camping-I know exactly."]

  tokenizer = AutoTokenizer.from_pretrained('txtcls_bert_base_uncased_mnli')
  txtcls_mnli_config = AutoConfig.from_pretrained('txtcls_bert_base_uncased_mnli')

  # Because batch_size parameter is required when bert model is created, and pipeline
  # function deals with samples one by one, the batch_size parameter is seted one.
  txtcls_mnli_config.batch_size = 1

  model = BertForMultipleChoice(txtcls_mnli_config)
  txtcls_pipeline = TextClassificationPipeline(task='text_classification',
                                               model=model,
                                               tokenizer=tokenizer,
                                               max_length=model.config.seq_length,
                                               padding="max_length")

  results = txtcls_pipeline(input_data, top_k=1)
  print(results)
  # 输出
  # [[{'label': 'neutral', 'score': 0.9714198708534241}], [{'label': 'contradiction', 'score': 0.9967639446258545}]]
  ```
