### Bert 下游任务微调

#### Bert 模型介绍

BERT:全名`Bidirectional Encoder Representations from Transformers`模型是谷歌在2018年基于Wiki数据集训练的Transformer模型。  

[论文](https://arxiv.org/abs/1810.04805): J Devlin，et al., Pre-training of Deep Bidirectional Transformers for Language Understanding, 2019

#### Bert 下游任务微调

下面以question_answering任务为例介绍Bert下游任务微调的流程。

- 数据集

  SQuAD v1.1数据集：该数据集包含 10 万个（问题，原文，答案）三元组，原文来自于 536 篇维基百科文章，而问题和答案的构建主要是通过众包的方式，让标注人员提出最多 5 个基于文章内容的问题并提供正确答案，且答案出现在原文中。

  下载地址：[SQuAD v1.1训练集](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)，[SQuAD v1.1验证集](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

  新建名为squad文件夹，将下载的json格式数据集文件放入文件夹中。

  ```bash
  └─squad  
   ├─train-v1.1.json
   └─dev-v1.1.json
  ```

- 初始化question_answering任务trainer

  使用mindformers.trainer.Trainer类，初始化question_answering任务的trainer。

  ```python
  from mindformers.trainer import Trainer
  
  # 初始化question_answering任务trainer
  trainer = Trainer(task='question_answering',
                    model='qa_bert_base_uncased',
                    train_dataset='./squad/',
                    eval_dataset='./squad/')
  ```

  参数含义如下：

  - task(str) - 任务名称，'question_answering'为问答任务。

  - model(str) - 模型名称， 'qa_bert_base_uncased'为Bert接question_answering下游任务模型。

  - train_dataset(str) - 训练数据集所在路径。

  - eval_dataset(str) - 评估数据集所在路径。

- 使用现有的预训练权重进行finetune微调

  从obs上下载bert_base_uncased预训练权重，加载预训练权重，并在下游任务qa_bert_base_uncased模型上进行微调。

  ```python
  # 使用现有的预训练权重进行finetune微调
  trainer.train(resume_or_finetune_from_checkpoint="qa_bert_base_uncased",
                do_finetune=True)
  ```

  参数含义如下：

  - resume_or_finetune_from_checkpoint(str) - 权重名称，'qa_bert_base_uncased'为问答任务对应的Bert预训练权重。

  - do_finetune(bool) - 是否进行微调，True表示以微调的方式加载权重。

  训练过程中会实时打印训练时长、Loss等信息。

- 使用finetune获得的权重进行eval评估

  从finetune保存的权重文件中，取最后一次保存的checkpoint文件的权重加载进网络中，并进行评估。

  ```python
  # 使用finetune获得的最新权重进行eval评估
  trainer.evaluate(eval_checkpoint=True)
  ```

  参数含义如下：

  - eval_checkpoint(bool) - 是否加载最后一次保存的权重进行评估，True表示加载最后一次保存的权重文件中的权重进网络中。

  obs上训练好的权重评估结果如下：

  ```text
  INFO - QA Metric = {'QA Metric': {'exact_match': 80.74739829706716, 'f1': 88.33552874684968}}
  ```

- 使用finetune获得的权重进行predict推理

  从finetune保存的权重文件中，取最后一次保存的checkpoint文件的权重加载进网络中，并进行推理。推理输入的文本包括context和question两部分，两者以短横线“-”为标志分隔开。

  ```python
  # 使用finetune获得的最新权重进行predict推理
  # 测试数据，测试数据分为context和question两部分，两者以 “-” 分隔
  input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]
  trainer.predict(predict_checkpoint=True, input_data=input_data)
  ```

  参数含义如下：

  - predict_checkpoint(bool) - 是否加载最后一次保存的权重进行推理，True表示加载最后一次保存的权重文件中的权重进网络中。

  - input_data(str) - 输入文本，分为context和question两部分，两者以 “-” 分隔。

  得到的输出为：

  ```text
  [{'text': 'Berlin', 'score': 0.9941, 'start': 34, 'end': 40}]
  ```
