# PanguAlpha

## 模型描述

「鹏程·盘古α」由以鹏城实验室为首的技术团队联合攻关，首次基于“鹏城云脑Ⅱ”和国产MindSpore框架的自动混合并行模式实现在2048卡算力集群上的大规模分布式训练，训练出业界首个2000亿参数以中文为核心的预训练生成语言模型。鹏程·盘古α预训练模型支持丰富的场景应用，在知识问答、知识检索、知识推理、阅读理解等文本生成领域表现突出，具备很强的小样本学习能力。

[论文](https://arxiv.org/abs/2104.12369)J Wei Zeng, Xiaozhe Ren, Teng Su，et al., PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation, 2021

## 数据集准备

以悟道数据集为例

- 数据集下载：[悟道数据集](https://data.baai.ac.cn/details/WuDaoCorporaText#a2)

- 词表下载：[model.vocab](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenizer/vocab.model)

- 参考[ModelZoo](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha#%E6%95%B0%E6%8D%AE%E9%9B%86%E7%94%9F%E6%88%90)，将数据处理成Mindrecord格式。注：训练数据处理时，长度应等于模型接收长度加一

```bash
# 数据预处理示例代码，代码来源于ModelZoo
# 生成Mindrecord数据，其中output_file需以字符串mindrecord结尾
python -m preprocess.py --input_glob  'data/*.txt' --tokenizer jieba --eot 40000 --data_column_name input_ids --seq_length 1025
```

## 快速使用

### 脚本启动

> 需开发者提前clone工程。

- 请参考[使用脚本启动](../../README.md#方式一使用已有脚本启动)

示例命令如下，将会执行2.6b大小的pangualpha模型训练

#### 单卡启动

```shell
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b.yaml \
                         --run_mode train \
                         --device_target Ascend \
                         --train_dataset_dir /your_path/wudao-mindrecord
```

其中`device_target`根据用户的运行设备不同，可选`GPU/Ascend`。另，模型和训练等相关配置可在`configs/pangualpha`目录下的yaml文件中配置。

#### 单机多卡启动

- 运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

```shell

# step1：机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：# 执行运行脚本：8卡分布式运行， DEVICE_RANGE = [0, 8]， 不包含8本身。
cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_STATUS

```

```python
# RANK_TABLE_FILE 参考样例
# 单机8卡
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.155.111.140",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

```text
# 参数说明
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的pangualpha/run_pangualpha_*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围，如[0,8]为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train\finetune\predict
```

其中，模型和训练等相关配置可在`configs/pangualpha`目录下的yaml文件中配置，如数据集路径，可在`configs/pangualpha/run_pangualpha_*.yaml`中配置`dataset_dir`参数。

#### 多机多卡启动

- 首先参考单机多卡启动方式，在每台机器上运行mindformers/tools/hccl_tools.py生成RANK_TABLE_FILE的json文件；

- 执行merge_hccl.py脚本将不同机器上生成的RANK_TABLE_FILE文件中的hccl*.json进行合并，包括server_list合并，server_count设为机器数，rank_id顺序增加，并保证不同机器上的RANK_TABLE_FILE相同；

- 在多机上同时拉起任务，每台机器拉起方式参考单机多卡启动方式，需注意的是，多机多卡的拉起方式，相对于单机多卡，多了一个总卡数`[RANK_SIZE]`的入参。

```shell
# step1：在每个机器上运行如下命令，生成各自的RANK_TABLE_FILE的json文件。
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"

# step2：运行如下命令，合并每个机器上的RANK_TABLE_FILE文件。
python ./mindformers/tools/merge_hccl.py hccl*.json

# step3：将step2得到的合并后的RANK_TABLE_FILE文件分别复制到所有的机器上。

# step4：根据服务器节点数等信息，修改相应的配置
'''
以pangualpha-13b模型两机训练为例，默认配置2机16卡，如果节点数有变，需要修改相应的配置。配置文件在../configs/pangualpha/run_pangualpha_13b.yaml

parallel_config:
  data_parallel: 4
  model_parallel: 4
  pipeline_stage: 1
  optimizer_shard: True
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
'''

# step5：执行运行脚本
# 第一台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the first device} ../configs/pangualpha/run_pangualpha_13b.yaml [0,8] train 32
# 第二台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the second device} ../configs/pangualpha/run_pangualpha_13b.yaml [8,16] train 32
# 第三台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the third device} ../configs/pangualpha/run_pangualpha_13b.yaml [16,24] train 32
# 第四台机器
bash run_distribute.sh {RANK_TABLE_FILE path of the forth device} ../configs/pangualpha/run_pangualpha_13b.yaml [24,32] train 32
```

### 调用API启动

> 需开发者提前pip安装。具体接口说明请参考[API接口](https://gitee.com/mindspore/transformer/wikis/API/)

- Model调用接口

```python
from mindformers import PanguAlphaHeadModel, PanguAlphaTokenizer

model = PanguAlphaHeadModel.from_pretrained('pangualpha_2_6b')
model.set_train(False)
tokenizer = PanguAlphaTokenizer.from_pretrained('pangualpha_2_6b')
inputs = tokenizer(["今天天气很好"],
                 padding='max_length',
                 max_length=model.config.seq_length,
                 return_tensors='ms')
output = model(inputs["input_ids"])
print(output)  # 计算输出的logits

model.set_train(True)
inputs = tokenizer(["今天天气很好"],
                   padding='max_length',
                   max_length=model.config.seq_length+1,
                   return_tensors='ms')
output = model(inputs["input_ids"])
print(output)  # 计算loss
```

- Trainer接口开启训练/推理：

```python
from mindformers.trainer import Trainer
# 方式1: 开启训练，并使用训练完成后的权重进行推理
trainer = Trainer(task='text_generation', model='pangualpha_2_6b', train_dataset="your data file path")
trainer.train()
res = trainer.predict(predict_checkpoint=True, input_data="我喜欢北京，因为")

# 方式2： 从obs下载训练好的权重并进行推理
trainer = Trainer(task='text_generation', model='pangualpha_2_6b')
res = trainer.predict(input_data="我喜欢北京，因为")
```

- pipeline接口开启快速推理

```python
from mindformers.pipeline import pipeline
pipeline_task = pipeline("text_generation", model='pangualpha_2_6b', max_length=50)
pipeline_result = pipeline_task("我喜欢北京，因为", top_k=3)
print(pipeline_result)
```

**注：**，要提高推理速度，可在`config`yaml文件中进行如下配置，设置增量推理`use_past`为True。

```python
# model config
use_past: True          # 开启增量推理
use_moe: False
expert_num: 1
per_token_num_experts_chosen: 1
checkpoint_name_or_path: "pangualpha_2_6b"
repetition_penalty: 1
max_decode_length: 1024
top_k: 3
top_p: 1
do_sample: False
```

## 模型权重下载

[盘古Alpha权重下载](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)

## 评测

盘古α支持TNEWS和CMRC2018两个中文数据集的评测。

- 文本分类：
    - 获取数据集:
        - [TNEWS数据集](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。
    - 处理数据成mindrecord格式
        - TNEWS：

        ```bash
        # 注：生成的数据集文件需以.mindrecord结尾
        cd mindformers/tools/dataset_preprocess/pangualpha
        python tnews_data_process.py --input_file {your_path/dev.json} \
                                     --label_file {your_path/labels.json} \
                                     --output_file {your_path/tnews.mindrecord}
        ```

    - 开启评测：评测指标为ACC
        - TNEWS

        ```bash
        python run_mindformers.py --config configs/pangualpha/run_pangualpha_2_6b_prompt_txtcls.yaml \
                                  --eval_dataset_dir {your_path/tnews.mindrecord} \
                                  --run_mode eval
        # ACC: 0.646, total_acc_num: 6458, total_num: 10000
        ```

- 阅读理解：
    - 获取数据集:
        - [CMRC2018数据集](https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip)是用于中文机器阅读理解的片段抽取任务(Span-Extraction)的数据，这个数据集由近20000个真实的问题组成，这些问题由人类专家在维基百科的段落中注释。
    - 处理数据成mindrecord格式
        - CMRC2018：

        ```bash
        # 注：生成的数据集文件需以.mindrecord结尾
        cd mindformers/tools/dataset_preprocess/pangualpha
        python wikitext2_data_process.py --train_file {your_path/train.json} \
                                         --dev_file {your_path/dev.json} \
                                         --output_file {your_path/cmrc2018.mindrecord}
        ```

    - 开启评测：评测指标为Em/F1
        - CMRC2018

        ```bash
        python run_mindformers.py --config configs/pangualpha/run_pangualpha_2_6b_em_f1.yaml \
                                  --eval_dataset_dir {your_path/cmrc2018.mindrecord} \
                                  --run_mode eval
        # F1 score: 21.12, Em score: 2.10, total_count: 1806
        ```
