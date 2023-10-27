# 离线权重转换

## 概述

目前分布式训练/推理，当预训练权重与分布式策略不匹配时，需要**将预训练权重转换为对应分布式策略的权重**，主要适用场景如下：

- 基于完整权重的分布式训练/推理：需要将完整权重转换为多卡分布式权重。
- 修改分布式策略进行训练/推理：需要将权重转换为对应分布式策略的权重。

**离线权重转换**需要以下3个步骤：

1、获取当前任务的分布式策略文件；

2、运行离线转换脚本获得目标权重；

3、配置load_checkpoint参数，启动分布式训练/推理。

主要参考：[mindspore分布式弹性训练与推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/parallel/resilience_train_and_predict.html)

## 方案1：源码执行

### step1：获取当前任务的分布式策略文件

在yaml文件中配置only_save_strategy=True，正常启动分布式任务，生成对应的分布式策略文件后，任务将会主动退出。

分布式策略文件保存为`output/strategy/ckpt_strategy_rank_x.ckpt`，**ckpt_strategy_rank_x.ckpt**数量和卡数相同。

```yaml
only_save_strategy: True
```

### step2：运行离线转换脚本获得目标权重

```shell
python mindformers/tools/transform_ckpt.py \
--src_ckpt_strategy src_strategy_path_or_dir \
--dst_ckpt_strategy dst_strategy_path_or_dir \
--src_ckpt_dir src_ckpt_dir \
--dst_ckpt_dir dst_ckpt_dir \
--prefix "checkpoint_"
```

参数说明:

- src_ckpt_strategy：源权重对应的分布式策略文件路径。**源权重为完整权重则不填写**；若为分布式权重，视以下情况填写：

  1. 源权重开启了流水线并行：权重转换基于**合并的策略文件**，填写**分布式策略文件夹路径**，脚本会自动将文件夹内所有**ckpt_strategy_rank_x.ckpt**合并，并在文件夹下生成**merged_ckpt_strategy.ckpt**；如果已有**merged_ckpt_strategy.ckpt**，可以直接填写该文件路径。

  2. 源权重未开启流水线并行：权重转换基于**任一策略文件**，填写任一**ckpt_strategy_rank_x.ckpt**路径即可。

  **注**：如果策略文件夹下存在**merged_ckpt_strategy.ckpt**，仍传入文件夹路径，脚本首先会将旧的**merged_ckpt_strategy.ckpt**删除，再合并一个新的**merged_ckpt_strategy.ckpt**用于权重转换，因此需要**确保文件夹有足够的写入权限**，否则将会报错。

- dst_ckpt_strategy：目标权重的分布式策略文件路径。**目标权重为完整权重则不填写**；若为分布式权重，请参考src_ckpt_strategy；

- src_ckpt_dir：源权重所在的文件夹路径，源权重须按照`model_dir/rank_x/xxx.ckpt`格式存放，文件夹路径填写为**model_dir**。

- dst_ckpt_dir：目标权重保存路径，为自定义空文件夹路径，目标权重的保存格式为`model_dir/rank_x/xxx.ckpt`。

- prefix：目标权重保存名前缀，默认为"checkpoint_"，即权重按照`model_dir/rank_x/checkpoint_x.ckpt`保存。

### step3：配置load_checkpoint参数

将yaml配置文件中`load_checkpoint`关键字指定为目标权重路径，视以下情况填写：

- 目标权重为分布式切分权重：填写权重文件夹路径，即model_dir；
- 目标权重为完整权重：填写权重文件路径，即model_dir/rank_0/xxx.ckpt；

```yaml
load_checkpoint: model_dir_or_path
```

## 方案2：高阶API 执行

参考Trainer API使用

使用`TrainingArguments`类打开`only_save_strategy`字段，获取当前任务的分布式策略文件，其余步骤可参考**方案1**。

# 自动权重转换（推荐）

## 概述

Mindformer支持**自动权重转换**，当预训练权重与分布式策略不匹配时，将**auto_trans_ckpt**开关置为True，并配置权重转换相关参数，由Mindformer自动完成权重转换，相比**权重离线切分转换**提升了任务启动效率。

**自动权重转换**只需要配置以下参数：

- **load_checkpoint**：源权重所在的文件夹路径，源权重须按照`model_dir/rank_x/xxx.ckpt`格式存放，文件夹路径填写为**model_dir**。

- **src_strategy_path_or_dir**：源权重对应的分布式策略文件路径。**源权重为完整权重则不填写**；若为分布式权重，视以下情况填写：

  1. 源权重开启了流水线并行：权重转换基于**合并的策略文件**，填写**分布式策略文件夹路径**，Mindformer会自动将文件夹内所有**ckpt_strategy_rank_x.ckpt**合并，并在文件夹下生成**merged_ckpt_strategy.ckpt**；如果已有**merged_ckpt_strategy.ckpt**，可以直接填写该文件路径。

  2. 源权重未开启流水线并行：权重转换基于**任一策略文件**，填写任一**ckpt_strategy_rank_x.ckpt**路径即可。

  **注**：如果策略文件夹下存在**merged_ckpt_strategy.ckpt**，仍传入文件夹路径，Mindformer首先会将旧的**merged_ckpt_strategy.ckpt**删除，再合并一个新的**merged_ckpt_strategy.ckpt**用于权重转换，因此需要确保文件夹有足够的写入权限，否则将会报错。

- **auto_trans_ckpt**：权重自动转换开关，为True开启，默认False。

**自动权重转换**会在`output`文件夹下输出两个结果文件夹，分别是**strategy**和 **transformed_checkpoint**：

- **strategy**：保存当前任务的**分布式策略文件**，文件夹内主要有以下两种文件：

  ① **ckpt_strategy_rank_x.ckpt**：rank_x的分布式策略文件；

  ② **merged_ckpt_strategy.ckpt**: 所有**ckpt_strategy_rank_x.ckpt**合并成的分布式策略文件；

  ​     只有开启流水线并行，才会有 **merged_ckpt_strategy.ckpt**。

- **transformed_checkpoint**：保存**转换后的目标权重**，目标权重的保存格式为`transformed_checkpoint/rank_x/checkpoint_x.ckpt`

## 适用场景

Mindformer的**自动权重转换**特性适用于以下三大任务场景，基本可以满足各种权重转换需求：

- **基于完整权重，启动分布式任务**
- **修改分布式策略，启动分布式任务**
- **基于分布式权重，启动单卡任务**

针对以上适用场景，本文档列举了六种使用**权重自动转换**特性的训练/推理案例，供用户参考：

- 训练案例一：基于完整权重，启动8卡分布式训练；
- 训练案例二：基于8卡分布式权重，启动4卡分布式训练；
- 推理案例一：基于8卡分布式权重，启动单卡推理；
- 推理案例二：基于8卡分布式权重，启动2卡分布式推理；
- 推理案例三：基于完整权重，启动2卡分布式推理；

- ModelArts训练案例：基于完整权重，启动2节点16卡分布式训练；

## 注意事项

**注1**：传入权重需要按照`model_dir/rank_x/xxx.ckpt`格式存放，确保每个`rank_x`文件夹下**仅存放一个ckpt文件**。

**注2**：开启**自动权重转换**后，任务首先会删除`output`下旧的strategy和transformed_checkpoint，然后保存当前任务的输出结果。因此上一次转换任务结束后，如有必要**请将strategy和transformed_checkpoint保存到自定义文件夹，避免误删**。

**注3**：**分布式推理案例仅使用"模型并行"**，主要原因有以下两点：

- 分布式推理暂时不支持"流水线并行"，pipeline_stage固定设置为1。

- 案例采用run_distribute.sh启动分布式推理，仅支持单batch输入，data_parallel固定设置为1。

## 自动转换案例

### 前期准备

- 权重

[llama7B完整权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/open_llama_7b.ckpt)

- 词表

llama7B的[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/tokenizer.model)

- 数据集

[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用mindformers/tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/wiki2048.mindrecord
```

- rank_table_file

案例中分别会用到8卡、4卡、2卡对应的rank_table_file

```shell
# 8卡的rank_table_file：自行重命名为rank_table_8.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,8]
mv hccl*.json rank_table_8.json

# 4卡的rank_table_file：自行重命名为rank_table_4_id04.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,4]
mv hccl*.json rank_table_4_id04.json

# 2卡的rank_table_file：自行重命名为rank_table_2_id02.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,2]
mv hccl*.json rank_table_2_id02.json
```

### 训练案例一：基于完整权重，启动8卡分布式训练

**案例描述**：基于一份完整的llama-7B预训练权重，使用8卡进行分布式训练。

完整权重：

![llama7b_autotrans_1to8_train_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_ckpt.png)

预训练数据集：

![llama7b_autotrans_1to8_train_ckpt](assets/Transform_Ckpt/wiki_dataset.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照model_dir/rank_x/xxx.ckpt格式存放，填写model_dir
load_checkpoint: "/worker/checkpoint/llama-7b/single/"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 配置数据集
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/worker/dataset/wikitext_2048/"
    shuffle: True

# 配置8卡分布式策略，仅供参考
parallel_config:
  data_parallel: 2
  model_parallel: 1
  pipeline_stage: 4
  micro_batch_num: 4
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

② 启动训练

```shell
cd script
bash run_distribute.sh ../rank_table_8.json ../configs/llama/run_llama_7b.yaml [0,8] train
```

③ 查看权重转换相关日志

![llama7b_autotrans_1to8_train_log](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_log1.png)

训练正常：

![llama7b_autotrans_1to8_train_log](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`output/strategy`文件夹下，由于开启了**流水线并行**，会对所有`ckpt_strategy_rank_x.ckpt`进行合并，得到`merged_ckpt_strategy.ckpt`。若不开启流水线并行，则不会合并。

![llama7b_autotrans_1to8_train_strategy](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![llama7b_autotrans_1to8_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_transformed_checkpoint.png)

⑤ 保存生成的**分布式策略文件**和**分布式权重**到自定义文件夹下，以供后续使用

```shell
mv ../output/transformed_checkpoint/ /worker/checkpoint/llama-7b/multi_dp2mp1pp4
mv ../output/strategy/ /worker/checkpoint/llama-7b/multi_dp2mp1pp4/
```

![llama7b_autotrans_1to8_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_save.png)

### 训练案例二：8卡分布式权重自动切分为4卡分布式权重

**案例描述**：基于8卡的分布式权重，转换到4卡进行分布式训练。

8卡分布式权重和策略文件：

- 用**训练案例一**保存好的

![llama7b_autotrans_1to8_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_save.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照model_dir/rank_x/xxx.ckpt格式存放，填写model_dir
load_checkpoint: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/"

# 配置分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/strategy/merged_ckpt_strategy.ckpt"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 设置数据集
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/worker/dataset/wikitext_2048/"
    shuffle: True

# 4卡分布式配置参考
# default parallel of device num = 8 910A
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 2
  micro_batch_num: 2
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

② 启动训练

```shell
cd script
bash run_distribute.sh ../rank_table_4_id04.json ../configs/llama/run_llama_7b.yaml [0,4] train
```

③ 查看权重转换相关日志

![llama7b_autotrans_8to4_train_log](assets/Transform_Ckpt/llama7b_autotrans_8to4_train_log1.png)

训练正常：

![llama7b_autotrans_8to4_train_log](assets/Transform_Ckpt/llama7b_autotrans_8to4_train_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`output/strategy`文件夹下，由于开启了**流水线并行**，会对所有`ckpt_strategy_rank_x.ckpt`进行合并，得到`merged_ckpt_strategy.ckpt`。若不开启流水线并行，则不会合并。

![llama7b_autotrans_8to4_train_strategy](assets/Transform_Ckpt/llama7b_autotrans_8to4_train_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![llama7b_autotrans_8to4_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_8to4_train_transformed_checkpoint.png)

注：**strategy**和**transformed_checkpoint**两个文件夹请及时保存到**自定义文件夹**中，以免被后续转换任务清空。

### 推理案例一：8卡分布式权重自动合并为完整权重

**案例描述**：基于8卡的分布式权重，合并为完整权重进行单卡推理。

8卡分布式权重和策略文件：

- 用**训练案例一**保存好的

![llama7b_autotrans_1to8_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_save.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照model_dir/rank_x/xxx.ckpt格式存放，填写model_dir
load_checkpoint: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/"

# 配置分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/strategy/merged_ckpt_strategy.ckpt"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 设置use_paralle为False
use_parallel: False

# 设置run_mode为predict
run_mode: 'predict'

# 打开增量推理
use_past: True

# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/worker/checkpoint/llama-7b/tokenizer.model"
```

② 启动推理

```shell
python run_mindformer.py --config configs/llama/run_llama_7b.yaml --predict_data "I love beijing, because"
```

③ 查看权重转换相关日志

![llama7b_autotrans_8to1_predict_log1](assets/Transform_Ckpt/llama7b_autotrans_8to1_predict_log1.png)

推理正常：

![llama7b_autotrans_8to1_predict_log1](assets/Transform_Ckpt/llama7b_autotrans_8to1_predict_log2.png)

④ 查看合并后的权重

**完整权重**：保存在`output/transformed_checkpoint`文件夹下

![llama7b_autotrans_8to1_predict_ckpt](assets/Transform_Ckpt/llama7b_autotrans_8to1_predict_transformed_checkpoint.png)

注：**transformed_checkpoint**请及时保存到**自定义文件夹**中，以免被后续转换任务清空。

### 推理案例二：8卡分布式权重自动切分为2卡分布式权重

**案例描述**：基于8卡的分布式权重，自动切分为2卡分布式权重进行分布式推理。

8卡分布式权重和策略文件：

- 用**训练案例一**保存好的

![llama7b_autotrans_1to8_train_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_save.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照model_dir/rank_x/xxx.ckpt格式存放，填写model_dir
load_checkpoint: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/"

# 配置分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama-7b/multi_dp2mp1pp4/strategy/merged_ckpt_strategy.ckpt"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 设置use_paralle为True
use_parallel: True

# 设置run_mode为predict
run_mode: 'predict'

# 打开增量推理
use_past: True

# 2卡分布式配置参考
# default parallel of device num = 8 910A
# 由于通过run_distribute.sh拉起推理，内部走的是pipeline推理流程，暂时不支持多batch推理，因此data_parallel设置为1
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/worker/checkpoint/llama-7b/tokenizer.model"
```

② 启动推理

```shell
cd script
bash run_distribute.sh rank_table_2_id02.json ../configs/llama/run_llama_7b.yaml [0,2] predict "I love beijing, because"
```

③ 查看权重转换相关日志

![](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_log1.png)

推理正常：

![llama7b_autotrans_8to2_predict_log2](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`output/strategy`文件夹下

![llama7b_autotrans_8to2_predict_strategy](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![llama7b_autotrans_8to2_predict_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_transformed_checkpoint.png)

注：**strategy**和**transformed_checkpoint**两个文件夹请及时保存到**自定义文件夹**中，以免被后续转换任务清空。

### 推理案例三：完整权重自动切分为2卡分布式权重

**案例描述**：基于一份完整的llama-7B预训练权重，使用2卡进行分布式推理。

完整权重：

![llama7b_autotrans_1to8_train_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_ckpt.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照model_dir/rank_x/xxx.ckpt格式存放，填写model_dir
load_checkpoint: "/worker/checkpoint/llama-7b/single/"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 设置use_paralle为True
use_parallel: True

# 设置run_mode为predict
run_mode: 'predict'

# 打开增量推理
use_past: True

# 2卡分布式配置参考
# default parallel of device num = 8 910A
# 由于通过run_distribute.sh拉起推理，内部走的是pipeline推理流程，暂时不支持多batch推理，因此data_parallel设置为1
parallel_config:
  data_parallel: 1
  model_parallel: 2
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/worker/checkpoint/llama-7b/tokenizer.model"
```

② 启动推理

```shell
cd script
bash run_distribute.sh rank_table_2_id02.json ../configs/llama/run_llama_7b.yaml [0,2] predict "I love beijing, because"
```

③ 查看权重转换相关日志

![llama7b_autotrans_1to2_predict_log1](assets/Transform_Ckpt/llama7b_autotrans_1to2_predict_log1.png)

推理正常：

![llama7b_autotrans_1to2_predict_log2](assets/Transform_Ckpt/llama7b_autotrans_1to2_predict_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`output/strategy`文件夹下

![](assets/Transform_Ckpt/llama7b_autotrans_1to2_predict_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![](assets/Transform_Ckpt/llama7b_autotrans_1to2_predict_transformed_checkpoint.png)

注：**strategy**和**transformed_checkpoint**两个文件夹请及时保存到**自定义文件夹**中，以免被后续转换任务清空。

### ModelArts训练案例

**案例描述**：基于一份完整的llama-7B预训练权重，在Modelarts上使用16卡进行分布式训练。

**步骤**：

① 配置参数

```yaml
# 16卡分布式配置参考
# default parallel of device num = 8 910A
parallel_config:
  data_parallel: 2
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 2
  vocab_emb_dp: True
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1
```

② 训练作业配置

![llama13b_autotrans_1to16_train_modelarts_inputs](assets/Transform_Ckpt/llama7b_autotrans_1to16_train_modelarts_config.png)

③ 提交训练作业，查看训练日志

![llama13b_autotrans_1to16_train_modelarts_log1](assets/Transform_Ckpt/llama7b_autotrans_1to16_train_modelarts_log1.png)

![llama13b_autotrans_1to16_train_modelarts_log2](assets/Transform_Ckpt/llama7b_autotrans_1to16_train_modelarts_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`remote_save_url/strategy`文件夹下，由于开启了**流水线并行**，会对所有`ckpt_strategy_rank_x.ckpt`进行合并，得到`merged_ckpt_strategy.ckpt`。若不开启流水线并行，则不会合并。

![llama13b_autotrans_1to16_train_modelarts_strategy](assets/Transform_Ckpt/llama7b_autotrans_1to16_train_modelarts_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![llama13b_autotrans_1to16_train_modelarts_distribute_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to16_train_modelarts_transformed_checkpoint.png)

注：**strategy**和**transformed_checkpoint**两个文件夹请及时保存到**自定义文件夹**中，以免被后续转换任务清空。