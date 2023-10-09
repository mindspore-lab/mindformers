# 权重离线切分转换

该特性适用于：1.  权重过大，单卡无法加载；2. 权重为切分权重且与目标网络和运行卡数不匹配；

此时可利用本特性进行权重的切分和转换，以适配目标网络运行；

使用场景：1. 分布式恢复训练（恢复时卡数或者并行策略发生改变）；2. 评估/推理场景（权重需要进一步切分或者合并）

## 方案1：源码执行

- step1（默认已有待切分权重相对应的策略文件，若没有，也可参考以下方法生成）

在config中配置`only_save_strategy: True`，正常启动分布式训练/评估/推理，生成目标卡数的分布式策略文件后将会退出。生成的分布式策略文件保存在`output/strategy`目录下。

```yaml
only_save_strategy: True
```

- step2

运行如下脚本完成权重切分转换

```shell
python mindformers/tools/transform_ckpt.py --src_ckpt_strategy SRC_CKPT_STRATEGY --dst_ckpt_strategy DST_CKPT_STRATEGY --src_ckpt_dir SRC_CKPT_DIR --dst_ckpt_dir DST_CKPT_DIR
```

参数说明:

`src_ckpt_strategy`：待转权重的分布式策略文件路径。
  若为None,表示待转权重为完整权重;
  若为切分策略文件,表示原始的权重对应的策略文件;
  若为文件夹,表示需要合并文件夹内策略文件(仅在流水并行生成的策略文件时需要),合并后的策略文件保存在`SRC_CKPT_STRATEGY/merged_ckpt_strategy.ckpt`路径下;

`dst_ckpt_strategy`：目标权重的分布式策略文件路径。即step1中生成的分布式策略文件路径。
  若为None,表示将待转权重合并为完整权重;
  若为切分策略文件,表示目标卡数对应的策略文件
  若为文件夹,表示需要合并文件夹内策略文件(仅在流水并行生成的策略文件时需要),合并后的策略文件保存在`DST_CKPT_STRATEGY/merged_ckpt_strategy.ckpt`路径下;

`src_ckpt_dir`: 待转权重路径，须按照`SRC_CKPT_DIR/rank_{i}/checkpoint_{i}.ckpt`存放，比如单一权重存放格式为`SRC_CKPT_DIR/rank_0/checkpoint_0.ckpt`。

`dst_ckpt_dir`：目标权重保存路径，为自定义空文件夹路径，转换后模型以`DST_CKPT_DIR/rank_{i}/xxx.ckpt`存放。

- step3

将`config`的配置文件中`load_checkpoint`关键字指定为转换的目标权重保存路径，若转换后仍为切分权重，传入转换后的权重文件夹路径即可；若转换后为完整权重，传入权重文件路径即可正常启动训练。

```yaml
load_checkpoint: "{转换后权重文件夹/文件路径}"
```

## 方案2：高阶API 执行

参考Trainer API使用

使用`TrainingArguments`类打开`only_save_strategy`字段，其余步骤可参考**方案1**

# 权重自动切分转换

## 概述

目前分布式训练/推理，当权重与分布式策略不匹配时，往往涉及到离线转ckpt，大致需要以下3个步骤：

1、获取分布式权重策略文件；

2、离线转换，将权重转换为分布式训练/推理所需权重；

3、启动分布式训练/推理。

以上流程对外部客户来说易用性较差，因此设计**分布式权重自动转换方案**，在分布式训练/推理时能够无痛转换，提升分布式训练/推理任务启动效率。

**权重自动切分转换**会在`output`文件夹下生成两个结果文件夹，分别是**strategy**和 **transformed_checkpoint**：

- strategy：保存**分布式策略文件**，主要有以下两种文件：

  ① **ckpt_strategy_rank_x.ckpt**：rank x的分布式策略文件；

  ② **merged_ckpt_strategy.ckpt**: 所有rank的分布式策略文件合并后的分布式策略文件；（开启流水线并行时才会合并）

- transformed_checkpoint：保存**转换后的权重**，权重文件按照`transformed_checkpoint/rank_x/checkpoint_x.ckpt`格式保存：

**注：**每次转换结束后需要**将strategy和transformed_checkpoint保存到自定义文件夹**，因为每次拉起新的任务，如果开启了权重自动转换，会将这两个文件夹清空，然后保存最新任务的转换结果。

## 适用场景

### 1. 完整权重转为分布式权重

- 适用：**使用完整权重进行分布式训练/推理**

① 配置`load_checkpoint`参数为**完整权重文件夹路径**，权重需要按照`{model_dir}/rank_0/xxx.ckpt`格式存放，路径填写到**model_dir**为止；

② 配置`auto_trans_ckpt`参数为**True**；

③ 正常拉起分布式训练/推理；

### 2.分布式权重转为分布式权重

- 适用：**修改分布式策略后训练/推理**，如：16卡训练，4卡推理；16卡预训练，8卡微调等场景。

① 配置`load_checkpoint`参数为**分布式权重文件夹路径**，权重需要按照`{model_dir}/rank_x/xxx.ckpt`格式存放，路径填写到**model_dir**为止；

② 配置`src_strategy_path_or_dir`参数为**分布式策略文件路径**，有两种情况：

- 如果**原分布式权重训练时开启了流水线并行**，则需要传入合并后的策略文件，找到原分布式权重训练时保存的`strategy`文件夹，如果有**merged_ckpt_strategy.ckpt**，`src_strategy_path_or_dir`填写为**merged_ckpt_strategy.ckpt**的路径；如果只有**ckpt_strategy_rank_x.ckpt**，说明原分布式权重训练时没有使用权重自动转换，`src_strategy_path_or_dir`填写为 `strategy`文件夹路径，权重自动转换会首先将 `strategy`文件夹内所有**ckpt_strategy_rank_x.ckpt**合并成一个**merged_ckpt_strategy.ckpt**后，再进行转换；

- 如果**原分布式权重训练时未开启流水线并行**，则`src_strategy_path_or_dir`填写任一**ckpt_strategy_rank_x.ckpt**路径即可。

③ 配置`auto_trans_ckpt`参数为**True**；

④ 正常拉起分布式训练/推理；

### 3.分布式权重转为完整权重

- 适用：**分布式训练结束，使用单卡推理**，如：16卡训练，单卡推理

① 配置`load_checkpoint`参数为**分布式权重文件夹路径**，权重需要按照`{model_dir}/rank_x/xxx.ckpt`格式存放，路径填写到**model_dir**为止；

② 配置`src_strategy_path_or_dir`参数为**分布式策略文件路径**，有两种情况：

- 如果**原分布式权重训练时开启了流水线并行**，则需要传入合并后的策略文件，找到原分布式权重训练时保存的`strategy`文件夹，如果有**merged_ckpt_strategy.ckpt**，`src_strategy_path_or_dir`填写为**merged_ckpt_strategy.ckpt**的路径；如果只有**ckpt_strategy_rank_x.ckpt**，说明原分布式权重训练时没有使用权重自动转换，`src_strategy_path_or_dir`填写为 `strategy`文件夹路径，权重自动转换会首先将 `strategy`文件夹内所有**ckpt_strategy_rank_x.ckpt**合并成一个**merged_ckpt_strategy.ckpt**后，再进行转换；

- 如果**原分布式权重训练时未开启流水线并行**，则`src_strategy_path_or_dir`填写任一**ckpt_strategy_rank_x.ckpt**路径即可。

② 配置`auto_trans_ckpt`参数为**True**，`use_parallel`参数为**False**；

③ 正常拉起推理；

## 注意事项

**注1**：权重需要按照`{model_dir}/rank_x/xxx.ckpt`格式存放，确保每个`rank_x`文件夹下**仅存放一个ckpt文件**。

**注2**：分布式推理的并行策略目前仅支持"数据并行"和"模型并行"，暂时不支持"流水线并行"，pipeline_stage需要固定设置为1。

**注3**：以下**自动转换案例**中的**推理案例二**和**推理案例三**，仅考虑"模型并行"的分布式推理场景，data_parallel固定设置为1，主要是因为通过run_distribute.sh启动分布式推理，暂时不支持batch推理。

## 自动转换案例

### 前期准备

- 数据集

数据集下载：[WikiText2数据集](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)

分词模型下载：例如下载huggingface的[tokenizer.model](https://huggingface.co/openlm-research/open_llama_7b/blob/main/tokenizer.model)

使用以下预处理脚本生成mindrecord训练数据

```bash
# 使用tools/dataset_preprocess/llama/llama_preprocess.py进行数据预处理+Mindrecord数据生成
python llama_preprocess.py \
--dataset_type wiki \
--input_glob  /{path}/wiki.train.tokens \
--model_file /{path}/tokenizer.model \
--seq_length 2048 \
--output_file /{path}/wiki2048.mindrecord
```

- 权重

[llama7B完整权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/llama/open_llama_7b.ckpt)

- rank_table_file

案例中分别会用到8卡、4卡、2卡对应的rank_table_file

```shell
# 8卡的rank_table_file：自行重命名为rank_table_8.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,8]
mv hccl*.json rank_table_8.json

# 4卡的rank_table_file：自行重命名为rank_table_4_npu0-4.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,4]
mv hccl*.json rank_table_4_npu0-4.json

# 2卡的rank_table_file：自行重命名为rank_table_2_npu0-2.json，原文件为hccl_xxx.json
python mindformers/tools/hccl_tools.py --device_num [0,2]
mv hccl*.json rank_table_2_npu0-2.json
```

### 训练案例一：完整权重自动切分为8卡分布式权重

**案例描述**：基于一份完整的llama-7B预训练权重，使用8卡进行分布式训练。

完整权重：

![llama7b_autotrans_1to8_train_ckpt](assets/Transform_Ckpt/llama7b_autotrans_1to8_train_ckpt.png)

预训练数据集：

![llama7b_autotrans_1to8_train_ckpt](assets/Transform_Ckpt/wiki_dataset.png)

**步骤**：

① 配置参数

```yaml
# 配置预训练权重路径，预训练权重需要按照{model_dir}/rank_0/xxx.ckpt格式存放，填写到model_dir为止
load_checkpoint: "/worker/checkpoint/llama-7b/single/"

# 设置auto_trans_ckpt为True
auto_trans_ckpt: True

# 设置数据集
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/worker/dataset/wikitext_2048/"
    shuffle: True

# 8卡分布式配置参考
# default parallel of device num = 8 910A
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
./run_distribute.sh ../rank_table_8.json ../configs/llama/run_llama_7b.yaml [0,8] train
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
# 配置预训练权重路径，预训练权重需要按照{model_dir}/rank_x/xxx.ckpt格式存放
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
./run_distribute.sh ../rank_table_4.json ../configs/llama/run_llama_7b.yaml [0,4] train
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
# 配置预训练权重路径，预训练权重需要按照{model_dir}/rank_x/xxx.ckpt格式存放
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

③ 启动推理

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
# 配置预训练权重路径，预训练权重需要按照{model_dir}/rank_x/xxx.ckpt格式存放
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
./run_distribute.sh rank_table_2.json configs/llama/run_llama_7b.yaml [0,2] predict "I love beijing, because"
```

④ 查看权重转换相关日志

![](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_log1.png)

推理正常：

![llama7b_autotrans_8to2_predict_log2](assets/Transform_Ckpt/llama7b_autotrans_8to2_predict_log2.png)

⑤ 查看转换生成的文件

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
# 配置预训练权重路径，预训练权重需要按照{dir}/rank_0/xxx.ckpt格式存放
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
./run_distribute.sh rank_table_2.json configs/llama/run_llama_13b.yaml [0,2] predict "<human>:你是谁？\n<bot>:"
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

**案例描述**：基于一份完整的llama-13B预训练权重，在Modelarts上使用16卡进行分布式训练。

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

![llama13b_autotrans_1to16_train_modelarts_inputs](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_inputs.png)

![llama13b_autotrans_1to16_train_modelarts_params](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_params.png)

③ 提交训练作业，查看训练日志

![llama13b_autotrans_1to16_train_modelarts_log1](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_log1.png)

![llama13b_autotrans_1to16_train_modelarts_log2](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_log2.png)

④ 查看转换生成的文件

**分布式策略文件**：保存在`remote_save_url/strategy`文件夹下，由于开启了**流水线并行**，会对所有`ckpt_strategy_rank_x.ckpt`进行合并，得到`merged_ckpt_strategy.ckpt`。若不开启流水线并行，则不会合并。

![llama13b_autotrans_1to16_train_modelarts_strategy](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_strategy.png)

**分布式权重**：保存在`output/transformed_checkpoint`文件夹下

![llama13b_autotrans_1to16_train_modelarts_distribute_ckpt](assets/Transform_Ckpt/llama13b_autotrans_1to16_train_modelarts_distribute_ckpt.png)

注：**strategy**和**transformed_checkpoint**两个文件夹请及时保存到**自定义文件夹**中，以免被后续转换任务清空。