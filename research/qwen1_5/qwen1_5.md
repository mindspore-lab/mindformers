# 通义千问

## 模型描述

通义千问是阿里云研发的通义千问大模型系列。Qwen1.5是Qwen2的beta版本, 基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

```text
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```

## 仓库介绍

`Qwen1.5` 基于 `MindFormers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   qwen1_5
     └── qwen1_5_tokenizer.py          # tokenizer
   ```

2. 模型配置：

   ```text
   qwen1_5
     ├── run_qwen1_5_72b.yaml          # 72B 全参微调启动配置
     └── run_qwen1_5_72b_infer.yaml    # 72B 在线推理启动配置
   ```

3. 环境准备和任务启动脚本：

   ```text
   qwen1_5
     ├── alpaca_converter.py           # alpaca数据集格式转换脚本
     ├── qwen1_5_preprocess.py         # 数据集预处理脚本
     ├── convert_weight.py             # 权重转换脚本
     └── run_qwen1_5.py                # Qwen高阶接口脚本
   ```

## 前期准备

### [mindformers安装](path/to/README.md#二mindformers安装)

### 环境要求

- 硬件：Atlas 800T A2
- MindSpore：2.2.12
- MindFormers版本：r1.0
- Python：3.7+

注：

环境搭建参考 [MindSpore官网](https://www.mindspore.cn/install/)，安装MindSpore及CANN社区版配套版本。

### RANK_TABLE_FILE准备

运行`mindformers/tools/hccl_tools.py`，生成`RANK_TABLE_FILE`文件

```shell
# 运行如下命令，生成当前机器的RANK_TABLE_FILE的json文件
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

**注：若使用ModelArts的notebook环境，可从 `/user/config/jobstart_hccl.json` 路径下直接获取rank table，无需手动生成**

RANK_TABLE_FILE 单机8卡参考样例:

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "xx.xx.xx.xx",
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

### 模型权重准备

#### torch权重转mindspore权重

从huggingface下载预训练权重后根据以下步骤进行权重转换，需要下载整个工程，huggingface权重的链接如下：

- [Qwen1.5-72B-Base](https://huggingface.co/Qwen/Qwen1.5-72B/tree/main)

**注**: 请安装`convert_weight.py`依赖包。后续所用的vocab.json和merges.txt文件在此工程中获取。

```shell
pip install torch transformers transformers_stream_generator einops accelerate
# transformers版本不低于4.37.2
```

下载完成后，运行`research/qwen1_5/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python research/qwen1_5/convert_weight.py \
--torch_ckpt_dir <torch_ckpt_dir> \
--mindspore_ckpt_path <mindspore_ckpt_path>
# 参数说明：
# torch_ckpt_dir: 预训练权重文件所在的目录，此参数必须。
# mindspore_ckpt_path: 转换后的输出文件存放路径。可选，如果不给出，默认为`./transform.ckpt`
```

#### mindspore权重转torch权重

在生成mindspore权重之后如需使用torch运行，可根据如下命令转换：

```shell
python convert_reversed.py --mindspore_ckpt_path /path/your.ckpt --torch_ckpt_path /path/your.bin
# 参数说明：
# mindspore_ckpt_path: 待转换的mindspore权重，此参数必须。
# torch_ckpt_path: 转换后的输出文件存放路径，此参数必须。
```

### 数据集准备

目前提供alpaca数据集的预处理脚本用于全参微调任务。

数据集下载链接如下：

- [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

执行`alpaca_converter.py`，将原始数据集转换为指定格式。

``` bash
python qwen1_5/alpaca_converter.py \
--data_path path/alpaca_data.json \
--output_path /path/alpaca-data-messages.json
# 参数说明
# data_path: 存放alpaca数据的路径
# output_path: 输出转换后对话格式的数据路径
```

转换后格式样例：

```text
  {
    "type": "chatml",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Give three tips for staying healthy."
      },
      {
        "role": "assistant",
        "content": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
      }
    ],
    "source": "unknown"
  },
```

执行`qwen1_5_preprocess.py`，进行数据预处理和Mindrecord数据生成。

```bash
python qwen1_5/qwen1_5_preprocess.py \
--input_glob /path/alpaca-data-messages.json \
--vocab_file /path/vocab.json \
--merges_file /path/merges.txt \
--seq_length 2048 \
--output_file /path/alpaca-messages.mindrecord
```

### [模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

从hugging face或官方github仓库转换而来的权重通常是单卡权重，基于该权重进行多卡微调，评测，推理，涉及ckpt从单机策略到分布式策略的切换。

通常训练采用分布式训练，基于该权重进行评测，推理多采用单卡，涉及ckpt从分布式策略到单机策略的切换。

以上涉及到ckpt的单卡，多卡转换，详细教程请参考特性文档[模型权重切分与合并](../../docs/feature_cards/Transform_Ckpt.md)

## 全参微调

### 微调性能

| config | task | Datasets | SeqLength | metric | phase |score | performance(tokens/s/p) |
|-------|-------|-------|-------|-------|-------|-------|-------|
| [qwen1.5-72b](./run_qwen1_5_72b.yaml)| text_generation | alpaca |2048 | - | [finetune](#全参微调) | - | 180.2 |

### 操作步骤

请参照[数据集准备](#数据集准备)章节获取mindrecord格式的alpaca数据集，参照[模型权重准备](#模型权重准备)章节获取权重。

1. 当前支持模型已提供yaml文件，下文以Qwen-72B为例，即使用`run_qwen1_5_72b.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

   当前模型已支持使用**Flash Attention算法**进行全参微调，请参考 [Flash Attention使用文档](../../docs/feature_cards/Training_Algorithms.md#flash-attention)

2. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机8卡的`RANK_TABLE_FILE`文件。

3. 设置如下环境变量：

   ```bash
   export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
   ```

4. 修改`run_qwen1_5_72b.yaml`中相关配置，默认开启自动权重转换，使用完整权重。

   ```yaml
   load_checkpoint: '/path/model_dir' # 使用完整权重，权重按照`model_dir/rank_0/xxx.ckpt`格式存放
   auto_trans_ckpt: True              # 打开自动权重转换
   use_parallel: True
   run_mode: 'finetune'

   model_config:
      seq_length: 8192 # 与数据集长度保持相同

   train_dataset: &train_dataset
     data_loader:
       type: MindDataset
       dataset_dir: "/path/alpaca.mindrecord"  # 配置训练数据集文件夹路径

   # 8卡分布式策略配置
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 4
     micro_batch_num: 48
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

5. 启动微调任务。

4节点32卡启动方式如下。

   ```shell
   # node1
   cd mindformers/research
   bash run_multinode.sh "python qwen1_5/run_qwen1_5.py \
   --config qwen1_5/run_qwen1_5_72b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [0,8] 32

   # node2
   cd mindformers/research
   bash run_multinode.sh "python qwen1_5/run_qwen1_5.py \
   --config qwen1_5/run_qwen1_5_72b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [8,16] 32

   # node3
   cd mindformers/research
   bash run_multinode.sh "python qwen1_5/run_qwen1_5.py \
   --config qwen1_5/run_qwen1_5_72b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [16,24] 32

   # node4
   cd mindformers/research
   bash run_multinode.sh "python qwen1_5/run_qwen1_5.py \
   --config qwen1_5/run_qwen1_5_72b.yaml \
   --load_checkpoint /path/model_dir \
   --use_parallel True \
   --run_mode finetune \
   --auto_trans_ckpt True \
   --train_data /path/alpaca.mindrecord" \
   RANK_TABLE_FILE [24,32] 32

   # 参数说明
   # config: 配置文件路径
   # load_checkpoint: 权重文件夹路径，权重按照'model_dir/rank_0/xxx.ckpt'格式存放
   # auto_trans_ckpt: 自动权重转换开关
   # run_mode: 运行模式，微调时设置为finetune
   # train_data: 训练数据集文件夹路径
   ```

## MindSpore推理

注意事项：

1. 当前支持模型已提供yaml文件，下文以Qwen1_5-72B为例，即使用`run_qwen1_5_72b_infer.yaml`配置文件进行介绍，请根据实际使用模型更改配置文件。

2. 运行下面的代码需要在`research/qwen1_5`目录下，或者先将`research/qwen1_5`目录所在路径加入到`PYTHONPATH`环境变量中。

3. Atlas 800T A2上运行时需要设置如下环境变量，否则推理结果会出现精度问题。

   ```shell
   export MS_GE_TRAIN=0
   export MS_ENABLE_GE=1
   export MS_ENABLE_REF_MODE=1
   ```

### 基于高阶接口推理

- **多卡推理**

1. 主要参数配置参考：

   以单机8卡，模型并行的多卡推理为例，请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机8卡的`RANK_TABLE_FILE`文件。

   ```yaml
   load_checkpoint: '/path/model_dir'       # 使用完整权重，权重存放格式为"model_dir/rank_0/xxx.ckpt"
   auto_trans_ckpt: True                    # 打开自动权重转换
   use_past: True                           # 使用增量推理
   use_parallel: True                       # 使用并行模式

   processor:
     tokenizer:
       vocab_file: "/{path}/vocab.json"     # vocab.json文件路径
       merges_file: "/{path}/merges.txt"    # merges.txt文件路径

   # parallel of device num = 2
   parallel_config:
     data_parallel: 1
     model_parallel: 8
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: True
     gradient_aggregation_group: 4
   ```

   *注*：可配置`model_config:param_init_type`为`float32`提高推理精度，但同时会影响在线推理性能。

2. 启动推理：

   ```shell
   cd mindformers/research
   # 推理命令中参数会覆盖yaml文件中的相同参数
   bash run_singlenode.sh \
   "python qwen1_5/run_qwen1_5.py \
   --config qwen1_5/run_qwen1_5_72b_infer.yaml \
   --run_mode predict \
   --use_parallel True \
   --load_checkpoint /path/model_dir \
   --auto_trans_ckpt True \
   --predict_data 帮助我制定一份去上海的旅游攻略" \
   RANK_TABLE_FILE [0,8] 8

   # 帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息……
   ```

## MindSpore Lite推理

## 多卡导出与推理

### 从完整权重导出mindir

1. 修改`run_qwen1_5_72b_infer.yaml`, 设置并行方式（下面以两卡下的模型并行为例）：

   ```yaml
   load_checkpoint: ''
   src_strategy_path_or_dir: ''

   model:
     model_config:
       seq_length: 8192
       batch_size: 1
       checkpoint_name_or_path: "/path/to/run_qwen1_5_72b_infer.yaml"

   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
   ```

2. RANK_TABLE_FILE准备：请参照[RANK_TABLE_FILE准备](#RANK_TABLE_FILE准备)获取单机4卡的`RANK_TABLE_FILE`文件。

3. 执行多卡导出

```shell
export MF_DIR=/path/to/mindformers/
cd $MF_DIR/research/qwen1_5
rm -rf output/*

PYTHONPATH=$MF_DIR:$PYTHONPATH
bash ../run_singlenode.sh " \
python run_qwen2.py \
--run_mode export \
--config run_qwen2_72b.yaml \
--use_parallel True \
--auto_trans_ckpt True  \
--load_checkpoint /path/to/qwen_1.5_72b.ckpt" \
<RANK_TABLE_FILE> [0,4] 4

sleep 3
tail -f output/log/rank_*/mindformer.log
# 看到 '...Export Over!...' 字样时用ctrl-c退出tail
```

四卡导出时，导出过程生成的文件列表如下：

```text
output
├── strategy/
│   ├── ckpt_strategy_rank_0_rank_0.ckpt
│   ├── ckpt_strategy_rank_1_rank_1.ckpt
│   ├── ckpt_strategy_rank_2_rank_2.ckpt
│   └── ckpt_strategy_rank_3_rank_3.ckpt
├── mindir_full_checkpoint/
│   ├── rank_0_graph.mindir
│   ├── rank_0_variables/
│   │   └── data_0
│   ├── rank_1_graph.mindir
│   ├── rank_1_variables/
│   │   └── data_0
│   ├── rank_2_graph.mindir
│   ├── rank_2_variables/
│   │   └── data_0
│   ├── rank_3_graph.mindir
│   └── rank_3_variables/
│       └── data_0
├── mindir_inc_checkpoint/
│   ├── rank_0_graph.mindir
│   ├── rank_0_variables/
│   │   └── data_0
│   ├── rank_1_graph.mindir
│   ├── rank_1_variables/
│   │   └── data_0
│   ├── rank_2_graph.mindir
│   ├── rank_2_variables/
│   │   └── data_0
│   ├── rank_3_graph.mindir
│   └── rank_3_variables/
│       └── data_0
└── transformed_checkpoint/
    └── qwen_1.5_72b/
        ├── rank_0/
        │   └── checkpoint_0.ckpt
        ├── rank_1/
        │   └── checkpoint_1.ckpt
        ├── rank_2/
        │   └── checkpoint_2.ckpt
        ├── rank_3/
        │   └── checkpoint_3.ckpt
        └── transform_succeed_rank_0.txt
```

后面运行mslite推理时需要`mindir_full_checkpoint`和`mindir_inc_checkpoint`这两个目录，建议将它们移动到其它位置，以避免被无意中其它操作删除或者覆盖；而`output/`目录下的其它目录可以删除。

### 执行Lite推理

1. 准备mslite推理的配置文件`lite.ini`

   ```ini
   [ascend_context]
   provider=ge
   rank_table_file=<RANK_TABLE_FILE>

   [ge_session_options]
   ge.externalWeight=1
   ge.exec.atomicCleanPolicy=1
   ge.event=notify
   ge.exec.staticMemoryPolicy=2
   ge.exec.formatMode=1
   ge.exec.precision_mode=must_keep_origin_dtype

   ```

   说明：与mslite单卡推理不同的是，我们需要添加`rank_table_file=<RANK_TABLE_FILE>`这行（注意将`<RANK_TABLE_FILE>`替换为实际的`json`文件名）。

2. 执行推理脚本：

   ```shell
   export MF_DIR=/path/to/mindformers-v1.0/
   cd $MF_DIR/research/qwen
   rm -rf output/log/rank_*

   PYTHONPATH=$MF_DIR:PYTHONPATH
   bash ../run_singlenode.sh " \
   python run_qwen1_5_mslite_infer.py \
   --mindir_root_dir output \
   --seq_length 8192 \
   --batch_size 1 \
   --do_sample False \
   --vocab_file /path/vocab.json \
   --merge_file /path/merges.txt \
   --ge_config_path lite.ini \
   --predict_data 帮助我制定一份去上海的旅游攻略 " \
   <RANK_TABLE_FILE> [0,4] 4

   sleep 3
   tail -f output/log/rank_*/mindformer.log
   ```

   注意: `seq_length`与`batch_size`必须与导出时YAML中设置的值相同，否则无法运行成功。
