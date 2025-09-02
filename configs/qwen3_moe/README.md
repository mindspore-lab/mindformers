# Qwen3-MoE

## 模型描述

Qwen3 是 Qwen 系列最新一代的大型语言模型。基于广泛的训练，Qwen3 在推理、指令跟随、代理能力和多语言支持方面实现了突破性进展。Qwen3-MoE 是 Qwen3 的高效混合专家版本，采用稀疏激活架构，在保持强大性能的同时显著降低计算成本。Qwen3-MoE 通过动态路由机制，仅激活部分专家网络，实现更高的推理效率，适用于大规模部署和高并发场景，同时延续了 Qwen3 在多任务处理、多语言理解和复杂推理上的卓越表现。

```text
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report},
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
```

## 支持规格

|    模型名称    |    规格     | 支持任务 | 模型架构  |                       支持设备                        |        模型级别         |
|:----------:|:---------:|:----:|:-----:|:-------------------------------------------------:|:-------------------:|
|Qwen3    | 235B-A22B |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD | [Released](#模型级别介绍) |
|Qwen3    | 30B-A3B   |  推理  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Validated](#模型级别介绍)     |
|Qwen3    | 30B-A3B   | 预训练  | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Validated](#模型级别介绍)     |

说明：

- 模型架构：`Mcore` 表示 1.6.0 发布的新模型架构，`Legacy` 表示原有模型架构。详见[架构说明](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/overview.html)。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://gitee.com/mindspore/mindformers/blob/master/README_CN.md#模型级别介绍)。

## 版本配套

Qwen3 当前支持的版本配套如下。

|           | Mindspore Transformers | MindSpore | CANN | HDK |
|:---------:|:----------------------:|:---------:|:----:|:---:|
| 当前支持的版本 |           在研版本           |    在研版本     |  在研版本  | 在研版本  |

## 使用样例

MindSpore Transformers 支持使用 Qwen3 进行推理。各任务的整体使用流程如下：

| 任务  | 前期准备                    | 使用流程                       |
|:---:|:------------------------|:---------------------------|
| 推理  |  环境安装 -> 模型下载                       |    修改任务配置 -> 启动推理任务                        |

### 前期准备

#### 环境安装

按照上述版本配套，参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从Modelers、Hugging Face、ModelScope等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等（重头预训练不需加载权重）。链接如下：

|         模型名称         | 下载链接                                             | 说明 |
|:--------------------:|:-------------------------------------------------|:---|
|  Qwen/Qwen3-30B-A3B  | [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/main) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)                                                                                                                                                                                          |    |
| Qwen/Qwen3-235B-A22B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-235B-A22B) |    |

#### 数据集下载

MindSpore Transformers 以下面的数据集为例提供了 Qwen3 的预训练流程的使用案例，实际训练时可参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节制作数据集。请在执行任务前提前下载所需数据集。链接如下：

| 任务  |    数据集名称     | 下载链接         | 说明 |
|:---:|:------------:|:-------------|:---|
| 预训练 | WikiText-103 | [Download](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens/wiki.train.tokens) | 用于预训练的大规模文本数据集 |

### 预训练样例

预训练是指在大规模无标注数据上训练模型，使其能够全面捕捉语言的广泛特性。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/pre_training.html)。

#### 1. 数据预处理

MindSpore Transformers 预训练阶段当前已支持[Megatron格式的数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html#megatron%E6%95%B0%E6%8D%AE%E9%9B%86)。用户可以参考[数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/dataset.html)章节，使用 MindSpore 提供的工具将原始数据集转换为 Megatron 格式。

制作Megatron格式数据集，需要经过两个步骤。首先将原始文本数据集转换为jsonl格式数据，然后使用MindSpore Transformers提供的脚本将jsonl格式数据转换为Megatron格式的.bin和.idx文件。

- `wiki.train.tokens` 转为 `jsonl`格式数据

用户需要**自行将`wiki.train.tokens`数据集处理成jsonl格式的文件**。作为参考，文档末尾的[FAQ](#faq)部分提供了一个临时转换方案，用户需要根据实际需求自行开发和验证转换逻辑。

下面是jsonl格式文件的示例：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
...
```

- `jsonl`格式数据 转为 `bin`格式数据

MindSpore Transformers提供了数据预处理脚本`toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py`用于将jsonl格式的原始文本预料转换成.bin或.idx文件。

> 这里需要提前下载[Qwen3-30B](https://huggingface.co/Qwen/Qwen3-30B-A3B)模型的tokenizer文件。

例如：

```shell
python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/wiki103-megatron \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-dir /path/to/Qwen3-30B-A3B # 其他规格的模型可以调整为对应的tokenizer路径
```

> 运行完成后会生成`/path/to/wiki103-megatron_text_document.bin`和`/path/to/wiki103-megatron_text_document.idx`文件。
> 填写数据集路径时需要使用`/path/to/wiki103-megatron_text_document`，不需要带后缀名。

#### 2. 修改任务配置

MindSpore Transformers 提供了预训练任务的配置文件，用户可以根据实际情况修改配置文件。以下是一个示例配置文件片段，用户需要根据自己的数据集路径和其他参数进行相应修改。

- 数据集配置

```yaml
# Dataset configuration
train_dataset: &train_dataset
  data_loader:
    ...
    sizes:
      - 8000  # 数据集的大小，可以根据实际数据集大小进行调整
      ...
    config:
      ...
      data_path:  # 采样比例和Megatron格式数据集路径
        - '1'
        - "/path/to/wiki103-megatron_text_document" # 替换为实际的Megatron格式数据集路径，此处不带后缀名
```

数据集路径需要替换为实际的Megatron格式数据集路径。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 3. 启动预训练任务

通过指定模型路径和配置文件[configs/qwen3/pretrain_qwen3_30b_a3b_4k.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3_moe/pretrain_qwen3_30b_a3b_4k.yaml)以msrun的方式启动[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，进行16卡分布式训练。可以参考如下方式拉起两台Atlas 800T A2（64G）训练。

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001
export MS_DEV_JIT_SYNTAX_LEVEL=0
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3_moe/pretrain_qwen3_moe_30b_a3b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`port`设置为当前进程的端口号。

上述命令执行完毕后，训练任务将在后台执行，过程日志保存在`./output/msrun_log`下，使用以下命令可查看训练状态（由于开启了流水并行，真实loss只显示在最后一个pipeline stage的日志中，其余pipeline stage会显示`loss`为`0`）

```shell
tail -f ./output/msrun_log/worker_15.log
```

训练过程中的权重checkpoint将会保存在`./output/checkpoint`下。

如有关于Qwen3预训练的相关问题，可以在MindSpore Transformers的Gitee仓库中[提交ISSUE](https://gitee.com/mindspore/mindformers/issues/new)以获取支持。

### 推理样例

推理是指在预训练模型的基础上，利用已学习到的语言知识对新的输入数据进行预测或生成。在MindSpore官网提供了详细的[指导](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/inference.html)。

#### 1. 修改任务配置

MindSpore Transformers 提供了推理任务的[配置文件](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3_moe/predict_qwen3_moe.yaml)，用户可以根据实际情况修改此配置文件中的权重路径和其他参数。

当前推理可以直接复用Hugging Face的配置文件和tokenizer，并且在线加载Hugging Face的safetensors格式的权重，使用时配置修改如下：

```yaml
pretrained_model_dir: '/path/hf_dir'
parallel_config:
  data_parallel: 1
  model_parallel: 1
```

参数说明：

- pretrained_model_dir：Hugging Face模型目录路径，放置模型配置、Tokenizer等文件。`/path/hf_dir`中的内容如下：

```text
📂Qwen3-0.6B
├── 📄config.json
├── 📄generation_config.json
├── 📄merges.txt
├── 📄model-xxx.safetensors
├── 📄model-xxx.safetensors
├── 📄model.safetensors.index.json
├── 📄tokenizer.json
├── 📄tokenizer_config.json
└── 📄vocab.json
```

- data_parallel：数据并行，当前推理并不支持此并行策略，默认为1；
- model_parallel：模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_nu（即实际使用的卡数）。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 启动推理任务

使用 `run_mindformer` 统一脚本执行推理任务。

单卡推理可以直接执行[run_mindformer.py](https://gitee.com/mindspore/mindformers/blob/master/run_mindformer.py)脚本，多卡推理需要借助[scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/master/scripts/msrun_launcher.sh)来启动。

run_mindformer.py的参数说明如下：

| 参数                             | 参数说明                                                      |
|:-------------------------------|:----------------------------------------------------------|
| config                         | yaml配置文件的路径                                               |
| run_mode                       | 运行的模式，推理设置为predict                                        |
| use_parallel                   | 是否使用多卡推理                                                  |
| predict_data                   | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入                  |
| predict_batch_size             | 多batch推理的batch_size大小                                     |
| pretrained_model_dir           | Hugging Face模型目录路径，放置模型配置、Tokenizer等文件                    |
| parallel_config.data_parallel  | 数据并行，当前推理们模式下设置为1                                         |
| parallel_config.model_parallel | 模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_nu（即实际使用的卡数） |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

单卡推理：

当使用完整权重推理时，推荐使用默认[配置](https://gitee.com/mindspore/mindformers/blob/master/configs/qwen3_moe/predict_qwen3_moe.yaml)，执行以下命令即可启动推理任务：

```shell
python run_mindformer.py \
--config configs/qwen3_moe/predict_qwen3_moe.yaml \
--run_mode predict \
--use_parallel False \
--pretrained_model_dir '/path/hf_dir' \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 1 \
--predict_data '帮助我制定一份去上海的旅游攻略'
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 `text_generation_result.txt` 文件中。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

多卡推理：

多卡推理的配置要求与单卡存在差异，需参考下面修改配置：

1. 模型并行model_parallel的配置和使用的卡数需保持一致，下文用例为2卡推理，需将model_parallel设置成2；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

当使用完整权重推理时，需要开启在线切分方式加载权重，参考以下命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3_moe/predict_qwen3_moe.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data '帮助我制定一份去上海的旅游攻略'" 2
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 text_generation_result.txt 文件中。详细日志可通过`./output/msrun_log`目录查看。

```text
'text_generation_text': [帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...]
```

多卡多batch推理：

多卡多batch推理的启动方式可参考上述[多卡推理](#多卡推理)，但是需要增加`predict_batch_size`的入参，并修改`predict_data`的入参。

`input_predict_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
帮助我制定一份去上海的旅游攻略
```

以完整权重推理为例，可以参考以下命令启动推理任务：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3_moe/predict_qwen3_moe.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data path/to/input_predict_data.txt" 2
```

推理结果查看方式，与多卡推理相同。

多机多卡推理：

在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/qwen3_moe/predict_qwen3_moe.yaml" \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data 帮助我制定一份去上海的旅游攻略" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`$local_worker`设置为当前节点上拉起的进程数(当前机器使用的卡数)；将`$worker_num`设置为参与任务的进程总数(使用的总卡数)；将`$port`设置为启动任务的端口号。

推理结果查看方式，与多卡推理相同。

## 附录

### 模型文件说明

Qwen3-MoE的模型文件包括以下内容：

```text
📦mindformers
├── 📂mindformers
│   └── 📂models
│       └── 📂qwen3_moe
│           ├── 📄__init__.py                   # Qwen3模块初始化文件
│           ├── 📄configuration_qwen3_moe.py        # Qwen3模型配置类定义
│           ├── 📄modeling_qwen3_moe.py             # Qwen3模型主体实现
│           ├── 📄modeling_qwen3_moe_infer.py       # Qwen3推理模型实现
│           ├── 📄modeling_qwen3_moe_train.py       # Qwen3训练模型实现
│           └── 📄utils.py                      # Qwen3工具函数和基础类
├── 📂configs
│   └── 📂qwen3_moe
│       ├── 📄pretrain_qwen3_30b_a3b_4k.yaml       # Qwen3-30B-A3B 4k 预训练配置
│       ├── 📄predict_qwen3_moe.yaml               # Qwen3推理配置
│       └── 📄parallel_speed_up.json           # 数据集并行通信配置
└── 📄run_mindformer.py                        # 主要执行脚本
```

### 并行配置建议

以下配置为训练或推理场景下，不同模型规格的推荐配置。其中部分配置为经过验证的最佳配置，部分配置为可以运行的配置。用户可根据实际情况选择合适的配置。

> 注意：max_device_memory 在 Atlas 800T A2 和 Atlas 900 A3 SuperPoD 等机器上一般设置≤60GB，在 Atlas 800I A2 上一般设置≤30GB。

- 预训练：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>序列长度</th>
    <th>并行配置</th>
    <th>重计算配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>32B</td>
    <td>2 × Atlas 800T A2 (8P)</td>
    <td>16</td>
    <td>4096</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 4
  pipeline_stage: 4
  micro_batch_num: 4
  vocab_emb_dp: True
  use_seq_parallel: False
  gradient_aggregation_group: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">recompute_config:
  recompute: True
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True
  recompute_slice_activation: True</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "58GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
</table>

- 推理：

<table>
  <tr>
    <th>模型</th>
    <th>规格</th>
    <th>设备</th>
    <th>卡数</th>
    <th>并行配置</th>
    <th>内存配置</th>
    <th>模型级别</th>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>235B-A22B</td>
    <td>2 × Atlas 800T A2 (8P)</td>
    <td>16</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 16</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Released </td>
  </tr>
  <tr>
    <td>Qwen3</td>
    <td>30B-A3B</td>
    <td>1 × Atlas 800T A2 (2P)</td>
    <td>2</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: &dp 1
  model_parallel: 2</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
</table>

### FAQ

Q1：我有两台Atlas 800T A2服务器，如何进行Qwen3的预训练？拉起任务的指令是什么？

A1：根据指导修改配置后，参考如下命令拉起任务：

- 机器1 IP: 192.168.1.1 （作为主节点）

```bash
# 机器1的启动指令
master_ip=192.168.1.1
node_rank=0
port=50001

export MS_DEV_JIT_SYNTAX_LEVEL=0
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/pretrain_qwen3_32b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

- 机器2 IP: 192.168.1.2

```bash
# 机器2的启动指令
master_ip=192.168.1.1
node_rank=1
port=50001

export MS_DEV_JIT_SYNTAX_LEVEL=0
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3_moe/pretrain_qwen3_30b_a3b_4k.yaml \
--auto_trans_ckpt False \
--use_parallel True \
--run_mode train" \
16 8 $master_ip $port $node_rank output/msrun_log False 7200
```

Q2: 数据集准备部分中，应该如何将`wiki.train.tokens` 转为 `jsonl`格式数据？

A2: [社区issue](https://gitee.com/mindspore/mindformers/issues/ICOKGY)中提供了一个临时转换脚本，仅作为参考使用。用户需要根据自己的数据特点和需求，自行开发和验证适合的转换逻辑。

Q3：如果修改了配置中的参数，使用`run_mindformer.py`拉起任务时，还需要重新传参吗？

A3：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py`会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

Q4：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？

A4：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从50000~65536中选取，避免端口号冲突的情况发生。