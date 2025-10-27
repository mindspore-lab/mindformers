# DeepSeek-V3

## 模型描述

DeepSeek-V3 系列模型是深度求索（DeepSeek）公司推出的一款高性能开源大语言模型，具有强大的自然语言处理能力。该模型在多个领域展现出了卓越的表现，包括代码生成、数学推理、逻辑推理和自然语言理解等。模型总参数 6710 亿，激活参数 370 亿， 其中DeepSeek-R1模型是基于 DeepSeek-V3 Base 模型进一步优化的推理特化模型，通过多阶段强化学习训练，在复杂推理、数学和编程任务上达到国际顶尖水平，同时大幅降低幻觉率。

具体的模型介绍，查看以下论文以及报告：  

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)  
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

## 支持规格

|    模型名称    |    规格     |  支持任务  | 模型架构  |                       支持设备                        |        模型级别         |
|:----------:|:---------:|:------:|:-----:|:-------------------------------------------------:|:-------------------:|
|DeepSeek-V3    | 671B |   推理   | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD | [Released](#模型级别介绍) |
|DeepSeek-R1    | 671B |   推理   | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD | [Released](#模型级别介绍) |

说明：

- 模型架构：`Mcore` 表示新模型架构。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。每个级别的介绍详见[模型级别介绍](https://gitee.com/mindspore/mindformers/blob/r1.7.0/README_CN.md#模型级别介绍)。

## 版本配套

DeepSeek-V3 当前支持的版本配套如下。

|           | Mindspore Transformers | MindSpore | CANN | HDK |
|:---------:|:----------------------:|:---------:|:----:|:---:|
| 当前支持的版本 |           1.7.0           |    2.7.1     |  8.3.RC1  | 25.3.RC1  |

## 使用样例

MindSpore Transformers 支持使用 DeepSeek-V3 进行推理。各任务的整体使用流程如下：

| 任务  | 前期准备                    | 使用流程                       |
|:---:|:------------------------|:---------------------------|
| 推理  |  环境安装 -> 模型下载                       |    修改任务配置 -> 启动推理任务                        |

### 前期准备

#### 环境安装

按照上述版本配套，参考[环境安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/installation.html)安装运行环境。

#### 模型下载

用户可以从Hugging Face等开源社区下载所需的模型文件，包括模型权重、Tokenizer、配置等。链接如下：

|         模型名称         | 下载链接                                             | 说明 |
|:--------------------:|:-------------------------------------------------|:---|
|  DeepSeek-V3  | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3) |
|  DeepSeek-V3-0324  | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324) |
|  DeepSeek-R1  | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1s) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1) |
|  DeepSeek-R1-0528  | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) / [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528) |

### 推理样例

DeepSeek-V3模型总参数量671B，Bfloat16权重参内存占用高达1.4T，最少需要四台Atlas 800T A2。MindSpore Transformers可以通过统一脚本实现单卡多卡以及多机的推理。

#### 1. 修改任务配置

MindSpore Transformers 提供了推理任务的配置文件[predict_deepseek3_671b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/configs/deepseek3/predict_deepseek3_671b.yaml)，用户可以根据实际情况修改此配置文件中的权重路径和其他参数。

当前推理可以直接复用Hugging Face的配置文件和tokenizer，并且在线加载Hugging Face的safetensors格式的权重，使用时配置修改如下：

```yaml
pretrained_model_dir: '/path/hf_dir'
parallel_config:
  data_parallel: 1
  model_parallel: 32
```

参数说明：

- pretrained_model_dir：Hugging Face模型目录路径，放置模型配置、Tokenizer等文件。`/path/hf_dir`中的内容如下：

```text
📂DeepSeek-V3
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

- data_parallel：数据并行，默认值为 1，执行大小EP推理时需要修改此配置。
- model_parallel：模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数）。
- expert_parallel：专家并行，默认值为 1，当执行大小EP推理时需要修改此配置。

> 当执行大小EP推理的时候，data_parallel及model_parallel指定attn及ffn-dense部分的并行策略，expert_parallel指定moe部分路由专家并行策略，data_parallel * model_parallel可被expert_parallel整除。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 本地纯TP推理

使用 `run_mindformer` 统一脚本执行推理任务。

DeepSeek-V3因为参数量只能用多卡推理，多卡推理需要借助scripts/msrun_launcher.sh来启动。

run_mindformer.py的参数说明如下：

| 参数                             | 参数说明                                                      |
|:-------------------------------|:----------------------------------------------------------|
| config                         | yaml配置文件的路径                                               |
| run_mode                       | 运行的模式，推理设置为predict                                        |
| use_parallel                   | 是否使用多卡推理                                                  |
| predict_data                   | 推理的输入数据，多batch推理时需要传入输入数据的txt文件路径，包含多行输入                  |
| predict_batch_size             | 多batch推理的batch_size大小                                     |
| pretrained_model_dir           | Hugging Face模型目录路径，放置模型配置、Tokenizer等文件                    |
| parallel_config.data_parallel  | 数据并行，当前推理模式下设置为1                                         |
| parallel_config.model_parallel | 模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数） |
| parallel_config.expert_parallel  | 数据并行，当前推理模式下设置为1                                         |

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

多机多卡推理：

DeepSeek-V3总参数量671B，只能进行多机多卡推理，在每台服务器上执行如下命令。设置`master_ip`为主节点IP地址，即`Rank 0`服务器的IP；`node_rank`为每个节点的序号；`port`为当前进程的端口号（可在50000~65536中选择）。

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 32 \
 --predict_data 请介绍一下北京" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

> 此处样例代码假设主节点为`192.168.1.1`、当前Rank序号为`0`。实际执行时请将`master_ip`设置为实际的主节点IP地址；将`node_rank`设置为当前节点的Rank序号；将`$local_worker`设置为当前节点上拉起的进程数(当前机器使用的卡数)；将`$worker_num`设置为参与任务的进程总数(使用的总卡数)；将`$port`设置为启动任务的端口号；`$parallel_config.model_parallel`需要设置成实际卡数。

推理结果会保存到当前目录下的 text_generation_result.txt 文件中，推理过程中的日志可通过如下命令查看：

```shell
tail -f ./output/msrun_log/worker_0.log
```

#### 3. 本地大EP推理

大EP，指的是路由专家仅仅按EP分组，不做其他切分。DeepSeek-V3总参数量671B，非MoE参数量大致为20B，大EP浮点推理至少为64卡，即四台A3机器或者八台A2机器。相较于纯tp推理，启动命令的入参需要修改并行配置和`predict_data`的入参，并且增加`predict_batch_size`的入参为DP的倍数，具体执行命令如下:

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 64 \
 --parallel_config.model_parallel 1 \
 --parallel_config.expert_parallel 64 \
 --predict_data path/to/input_data.txt \
 --predict_batch_size 64" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

`input_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
请介绍一下北京
请介绍一下北京
请介绍一下北京
......
请介绍一下北京
```

推理结果和过程日志查看同本地纯TP推理。

#### 4. 本地小EP推理

小EP推理，指的是路由专家不仅仅按EP分组，同时专家本身被TP切分，浮点推理至少为32卡，即两台台A3机器或者八台A2机器。相较于纯tp推理，启动命令的入参需要修改并行配置和`predict_data`的入参，并且增加`predict_batch_size`的入参为DP的倍数，具体执行命令如下:

```shell
master_ip=192.168.1.1
node_rank=0
port=50001

bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/deepseek3/predict_deepseek3_671b.yaml \
 --run_mode predict \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 4 \
 --parallel_config.model_parallel 8 \
 --parallel_config.expert_parallel 4 \
 --predict_data path/to/input_data.txt \
 --predict_batch_size 4" $worker_num $local_worker $master_ip $port $node_rank output/msrun_log False 300
```

`input_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
请介绍一下北京
请介绍一下北京
请介绍一下北京
......
请介绍一下北京
```

推理结果和过程日志查看同本地纯TP推理。

#### 3. 启动服务化推理任务

服务化推理支持量化、大小ep等特性，可以查看以下文档：[服务化推理](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/deployment.html)

## 附录

### 模型文件说明

DeepSeek-V3的模型文件包括以下内容：

```text
📦deepseek3
├── 📄__init__.py                   # deepseek3模块初始化文件
├── 📄configuration_deepseek3.py     # deepseek3模型配置类定义
├── 📄modeling_deepseek3.py          # deepseek3模型主体实现
├── 📄modeling_deepseek3_infer.py    # deepseek3推理模型实现
└── 📄utils.py                      # deepseek3工具函数和基础类
```

### 并行配置建议

以下配置为推理场景下，不同模型规格的推荐配置。

> 注意：max_device_memory 在 Atlas 800T A2 和 Atlas 900 A3 SuperPoD 等机器上一般设置≤60GB，在 Atlas 800I A2 上一般设置≤30GB。

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
    <td>DeepSeek-V3/R1</td>
    <td>671B</td>
    <td>4 × Atlas 800T A2 (8P)</td>
    <td>32</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: 1
  model_parallel: 32</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Released </td>
  </tr>
</table>

### FAQ

Q1：如果修改了配置中的参数，使用`run_mindformer.py`拉起任务时，还需要重新传参吗？

A1：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py`会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

Q2：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？

A2：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从50000~65536中选取，避免端口号冲突的情况发生。