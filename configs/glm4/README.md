# Glm4

## 模型描述

GLM-4 系列模型是专为智能代理设计的基础模型, 其性能可与OpenAI的GPT系列和DeepSeek的V3/R1系列相媲美, 它还支持非常用户友好的本地部署功能。GLM-4在15T的高质量数据上进行了预训练，其中包括大量的推理型合成数据。这为后续的强化学习扩展奠定了基础。在训练后阶段，采用了人类偏好调整对话场景。此外，还使用拒绝采样和强化学习等技术，增强了模型在指令遵循、工程代码和函数调用方面的性能，从而增强了代理任务所需的原子能力。在工程代码、工件生成、函数调用、基于搜索的问答和报告生成方面取得了良好的效果。  

## 支持规格

|    模型名称    |    规格     |  支持任务  | 模型架构  |                       支持设备                        |        模型级别         |
|:----------:|:---------:|:------:|:-----:|:-------------------------------------------------:|:-------------------:|
|GLM4-32B    | 32B |   推理   | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD | [Validated](#模型级别介绍) |
|GLM4-9B    | 9B   |   推理   | Mcore | Atlas 800T A2/Atlas 800I A2/Atlas 900 A3 SuperPoD |     [Validated](#模型级别介绍)     |

说明：

- 模型架构：`Mcore` 表示新模型架构。
- 模型级别：训练和推理各分为5个级别，分别代表该模型遵循不同的标准上线。

## 版本配套

GLM4 当前支持的版本配套如下。

|           | Mindspore Transformers | MindSpore | CANN | HDK |
|:---------:|:----------------------:|:---------:|:----:|:---:|
| 当前支持的版本 |           在研版本           |    在研版本     |  在研版本  | 在研版本  |

## 使用样例

MindSpore Transformers 支持使用 GLM4 进行推理。各任务的整体使用流程如下：

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
|  GLM-4-32B  | [Hugging Face](https://huggingface.co/zai-org/GLM-4-32B-0414) / [ModelScope](https://modelscope.cn/models/zai-org/GLM-4-32B-0414) |
| GLM-4-9B | [Hugging Face](https://huggingface.co/zai-org/GLM-4-9B-0414) / [ModelScope](https://modelscope.cn/models/zai-org/GLM-4-9B-0414) |    |

### 推理样例

GLM-4模型分为9B和32B两个版本，可根据需求选择对应的模型版本。mindformer可以通过统一脚本实现单卡多卡以及多机的推理。

#### 1. 修改任务配置

MindSpore Transformers 提供了推理任务的配置文件，用户可以根据实际情况修改此配置文件中的权重路径和其他参数。

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
📂GLM4
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
- model_parallel：模型并行，默认值为 1。需根据实际模型规模及硬件资源情况，调整该参数为相应的device_num（即实际使用的卡数）。

不同规格和序列长度的并行配置可参考[并行配置建议](#并行配置建议)。

#### 2. 启动推理任务

使用 `run_mindformer` 统一脚本执行推理任务。

单卡推理可以直接执行run_mindformer.py脚本，多卡推理需要借助scripts/msrun_launcher.sh来启动。

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

msrun_launcher.sh包括run_mindformer.py命令和推理卡数两个参数。

单卡推理：

当使用9B权重推理时，推荐使用默认[配置](https://gitee.com/mindspore/mindformers/blob/master/configs/glm4/predict_glm4.yaml)进行单卡推理，执行以下命令即可启动推理任务：

```shell
python run_mindformer.py \
--config configs/glm4/predict_glm4.yaml \
--run_mode predict \
--use_parallel False \
--pretrained_model_dir '/path/hf_dir' \
--parallel_config.data_parallel 1 \
--parallel_config.model_parallel 1 \
--predict_data '请介绍一下北京'
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 `text_generation_result.txt` 文件中。

```text
'text_generation_text': [好的，为您介绍一下北京：北京，简称“京”，是中国的首都，也是中国的直辖市之一，同时还兼具...]
```

多卡推理：

Glm4的32B规模模型，只能进行多卡推理，多卡推理的配置需参考下面修改配置：

1. 模型并行model_parallel的配置和使用的卡数需保持一致，下文用例为8卡推理，需将model_parallel设置成8；
2. 当前版本的多卡推理不支持数据并行，需将data_parallel设置为1。

当使用完整权重推理时，需要在yaml中开启在线切分方式加载权重，使用以下命令：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/glm4/predict_glm4.yaml \
 --run_mode predict \
 --use_parallel True \
 --auto_trans_ckpt True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data '请介绍一下北京'" 2
```

出现如下结果，证明推理成功。推理结果也会保存到当前目录下的 text_generation_result.txt 文件中。详细日志可通过`./output/msrun_log`目录查看。

```text
'text_generation_text': [好的，为您介绍一下北京：北京，简称“京”，是中国的首都，也是中国的直辖市之一，同时还兼具...]
```

多卡多batch推理：

多卡多batch推理的启动方式可参考上述[多卡推理](#多卡推理)，但是需要增加`predict_batch_size`的入参，并修改`predict_data`的入参。

`input_predict_data.txt`文件的内容和格式是每一行都是一个输入，问题的个数与`predict_batch_size`一致，可以参考以下格式：

```text
请介绍一下北京
请介绍一下北京
请介绍一下北京
请介绍一下北京
```

以完整权重推理为例，可以参考以下命令启动推理任务：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/glm4/predict_glm4.yaml \
 --run_mode predict \
 --predict_batch_size 4 \
 --use_parallel True \
 --pretrained_model_dir '/path/hf_dir' \
 --parallel_config.data_parallel 1 \
 --parallel_config.model_parallel 2 \
 --predict_data path/to/input_predict_data.txt" 2
```

推理结果查看方式，与多卡推理相同。

## 附录

### 模型文件说明

Glm4的模型文件包括以下内容：

```text
📦glm4
├── 📄__init__.py                   # glm4模块初始化文件
├── 📄configuration_glm4.py         # glm4模型配置类定义
├── 📄modeling_glm4.py              # glm4模型主体实现
├── 📄modeling_glm4_infer.py        # glm4推理模型实现
└── 📄utils.py                      # glm4工具函数和基础类
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
    <td>GLM4-32B</td>
    <td>32B</td>
    <td>1 × Atlas 800T A2 (2P)</td>
    <td>2</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: 1
  model_parallel: 2</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
  <tr>
    <td>GLM4-9B</td>
    <td>9B</td>
    <td>1 × Atlas 800T A2 (1P)</td>
    <td>1</td>
    <td>
      <pre><code class="language-yaml">parallel_config:
  data_parallel: 1
  model_parallel: 1</code></pre>
    </td>
    <td>
      <pre><code class="language-yaml">context:
  ...
  max_device_memory: "59GB"</code></pre>
    </td>
    <td> Validated </td>
  </tr>
</table>

### 模型级别介绍

#### 推理

- `Released`（发布级）：通过测试团队验收，评测精度与标杆满足对齐标准；
- `Validated`（验证级）：通过开发团队自验证，评测精度与标杆满足对齐标准；
- `Preliminary`（初步级）：通过开发者初步自验证，功能完整可试用，推理输出符合逻辑但精度未严格验证；
- `Untested`（未测试级）：功能可用但未经系统测试，精度未验证，支持用户自定义开发使能；
- `Community`（社区级）：社区贡献的 MindSpore 原生模型，由社区开发维护。

### FAQ

Q1：如果修改了配置中的参数，使用`run_mindformer.py`拉起任务时，还需要重新传参吗？

A1：根据指导修改配置后，参数值已被修改，无需重复传参，`run_mindformer.py`会自动读取解析配置中的参数；如果没有修改配置中的参数，则需要在命令中添加参数。

Q2：用户使用同一个服务器拉起多个推理任务时，端口号冲突怎么办？

A2：用户使用同一个服务器拉起多个推理任务时，要注意不能使用相同的端口号，建议将端口号从50000~65536中选取，避免端口号冲突的情况发生。