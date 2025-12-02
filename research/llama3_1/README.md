# Llama 3.1

> ## 🚨 弃用说明
>
> 本文档中的模型已过时，不再进行维护，并将在 *1.8.0* 版本下架。如需使用此模型，建议根据官方文档中的 **[模型库](https://www.mindspore.cn/mindformers/docs/zh-CN/master/introduction/models.html)** 选择合适的版本进行使用。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 模型描述

Llama 3.1，是开源Llama系列的最新产品，目前有三个版本：Llama 3.1-8B，Llama 3.1-70B，Llama 3.1-405B。
Llama 3.1在来自公开可用来源的超过15T的数据上进行了预训练。微调数据包括公开可用的指令数据集，以及超过1000万个人工标注的示例。
模型支持上下文窗口长度128K，并使用了新的分词器，词汇表大小达到128256个，采用了分组查询注意力机制(GQA)。
Llama 3.1模型是类GPT模型，是一个生成式的语言模型，主要是用于预测下一个单词。
目前Mindformers支持Llama 3.1-8B，Llama 3.1-70B，敬请期待Llama 3.1-405B。

## 模型性能

以下模型性能均由Atlas 800T A2硬件环境下测试得出。

| Config                                                 |      Task       | Datasets | SeqLength | Performance  |  Phase  |
|:-------------------------------------------------------|:---------------:|:--------:|:---------:|:------------:|:-------:|
| [llama3_1_8b](llama3_1_8b/predict_llama3_1_8b.yaml)    | text_generation |    -     |   2048    | 591 tokens/s | Predict |
| [llama3_1_70b](llama3_1_70b/predict_llama3_1_70b.yaml) | text_generation |    -     |   4096    | 509 tokens/s | Predict |

以下模型性能均由Atlas 900 A2 PoDc硬件环境下测试得出。

| Config                                                  |      Task       | Datasets | SeqLength |   Performance   |  Phase   |
|:--------------------------------------------------------|:---------------:|:--------:|:---------:|:---------------:|:--------:|
| [llama3_1_8b](llama3_1_8b/finetune_llama3_1_8b.yaml)    | text_generation |  alpaca  |   8192    | 2703 tokens/s/p | Finetune |
| [llama3_1_70b](llama3_1_70b/finetune_llama3_1_70b.yaml) | text_generation |  alpaca  |   8192    | 337 tokens/s/p  | Finetune |

## 模型文件

`Llama 3.1` 基于 `mindformers` 实现，主要涉及的文件有：

1. 模型具体实现：

   ```text
   mindformers/models/llama
       ├── __init__.py
       ├── llama.py                  # 模型实现
       ├── llama_config.py           # 模型配置项
       ├── llama_layer.py            # llama网络层定义
       ├── llama_processor.py        # llama预处理
       └── llama_transformer.py      # transformer层实现
   ```

2. 模型配置：

   ```text
   research/llama3_1
       ├──llama3_1_8b
       │    ├── predict_llama3_1_8b.yaml     # 8B推理配置
       │    └── finetune_llama3_1_8b.yaml    # 8B全量微调启动配置
       └──llama3_1_70b
            ├── predict_llama3_1_70b.yaml    # 70B推理配置
            └── finetune_llama3_1_70b.yaml   # 70B全量微调启动配置
   ```

3. 数据预处理脚本和任务启动脚本：

   ```text
   research/llama3_1
       ├── llama3_1_tokenizer.py      # llama3_1 tokenizer处理脚本
       ├── llama3_1_conversation.py   # 微调数据集处理，将原始alpaca转换为对话形式alpaca
       └── llama3_1_preprocess.py     # llama模型的mindrecord数据处理脚本
   ```

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)
和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 数据集及权重准备

#### 数据集下载

MindFormers提供**alpaca**作为[微调](#微调)数据集。

| 数据集名称   |              适用模型              |   适用阶段   |                                      下载链接                                       |
|:--------|:------------------------------:|:--------:|:-------------------------------------------------------------------------------:|
| alpaca  | llama3_1-8b <br/> llama3_1-70b | Finetune | [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) |

数据预处理中所用的`tokenizer.model`可以参考[模型权重下载](#模型权重下载)进行下载。

- **alpaca 数据预处理**

    1. 执行`mindformers/tools/dataset_preprocess/llama/alpaca_converter.py`，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。

       ```shell
       python alpaca_converter.py \
         --data_path /{path}/alpaca_data.json \
         --output_path /{path}/alpaca-data-conversation.json

       # 参数说明
       data_path:   输入下载的文件路径
       output_path: 输出文件的保存路径
       ```

    2. 执行`research/llama3_1/llama3_1_preprocess.py`，生成Mindrecord数据，将带有prompt模板的数据转换为mindrecord格式。

       ```shell
       # 此工具依赖fschat工具包解析prompt模板, 请提前安装fschat >= 0.2.13 python = 3.9
       python llama3_1_preprocess.py \
         --dataset_type qa \
         --input_glob /{path}/alpaca-data-conversation.json \
         --model_file /{path}/tokenizer.model \
         --seq_length 8192 \
         --output_file /{path}/alpaca-fastchat8192.mindrecord

       # 参数说明
       dataset_type: 预处理数据类型
       input_glob:   转换后的alpaca的文件路径
       model_file:   模型tokenizer.model文件路径
       seq_length:   输出数据的序列长度
       output_file:  输出文件的保存路径
       ```

> 数据处理时候注意bos，eos，pad等特殊`ids`要和配置文件中`model_config`里保持一致。

#### 模型权重下载

MindFormers暂时没有提供权重，用户可以下载HuggingFace官方权重经过[模型权重转换](#模型权重转换)后进行使用。

词表下载链接：[tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

| 模型名称         | MindSpore权重 |                        HuggingFace权重                         |
|:-------------|:-----------:|:------------------------------------------------------------:|
| Llama3_1-8B  |      -      | [Link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)  |
| Llama3_1-70B |      -      | [Link](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) |

> 注: 请自行申请huggingface上llama3_1使用权限，并安装transformers=4.40版本

#### 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 微调

### 全参微调

MindSpore Transformers提供 `Llama3.1-8B` 单机多卡以及 `Llama3.1-70B` 多机多卡的微调示例，过程中使用 `alpaca`
数据集对模型进行微调，数据集可以参考[数据集下载](#数据集下载)获得。

#### 单机训练

以Llama3_1-8b为例，Llama3_1-8B在Atlas 800T A2上训练，支持**单机/多机训练**。

使用`finetune_llama3_1_8b.yaml`进行训练，或修改默认配置文件中的`model_config.seq_length`
，使训练配置与数据集的`seq_length`保持一致。

执行命令启动微调任务，在单机上拉起任务。

```shell
# 单机8卡默认快速启动
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3_1 \
 --config research/llama3_1/llama3_1_8b/finetune_llama3_1_8b.yaml \
 --load_checkpoint model_dir/xxx.ckpt \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune \
 --train_data dataset_dir"

# 参数说明
config:          配置文件路径
load_checkpoint: 权重文件路径
auto_trans_ckpt: 自动权重转换开关
run_mode:        运行模式, 微调时设置为finetune
train_data:      训练数据集路径
```

#### 多机训练

以llama3_1-70b为例，使用`finetune_llama3_1_70b.yaml`配置文件，执行8机64卡微调。需要先对权重进行切分，切分权重可以参见[权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html#%E6%9D%83%E9%87%8D%E5%88%87%E5%88%86%E4%B8%8E%E5%90%88%E5%B9%B6)（如果是共享盘也可以开启自动权重转换，使用完整权重）。

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同，各个参数位置含义参见[使用指南](../../README_CN.md#三使用指南)。

在每台机器上运行以下命令，多机运行命令在每台机器上仅`node_num` 不同，从0开始计数，命令中主节点ip为第0个节点ip。

```shell
# 节点0，设0节点ip为192.168.1.1，作为主节点ip，总共64卡且每个节点8卡
# 节点0、节点1、...节点7 依此修改node_num，比如8机，node_num为0~7。
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/llama3_1 \
 --config research/llama3_1/llama3_1_70b/finetune_llama3_1_70b.yaml \
 --load_checkpoint model_dir/xxx.ckpt \
 --train_data dataset_dir \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" \
 64 8 {主节点ip} 8118 {node_num} output/msrun_log False 300
```

## 推理

MindFormers提供`Llama3_1-8b`的快速推理脚本，脚本主要通过generate高阶接口实现，支持单卡推理。推理输入默认不添加bos字符，如果需要添加可在config中增加add_bos_token选项。

```shell
# 脚本使用
bash scripts/examples/llama3/run_llama3_predict.sh PARALLEL CONFIG_PATH CKPT_PATH DEVICE_NUM

# 参数说明
PARALLEL:    是否使用多卡推理, 'single'表示单卡推理, 'parallel'表示多卡推理
CONFIG_PATH: 模型配置文件路径
CKPT_PATH:   模型权重文件路径
VOCAB_FILE:  词表路径
DEVICE_NUM:  使用卡数, 仅开启多卡推理时生效
```

### 单卡推理

以`Llama3_1-8b`单卡推理为例。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh single \
 research/llama3_1/llama3_1_8b/predict_llama3_1_8b.yaml \
 path/to/llama3_1_8b.ckpt \
 path/to/tokenizer.model
```

### 多卡推理

以`Llama3_1-70b`4卡推理为例。Llama3_1-70b权重较大，建议先进行权重切分，参见[权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html#%E6%9D%83%E9%87%8D%E5%88%87%E5%88%86%E4%B8%8E%E5%90%88%E5%B9%B6)。

```shell
bash scripts/examples/llama3/run_llama3_predict.sh parallel \
 research/llama3_1/llama3_1_70b/predict_llama3_1_70b.yaml \
 path/to/model_dir \
 path/to/tokenizer.model 4
```

## 基于MindIE的服务化推理

MindIE，全称Mind Inference Engine，是华为昇腾针对AI全场景业务的推理加速套件。

MindFormers承载在模型应用层MindIE-LLM中，MindIE-LLM是大语言模型推理框架，提供API支持大模型推理能力。

MindIE安装流程请参考[MindIE服务化部署文档](https://www.mindspore.cn/mindformers/docs/zh-CN/master/guide/deployment.html)。

以下例子默认已完成MindIE安装部署且仅适用于**MindIE RC3版本**，且安装路径均为默认路径`/usr/local/Ascend/`。

### 单卡推理

此例子使用llama3_1-8B模型演示。

#### 修改MindIE启动配置

打开mindie-service中的config.json文件，修改server相关配置。

```bash
vim /usr/local/Ascend/mindie/1.0.RC3/mindie-service/conf/config.json
```

需要关注以下字段的配置

1. `ModelDeployConfig.ModelConfig.backendType`

   该配置为对应的后端类型，必填"ms"。

   ```json
   "backendType": "ms"
   ```

   2. `ModelDeployConfig.ModelConfig.modelWeightPath`

      该配置为模型配置文件目录，放置模型和tokenizer等相关文件。

      以llama3_1-8B为例，`modelWeightPath`的组织结构如下：

      ```reStructuredText
      mf_model
       └── llama3_1_8b
              ├── config.json                             # 模型json配置文件
              ├── tokenizer.model                         # 模型vocab文件，hf上对应模型下载
              ├── predict_llama3_1_8b.yaml                # 模型yaml配置文件
              ├── llama3_1_tokenizer.py                   # 模型tokenizer文件,从mindformers仓中research目录下找到对应模型复制
              └── llama3_1_8b.ckpt                        # 单卡模型权重文件
      ```

      predict_llama3_1_8b.yaml需要关注以下配置：

      ```yaml
      load_checkpoint: '/mf_model/llama3_1_8b/llama3_1_8b.ckpt' # 为存放模型单卡权重文件路径
      use_parallel: False
      model:
        model_config:
          type: LlamaConfig
          auto_map:
            AutoTokenizer: [llama3_1_tokenizer.Llama3Tokenizer, null]
      processor:
        tokenizer:
          vocab_file: "/mf_model/llama3_1_8b/tokenizer.model"  #vocab文件路径
      ```

      模型的config.json文件可以使用`save_pretrained`接口生成，示例如下：

      ```python
      from mindformers import AutoConfig

      model_config = AutoConfig.from_pretrained("/mf_model/llama3_1_8b/predict_llama3_1_8b.yaml ")
      model_config.save_pretrained(save_directory="/mf_model/llama3_1_8b", save_json=True)
      ```

      模型权重下载和转换可参考 [权重格式转换](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.ckpt_to_safetensors.html)。

      准备好模型配置目录后，设置参数`modelWeightPath`为该目录路径。

```json
   "modelWeightPath": "/mf_model/llama3_1_8b"
```

最终修改完后的config.json如下：

```json
{
    "Version": "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindservice.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress": "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrl" : "security/certs/server_crl.pem",
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrl" : "security/certs/management/server_crl.pem",
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "pdInterNodeTLSEnabled": false,
        "pdCommunicationPort": 1121,
        "interNodeTlsCaFile" : "security/grpc/ca/ca.pem",
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrl" : "security/certs/server_crl.pem",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb"
    },

    "BackendConfig": {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled": false,
        "multiNodesInferPort": 1120,
        "interNodeTLSEnabled": true,
        "interNodeTlsCaFile": "security/grpc/ca/ca.pem",
        "interNodeTlsCert": "security/grpc/certs/server.pem",
        "interNodeTlsPk": "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrl" : "security/grpc/certs/server_crl.pem",
        "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
        "ModelDeployConfig":
        {
            "maxSeqLen" : 2560,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType": "Standard",
                    "modelName" : "llama3_1_8b",
                    "modelWeightPath" : "/mf_model/llama3_1_8b",
                    "worldSize" : 1,
                    "cpuMemSize" : 16,
                    "npuMemSize" : 16,
                    "backendType": "ms"
                }
            ]
        },

        "ScheduleConfig":
        {
            "templateType": "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 512,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

> 注：为便于测试，`httpsEnabled`参数设置为`false`，忽略后续https通信相关参数。

#### 启动服务

```bash
cd /usr/local/Ascend/mindie/1.0.RC3/mindie-service
nohup ./bin/mindieservice_daemon > output.log 2>&1 &
tail -f output.log
```

打印如下信息，启动成功。

```json
Daemon start success!
```

#### 请求测试

服务启动成功后，可使用curl命令发送请求验证，样例如下：

```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请介绍一下自己<|im_end|>\n<|im_start|>assistant\n","stream": false}' http://127.0.0.1:1035/generate
```

返回推理结果验证成功：

```json
{"generated_text":"我叫小助手，专门为您服务的。<|im_end|>\n<"}
```
