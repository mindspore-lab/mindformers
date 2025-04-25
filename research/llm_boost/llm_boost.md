# Llm_boost

> ## 🚨 弃用说明
>
> 本文档已过时，不再进行维护，并将在 *1.6.0* 版本下架，其中可能包含过时的信息或已被更新的功能替代。建议参考最新的 **[官方文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/index.html)** ，以获取准确的信息。
>
> 如有任何问题或建议，请通过 **[社区Issue](https://gitee.com/mindspore/mindformers/issues/new)** 提交反馈。感谢您的理解与支持！

## 功能描述

llm_boost为大模型推理加速模块, 支持对接第三方推理框架进行推理

## 支持模型

|   模型    |     硬件      | 推理  |  后端   |
| :-------: | :-----------: | :---: | :-----: |
| Llama2-7b | Atlas 800T A2 | 单卡  | BuildIn |
| Qwen2-7b  | Atlas 800T A2 | 单卡  | BuildIn |

## 环境及数据准备

### 安装环境

MindFormers软硬件配套关系以及安装参考[环境安装指南](../../README_CN.md#源码编译安装)和[版本匹配关系](../../README_CN.md#版本匹配关系)。

### 1. 安装CANN

- 详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)
- 安装顺序：先安装toolkit 再安装kernel

#### 1.1 安装toolkit

- 下载

| cpu     | 包名（其中`${version}`为实际版本）               |
| ------- | ------------------------------------------------ |
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_${version}_linux-x86_64.run  |

- 安装

  ```bash
  # 安装toolkit
  chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
  ./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

#### 1.2 安装kernel

- 下载

| 包名                                       |
| ------------------------------------------ |
| Ascend-cann-kernels-*_${version}_linux.run |

- 根据芯片型号选择对应的安装包

- 安装

  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels-*_${version}_linux.run --install
  ```

#### 1.3 安装加速库

- 下载加速库

  | 包名（其中`${version}`为实际版本）            |
  | --------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |
  | Ascend-cann-nnal_${version}_linux-x86_64.run  |
  | ...                                           |

- 将文件放置在\${working_dir}路径下

- 安装

    ```bash
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run --install --install-path=${working_dir}
    source ${working_dir}/nnal/atb/set_env.sh
    ```

### 2. 安装atb_models

  ```bash
  mkdir atb-models
  cd atb-models
  tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi0.tar.gz
  sed -i '/PYTORCH/s/^/#/' set_env.sh
  source set_env.sh
  ```

## 模型权重下载

用户可以从HuggingFace官方下载预训练权重，经过[模型权重转换](#模型权重转换)后进行使用，`vocab.json`和`merges.txt`文件也在链接中下载。

词表下载链接：[vocab.json](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/vocab.json)和[merges.txt](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/merges.txt)

| 模型名称          |                                     Base权重（建议训练和微调使用）                                     |                  Instruct权重（建议推理使用）                   |
| :---------------- | :----------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------: |
| llama2-7b         | [Link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt) |     [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf)     |
| qwen2-7b-Instruct |                         [Link](https://huggingface.co/Qwen/Qwen2-7B/tree/main)                         | [Link](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main) |

## 模型权重转换

下载完成后，运行`mindformers/convert_weight.py`转换脚本，将huggingface的权重转换为完整的ckpt权重。

```shell
以Llama2-7b为例。
python convert_weight.py --model llama --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```

## 模型权重切分

在分布式推理场景下，常需要将模型权重重新切分以适应目标切分策略，常见场景为：

**场景一**：从完整模型权重切分至分布式权重

通常是已有完整权重，但目标切分策略存在mp切分，此时需要先生成目标strategy，然后参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)，将完整权重转换为目标切分权重。

以`Qwen2-7b`2卡推理为例, 生成目标strategy。

```shell
  cd research/llm_boost
  # 推理命令中参数会覆盖yaml文件中的相同参数
  python run_llm_boost.py \
    --config_path ./predict_qwen2_7b_instruct_llm_boost.yaml \
    --only_save_strategy True \
    --load_checkpoint /path/model_dir \
    --vocab_file /path/vocab.json \
    --merges_file /path/merges.txt \
    --use_parallel True \
    --device_num 2
```

**场景二**：从分布式训练获得的已切分权重转化为另一策略的分布式权重

通常是在分布式训练完成后获取了按训练切分策略进行切分的权重，在推理阶段模型需要转换为另一切分策略；
同样需要生成目标strategy，参考[权重转换文档](../../docs/feature_cards/Transform_Ckpt.md)，与原有切分startegy一同，转换模型切分策略

## 推理

  主要参数配置参考：

  ```shell
  # model config
  model:
    model_config:
      type: LlmBoostConfig
      llm_backend: BuildIn  # llm backend
      boost_model_name: Llama # model name
    arch:
      type: LlmBoostForCausalLM
  ```

  运行下面的代码需要先将`mindformers`目录所在路径加入到`PYTHONPATH`环境变量中。

### 单卡推理

以`Qwen2-7b`单卡推理为例。

```shell
  cd research/llm_boost
  # 推理命令中参数会覆盖yaml文件中的相同参数
  python run_llm_boost.py \
    --predict_data "帮我制定一份去上海的旅游攻略" \
    --config_path ./predict_qwen2_7b_instruct_llm_boost.yaml \
    --load_checkpoint /path/model_dir \
    --vocab_file /path/vocab.json \
    --merges_file /path/merges.txt \
    --use_parallel False \
    --batch_size 4

  # 输出推理结果：帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
```

### 多卡推理

以`Qwen2-7b`多卡推理为例。

1. 主要参数配置参考：

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 4
     pipeline_stage: 1
     micro_batch_num: 1
     vocab_emb_dp: False
     gradient_aggregation_group: 4
   ```

2. 启动多卡推理

```shell
  cd research/llm_boost
  # 推理命令中参数会覆盖yaml文件中的相同参数
  bash ../../scripts/msrun_launcher.sh "run_llm_boost.py \
    --predict_data "帮我制定一份去上海的旅游攻略" \
    --config /path/predict_qwen2_7b_instruct_llm_boost.yaml \
    --load_checkpoint /path/model_dir \
    --vocab_file /path/vocab.json \
    --merges_file /path/merges.txt \
    --use_parallel True \
    --batch_size 4 \
    --device_num 4" 4

  # 输出推理结果：帮助我制定一份去上海的旅游攻略，包括景点、美食、住宿等信息...
```

## 基于MindIE的服务化推理

MindIE，全称Mind Inference Engine，是华为昇腾针对AI全场景业务的推理加速套件。

MindFormers承载在模型应用层MindIE-LLM中，MindIE-LLM是大语言模型推理框架，提供API支持大模型推理能力。

MindIE安装流程请参考[MindIE服务化部署文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/usage/mindie_deployment.html)。

以下例子默认已完成MindIE安装部署且仅适用于**MindIE RC3版本**，且安装路径均为默认路径`/usr/local/Ascend/`。

此例子使用Qwen2-7B模型演示。

### 修改MindIE启动配置

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

      以Qwen2-7B为例，`modelWeightPath`的组织结构如下：

      ```reStructuredText
      mf_model
       └── qwen2_7b
              ├── config.json                              # 模型json配置文件
              ├── vocab.json                               # 模型vocab文件，hf上对应模型下载
              ├── merges.txt                               # 模型merges文件，hf上对应模型下载
              ├── predict_qwen2_7b_instruct_llm_boost.yaml # 模型yaml配置文件
              ├── qwen2_tokenizer.py                       # 模型tokenizer文件,从mindformers仓中research目录下找到对应模型复制
              ├── llm_boost.py                             # llm_boost模型文件，从mindformers仓中research/llm_boost目录下复制
              ├── llm_boost_config.py                      # llm_boost配置定义文件， 从mindformers仓中research/llm_boost目录下复制
              └── qwen2_7b_ckpt_dir                        # 模型的权重文件路径
      ```

      predict_qwen2_7b_instruct_llm_boost.yaml需要关注以下配置：

      ```yaml
      load_checkpoint: '/mf_model/qwen2_7b/qwen2_7b_ckpt_dir' # 为存放模型单卡权重文件路径
      use_parallel: False  # 是否开启多卡并行推理
      parallel_config:
        model_parallel: 1  # 多卡推理配置模型切分，一般与使用卡数一致
      model:
        model_config:
          type: LlmBoostConfig
          llm_backend: BuildIn
          boost_model_name: Qwen
          auto_map:
            AutoModel: llm_boost.LlmBoostForCausalLM
            AutoTokenizer: [qwen2_tokenizer.Qwen2Tokenizer, null]
            AutoConfig: llm_boost_config.LlmBoostConfig
        arch:
          type: LlmBoostForCausalLM
      processor:
        tokenizer:
          vocab_file: "/path/vocab.json"  #vocab文件路径
          merges_file: "/path/merges.txt" #merges文件路径
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
                    "modelName" : "qwen2_7b",
                    "modelWeightPath" : "/mf_model/qwen2_7b",
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
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "帮我制定一份去上海的旅游攻略","stream": false}' http://127.0.0.1:1025/generate
```

返回推理结果验证成功：

```json
{"generated_text":"，包括景点、美食和住宿推荐。\n当然！以下是一个简要的上海旅游攻略：\n\n"}
```
