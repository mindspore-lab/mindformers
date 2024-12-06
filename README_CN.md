# 欢迎来到MindSpore Transformers（MindFormers）

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 一、介绍

MindSpore Transformers套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

MindSpore Transformers套件基于MindSpore内置的并行技术和组件化设计，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换；
- 提供灵活易用的个性化并行配置；
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略；
- 一键启动任意任务的单卡/多卡训练、微调、评估、推理流程；
- 支持用户进行组件化配置任意模块，如优化器、学习策略、网络组装等；
- 提供Trainer、pipeline、AutoClass等高阶易用性接口；
- 提供预置SOTA权重自动下载及加载功能；
- 支持人工智能计算中心无缝迁移部署；

欲获取MindFormers相关使用教程以及API文档，请参阅[**MindFormers文档**](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/index.html)，以下提供部分内容的快速跳转链接：

- 📝 [快速启动](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/quick_start/source_code_start.html)
- 📝 [大模型预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/usage/pre_training.html)
- 📝 [大模型低参微调](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/usage/parameter_efficient_fine_tune.html)
- 📝 [MindIE服务化部署](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/usage/mindie_deployment.html)

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

### 支持模型

当前MindFormers支持的模型列表如下：

| 模型名                                                                                                    | 支持规格                          | 模型类型         |
|--------------------------------------------------------------------------------------------------------|-------------------------------|--------------|
| [Llama2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/llama2.md)               | 7B/13B/70B                    | 稠密LLM        |
| [Llama3](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/llama3)                          | 8B/70B                        | 稠密LLM        |
| [Llama3.1](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/llama3_1)                      | 8B/70B                        | 稠密LLM        |
| [Qwen](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen)                              | 7B/14B                        | 稠密LLM        |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen1_5)                        | 7B/14B/72B                    | 稠密LLM        |
| [Qwen2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen2)                            | 0.5B/1.5B/7B/57B/57B-A14B/72B | 稠密/稀疏MoE LLM |
| [Qwen-VL](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwenvl)                         | 9.6B                          | 多模态          |
| [GLM2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm2.md)                   | 6B                            | 稠密LLM        |
| [GLM3](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm3.md)                   | 6B                            | 稠密LLM        |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/glm32k)                        | 6B                            | 稠密LLM        |
| [GLM4](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm4.md)                   | 9B                            | 稠密LLM        |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_video.md) | 13B                           | 多模态          |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_image.md) | 19B                           | 多模态          |
| [InternLM](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md)          | 7B/20B                        | 稠密LLM        |
| [InternLM2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2)                    | 7B/20B                        | 稠密LLM        |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek)                | 33B                           | 稠密LLM        |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5)        | 7B                            | 稠密LLM        |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2)                  | 236B                          | 稀疏MoE LLM    |
| [CodeLlama](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/codellama.md)         | 34B                           | 稠密LLM        |
| [Mixtral](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral)                        | 8x7B                          | 稀疏MoE LLM    |
| [Baichuan2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2)                    | 7B/13B                        | 稠密LLM        |
| [Yi](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi)                                  | 6B/34B                        | 稠密LLM        |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md)                   | 13B                           | 稠密LLM        |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/whisper.md)             | 1.5B                          | 多模态          |

## 二、安装

### 版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.10。

| MindFormers | MindPet |                 MindSpore                  |                                                     CANN                                                     |                                   驱动固件                                   |                                 镜像链接                                 |
|:-----------:|:-------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|    1.3.0    |  1.0.4  | [2.4.0](https://www.mindspore.cn/install/) | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) | [24.1.RC3](https://www.hiascend.com/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |

当前MindFormers建议使用如上的软件配套关系。其中CANN和固件驱动的安装需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本。

### 源码编译安装

MindFormers目前支持源码编译安装，用户可以执行如下命令进行安装。

```shell
git clone -b r1.3.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、使用指南

MindFormers支持模型启动预训练、微调、推理、评测等功能，可点击[支持模型](#支持模型)中模型名称查看文档完成上述任务，以下为模型分布式启动方式的说明与示例。

MindFormers推荐使用分布式方式拉起模型训练、推理等功能，目前提供`scripts/msrun_launcher.sh`分布式启动脚本作为模型的主要启动方式，`msrun`特性说明可以参考[msrun启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)。
该脚本主要输入参数说明如下：

  | **参数**           | **单机是否必选** | **多机是否必选** |     **默认值**      | **说明**           |
  |------------------|:----------:|:----------:|:----------------:|------------------|
  | WORKER_NUM       |  &check;   |  &check;   |        8         | 所有节点中使用计算卡的总数    |
  | LOCAL_WORKER     |     -      |  &check;   |        8         | 当前节点中使用计算卡的数量    |
  | MASTER_ADDR      |     -      |  &check;   |    127.0.0.1     | 指定分布式启动主节点的ip    |
  | MASTER_PORT      |     -      |  &check;   |       8118       | 指定分布式启动绑定的端口号    |
  | NODE_RANK        |     -      |  &check;   |        0         | 指定当前节点的rank id   |
  | LOG_DIR          |     -      |  &check;   | output/msrun_log | 日志输出路径，若不存在则递归创建 |
  | JOIN             |     -      |  &check;   |      False       | 是否等待所有分布式进程退出    |
  | CLUSTER_TIME_OUT |     -      |  &check;   |       7200       | 分布式启动的等待时间，单位为秒  |

> 注：如果需要指定`device_id`启动，可以设置环境变量`ASCEND_RT_VISIBLE_DEVICES`，如要配置使用2、3卡则输入`export ASCEND_RT_VISIBLE_DEVICES=2,3`。

### 单机多卡

```shell
# 1. 单机多卡快速启动方式，默认8卡启动
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}"

# 2. 单机多卡快速启动方式，仅设置使用卡数即可
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" WORKER_NUM

# 3. 单机多卡自定义启动方式
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" \
  WORKER_NUM MASTER_PORT LOG_DIR JOIN CLUSTER_TIME_OUT
 ```

- 使用示例

  ```shell
  # 单机多卡快速启动方式，默认8卡启动
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune"

  # 单机多卡快速启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" 8

  # 单机多卡自定义启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" \
    8 8118 output/msrun_log False 300
  ```

### 多机多卡

多机多卡执行脚本进行分布式训练需要分别在不同节点运行脚本，并将参数MASTER_ADDR设置为主节点的ip地址，
所有节点设置的ip地址相同，不同节点之间仅参数NODE_RANK不同。

  ```shell
  # 多机多卡自定义启动方式
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config {CONFIG_PATH} \
   --run_mode {train/finetune/eval/predict}" \
   WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK LOG_DIR JOIN CLUSTER_TIME_OUT
  ```

- 使用示例

  ```shell
  # 节点0，节点ip为192.168.1.1，作为主节点，总共8卡且每个节点4卡
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 0 output/msrun_log False 300

  # 节点1，节点ip为192.168.1.2，节点0与节点1启动命令仅参数NODE_RANK不同
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 1 output/msrun_log False 300
  ```

### 单卡启动

MindFormers提供`run_mindformer.py`脚本作为单卡启动方法，该脚本可以根据模型配置文件，完成支持模型的单卡训练、微调、评估、推理流程。

```shell
# 运行run_mindformer.py的入参会覆盖模型配置文件中的参数
python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
```

## 四、贡献

欢迎参与社区贡献，可参考[MindFormers贡献指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.0/faq/mindformers_contribution.html)。

## 五、许可证

[Apache 2.0许可证](LICENSE)
