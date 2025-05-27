# 欢迎来到MindSpore Transformers（MindFormers）

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 一、介绍

MindSpore Transformers套件的目标是构建一个大模型预训练、微调、评测、推理、部署的全流程开发套件，提供业内主流的Transformer类大语言模型（Large Language Models, LLMs）和多模态理解模型（Multimodal Models, MMs）。期望帮助用户轻松地实现大模型全流程开发。

MindSpore Transformers套件基于MindSpore内置的多维混合并行技术和组件化设计，具备如下特点：

- 一键启动模型单卡或多卡预训练、微调、评测、推理、部署流程；
- 提供丰富的多维混合并行能力可供灵活易用地进行个性化配置；
- 大模型训推系统级深度优化，原生支持超大规模集群高效训推，故障快速恢复；
- 支持任务组件配置化开发。任意模块可通过统一配置进行使能，包括模型网络、优化器、学习率策略等；
- 提供训练精度/性能监控指标实时可视化能力等。

欲获取MindSpore Transformers相关使用教程以及API文档，请参阅[**MindFormers文档**](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/index.html)，以下提供部分内容的快速跳转链接：

- 📝 [快速启动](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/quick_start/source_code_start.html)
- 📝 [大模型预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/usage/pre_training.html)
- 📝 [大模型微调](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/usage/sft_tuning.html)
- 📝 [大模型评测](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/usage/evaluation.html)
- 📝 [服务化部署](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/usage/deployment.html)

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

### 模型列表

MindSpore Transformers当前版本支持的模型列表如下：

| 模型名                                                                                                     | 支持规格                          |   模型类型   |
|:--------------------------------------------------------------------------------------------------------|:------------------------------|:--------:|
| [CodeLlama](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md)          | 34B                           |  稠密LLM   |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md)  | 19B                           |    MM    |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md)  | 13B                           |    MM    |
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek3)                   | 671B                          |  稀疏LLM   |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2)                   | 236B                          |  稀疏LLM   |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5)         | 7B                            |  稠密LLM   |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek)                 | 33B                           |  稠密LLM   |
| [GLM4](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm4.md)                    | 9B                            |  稠密LLM   |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/glm32k)                         | 6B                            |  稠密LLM   |
| [GLM3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md)                    | 6B                            |  稠密LLM   |
| [InternLM2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/internlm2)                     | 7B/20B                        |  稠密LLM   |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md)            | 3B                            |  稠密LLM   |
| [Llama3.2-Vision](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md)       | 11B                           |    MM    |
| [Llama3.1](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3_1)                       | 8B/70B                        |  稠密LLM   |
| [Llama3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3)                           | 8B/70B                        |  稠密LLM   |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md)                | 7B/13B/70B                    |  稠密LLM   |
| [Mixtral](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/mixtral)                         | 8x7B                          |  稀疏LLM   |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2_5)                         | 0.5B/1.5B/7B/14B/32B/72B      |  稠密LLM   |
| [Qwen2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2)                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | 稠密/稀疏LLM |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5)                         | 0.5B/1.8B/4B/7B/14B/72B       |  稠密LLM   |
| [Qwen-VL](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl)                          | 9.6B                          |    MM    |
| [TeleChat2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/telechat2)                     | 7B/35B/115B                   |  稠密LLM   |
| [TeleChat](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/telechat)                       | 7B/12B/52B                    |  稠密LLM   |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md)              | 1.5B                          |    MM    |
| [Yi](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yi)                                   | 6B/34B                        |  稠密LLM   |
| [YiZhao](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yizhao)                           | 12B                           |  稠密LLM   |

## 二、安装

### 版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.11.4。

| MindFormers |                   MindSpore                   |                                                      CANN                                                      |                                                      固件与驱动                                                      | 镜像链接 |
|:-----------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:----:|
|    1.5.0    | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/187.html) |

历史版本配套关系：

| MindFormers |                 MindSpore                  |                                                     CANN                                                     |                                                    固件与驱动                                                    |                                 镜像链接                                 |
|:-----------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|    1.3.2    | [2.4.10](https://www.mindspore.cn/install) |  [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)  | [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/168.html) |
|    1.3.0    | [2.4.0](https://www.mindspore.cn/versions) | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) |                  [24.1.RC3](https://www.hiascend.com/hardware/firmware-drivers/community)                   | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |
|    1.2.0    | [2.3.0](https://www.mindspore.cn/versions) | [8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) |                  [24.1.RC2](https://www.hiascend.com/hardware/firmware-drivers/community)                   | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/138.html) |

### 源码编译安装

MindFormers目前支持源码编译安装，用户可以执行如下命令进行安装。

```shell
git clone -b v1.5.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、使用指南

MindFormers支持模型启动预训练、微调、推理、评测等功能，可点击[模型列表](#模型列表)中模型名称查看文档完成上述任务，以下为模型分布式启动方式的说明与示例。

MindFormers推荐使用分布式方式拉起模型训练、推理等功能，目前提供`scripts/msrun_launcher.sh`分布式启动脚本作为模型的主要启动方式，`msrun`特性说明可以参考[msrun启动](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html)。
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

## 四、生命周期及版本配套策略

MindFormers版本有以下五个维护阶段：

| **状态**        | **期限**  | **说明**                                                                  |
|---------------|---------|-------------------------------------------------------------------------|
| 计划            | 1-3 个月  | 规划功能。                                                                   |
| 开发            | 3 个月    | 构建功能。                                                                   |
| 维护            | 6-12 个月 | 合入所有已解决的问题并发布新版本，对于不同版本的MindFormers，实施差异化的维护计划：标准版维护期为6个月，而长期支持版则为12个月。 |
| 无维护           | 0-3 个月  | 合入所有已解决的问题，没有专职维护团队，且不计划发布新版本。                                          |
| 生命周期终止（EOL）   | N/A     | 分支进行封闭，不再接受任何修改。                                                        |

MindFormers已发布版本维护策略：

| **MindFormers版本** | **对应标签** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**           | **EOL日期**  |
|-------------------|----------|----------|----------|------------|--------------------|------------|
| 1.5.0             | v1.5.0   | 常规版本     | 维护       | 2024/04/20 | 预计2025/10/20起无维护   |            |
| 1.3.2             | v1.3.2   | 常规版本     | 维护       | 2024/12/20 | 预计2025/06/20起无维护   |            |
| 1.2.0             | v1.2.0   | 常规版本     | 无维护      | 2024/07/12 | 预计2025/07/12生命周期终止 | 2025/07/12 |
| 1.1.0             | v1.1.0   | 常规版本     | 无维护      | 2024/04/15 | 预计2025/01/15生命周期终止 | 2025/01/15 |

## 五、免责声明

1. `scripts/examples`目录下的内容是作为参考示例提供的，并不构成商业发布产品的一部分，仅供用户参考。如需使用，需要用户自行负责将其转化为适合商业用途的产品，并确保进行安全防护，对于由此产生的安全问题，MindSpore不承担安全责任。
2. 关于数据集， MindSpore Transformers 仅提示性地建议可用于训练的数据集， MindSpore Transformers 不提供任何数据集。如用户使用这些数据集进行训练，请特别注意应遵守对应数据集的License，如因使用数据集而产生侵权纠纷， MindSpore Transformers 不承担任何责任。
3. 如果您不希望您的数据集在 MindSpore Transformers 中被提及，或希望更新 MindSpore Transformers 中关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对 MindSpore Transformers 的理解和贡献。

## 六、贡献

欢迎参与社区贡献，可参考[MindFormers贡献指南](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/faq/mindformers_contribution.html)。

## 七、许可证

[Apache 2.0许可证](LICENSE)