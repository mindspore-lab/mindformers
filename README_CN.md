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

欲获取MindSpore Transformers相关使用教程以及API文档，请参阅[**MindSpore Transformers文档**](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)，以下提供部分内容的快速跳转链接：

- 📝 [大模型预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html)
- 📝 [大模型微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/supervised_fine_tuning.html)
- 📝 [大模型评测](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/evaluation.html)
- 📝 [服务化部署](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/deployment.html)

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

### 模型列表

当前MindSpore Transformers全量的模型列表如下：

| 模型名                                                                                                     | 支持规格                                      |    模型类型     | 最新支持版本 |
|:--------------------------------------------------------------------------------------------------------|:------------------------------------------|:-----------:|:------:|
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/blob/r1.6.0/research/deepseek3)                   | 671B                                      |    稀疏LLM    | 1.6.0  |
| [GLM4](https://gitee.com/mindspore/mindformers/blob/r1.6.0/docs/model_cards/glm4.md)                    | 9B                                        |    稠密LLM    | 1.6.0  |
| [Llama3.1](https://gitee.com/mindspore/mindformers/blob/r1.6.0/research/llama3_1)                       | 8B/70B                                    |    稠密LLM    | 1.6.0  |
| [Mixtral](https://gitee.com/mindspore/mindformers/blob/r1.6.0/research/mixtral)                         | 8x7B                                      |    稀疏LLM    | 1.6.0  |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/blob/r1.6.0/research/qwen2_5)                         | 0.5B/1.5B/7B/14B/32B/72B                  |    稠密LLM    | 1.6.0  |
| [TeleChat2](https://gitee.com/mindspore/mindformers/blob/r1.6.0/research/telechat2)                     | 7B/35B/115B                               |    稠密LLM    | 1.6.0  |
| [CodeLlama](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md)          | 34B                                       |    稠密LLM    | 1.5.0  |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md)  | 19B                                       |     MM      | 1.5.0  |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md)  | 13B                                       |     MM      | 1.5.0  |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2)                   | 236B                                      |    稀疏LLM    | 1.5.0  |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5)         | 7B                                        |    稠密LLM    | 1.5.0  |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek)                 | 33B                                       |    稠密LLM    | 1.5.0  |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/glm32k)                         | 6B                                        |    稠密LLM    | 1.5.0  |
| [GLM3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md)                    | 6B                                        |    稠密LLM    | 1.5.0  |
| [InternLM2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/internlm2)                     | 7B/20B                                    |    稠密LLM    | 1.5.0  |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md)            | 3B                                        |    稠密LLM    | 1.5.0  |
| [Llama3.2-Vision](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md)       | 11B                                       |     MM      | 1.5.0  |
| [Llama3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3)                           | 8B/70B                                    |    稠密LLM    | 1.5.0  |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md)                | 7B/13B/70B                                |    稠密LLM    | 1.5.0  |
| [Qwen2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2)                             | 0.5B/1.5B/7B/57B/57B-A14B/72B             |  稠密/稀疏LLM   | 1.5.0  |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5)                         | 7B/14B/72B                                |    稠密LLM    | 1.5.0  |
| [Qwen-VL](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl)                          | 9.6B                                      |     MM      | 1.5.0  |
| [TeleChat](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/telechat)                       | 7B/12B/52B                                |    稠密LLM    | 1.5.0  |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md)              | 1.5B                                      |     MM      | 1.5.0  |
| [Yi](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yi)                                   | 6B/34B                                    |    稠密LLM    | 1.5.0  |
| [YiZhao](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yizhao)                           | 12B                                       |    稠密LLM    | 1.5.0  |
| [Baichuan2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md)        | 7B/13B                                    |    稠密LLM    | 1.3.2  |
| [GLM2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md)                    | 6B                                        |    稠密LLM    | 1.3.2  |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md)                    | 124M/13B                                  |    稠密LLM    | 1.3.2  |
| [InternLM](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md)           | 7B/20B                                    |    稠密LLM    | 1.3.2  |
| [Qwen](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md)                       | 7B/14B                                    |    稠密LLM    | 1.3.2  |
| [CodeGeex2](https://gitee.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md)          | 6B                                        |    稠密LLM    | 1.1.0  |
| [WizardCoder](https://gitee.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md)  | 15B                                       |    稠密LLM    | 1.1.0  |
| [Baichuan](https://gitee.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md)             | 7B/13B                                    |    稠密LLM    |  1.0   |
| [Blip2](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md)                    | 8.1B                                      |     MM      |  1.0   |
| [Bloom](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md)                    | 560M/7.1B/65B/176B                        |    稠密LLM    |  1.0   |
| [Clip](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md)                      | 149M/428M                                 |     MM      |  1.0   |
| [CodeGeex](https://gitee.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md)             | 13B                                       |    稠密LLM    |  1.0   |
| [GLM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md)                        | 6B                                        |    稠密LLM    |  1.0   |
| [iFlytekSpark](https://gitee.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) | 13B                                       |    稠密LLM    |  1.0   |
| [Llama](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md)                    | 7B/13B                                    |    稠密LLM    |  1.0   |
| [MAE](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md)                        | 86M                                       |     MM      |  1.0   |
| [Mengzi3](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md)                | 13B                                       |    稠密LLM    |  1.0   |
| [PanguAlpha](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md)          | 2.6B/13B                                  |    稠密LLM    |  1.0   |
| [SAM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md)                        | 91M/308M/636M                             |     MM      |  1.0   |
| [Skywork](https://gitee.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md)                | 13B                                       |    稠密LLM    |  1.0   |
| [Swin](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md)                      | 88M                                       |     MM      |  1.0   |
| [T5](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md)                          | 14M/60M                                   |    稠密LLM    |  1.0   |
| [VisualGLM](https://gitee.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md)          | 6B                                        |     MM      |  1.0   |
| [Ziya](https://gitee.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md)                         | 13B                                       |    稠密LLM    |  1.0   |
| [Bert](https://gitee.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md)                      | 4M/110M                                   |    稠密LLM    |  0.8   |

模型维护策略跟随最新支持版本的[生命周期及版本配套策略](#四生命周期及版本配套策略)。

## 二、安装

### 版本匹配关系

当前支持的硬件为 Atlas 800T A2、Atlas 800I A2、Atlas 900 A3 SuperPoD。

当前套件建议使用的Python版本为3.11.4。

| MindSpore Transformers | MindSpore |  CANN   | 固件与驱动  |
|:----------------------:|:---------:|:-------:|:------:|
|         1.6.0          | 2.7.0-rc1 | 8.2.RC1 | 25.2.0 |

历史版本配套关系：

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                                      固件与驱动                                                      |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

### 源码编译安装

MindSpore Transformers目前支持源码编译安装，用户可以执行如下命令进行安装。

```shell
git clone -b r1.6.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、使用指南

MindSpore Transformers支持一键启动大模型的分布式[预训练](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/pre_training.html)、[SFT 微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/supervised_fine_tuning.html)、[推理](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/inference.html)任务，可点击[模型列表](#模型列表)中各模型的链接查看对应使用文档。

关于MindSpore Transformers的更多功能说明可参阅[MindSpore Transformers文档](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/index.html)。

## 四、生命周期及版本配套策略

MindSpore Transformers版本有以下五个维护阶段：

|   **状态**    | **期限** | **说明**                         |
|:-----------:|:------:|:-------------------------------|
|     计划      | 1-3 个月 | 规划功能。                          |
|     开发      |  3 个月  | 构建功能。                          |
|     维护      |  6 个月  | 合入所有已解决的问题并发布新版本。              |
|     无维护     | 0-3 个月 | 合入所有已解决的问题，没有专职维护团队，且不计划发布新版本。 |
| 生命周期终止（EOL） |  N/A   | 分支进行封闭，不再接受任何修改。               |

MindSpore Transformers已发布版本维护策略：

| **MindSpore Transformers版本** | **对应标签** | **当前状态** |  **发布时间**  |     **后续状态**     | **EOL日期**  |
|:----------------------------:|:--------:|:--------:|:----------:|:----------------:|:----------:|
|            1.5.0             |  v1.5.0  |    维护    | 2025/04/29 | 预计2025/10/29起无维护 | 2026/01/29 |
|            1.3.2             |  v1.3.2  |    维护    | 2024/12/20 | 预计2025/06/20起无维护 | 2025/09/20 |
|            1.2.0             |  v1.2.0  |  生命周期终止  | 2024/07/12 |        -         | 2025/04/12 |
|            1.1.0             |  v1.1.0  |  生命周期终止  | 2024/04/15 |        -         | 2025/01/15 |

## 五、免责声明

1. `scripts/examples`目录下的内容是作为参考示例提供的，并不构成商业发布产品的一部分，仅供用户参考。如需使用，需要用户自行负责将其转化为适合商业用途的产品，并确保进行安全防护，对于由此产生的安全问题，MindSpore Transformers 不承担安全责任。
2. 关于数据集， MindSpore Transformers 仅提示性地建议可用于训练的数据集， MindSpore Transformers 不提供任何数据集。用户使用任何数据集进行训练，都需确保训练数据的合法性与安全性，并自行承担以下风险：
   1. 数据投毒（Data Poisoning）：恶意篡改的训练数据可能导致模型产生偏见、安全漏洞或错误输出。
   2. 数据合规性：用户应确保数据采集、处理过程符合相关法律法规及隐私保护要求。
3. 如果您不希望您的数据集在 MindSpore Transformers 中被提及，或希望更新 MindSpore Transformers 中关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对 MindSpore Transformers 的理解和贡献。
4. 关于模型权重，用户下载、分发的模型权重需经可信来源验证，MindSpore Transformers 无法保证第三方权重的安全性。权重文件在传输、加载过程中可能被篡改，导致模型产生预期外的输出或安全漏洞。用户应自行承担使用第三方权重的风险，并确保在使用前对权重文件进行安全验证。

## 六、贡献

欢迎参与社区贡献，可参考[MindSpore Transformers贡献指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/contribution/mindformers_contribution.html)。

## 七、许可证

[Apache 2.0许可证](LICENSE)