# MindSpore Transformers (MindFormers)

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 1. Introduction

The goal of the MindSpore Transformers suite is to build a full-process development suite for Large model pre-training, fine-tuning, evaluation, inference, and deployment. It provides mainstream Transformer-based Large Language Models (LLMs) and Multimodal Models (MMs). It is expected to help users easily realize the full process of large model development.

Based on MindSpore's built-in parallel technology and component-based design, the MindSpore Transformers suite has the following features:

- One-click initiation of single or multi card pre-training, fine-tuning, evaluation, inference, and deployment processes for large models;
- Provides rich multi-dimensional hybrid parallel capabilities for flexible and easy-to-use personalized configuration;
- System-level deep optimization on large model training and inference, native support for ultra-large-scale cluster efficient training and inference, rapid fault recovery;
- Support for configurable development of task components. Any module can be enabled by unified configuration, including model network, optimizer, learning rate policy, etc.;
- Provide real-time visualization of training accuracy/performance monitoring indicators.

For details about MindSpore Transformers tutorials and API documents, see **[MindSpore Transformers Documentation](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/index.html)**. The following are quick jump links to some of the key content:

- 📝 [Pre-training](https://www.mindspore.cn/mindformers/docs/en/dev/guide/pre_training.html)
- 📝 [Supervised Fine-Tuning](https://www.mindspore.cn/mindformers/docs/en/dev/guide/supervised_fine_tuning.html)
- 📝 [Evaluation](https://www.mindspore.cn/mindformers/docs/en/dev/feature/evaluation.html)
- 📝 [Service-oriented Deployment](https://www.mindspore.cn/mindformers/docs/en/dev/guide/deployment.html)

If you have any suggestions on MindSpore Transformers, contact us through an issue, and we will address it promptly.

### Models List

The following table lists models supported by MindSpore Transformers.

| Model                                                                                                   | Specifications                |    Model Type    |        Latest Version         |
|:--------------------------------------------------------------------------------------------------------|:------------------------------|:----------------:|:-----------------------------:|
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3)                      | 671B                          |    Sparse LLM    | In-development version, 1.5.0 |
| [GLM4](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm4.md)                       | 9B                            |    Dense LLM     | In-development version, 1.5.0 |
| [Llama3.1](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1)                          | 8B/70B                        |    Dense LLM     | In-development version, 1.5.0 |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5)                            | 0.5B/1.5B/7B/14B/32B/72B      |    Dense LLM     | In-development version, 1.5.0 |
| [TeleChat2](https://gitee.com/mindspore/mindformers/blob/dev/research/telechat2)                        | 7B/35B/115B                   |    Dense LLM     | In-development version, 1.5.0 |
| [CodeLlama](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md)          | 34B                           |    Dense LLM     |             1.5.0             |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md)  | 19B                           |        MM        |             1.5.0             |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md)  | 13B                           |        MM        |             1.5.0             |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2)                   | 236B                          |    Sparse LLM    |             1.5.0             |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5)         | 7B                            |    Dense LLM     |             1.5.0             |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/deepseek)                 | 33B                           |    Dense LLM     |             1.5.0             |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/glm32k)                         | 6B                            |    Dense LLM     |             1.5.0             |
| [GLM3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md)                    | 6B                            |    Dense LLM     |             1.5.0             |
| [InternLM2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/internlm2)                     | 7B/20B                        |    Dense LLM     |             1.5.0             |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md)            | 3B                            |    Dense LLM     |             1.5.0             |
| [Llama3.2-Vision](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md)       | 11B                           |        MM        |             1.5.0             |
| [Llama3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/llama3)                           | 8B/70B                        |    Dense LLM     |             1.5.0             |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md)                | 7B/13B/70B                    |    Dense LLM     |             1.5.0             |
| [Mixtral](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/mixtral)                         | 8x7B                          |    Sparse LLM    |             1.5.0             |
| [Qwen2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2)                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | Dense/Sparse LLM |             1.5.0             |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5)                         | 7B/14B/72B                    |    Dense LLM     |             1.5.0             |
| [Qwen-VL](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl)                          | 9.6B                          |        MM        |             1.5.0             |
| [TeleChat](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/telechat)                       | 7B/12B/52B                    |    Dense LLM     |             1.5.0             |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md)              | 1.5B                          |        MM        |             1.5.0             |
| [Yi](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yi)                                   | 6B/34B                        |    Dense LLM     |             1.5.0             |
| [YiZhao](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/yizhao)                           | 12B                           |    Dense LLM     |             1.5.0             |
| [Baichuan2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md)        | 7B/13B                        |    Dense LLM     |             1.3.2             |
| [GLM2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md)                    | 6B                            |    Dense LLM     |             1.3.2             |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md)                    | 124M/13B                      |    Dense LLM     |             1.3.2             |
| [InternLM](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md)           | 7B/20B                        |    Dense LLM     |             1.3.2             |
| [Qwen](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md)                       | 7B/14B                        |    Dense LLM     |             1.3.2             |
| [CodeGeex2](https://gitee.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md)          | 6B                            |    Dense LLM     |             1.1.0             |
| [WizardCoder](https://gitee.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md)  | 15B                           |    Dense LLM     |             1.1.0             |
| [Baichuan](https://gitee.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md)             | 7B/13B                        |    Dense LLM     |              1.0              |
| [Blip2](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md)                    | 8.1B                          |        MM        |              1.0              |
| [Bloom](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md)                    | 560M/7.1B/65B/176B            |    Dense LLM     |              1.0              |
| [Clip](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md)                      | 149M/428M                     |        MM        |              1.0              |
| [CodeGeex](https://gitee.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md)             | 13B                           |    Dense LLM     |              1.0              |
| [GLM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md)                        | 6B                            |    Dense LLM     |              1.0              |
| [iFlytekSpark](https://gitee.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) | 13B                           |    Dense LLM     |              1.0              |
| [Llama](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md)                    | 7B/13B                        |    Dense LLM     |              1.0              |
| [MAE](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md)                        | 86M                           |        MM        |              1.0              |
| [Mengzi3](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md)                | 13B                           |    Dense LLM     |              1.0              |
| [PanguAlpha](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md)          | 2.6B/13B                      |    Dense LLM     |              1.0              |
| [SAM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md)                        | 91M/308M/636M                 |        MM        |              1.0              |
| [Skywork](https://gitee.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md)                | 13B                           |    Dense LLM     |              1.0              |
| [Swin](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md)                      | 88M                           |        MM        |              1.0              |
| [T5](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md)                          | 14M/60M                       |    Dense LLM     |              1.0              |
| [VisualGLM](https://gitee.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md)          | 6B                            |        MM        |              1.0              |
| [Ziya](https://gitee.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md)                         | 13B                           |    Dense LLM     |              1.0              |
| [Bert](https://gitee.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md)                      | 4M/110M                       |    Dense LLM     |              0.8              |

The model maintenance strategy follows the [Life Cycle And Version Matching Strategy](#4-life-cycle-and-version-matching-strategy) of the corresponding latest supported version.

## 2. Installation

### Version Mapping

Currently, the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server is supported.

Python 3.11.4 is recommended for the current suite.

| MindSpore Transformers |       MindSpore        |          CANN          |    Driver/Firmware     |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|
| In-development version | In-development version | In-development version | In-development version |

Historical Version Supporting Relationships:

| MindSpore Transformers |                   MindSpore                   |                                                      CANN                                                      |                                                 Driver/Firmware                                                 |
|:----------------------:|:---------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|
|         1.5.0          | [2.6.0-rc1](https://www.mindspore.cn/install) | [8.1.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) | [25.0.RC1](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html) |
|         1.3.2          |  [2.4.10](https://www.mindspore.cn/versions)  |   [8.0.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |   [24.1.0](https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html)   |
|         1.3.0          |  [2.4.0](https://www.mindspore.cn/versions)   | [8.0.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) | [24.1.RC3](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html) |
|         1.2.0          |  [2.3.0](https://www.mindspore.cn/versions)   | [8.0.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) | [24.1.RC2](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html) |

### Installation Using the Source Code

Currently, MindSpore Transformers can be compiled and installed using the source code. You can run the following commands to install MindSpore Transformers:

```shell
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 3. User Guide

MindSpore Transformers supports distributed [pre-training](https://www.mindspore.cn/mindformers/docs/en/dev/guide/pre_training.html), [supervised fine-tuning](https://www.mindspore.cn/mindformers/docs/en/dev/guide/supervised_fine_tuning.html), and [inference](https://www.mindspore.cn/mindformers/docs/en/dev/guide/inference.html) tasks for large models with one click. You can click the link of each model in [Model List](#models-list) to see the corresponding documentation.

For more information about the functions of MindSpore Transformers, please refer to [MindSpore Transformers Documentation](https://www.mindspore.cn/mindformers/docs/en/dev/index.html).

## 4. Life Cycle And Version Matching Strategy

MindSpore Transformers version has the following five maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                             |
|-------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------|
| Plan              | 1-3 months   | Planning function.                                                                                                          |
| Develop           | 3 months     | Build function.                                                                                                             |
| Preserve          | 6 months     | Incorporate all solved problems and release new versions.                                                                   |
| No Preserve       | 0—3 months   | Incorporate all the solved problems, there is no full-time maintenance team, and there is no plan to release a new version. |
| End of Life (EOL) | N/A          | The branch is closed and no longer accepts any modifications.                                                               |

MindSpore Transformers released version preservation policy:

| **MindSpore Transformers Version** | **Corresponding Label** | **Current Status** | **Release Time** |        **Subsequent Status**         | **EOL Date** |
|:----------------------------------:|:-----------------------:|:------------------:|:----------------:|:------------------------------------:|:------------:|
|               1.5.0                |         v1.5.0          |      Preserve      |    2025/04/29    | No preserve expected from 2025/10/29 |  2026/01/29  |
|               1.3.2                |         v1.3.2          |      Preserve      |    2024/12/20    | No preserve expected from 2025/06/20 |  2025/09/20  |
|               1.2.0                |         v1.2.0          |    End of Life     |    2024/07/12    |                  -                   |  2025/04/12  |
|               1.1.0                |         v1.1.0          |    End of Life     |    2024/04/15    |                  -                   |  2025/01/15  |

## 5. Disclaimer

1. `scripts/examples` directory are provided as reference examples and do not form part of the commercially released products. They are only for users' reference. If it needs to be used, the user should be responsible for transforming it into a product suitable for commercial use and ensuring security protection. MindSpore Transformers does not assume security responsibility for the resulting security problems.
2. Regarding datasets, MindSpore Transformers only provides suggestions for datasets that can be used for training. MindSpore Transformers does not provide any datasets. Users who use any dataset for training must ensure the legality and security of the training data and assume the following risks:  
   1. Data poisoning: Maliciously tampered training data may cause the model to produce bias, security vulnerabilities, or incorrect outputs.
   2. Data compliance: Users must ensure that data collection and processing comply with relevant laws, regulations, and privacy protection requirements.
3. If you do not want your dataset to be mentioned in MindSpore Transformers, or if you want to update the description of your dataset in MindSpore Transformers, please submit an issue to Gitee, and we will remove or update the description of your dataset according to your issue request. We sincerely appreciate your understanding and contribution to MindSpore Transformers.
4. Regarding model weights, users must verify the authenticity of downloaded and distributed model weights from trusted sources. MindSpore Transformers cannot guarantee the security of third-party weights. Weight files may be tampered with during transmission or loading, leading to unexpected model outputs or security vulnerabilities. Users should assume the risk of using third-party weights and ensure that weight files are verified for security before use.

## 6. Contribution

We welcome contributions to the community. For details, see [MindSpore Transformers Contribution Guidelines](https://www.mindspore.cn/mindformers/docs/en/dev/contribution/mindformers_contribution.html).

## 7. License

[Apache 2.0 License](LICENSE)