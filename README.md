# MindSpore Transformers (MindFormers)

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)

## 1. Introduction

The goal of the MindFormers suite is to build a full-process development suite for foundation model training, fine-tuning, evaluation, inference, and deployment. It provides mainstream Transformer-based pre-trained models and SOTA downstream task applications in the industry, covering various parallel features. It is expected to help users easily implement foundation model training and innovative R&D.

Based on MindSpore's built-in parallel technology and component-based design, the MindFormers suite has the following features:

- Seamless switch from single-device to large-scale cluster training with just one line of code
- Flexible and easy-to-use personalized parallel configuration
- Automatic topology awareness, efficiently combining data parallelism and model parallelism strategies
- One-click launch for single-device/multi-device training, fine-tuning, evaluation, and inference for any task
- Support for users to configure any module in a modular way, such as optimizers, learning strategies, and network assembly
- High-level usability APIs such as Trainer, pipeline, and AutoClass.
- Built-in SOTA weight auto-download and loading functionality
- Seamless migration and deployment support for AI computing centers

For details about MindFormers tutorials and API documents, see **[MindFormers Documentation](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/index.html)**. The following are quick jump links to some of the key content:

- [Calling Source Code to Start](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/quick_start/source_code_start.html)
- [Pre-training](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/usage/pre_training.html)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/usage/parameter_efficient_fine_tune.html)
- [MindIE Service Deployment](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/usage/mindie_deployment.html)

If you have any suggestions on MindFormers, contact us through an issue, and we will address it promptly.

### Supported Models

The following table lists models supported by MindFormers.

| Model                                                                                                   | Specifications                         | Model Type        |
|--------------------------------------------------------------------------------------------------------|-------------------------------|--------------|
| [Llama2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/llama2.md)               | 7B/13B/70B                    | Dense LLM       |
| [Llama3](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/llama3)                          | 8B/70B                        | Dense LLM       |
| [Llama3.1](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/llama3_1)                      | 8B/70B                        | Dense LLM       |
| [Qwen](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen)                              | 7B/14B                        | Dense LLM       |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen1_5)                        | 7B/14B/72B                    | Dense LLM       |
| [Qwen2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwen2)                            | 0.5B/1.5B/7B/57B/57B-A14B/72B | Dense/Sparse MoE LLM|
| [Qwen-VL](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/qwenvl)                         | 9.6B                          | Multimodal         |
| [GLM2](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm2.md)                   | 6B                            | Dense LLM       |
| [GLM3](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm3.md)                   | 6B                            | Dense LLM       |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/tree/r1.3.0/research/glm32k)                        | 6B                            | Dense LLM       |
| [GLM4](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/glm4.md)                   | 9B                            | Dense LLM       |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_video.md) | 13B                           | Multimodal         |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/cogvlm2_image.md) | 19B                           | Multimodal         |
| [InternLM](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md)          | 7B/20B                        | Dense LLM       |
| [InternLM2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm2)                    | 7B/20B                        | Dense LLM       |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek)                | 33B                           | Dense LLM       |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek1_5)        | 7B                            | Dense LLM       |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/deepseek2)                  | 236B                          | Sparse MoE LLM   |
| [CodeLlama](https://gitee.com/mindspore/mindformers/tree/r1.3.0/docs/model_cards/codellama.md)         | 34B                           | Dense LLM       |
| [Mixtral](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/mixtral)                        | 8x7B                          | Sparse MoE LLM   |
| [Baichuan2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2)                    | 7B/13B                        | Dense LLM       |
| [Yi](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/yi)                                  | 6B/34B                        | Dense LLM       |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md)                   | 13B                           | Dense LLM       |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/whisper.md)             | 1.5B                          | Multimodal         |

## 2. Installation

### Version Mapping

Currently, the [Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2) training server is supported.

Python 3.10 is recommended for the current suite.

| MindFormers | MindPet |                 MindSpore                  |                                                     CANN                                                     |                                   Driver/Firmware                                  |                                 Image Link                                |
|:-----------:|:-------:|:------------------------------------------:|:------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|    1.3.0    |  1.0.4  | [2.4.0](https://www.mindspore.cn/install/en) | | [24.1.RC3](https://www.hiascend.com/en/hardware/firmware-drivers/community) | [Link](http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html) |

The preceding software mapping is recommended for MindFormers. The CANN and firmware/driver must match the machine in use. You need to identify the machine model and select the version of the corresponding architecture.

### Installation Using the Source Code

Currently, MindFormers can be compiled and installed using the source code. You can run the following commands to install MindFormers:

```shell
git clone -b r1.3.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 3. User Guide

MindFormers supports model pre-training, fine-tuning, inference, and evaluation. You can click a model name in [Supported Models](#supported-models) to view the document and complete the preceding tasks. The following describes the distributed startup mode and provides an example.

It is recommended that MindFormers launch model training and inference in distributed mode. Currently, the `scripts/msrun_launcher.sh` distributed launch script is provided as the main way to launch models. For details about the `msrun` feature, see [msrun Launching](https://www.mindspore.cn/tutorials/experts/en/r2.3.1/parallel/msrun_launcher.html).
The input parameters of the script are described as follows.

  | **Parameter**          | **Required on Single-Node**| **Required on Multi-Node**|     **Default Value**     | **Description**          |
  |------------------|:----------:|:----------:|:----------------:|------------------|
  | WORKER_NUM       |  &check;   |  &check;   |        8         | Total number of compute devices used on all nodes   |
  | LOCAL_WORKER     |     -      |  &check;   |        8         | Number of compute devices used on the current node   |
  | MASTER_ADDR      |     -      |  &check;   |    127.0.0.1     | IP address of the primary node to be started in distributed mode   |
  | MASTER_PORT      |     -      |  &check;   |       8118       | Port number bound for distributed startup   |
  | NODE_RANK        |     -      |  &check;   |        0         | Rank ID of the current node  |
  | LOG_DIR          |     -      |  &check;   | output/msrun_log | Log output path. If the path does not exist, create it recursively.|
  | JOIN             |     -      |  &check;   |      False       | Specifies whether to wait for all distributed processes to exit.   |
  | CLUSTER_TIME_OUT |     -      |  &check;   |       7200       | Waiting time for distributed startup, in seconds. |

> Note: If you need to specify `device_id` for launching, you can set the environment variable `ASCEND_RT_VISIBLE_DEVICES`. For example, to use devices 2 and 3, input `export ASCEND_RT_VISIBLE_DEVICES=2,3`.

### Single-Node Multi-Device

```shell
# 1. Single-node multi-device quick launch mode. Eight devices are launched by default.
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}"

# 2. Single-node multi-device quick launch mode. You only need to set the number of devices to be used.
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" WORKER_NUM

# 3. Single-node multi-device custom launch mode.
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config {CONFIG_PATH} \
  --run_mode {train/finetune/eval/predict}" \
  WORKER_NUM MASTER_PORT LOG_DIR JOIN CLUSTER_TIME_OUT
 ```

- Examples

  ```shell
  # Single-node multi-device quick launch mode. Eight devices are launched by default.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune"

  # Single-node multi-device quick launch mode.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" 8

  # Single-node multi-device custom launch mode.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config path/to/xxx.yaml \
    --run_mode finetune" \
    8 8118 output/msrun_log False 300
  ```

### Multi-Node Multi-Device

To execute the multi-node multi-device script for distributed training, you need to run the script on different nodes and set `MASTER_ADDR` to the IP address of the primary node. The IP address should be the same across all nodes, and only the `NODE_RANK` parameter varies across nodes.

  ```shell
  # Multi-node multi-device custom launch mode.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config {CONFIG_PATH} \
   --run_mode {train/finetune/eval/predict}" \
   WORKER_NUM LOCAL_WORKER MASTER_ADDR MASTER_PORT NODE_RANK LOG_DIR JOIN CLUSTER_TIME_OUT
  ```

- Examples

  ```shell
  # Node 0, with IP address 192.168.1.1, serves as the primary node. There are a total of 8 devices, with 4 devices allocated per node.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 0 output/msrun_log False 300

  # Node 1, with IP address 192.168.1.2, has the same launch command as node 0, with the only difference being the NODE_RANK parameter.
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode {train/finetune/eval/predict}" \
    8 4 192.168.1.1 8118 1 output/msrun_log False 300
  ```

### Single-Device Launch

MindFormers provides the `run_mindformer.py` script as the single-device launch method. This script can be used to complete the single-device training, fine-tuning, evaluation, and inference of a model based on the model configuration file.

```shell
# The input parameters for running run_mindformer.py will override the parameters in the model configuration file.
python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
```

## 4. Contribution

We welcome contributions to the community. For details, see [MindFormers Contribution Guidelines](https://www.mindspore.cn/mindformers/docs/en/r1.3.0/faq/mindformers_contribution.html).

## 5. License

[Apache 2.0 License](LICENSE)
