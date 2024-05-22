# 欢迎来到MindSpore Transformers（MindFormers）

[![LICENSE](https://img.shields.io/github/license/mindspore-lab/mindformers.svg?style=flat-square)](https://github.com/mindspore-lab/mindformers/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/mindformers)](https://pepy.tech/project/mindformers)
[![PyPI](https://badge.fury.io/py/mindformers.svg)](https://badge.fury.io/py/mindformers)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindformers.svg)](https://pypi.org/project/mindformers)

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

如果您对MindSpore Transformers有任何建议，请通过issue与我们联系，我们将及时处理。

- 📝 **[MindFormers教程文档](https://mindformers.readthedocs.io/zh_CN/latest)**
- 📝 [大模型能力表一览](https://mindformers.readthedocs.io/zh-cn/latest/docs/model_support_list.html#llm)
- 📝 [MindPet指导教程](docs/feature_cards/Pet_Tuners.md)
- 📝 [AICC指导教程](docs/readthedocs/source_zh_cn/docs/practice/AICC.md)

目前支持的模型列表如下：

|                         模型                         | model name                                                         |
|:--------------------------------------------------:|:-------------------------------------------------------------------|
|        [LLama2](docs/model_cards/llama2.md)        | llama2_7b, llama2_13b, llama2_7b_lora, llama2_13b_lora, llama2_70b |
|        [LLama3](research/llama3/llama3.md)         | llama3_8b                                                          |
|          [GLM2](docs/model_cards/glm2.md)          | glm2_6b, glm2_6b_lora                                              |
|          [GLM3](docs/model_cards/glm3.md)          | glm3_6b, glm3_6b_lora                                              |
|          [GPT2](docs/model_cards/gpt2.md)          | gpt2, gpt2_13b                                                     |
|    [Baichuan2](research/baichuan2/baichuan2.md)    | baichuan2_7b, baichuan2_13b, baichuan2_7b_lora, baichuan2_13b_lora |
|           [Qwen](research/qwen/qwen.md)            | qwen_7b, qwen_14b, qwen_7b_lora, qwen_14b_lora                     |
|       [Qwen1.5](research/qwen1_5/qwen1_5.md)       | qwen1.5-14b, qwen1.5-72b                                           |
|     [CodeGeex2](docs/model_cards/codegeex2.md)     | codegeex2_6b                                                       |
|     [CodeLlama](docs/model_cards/codellama.md)     | codellama_34b                                                      |
|     [DeepSeek](research/deepseek/deepseek.md)      | deepseek-coder-33b-instruct                                        |
|     [Internlm](research/internlm/internlm.md)      | internlm_7b, internlm_20b, internlm_7b_lora                        |
|       [Mixtral](research/mixtral/mixtral.md)       | mixtral-8x7b                                                       |
| [Wizardcoder](research/wizardcoder/wizardcoder.md) | wizardcoder_15b                                                    |
|              [Yi](research/yi/yi.md)               | yi_6b, yi_34b                                                      |

## 二、MindFormers安装

### Linux源码编译方式安装

支持源码编译安装，用户可以执行下述的命令进行包的安装。

```bash
git clone -b r1.1.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、版本匹配关系

当前支持的硬件为[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.9。

| MindFormers | MindPet |                  MindSpore                   |                                                                                                                                          CANN                                                                                                                                          |                                  驱动固件                                  |                                 镜像链接                                  | 备注   |
|:-----------:|:-------:|:--------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|:---------------------------------------------------------------------:|------|
|   r1.1.0    |  1.0.4  | [2.3.0rc2](https://www.mindspore.cn/install) | 8.0.RC1.beta1:<br> [aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1/Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run)<br> [x86_64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC1/Ascend-cann-toolkit_8.0.RC1_linux-x86_64.run) | [driver](https://www.hiascend.com/hardware/firmware-drivers/community) | [image](http://mirrors.cn-central-221.ovaijisuan.com/detail/129.html) | 版本分支 |

**当前MindFormers仅支持如上的软件配套关系**。其中CANN和固件驱动的安装需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本。

## 四、快速使用

MindFormers套件对外提供两种使用和开发形式，为开发者提供灵活且简洁的使用方式和高阶开发接口。

### 方式一：使用[msrun方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/msrun_launcher.html)（仅适用于配套MindSpore2.3以上版本）

用户可以直接clone整个仓库，按照以下步骤即可运行套件中已支持的任意`configs`模型任务配置文件，方便用户快速进行使用和开发：

目前msrun方式启动不支持指定device_id启动，msrun命令会按当前节点所有显卡顺序设置rank_id。

- 参数说明

  | **参数**           | **单机是否必选**  | **多机是否必选** |     **默认值**      | **说明**           |
  |------------------|:-----------:|:----------:|:----------------:|------------------|
  | WORKER_NUM       |      √      |     √      |        8         | 所有节点中使用计算卡的总数    |
  | LOCAL_WORKER     |      ×      |     √      |        8         | 当前节点中使用计算卡的数量    |
  | MASTER_ADDR      |      ×      |     √      |    127.0.0.1     | 指定分布式启动主节点的ip    |
  | MASTER_PORT      |      ×      |     √      |       8118       | 指定分布式启动绑定的端口号    |
  | NODE_RANK        |      ×      |     √      |        0         | 指定当前节点的rank id   |
  | LOG_DIR          |      ×      |     √      | output/msrun_log | 日志输出路径，若不存在则递归创建 |
  | JOIN             |      ×      |     √      |      False       | 是否等待所有分布式进程退出    |
  | CLUSTER_TIME_OUT |      ×      |     √      |       600        | 分布式启动的等待时间，单位为秒  |

#### 单机多卡

  ```shell
  # 单机多卡快速启动方式，默认8卡启动
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config {CONFIG_PATH} \
   --run_mode {train/finetune/eval/predict}"

  # 单机多卡快速启动方式，仅设置使用卡数即可
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config {CONFIG_PATH} \
   --run_mode {train/finetune/eval/predict}" WORKER_NUM

  # 单机多卡自定义启动方式
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

#### 多机多卡

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

#### 单卡启动

通过统一接口启动，根据模型的config配置，完成任意模型的单卡训练、微调、评估、推理流程。

  ```shell
  # 训练启动，run_mode支持train、finetune、eval、predict四个关键字，以分别完成模型训练、评估、推理功能，默认使用配置文件中的run_mode
  python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
  ```

### 方式二：调用API启动

**详细高阶API使用教程请参考：**[MindFormers大模型使用教程](docs/readthedocs/source_zh_cn/docs/practice/Develop_With_Api.md)

- 准备工作

    - step 1：安装mindformers

      具体安装请参考[第二章](https://gitee.com/mindspore/mindformers/blob/dev/README.md#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)。

    - step2: 准备数据

      准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集。

- Trainer 快速入门

  用户可以通过以上方式安装mindformers库，然后利用Trainer高阶接口执行模型任务的训练、微调、评估、推理功能。

  ```python
  # 以gpt2模型为例
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers import Trainer

  # 初始化预训练任务
  trainer = Trainer(task='text_generation',
                    model='gpt2',
                    train_dataset='path/to/train_dataset',
                    eval_dataset='path/to/eval_dataset')
  # 开启预训练
  trainer.train()

  # 开启全量微调
  trainer.finetune()

  # 开启评测
  trainer.evaluate()

  # 开启推理
  predict_result = trainer.predict(input_data="An increasing sequence: one,", do_sample=False, max_length=20)
  print(predict_result)
  # output result is: [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight,']}]

  # Lora微调
  trainer = Trainer(task="text_generation", model="gpt2", pet_method="lora",
                    train_dataset="path/to/train_dataset")
  trainer.finetune(finetune_checkpoint="gpt2")
  ```

- pipeline 快速入门

  MindFormers套件为用户提供了已集成模型的pipeline推理接口，方便用户体验大模型推理服务。

  pipeline使用样例如下：

  ```python
  # 以gpt2 small为例
  import mindspore; mindspore.set_context(mode=0, device_id=0)
  from mindformers.pipeline import pipeline

  pipeline_task = pipeline(task="text_generation", model="gpt2")
  pipeline_result = pipeline_task("An increasing sequence: one,", do_sample=False, max_length=20)
  print(pipeline_result)
  ```

  结果打印示例(已集成的gpt2模型权重推理结果)：

  ```text
  [{'text_generation_text': ['An increasing sequence: one, two, three, four, five, six, seven, eight,']}]
  ```

- AutoClass 快速入门

  MindFormers套件为用户提供了高阶AutoClass类，包含AutoConfig、AutoModel、AutoProcessor、AutoTokenizer四类，方便开发者进行调用。

    - AutoConfig获取已支持的任意模型配置

      ```python
      from mindformers import AutoConfig

      # 获取gpt2的模型配置
      gpt2_config = AutoConfig.from_pretrained('gpt2')
      # 获取vit_base_p16的模型配置
      vit_base_p16_config = AutoConfig.from_pretrained('vit_base_p16')
      ```

    - AutoModel获取已支持的网络模型

      ```python
      from mindformers import AutoModel

      # 利用from_pretrained功能实现模型的实例化（默认加载对应权重）
      gpt2 = AutoModel.from_pretrained('gpt2')
      # 利用from_config功能实现模型的实例化（默认加载对应权重）
      gpt2_config = AutoConfig.from_pretrained('gpt2')
      gpt2 = AutoModel.from_config(gpt2_config)
      # 利用save_pretrained功能保存模型对应配置
      gpt2.save_pretrained('./gpt2', save_name='gpt2')
      ```

    - AutoProcessor获取已支持的预处理方法

      ```python
      from mindformers import AutoProcessor

      # 通过模型名关键字获取对应模型预处理过程（实例化gpt2的预处理过程，通常用于Trainer/pipeline推理入参）
      gpt2_processor_a = AutoProcessor.from_pretrained('gpt2')
      # 通过yaml文件获取相应的预处理过程
      gpt2_processor_b = AutoProcessor.from_pretrained('configs/gpt2/run_gpt2.yaml')
      ```

    - AutoTokenizer获取已支持的tokenizer方法

      ```python
      from mindformers import AutoTokenizer
      # 通过模型名关键字获取对应模型预处理过程（实例化gpt2的tokenizer，通常用于Trainer/pipeline推理入参）
      gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
      ```

## 五、贡献

欢迎参与社区贡献，可参考MindSpore贡献要求[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 六、许可证

[Apache 2.0许可证](LICENSE)
