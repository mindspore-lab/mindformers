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
|          [GLM2](docs/model_cards/glm2.md)          | glm2_6b, glm2_6b_lora                                              |
|     [CodeLlama](docs/model_cards/codellama.md)     | codellama_34b                                                       |
|     [CodeGeex2](docs/model_cards/codegeex2.md)     | codegeex2_6b                                                       |
|         [LLama](docs/model_cards/llama.md)         | llama_7b, llama_13b, llama_7b_lora                                 |
|           [GLM](docs/model_cards/glm.md)           | glm_6b, glm_6b_lora                                                |
|         [Bloom](docs/model_cards/bloom.md)         | bloom_560m, bloom_7.1b                                             |
|          [GPT2](docs/model_cards/gpt2.md)          | gpt2, gpt2_13b                                                     |
|    [PanGuAlpha](docs/model_cards/pangualpha.md)    | pangualpha_2_6_b, pangualpha_13b                                   |
|         [BLIP2](docs/model_cards/blip2.md)         | blip2_stage1_vit_g                                                 |
|          [CLIP](docs/model_cards/clip.md)          | clip_vit_b_32, clip_vit_b_16, clip_vit_l_14, clip_vit_l_14@336     |
|            [T5](docs/model_cards/t5.md)            | t5_small                                                           |
|           [sam](docs/model_cards/sam.md)           | sam_vit_b, sam_vit_l, sam_vit_h                                    |
|           [MAE](docs/model_cards/mae.md)           | mae_vit_base_p16                                                   |
|           [VIT](docs/model_cards/vit.md)           | vit_base_p16                                                       |
|          [Swin](docs/model_cards/swin.md)          | swin_base_p4w7                                                     |
|       [skywork](research/skywork/skywork.md)       | skywork_13b                                                        |
|    [Baichuan2](research/baichuan2/baichuan2.md)    | baichuan2_7b, baichuan2_13b, baichuan2_7b_lora, baichuan2_13b_lora |
|     [Baichuan](research/baichuan/baichuan.md)      | baichuan_7b, baichuan_13b                                          |
|           [Qwen](research/qwen/qwen.md)            | qwen_7b, qwen_14b, qwen_7b_lora, qwen_14b_lora                     |
| [Wizardcoder](research/wizardcoder/wizardcoder.md) | wizardcoder_15b                                                    |
|     [Internlm](research/internlm/internlm.md)      | internlm_7b, internlm_20b, internlm_7b_lora                        |
|           [ziya](research/ziya/ziya.md)            | ziya_13b                                                           |
|    [VisualGLM](research/visualglm/visualglm.md)    | visualglm                                                          |

## 二、mindformers安装

### 方式一：Linux源码编译方式安装

支持源码编译安装，用户可以执行下述的命令进行包的安装。

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

## 三、版本匹配关系

当前支持的硬件为Atlas 800训练服务器与[Atlas 800T A2](https://www.hiascend.com/hardware/ai-server?tag=900A2)训练服务器。

当前套件建议使用的Python版本为3.9。

| MindFormers | MindPet |                 MindSpore                  |                                                                                                                                               CANN                                                                                                                                               |                               驱动固件                               |                               镜像链接                               | 备注                 |
| :---------: | :-----: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------: | :------------------------------------------------------------------: | -------------------- |
|     dev     |  1.0.4  | 2.3版本(尚未发布) | 尚未发布   | 尚未发布 |                                  /                                   | 开发分支(非稳定版本) |

其中CANN，固件驱动的安装需与使用的机器匹配，请注意识别机器型号，选择对应架构的版本

## 四、快速使用

MindFormers套件对外提供两种使用和开发形式，为开发者提供灵活且简洁的使用方式和高阶开发接口。

### 方式一：使用已有脚本启动

用户可以直接clone整个仓库，按照以下步骤即可运行套件中已支持的任意`configs`模型任务配置文件，方便用户快速进行使用和开发：

**一、使用[msrun方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/msrun_launcher.html)（推荐，仅适用于配套MindSpore2.3以上版本）**

目前msrun方式启动不支持指定device_id启动，msrun命令会按当前节点所有显卡顺序设置rank_id。

- 单机多卡

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

- 多机多卡

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

**二、使用rank table或动态组网方式启动**

- 准备工作

    - step1：克隆mindformers仓库。

      ```shell
      git clone -b dev https://gitee.com/mindspore/mindformers.git
      cd mindformers
      ```

    - step2: 准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集。

    - step3：修改配置文件`configs/{model_name}/run_{model_name}_***.yaml`中数据集路径。

    - step4：如果要使用分布式训练，则需提前生成RANK_TABLE_FILE。
    **注意**：不支持在镜像容器中执行该命令，请在容器外执行。

      ```shell
      # 不包含8本身，生成0~7卡的hccl json文件
      python mindformers/tools/hccl_tools.py --device_num [0,8)
      ```

- 单卡启动：统一接口启动，根据模型的config配置，完成任意模型的单卡训练、微调、评估、推理流程。

  ```shell
  # 训练启动，run_mode支持train、finetune、eval、predict四个关键字，以分别完成模型训练、评估、推理功能，默认使用配置文件中的run_mode
  python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
  ```

- 多卡启动：scripts脚本启动，根据模型的config配置，完成任意模型的单卡/多卡训练、微调、评估、推理流程。

    - 使用 [rank table方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/rank_table.html)

      ```shell
      # 8卡分布式运行， DEVICE_RANGE = [0,8), 不包含8本身
      cd scripts
      bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_MODE
      ```

    - 使用[动态组网方式启动](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/dynamic_cluster.html)

      ```shell
      # 8卡分布式运行
      启动前的准备:
      1. 使用hostname命令将每台服务器hostname设置为各自的ip:  hostname [host ip], 如果在docker内需求设置为docker内部ip,同时保证各个服务器之间docker网络互通
      2. 设置环境变量: export SERVER_ID=0; export SERVER_NUM=1; export PER_DEVICE_NUMS=8; export MS_SCHED_HOST=[HOST IP]; export MS_SCHED_PORT=[PORT]
      cd scripts
      # SERVER_ID为当前服务器序号，SERVER_NUM为服务器的总数，PER_DEVICE_NUMS为每台服务器使用的卡数默认值为8，MS_SCHED_HOST为调度节点的ip，MS_SCHED_PORT为通信端口
      bash run_distribute_ps_auto.sh CONFIG_PATH RUN_MODE
      ```

- 常用参数说明

  ```text
  RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
  CONFIG_PATH: 为configs文件夹下面的{model_name}/run_*.yaml配置文件
  DEVICE_ID: 为设备卡，范围为0~7
  DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
  RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict\export
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

    - Trainer 训练/微调启动

      用户可使用`Trainer.train`或者`Trainer.finetune`接口完成模型的训练/微调/断点续训。

      ```python
      import mindspore; mindspore.set_context(mode=0, device_id=0)
      from mindformers import Trainer

      cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                            model='vit_base_p16', # 已支持的模型名
                            train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                            eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式
      # Example 1： 开启训练复现流程
      cls_trainer.train()
      # Example 2： 加载集成的mae权重，开启微调流程
      cls_trainer.finetune(finetune_checkpoint='mae_vit_base_p16')
      # Example 3： 开启断点续训功能
      cls_trainer.train(train_checkpoint=True, resume_training=True)
      ```

    - Trainer 评估启动

      用户可使用`Trainer.evaluate`接口完成模型的评估流程。

      ```python
      import mindspore; mindspore.set_context(mode=0, device_id=0)
      from mindformers import Trainer

      cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                            model='vit_base_p16', # 已支持的模型名
                            eval_dataset="/data/imageNet-1k/val") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式
      # Example 1： 开启评估已集成模型权重的复现流程
      cls_trainer.evaluate()
      # Example 2： 开启评估训练得到的最后一个权重
      cls_trainer.evaluate(eval_checkpoint=True)
      # Example 3： 开启评估指定的模型权重
      cls_trainer.evaluate(eval_checkpoint='./output/checkpoint/rank_0/mindformers.ckpt')
      ```

      结果打印示例(已集成的vit_base_p16模型权重评估分数)：

      ```text
      Top1 Accuracy=0.8317
      ```

    - Trainer推理启动

      用户可使用`Trainer.predict`接口完成模型的推理流程。

      ```python
      import mindspore; mindspore.set_context(mode=0, device_id=0)
      from mindformers import Trainer

      cls_trainer = Trainer(task='image_classification', # 已支持的任务名
                            model='vit_base_p16') # 已支持的模型名
      input_data = './cat.png' # 一张猫的图片
      # Example 1： 指定输入的数据完成模型推理
      predict_result_d = cls_trainer.predict(input_data=input_data)
      # Example 2： 开启推理（自动加载训练得到的最后一个权重）
      predict_result_b = cls_trainer.predict(input_data=input_data, predict_checkpoint=True)
      # Example 3： 加载指定的权重以完成推理
      predict_result_c = cls_trainer.predict(input_data=input_data, predict_checkpoint='./output/checkpoint/rank_0/mindformers.ckpt')
      print(predict_result_d)
      ```

      结果打印示例(已集成的vit_base_p16模型权重推理结果)：

      ```text
      {‘label’: 'cat', score: 0.99}
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
