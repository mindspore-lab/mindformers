# 欢迎来到MindSpore Transformers（MindFormers）

## 一、介绍

MindSpore Transformers套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件：
提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

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

- 模型README：[docs/model_cards](https://gitee.com/mindspore/mindformers/tree/dev/docs/model_cards)
- 任务README：[docs/task_cards](https://gitee.com/mindspore/mindformers/tree/dev/docs/task_cards)
- MindPet指导教程：[docs/pet_tuners](https://gitee.com/mindspore/mindformers/blob/dev/docs/feature_cards/Pet_Tuners.md)
- AICC指导教程：[docs/aicc_cards](https://gitee.com/mindspore/mindformers/blob/dev/docs/aicc_cards/README.md)
- 高阶API指导文档: [docs/mindformers_tutorial.md](https://gitee.com/mindspore/mindformers/blob/dev/docs/README.md)
- 在线文档：[mindformers](https://mindformers.readthedocs.io/zh_CN/latest)

目前支持的模型列表如下：

|                     模型                     |                                                                                                         任务（task name）                                                                                                         | 模型（model name）                                                                                                                                                                                            |
| :------------------------------------------: |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       [BERT](docs/model_cards/bert.md)       | masked_language_modeling<br>[text_classification](docs/task_cards/text_classification.md) <br>[token_classification](docs/task_cards/token_classification.md) <br>[question_answering](docs/task_cards/question_answering.md) | bert_base_uncased <br>txtcls_bert_base_uncased<br>txtcls_bert_base_uncased_mnli <br>tokcls_bert_base_chinese<br>tokcls_bert_base_chinese_cluener <br>qa_bert_base_uncased<br>qa_bert_base_chinese_uncased |
|         [T5](docs/model_cards/t5.md)         |                                                                                                          translation                                                                                                          | t5_small                                                                                                                                                                                                  |
|       [GPT2](docs/model_cards/gpt2.md)       |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | gpt2_small <br>gpt2_13b <br>gpt2_52b                                                                                                                                                                      |
| [PanGuAlpha](docs/model_cards/pangualpha.md) |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | pangualpha_2_6_b<br>pangualpha_13b                                                                                                                                                                        |
|        [GLM](docs/model_cards/glm.md)        |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | glm_6b<br>glm_6b_lora                                                                                                                                                                                     |
|       [GLM2](docs/model_cards/glm2.md)       |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | glm2_6b<br>glm2_6b_lora                                                                                                                                                                                   |
|      [LLama](docs/model_cards/llama.md)      |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | llama_7b <br>llama_13b <br>llama_65b <br>llama_7b_lora                                                                                                                                                    |
|     [LLama2](docs/model_cards/llama.md)      |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | llama2_7b <br>llama2_13b                                                                                                                                                                                  |
|      [Bloom](docs/model_cards/bloom.md)      |                                                                                     [text_generation](docs/task_cards/text_generation.md)                                                                                     | bloom_560m<br>bloom_7.1b <br>bloom_65b<br>bloom_176b                                                                                                                                                      |
|        [MAE](docs/model_cards/mae.md)        |                                                                                                     masked_image_modeling                                                                                                     | mae_vit_base_p16                                                                                                                                                                                          |
|        [VIT](docs/model_cards/vit.md)        |                                                                                [image_classification](docs/task_cards/image_classification.md)                                                                                | vit_base_p16                                                                                                                                                                                              |
|       [Swin](docs/model_cards/swin.md)       |                                                                                [image_classification](docs/task_cards/image_classification.md)                                                                                | swin_base_p4w7                                                                                                                                                                                            |
|       [CLIP](docs/model_cards/clip.md)       |                     [contrastive_language_image_pretrain](docs/task_cards/contrastive_language_image_pretrain.md)<br> [zero_shot_image_classification](docs/task_cards/zero_shot_image_classification.md)                     | clip_vit_b_32<br>clip_vit_b_16 <br>clip_vit_l_14<br>clip_vit_l_14@336                                                                                                                                     |
|      [BLIP2](docs/model_cards/blip2.md)      |                    [contrastive_language_image_pretrain](docs/task_cards/contrastive_language_image_pretrain.md)<br> [zero_shot_image_classification](docs/task_cards/zero_shot_image_classification.md)                      | blip2_stage1_vit_g                                                                                                                                                                                        |

目前在research中支持的模型列表如下：

|                     模型                     |                   任务（task name）                   | 模型（model name）              |
| :------------------------------------------: | :---------------------------------------------------: | :------------------------------ |
|  [Baichuan](research/baichuan/baichuan.md)   | [text_generation](docs/task_cards/text_generation.md) | baichuan_7b <br> baichuan_13b   |
| [Baichuan2](research/baichuan2/baichuan2.md) | [text_generation](docs/task_cards/text_generation.md) | baichuan2_7b <br> baichuan2_13b |
|  [Internlm](research/internlm/internlm.md)   | [text_generation](docs/task_cards/text_generation.md) | InternLM-7B                     |
|        [ziya](research/ziya/ziya.md)         | [text_generation](docs/task_cards/text_generation.md) | ziya-13B                        |

## 二、mindformers安装

- 方式1：源码编译安装

支持源码编译安装，用户可以执行下述的命令进行包的安装

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

- 方式2：镜像

docker下载命令

```shell
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_2_0:mindformers_0.6.0dev_20230616_py39_37
```

创建容器

```shell
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers_dev_mindspore_2_0:mindformers_0.6.0dev_20230616_py39_37 \
/bin/bash
```

## 三、版本匹配关系

| 版本对应关系 | MindFormers | MindPet | MindSpore |  Python   |    芯片     |
| :----------: | :---------: | :-----: | :-------: | :-------: | :---------: |
|    版本号    |     dev     |  1.0.0  | 2.0/1.10  | 3.7.5/3.9 | Ascend 910A |

## 四、快速使用

MindFormers套件对外提供两种使用和开发形式，为开发者提供灵活且简洁的使用方式和高阶开发接口。

### 方式一：使用已有脚本启动

用户可以直接clone整个仓库，按照以下步骤即可运行套件中已支持的任意`configs`模型任务配置文件，方便用户快速进行使用和开发：

- 准备工作

    - step1：git clone mindformers

  ```shell
  git clone -b dev https://gitee.com/mindspore/mindformers.git
  cd mindformers
  ```

    - step2:  准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集

    - step3：修改配置文件`configs/{model_name}/run_{model_name}_***.yaml`中数据集路径

    - step4：如果要使用分布式训练，则需提前生成RANK_TABLE_FILE

  ```shell
  # 不包含8本身，生成0~7卡的hccl json文件
  python mindformers/tools/hccl_tools.py --device_num [0,8]
  ```

- 单卡启动：统一接口启动，根据模型 CONFIG 完成任意模型的单卡训练、微调、评估、推理流程

```shell
# 训练启动，run_status支持train、finetuen、eval、predict四个关键字，以分别完成模型训练、评估、推理功能，默认使用配置文件中的run_mode
python run_mindformer.py --config {CONFIG_PATH} --run_mode {train/finetune/eval/predict}
```

- 多卡启动： scripts 脚本启动，根据模型 CONFIG 完成任意模型的单卡/多卡训练、微调、评估、推理流程

```shell
# 8卡分布式运行， DEVICE_RANGE = [0, 8], 不包含8本身
cd scripts
bash run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_MODE
```

- 常用参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的{model_name}/run_*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
RUN_MODE: 为任务运行状态，支持关键字 train\finetune\eval\predict
```

### 方式二：调用API启动

**详细高阶API使用教程请参考：**[MindFormers大模型使用教程](docs/README.md)

- 准备工作

    - step 1：安装mindformers

  具体安装请参考[第二章](https://gitee.com/mindspore/mindformers/blob/dev/README.md#%E4%BA%8Cmindformers%E5%AE%89%E8%A3%85)

    - step2: 准备数据

  准备相应任务的数据集，请参考`docs`目录下各模型的README.md文档准备相应数据集。

- Trainer 快速入门

  用户可以通过以上方式安装mindformers库，然后利用Trainer高阶接口执行模型任务的训练、微调、评估、推理功能。

    - Trainer 训练\微调启动

  用户可使用`Trainer.train`接口完成模型的训练\微调\断点续训。

  ```python
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

  ```text
  结果打印示例(已集成的vit_base_p16模型权重评估分数)：
  Top1 Accuracy=0.8317
  ```

    - Trainer 推理启动

  用户可使用`Trainer.predict`接口完成模型的推理流程。

  ```python
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

  ```text
  结果打印示例(已集成的vit_base_p16模型权重推理结果)：
  {‘label’: 'cat', score: 0.99}
  ```

- pipeline 快速入门

  MindFormers套件为用户提供了已集成模型的pipeline推理接口，方便用户体验大模型推理服务。

    - pipeline 使用

  ```python
  from mindformers import pipeline
  from mindformers.tools.image_tools import load_image

  test_img = load_image("./sunflower.png") # 一朵太阳花图片
  classifier = pipeline("zero_shot_image_classification",
                        model='clip_vit_b_32',
                        candidate_labels=["sunflower", "tree", "dog", "cat", "toy"])
  predict_result = classifier(test_img)
  print(predict_result)
  ```

  ```text
  结果打印示例(已集成的clip_vit_b_32模型权重推理结果)：
   [[{'score': 0.9999547, 'label': 'sunflower'}, {'score': 1.8684346e-05, 'label': 'toy'}, {'score': 1.3045716e-05, 'label': 'dog'}, {'score': 1.129241e-05, 'label': 'tree'}, {'score': 2.1734568e-06, 'label': 'cat'}]]
  ```

- AutoClass 快速入门

  MindFormers套件为用户提供了高阶AutoClass类，包含AutoConfig、AutoModel、AutoProcessor、AutoTokenizer四类，方便开发者进行调用。

    - AutoConfig 获取已支持的任意模型配置

  ```python
  from mindformers import AutoConfig

  # 获取clip_vit_b_32的模型配置
  clip_vit_b_32_config = AutoConfig.from_pretrained('clip_vit_b_32')
  # 获取vit_base_p16的模型配置
  vit_base_p16_config = AutoConfig.from_pretrained('vit_base_p16')
  ```

    - AutoModel 获取已支持的网络模型

  ```python
  from mindformers import AutoModel

  # 利用from_pretrained功能实现模型的实例化（默认加载对应权重）
  clip_vit_b_32_a = AutoModel.from_pretrained('clip_vit_b_32')
  # 利用from_config功能实现模型的实例化（默认加载对应权重）
  clip_vit_b_32_config = AutoConfig.from_pretrained('clip_vit_b_32')
  clip_vit_b_32_b = AutoModel.from_config(clip_vit_b_32_config)
  # 利用save_pretrained功能保存模型对应配置
  clip_vit_b_32_b.save_pretrained('./clip', save_name='clip_vit_b_32')
  ```

    - AutoProcessor 获取已支持的预处理方法

  ```python
  from mindformers import AutoProcessor

  # 通过模型名关键字获取对应模型预处理过程（实例化clip的预处理过程，通常用于Trainer/pipeline推理入参）
  clip_processor_a = AutoProcessor.from_pretrained('clip_vit_b_32')
  # 通过yaml文件获取相应的预处理过程
  clip_processor_b = AutoProcessor.from_pretrained('configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml')
  ```

    - AutoTokenizer 获取已支持的tokenizer方法

  ```python
  from mindformers import AutoTokenizer
  # 通过模型名关键字获取对应模型预处理过程（实例化clip的tokenizer，通常用于Trainer/pipeline推理入参）
  clip_tokenizer = AutoTokenizer.from_pretrained('clip_vit_b_32')
  ```

## 五、贡献

欢迎参与社区贡献，可参考MindSpore贡献要求[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/dev/CONTRIBUTING_CN.md)。

## 六、许可证

[Apache 2.0许可证](LICENSE)