# 欢迎来到MindSpore MindFormers

## 介绍

MindSpore MindFormers套件的目标是构建一个大模型训练、推理、部署的全流程开发套件：
提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。 期望帮助用户轻松的实现大模型训练和创新研发。

MindSpore MindFormers套件基于MindSpore内置的并行技术和组件化设计，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换。
- 提供灵活易用的个性化并行配置。
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略。
- 一键启动任意任务的训练、评估、推理流程。
- 支持用户进行组件化配置任意模块，如优化器、学习策略、网络组装等。
- 提供Trainer、ModelClass、ConfigClass、pipeline等高阶易用性接口。

如果您对MindSpore MindFormers有任何建议，请通过Gitee或MindSpore与我们联系，我们将及时处理。

目前支持的模型列表如下：

- BERT
- GPT
- OPT
- T5
- MAE
- SimMIM
- CLIP
- FILIP
- Vit
- Swin

### 安装

目前仅支持源码编译安装，用户可以执行下述的命令进行包的安装

```bash
git clone https://gitee.com/mindspore/mindformers.git
cd mindformers
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
sh build.sh
```

#### 版本匹配关系

|版本对应关系| Mindformer  | MindSpore |
|-----------| -----------| ----------|
|版本号      | 0.2.0      | 1.8.1 |

### 快速使用

目前该库提供两种方式供用户使用，套件详细设计请阅：[MindFormers套件设计](https://gitee.com/mindspore/transformer/wikis/%E7%89%B9%E6%80%A7%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3?sort_id=6569071)

#### 方式一：clone 工程代码

用户可以直接clone整个仓库，按照以下步骤即可运行套件中已支持的任意`configs`模型任务配置文件，方便用户快速进行使用和开发

- 准备工作

    - step1：git clone mindformers

    ```shell
    git clone https://gitee.com/mindspore/mindformers.git
    cd mindformers
    ```

    - step2:  准备相应任务的数据集，请参考`configs`目录下各模型的README.md文档准备相应数据集

    - step3：修改配置文件`configs/{model_name}/task_config/{model_name}_dataset.yaml`中数据集路径

    - step4：如果要使用分布式训练，则需提前生成RANK_TABLE_FILE

    ```shell
    # 不包含8本身，生成0~7卡的hccl json文件
    python mindformers/tools/hccl_tools --device_num [0,8]
    ```

- 常用参数说明

```text
RANK_TABLE_FILE: 由mindformers/tools/hccl_tools.py生成的分布式json文件
CONFIG_PATH: 为configs文件夹下面的{model_name}/run_*.yaml配置文件
DEVICE_ID: 为设备卡，范围为0~7
DEVICE_RANGE: 为单机分布式卡的范围, 如[0,8]为8卡分布式，不包含8本身
RUN_STATUS: 为任务运行状态，支持关键字 train、eval、predict
```

- 快速使用方式 1：统一接口启动，根据模型 CONFIG 完成任意模型的单卡训练、评估、推理流程

```shell
# 训练启动，run_status支持train、eval、predict三个关键字，以分别完成模型训练、评估、推理功能，默认使用配置文件中的run_status
python run_mindformer.py --config {CONFIG_PATH} --run_status {train/eval/predict}
```

- 快速使用方式 2： scripts 脚本启动，根据模型 CONFIG 完成任意模型的单卡/多卡训练、评估、推理流程

```shell
# 单卡启动脚本
cd scripts
sh run_standalone.sh CONFIG_PATH DEVICE_ID RUN_STATUS

# 多卡启动脚本
# 8卡分布式运行， DEVICE_RANGE = [0, 8], 不包含8本身
cd scripts
sh run_distribute.sh RANK_TABLE_FILE CONFIG_PATH DEVICE_RANGE RUN_STATUS
```

#### 方式二：pip 安装使用

用户可以通过`pip install mindformers`的方式利用Trainer高阶接口执行模型任务的训练、评估、推理功能。

Trainer接口详细设计请阅：[Trainer接口使用案例及接口设计说明](https://gitee.com/mindspore/transformer/wikis/%E7%89%B9%E6%80%A7%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3?sort_id=6569071)

- 准备工作

    - step1: 安装mindformers包

    ```shell
    pip install mindformers
    ```

    - step2: 准备相应任务的数据集，请参考`configs`目录下各模型的README.md文档准备相应数据集

- 小白体验使用方式：准备数据集，直接开启已有任务的训练、评估、推理流程

```python
from mindformers import Trainer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ContextConfig

## Step 1 MindSpore 环境初始化
context_config = ContextConfig(device_id=0, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
init_context(seed=2022, use_parallel=False, context_config=context_config)  # 进行环境初始化, 单卡设定

## Step 2 输入对应任务的标准数据集路径，自动创建已有任务的训练、评估、推理流程 (需提前准备好对应的数据集)
mim_trainer = Trainer(task_name='masked_image_modeling', # 已集成的任务名
                      model='mae_vit_base_p16', # 已集成的模型名
                      train_dataset="/data/imageNet-1k/train", # 传入标准的训练数据集路径，默认支持ImageNet数据集格式
                      eval_dataset="/data/imageNet-1k/eval") # 传入标准的评估数据集路径，默认支持ImageNet数据集格式
mim_trainer.train() # 开启训练流程
# mim_trainer.eval() # 开启评估流程
# mim_trainer.predict(input_data) # 输入要执行推理的数据，开启推理流程
```

- 初阶开发使用方式: 通过config类配置参数完成已有任务的训练、评估、推理流程

```python
from mindformers.trainer import Trainer
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    OptimizerConfig, DatasetConfig, DataLoaderConfig, RunnerConfig, \
    ContextConfig, LRConfig

## Step 1 MindSpore 环境初始化
context_config = ContextConfig(device_id=1, device_target='Ascend', mode=0)  # 支持MindSpore context的环境配置
init_context(seed=2022, use_parallel=False, context_config=context_config)  # 进行环境初始化, 单卡设定

## Step 2 通过支持的Config类设定支持的超参数
runner_config = RunnerConfig(epochs=10, batch_size=2, image_size=224)  # 自定义运行超参
lr_schedule_config = LRConfig(lr_type='WarmUpLR', learning_rate=0.001, warmup_steps=10)  # 自定义学习策略
optim_config = OptimizerConfig(optim_type='Adam', beta1=0.009, learning_rate=lr_schedule_config) # 自定义优化器策略
train_loader_config = DataLoaderConfig(dataset_dir="/data/imageNet-1k/train")   # 数据加载参数设定， 默认ImageFolderDataset加载方式
eval_loader_config = DataLoaderConfig(dataset_dir="/data/imageNet-1k/eval")
train_dataset_config = DatasetConfig(data_loader=train_loader_config,
                               input_columns=["image"],
                               output_columns=["image"],
                               column_order=["image"],
                               batch_size=2,
                               image_size=224) # 设定训练数据集的输入、输出、bs等超参数
eval_dataset_config = DatasetConfig(data_loader=eval_loader_config,
                                    input_columns=["image"],
                                    output_columns=["image"],
                                    column_order=["image"],
                                    batch_size=2,
                                    image_size=224) # 设定评估数据集的输入、输出、bs等超参数

config = ConfigArguments(output_dir="./output_dir",
                         runner_config=runner_config,
                         train_dataset=train_dataset_config,
                         eval_dataset=eval_dataset_config,
                         optimizer=optim_config) # 统一超参配置接口

## Step 3 通过config配置拉起相应任务的训练、评估、推理功能
mim_trainer = Trainer(task_name='masked_image_modeling', model='mae_vit_base_p16', config=config)
mim_trainer.train() # 开启训练流程
# mim_trainer.eval() # 开启评估流程
# mim_trainer.predict(input_data) # 输入要执行推理的数据，开启推理流程
```

- 中阶开发使用方式: 用户通过自定义开发的网络、数据集、优化器等模块完成已有任务的训练、评估、推理流程

```python
import numpy as np

from mindspore.nn import AdamWeightDecay, WarmUpLR
from mindspore.train.callback import LossMonitor, TimeMonitor,\
    CheckpointConfig, ModelCheckpoint
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.models import MaeModel
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig, ContextConfig


class MyDataLoader:
    """Self-Define DataLoader."""
    def __init__(self):
        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

## Step 1 MindSpore 环境初始化
context_config = ContextConfig(device_id=1, device_target='Ascend', mode=0)
init_context(seed=2022, use_parallel=False, context_config=context_config)

#  Step 2 运行超参配置定义
runner_config = RunnerConfig(epochs=10, batch_size=8, image_size=224, sink_mode=True, per_epoch_size=10)
config = ConfigArguments(output_dir="./output_dir", seed=2022, runner_config=runner_config)

#  Step 3 自定义网络实例
mae_model = MaeModel()

#  Step 4 自定义数据集加载及预处理流程
dataset = GeneratorDataset(source=MyDataLoader(), column_names='image')
dataset = dataset.batch(batch_size=8)

# Step 5 自定义学习策略和优化器
lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                            learning_rate=lr_schedule,
                            params=mae_model.trainable_params())

# Step 6 自定义callback函数
loss_cb = LossMonitor(per_print_times=2)
time_cb = TimeMonitor()
ckpt_config = CheckpointConfig(save_checkpoint_steps=10, integrated_save=True)
ckpt_cb = ModelCheckpoint(directory="./output/checkpoint", prefix="my_model", config=ckpt_config)
callbacks = [loss_cb, time_cb, ckpt_cb]

# 通过自定义任意模块完成masked_image_modeling任务的训练、评估、推理流程
mim_trainer = Trainer(task_name='masked_image_modeling',
                      model=mae_model,  # 包含loss计算
                      config=config,
                      optimizers=optimizer,
                      train_dataset=dataset,
                      eval_dataset=dataset,
                      callbacks=callbacks)
mim_trainer.train() # 开启训练流程
# mim_trainer.eval() # 开启评估流程
# mim_trainer.predict(input_data) # 输入要执行推理的数据，开启推理流程
```

- 高阶使用方式：高阶类混合使用和组装.....

## 使用指南

目前提供下述的文档

- [使用指南](docs/how_to_config.md)
- [如何使用BERT进行微调](docs/how_to_train_bert.md)

## Benchmark

请[在此](docs/benchmark.md)查看每个模型的复现性能基准。

## FAQ

1. 如何迁移HuggingFace权重 ？

请查看[如何转换HuggingFace的权重](./tools/README.md)

## 贡献

欢迎参与社区贡献，详情参考[Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING_CN.md)。

## 许可证

[Apache 2.0许可证](LICENSE)
