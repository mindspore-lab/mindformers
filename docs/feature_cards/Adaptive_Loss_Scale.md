# Adaptive loss scaling

## 基本介绍

现有dynamic loss scaling方案使用固定scale window，在FP16或更低精度(8bit浮点格式)混合精度训练训练时，如果选用较大的scale window，
存在loss scaling 调整不及时的风险，影响模型收敛性和收敛速度；如果选用较小的scale window，loss scale调整至合适的值时，
仍会频繁上调，损失大量训练数据。 Adaptive loss scaling方案，通过动态调节scale window，实现自适应调整loss scale，
实时将loss scale调整至FP16和8bit浮点格式正常训练所需的合适的值，同时避免损失大量训练数据。

## 使用场景及针对的问题

### 使用场景

#### 大模型预训练

包含：FP16混合精度训练、全FP16训练、FP8混合精度训练、其他低精度浮点格式混合精度训练

#### 大模型断点续训

包含：FP16混合精度训练、全FP16训练、FP8混合精度训练、其他低精度浮点格式混合精度训练

#### 微调

FT、SFT、RLHF等模型微调场景

### 针对的问题

#### 大模型训练早期

由于使用FP16和FP8等低精度数据格式引入的数值动态范围不足或精度不足导致的梯度弥散及loss回升问题

#### 大模型训练中后期

loss scale不稳定，异常波动，需频繁手动调整scale window进行断点重训的现象

## 设计概述

根据用户输入的max scale window，和默认的min scale window 20。根据最大和最小scale window， 自动生成一个scale window list，包含多个档位的scale window。
(若max scale window 非法，如小于min scale window， 则使用1000作为max scale window)

scale window 1为隐藏窗口，下调scale window时，自动触发，其下一档scale window为 min scale window 20。

针对模型训练过程loss scale变化趋势，设计两种检测机制：

scale window上调检测机制：训练开始初始使用第一档scale window 20进行训练，新增一个上调计数growth_num，初始为0，每次上调loss scale时，计数+1；每上调三次loss scale (上调计数为3时)，窗口随之上调，同时重置上调计数，直到达到最大窗口；

scale window下调检测机制：新增一个下调计数down_num，初始为0，每次下调loss scale时，计数+1，出现连续三次loss scale下降 (若中间出现loss scale上调，则重置下调计数)，则将窗口调到1，同时重置下调计数。

![Adaptive_loss_scale_process](assets/Adaptive_loss_scale/Adaptive_loss_scale_process.png)

## 使用示例

使用方法与dynamic loss scaling和fixed loss scale基本一致，新增用户指定的超参max_scale_window和min_scale_window, 同时需要将更新后的scale window信息写入至断点中以支持断点续训

Mindspore用法:

```python
>>> import numpy as np
>>> import mindspore
>>> from mindspore import Tensor, Parameter, nn, ops
>>>
>>> class Net(nn.Cell):
    ...     def __init__(self, in_features, out_features):
    ...         super(Net, self).__init__()
...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
                                    ...                                 name='weight')
...         self.matmul = ops.MatMul()
...
...     def construct(self, x):
    ...         output = self.matmul(x, self.weight)
...         return output
...
>>> in_features, out_features = 16, 10
>>> net = Net(in_features, out_features)
>>> loss = nn.MSELoss()
>>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
>>> net_with_loss = nn.WithLossCell(net, loss)
>>> manager = nn.AdaptiveLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=20,
>>>                                          max_scale_window=1000, min_scale_window=20)
>>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
>>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
>>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
>>> output = train_network(input, labels)
```

Mindformers用法:

```python
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore.nn import Momentum
from mindformers import Trainer, TrainingArguments, AutoModel
from mindformers import init_context, ContextConfig
from mindformers.wrapper import MFTrainOneStepCell, AdaptiveLossScaleUpdateCell


def context_init():
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    rank_id, device_num = init_context(use_parallel=False, context_config=context_config)


def generator():
    """text dataset generator."""
    seq_len = 1025
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    for _ in range(512):
        yield input_ids

# 环境初始化
context_init()
# 自定义训练超参数
training_args = TrainingArguments(num_train_epochs=3, batch_size=2, learning_rate=0.001,
                                  warmup_steps=1000, sink_mode=True)
# 自定义模型
pangu_model = AutoModel.from_pretrained("pangualpha_2_6b")
opt = Momentum(learning_rate=0.1, momentum=0.9,
               params=pangu_model.trainable_params(),)
manager = AdaptiveLossScaleUpdateCell(1, 2, 20, 1000, 20)
train_network = MFTrainOneStepCell(pangu_model, opt, scale_sense=manager)
train_network.set_train()
# 自定义数据集
dataset = GeneratorDataset(generator, column_names=["input_ids"])
train_dataset = dataset.batch(batch_size=4)
eval_dataset = dataset.batch(batch_size=4)
# 定义文本生成任务，传入自定义模型、数据集、超参数
text_generation = Trainer(task='text_generation', model_name=='pangualpha_2_6b',
                          wrapper=train_network, args=training_args,
                          train_dataset=train_dataset, eval_dataset=eval_dataset)
```

模型训练yaml中设置方式runner_config中声明使用adaptive loss scaling

```yaml
# runner
runner_config:
  epochs: 3
  batch_size: 4
  sink_mode: True
  sink_size: 2
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense:
    type: AdaptiveLossScaleUpdateCell
    loss_scale_value: 4294967296
    scale_factor: 2
    scale_window: 20
    max_scale_window: 1000
    min_scale_window: 20
  use_clip_grad: True
```
