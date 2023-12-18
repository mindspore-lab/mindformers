# 训练优化算法

MindFormers套件集成了许多模型训练中通用的优化算法，并提供了便捷的使用方式，在本文档中集中进行说明。

目录：

- [训练优化算法](#训练优化算法)
    - [梯度累积](#梯度累积)
    - [梯度裁剪](#梯度裁剪)

## 梯度累积

梯度累积算法是业界常用的扩大batch_size，解决OOM的一种算法，可参考[MindSpore文档](https://www.mindspore.cn/tutorials/experts/zh-CN/master/optimize/gradient_accumulation.html)

MindSpore在2.1.1之后的版本中增加了 `mindspore.nn.wrap.cell_wrapper.GradAccumulationCell` 这一梯度累积实现接口，通过拆分MiniBatch的形式实现了梯度累积

MindFormers套件对上述实现接口进行了适配，在需要开启梯度累积的场景下，只需在配置文件中的 `runner_config` 项下新增 `gradient_accumulation_steps` 项，并配置为所需的梯度累积步数即可，如下：

```yaml
runner_config:
  ...
  gradient_accumulation_steps: 4
  ...
```

除配置文件外，其余几种常用使用方式也提供了梯度累积的配置接口：

1. run_mindformer.py脚本启动时，可指定 `--gradient_accumulation_steps` 入参；

2. trainer接口启动时，可通过 `TrainingArguments` 类指定 `gradient_accumulation_steps` 入参；

**限制**：由于 `GradAccumulationCell` 的实现依赖并行特性，梯度累积当前仅支持在**半自动并行模式**下使用；此外，pipeline并行场景下，梯度累积含义与micro_batch相同，将不会生效，请配置 `micro_batch_num` 项以增大训练batch_size

## 梯度裁剪

梯度裁剪算法可以避免反向梯度过大，跳过最优解的情况

MindFormers中，默认的训练流程 `MFTrainOneStepCell` 中集成了梯度裁剪逻辑，通过 `use_clip_grad` 配置项来控制在训练过程中是否开启梯度裁剪，默认为False；并可通过 `max_grad_norm` 项控制梯度裁剪的最大norm值，默认为1.0；如下以开启梯度裁剪：

```yaml
runner_wrapper:
  type: MFTrainOneStepCell
  ...
  use_clip_grad: True
  max_grad_norm: 1.0
  ...
```

## 优化器异构

在大模型训练过程中，优化器状态占用了大量的内存，进而限制了可训练的模型规模，使用优化器异构，将优化器指定到CPU上执行，可以极大扩展可训练模型规模

MindFormers中，FusedCastAdamWeightDecay优化器支持了异构的能力，通过配置`optimizer`的`type`项，选择FusedCastAdamWeightDecay，即可开启优化器异构训练，如下：

```yaml
optimizer:
  type: FusedCastAdamWeightDecay
  ...
  beta2: 0.95
  eps: 0.00000001
```

**限制**：目前仅支持在网络参数为fp16且并行配置为非pipeline场景使用