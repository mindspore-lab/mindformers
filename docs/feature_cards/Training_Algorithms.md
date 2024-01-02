# 训练优化算法

MindFormers套件集成了许多模型训练中通用的优化算法，并提供了便捷的使用方式，在本文档中集中进行说明。

目录：

- [训练优化算法](#训练优化算法)
    - [梯度累积](#梯度累积)
    - [梯度裁剪](#梯度裁剪)
    - [Token分布](#Token分布)

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

## Token分布

在MoE大模型训练过程中，常见的TopK Router算法会导致Token分发不均匀，存在Router给少数热门专家分配大量Token，多数冷门专家分配少量Token的情况。专家受限于专家容量，会将超过专家容量的Token丢弃，不足专家容量的Padding。所以获取Token分布情况能帮助用户合理确定专家容量。
MindFormers配置文件中的`MoE_config`新增了`save_token_distribution`配置项，默认False，需要搭配`callbacks`中的`SummaryMonitor`一同开启，如下：

```yaml
moe_config:
  expert_num: 8
  save_token_distribution: true

callbacks:
- type: SummaryMonitor
  summary_dir: "../summary_dir/token_distribution_dir"
  keep_default_action: False
  collect_freq: 1
  collect_tensor_freq: 1
  export_options: {'tensor_format':'npy'}
```

在开启该配置之后，会在`summary_dir`路径下生成`export_xxx/tensor`文件夹，其中包含每层MoE中Token分布数据，再使用`mindformers/tools/moe_token_distribution_tools.py`脚本，输入参数：`num_layers`、`hot_expert_num`、`npy_files_load_path`、`save_path_prefix`。会在保存路径中生成Token分布图。

## Flash Attention

Flash Attention（简称FA），是深度学习业界主流的注意力计算加速算法；MindSpore+Ascend架构也提供了FA实现，当前MindFormers对部分模型进行了FA的适配，可使用 `model_config` 中的 `use_flash_attention` 配置项控制模型是否使用FA

注意，FA特性依赖于MindSpore 2.2.10+版本，且目前仅针对Atlas A2训练系列硬件进行了适配，请使用正确的版本配套

由于FA特性并非全版本全硬件支持，当前默认关闭FA，需手动打开配置项以使用FA

举例如下，llama可通过修改配置项以使能FA，而后可使用该配置项进行训练

```yaml
# model config
model:
  model_config:
    type: LlamaConfig
    ...
    use_flash_attention: True   # True to enable FA, False to disable FA
    ...
  arch:
    type: LlamaForCausalLM
```

FA的模型支持度可参见 [模型能力表格](../model_support_list.md#llm大模型能力支持一览)
