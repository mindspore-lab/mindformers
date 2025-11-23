## 目录

- [MindFormers LLM 配置白名单](#mindformers-llm-配置白名单)
    - [训练配置说明](#训练配置说明)
        - [基础配置](#基础配置)
        - [权重配置](#权重配置)
        - [训练运行配置](#训练运行配置)
        - [学习策略配置](#学习策略配置)
        - [优化器配置](#优化器配置)
        - [数据集](#数据集)
            - [预训练数据集](#预训练数据集)
            - [微调数据集](#微调数据集)
                - [在线加载数据集](#在线加载数据集)
                - [离线加载数据集](#离线加载数据集)
        - [并行配置](#并行配置)
        - [重计算配置](#重计算配置)
        - [CPU OffLoad配置](#cpu-offload配置)
        - [训练监控配置](#训练监控配置)
        - [TensorBoard可视化配置](#tensorboard可视化配置)
        - [Profile配置](#profile配置)
        - [运行环境配置](#运行环境配置)
            - [Context](#context)
            - [Parallel Context](#parallel-context)
        - [模型配置](#模型配置)
    - [推理配置说明](#推理配置说明)
        - [基础配置](#基础配置-1)
        - [并行配置](#并行配置-1)
        - [模型配置](#模型配置-1)

## MindFormers LLM 配置白名单

注意当前为实验性特性，为了配套新版重写的LLMTrainer使用（仅支持静态图场景），旧Trainer无法直接使用，但相关参数含义可以参考，请勿直接使用于旧版配置文件中。

## 训练配置说明

### 基础配置

```yaml
# output_dir: 训练输出目录，用于保存模型检查点、MindFormers Logger日志、转换的权重、分布式策略等训练产出文件
# 建议使用绝对路径或相对于项目根目录的相对路径
output_dir: './output'

# run_mode: 运行模式，指定当前任务的执行模式
# 可选值: 'train'(训练模式) | 'finetune'(微调模式) | 'predict'(推理预测模式)
# 不同模式下会加载不同的组件和执行不同的训练/推理流程
run_mode: 'train'

# use_parallel: 是否启用并行训练功能
# 设置为True时启用分布式并行训练(包括数据并行、张量并行、流水线并行等)
# 设置为False时使用单卡训练模式，适用于调试或小规模训练场景
use_parallel: True

# train_precision_sync: 训练确定性计算开关，用于控制训练过程是否使用确定性计算以保证结果可复现
# 设置为True时启用确定性计算，确保每次训练结果一致，但可能影响训练性能
# 设置为False时使用非确定性计算，训练速度更快但结果可能有微小差异
train_precision_sync: False

# pretrained_model_dir: HuggingFace模型文件目录，用于直接读取HF的模型配置、词表、权重等内容
# 指定预训练模型的本地路径，支持从HuggingFace格式的模型文件加载权重和配置
# 路径应包含config.json、tokenizer.json、model.safetensors等标准HF模型文件
pretrained_model_dir: '/path/hf_dir'

# Trainer配置，用于指定训练器类型和任务相关信息
trainer:
  # type: 训练器类型，使用新的LLMTrainer简化训练流程
  # LLMTrainer是专门为大语言模型设计的训练器，提供简化的训练逻辑和优化流程
  type: LLMTrainer

  # task_name: 任务名称，用户可自定义，如qwen3、glm4等
  # 用于标识当前训练任务的名称，字符内容没有限制，建议使用模型名称便于识别和管理
  task_name: 'llm_model'  # qwen3 glm4等，用户可自己定义，字符内容没有限制
```

### 权重配置

```yaml
# 权重相关配置
checkpoint_config:
  # 指定模型权重文件的路径，空字符串表示不加载，pretrained_model_dir有效则会使用pretrained_model_dir下的权重
  # 当且仅当load_checkpoint == “not_load_any_ckpt”时, 不会加载任何权重，包括pretrained_model_dir下的
  # “not_load_any_ckpt”字段场景通常用于pretrained_model_dir有效的情况下，复用其下除权重之外的文件，如预训练场景或者需要不加载任何权重的场景
  load_checkpoint: ''

  # 权重保存配置，设置保存检查点的各项参数
  # prefix: 保存权重文件的前缀名，权重名字，字符内容没有限制，用户根据实际需要指定，建议写训练的模型名称即可 qwen3/glm4等
  prefix: "llm_model"

  # save_checkpoint_seconds: 按时间间隔保存检查点(秒)，默认0表示不启用该功能
  # 当设置为大于0的值时，每隔指定秒数自动保存一次模型检查点
  save_checkpoint_seconds: 0

  # save_checkpoint_steps: 按训练步数间隔保存检查点
  # 设置为正整数时，每训练指定步数就保存一次模型权重
  # 例如设置为100表示每训练100步保存一次检查点
  save_checkpoint_steps: 100

  # keep_checkpoint_max: 最多保留的检查点文件数量，默认只保留1个最新的检查点
  # 当保存的检查点数量超过该值时，会自动删除最旧的检查点文件
  keep_checkpoint_max: 1

  # keep_checkpoint_per_n_minutes: 每N分钟保留一个检查点，默认0表示不启用该功能
  # 当设置为大于0的值时，每隔指定分钟数保留一个检查点，不受keep_checkpoint_max限制
  # 该机制可以确保即使在频繁保存的情况下，也能保留关键时间点的检查点
  keep_checkpoint_per_n_minutes: 0

  # integrated_save: 集成保存模式，控制检查点保存方式
  # 设置为True时启用集成保存，会在单个设备上收集所有分片权重后统一保存，适用于小规模训练
  # 设置为False时使用分布式保存，各设备独立保存自己的权重分片，适用于大规模训练以避免内存溢出(OOM)
  # 注意：大集群大参数场景仅支持False
  integrated_save: False

  # save_network_params: 是否额外保存纯网络参数(不含优化器信息)，方便推理场景使用
  # 设置为True时会额外保存一份仅包含网络参数的文件，避免后续需要过滤优化器等训练信息，但会增加存储压力
  save_network_params: False

  # save_trainable_params: 是否只保存可训练参数，适用于LoRA等参数高效微调场景
  # 设置为True时只保存可训练的参数（如LoRA适配器），冻结的参数不会被保存，用于缓解存储压力
  save_trainable_params: False

  # async_save: 是否启用异步保存，可以减少保存时的IO等待时间
  # 设置为True时启用异步保存模式，训练过程不会被保存操作阻塞，提高训练效率
  async_save: False

  # remove_redundancy: 去冗余保存或加载开关，用于减少保存的权重文件大小
  # 设置为True时会去除权重文件中的冗余信息，减小文件体积，节省存储空间
  remove_redundancy: True
```

### 训练运行配置

```yaml
# Runner configuration
training_args:
  # epochs: 训练轮数，指定整个数据集需要训练的轮次数量
  epochs: 2

  # micro_batch_size: 微批次大小，每个设备上单次前向/反向传播处理的样本数量
  micro_batch_size: 1

  # global_batch_size: 全局批次大小，所有数据并行设备上的有效批次大小
  # 计算公式: global_batch_size = micro_batch_size × num_micro_batches × data_parallel_size × micro_batch_interleave_num
  # 该参数控制实际的训练批次大小，影响梯度更新的频率和稳定性
  global_batch_size: 512

  # training_seed: 训练过程中的随机种子，用于确保训练过程的可复现性
  training_seed: 1234

  # dataset_seed: 数据集采样的随机种子，null时使用training_seed
  # null或int，这个配置存在时，数据集采样时的随机数按照该设定进行，不存在时，以training_seed代替
  dataset_seed: 1234

  # check_for_nan_in_loss_and_grad: 是否检查loss和梯度中的NaN/Inf值，发现则报错退出
  # 设置为True时会检查训练过程中的数值异常，当检测到NaN或Inf值时立即报错并退出训练，便于快速定位训练问题
  check_for_nan_in_loss_and_grad: False

  # calculate_per_token_loss: 是否计算每个token的loss，用于更细粒度的损失计算
  # 设置为True时会计算每个token位置的损失值，可用于更精细的损失分析和可视化
  calculate_per_token_loss: False

  # scale_sense: 损失缩放因子，用于混合精度训练时防止梯度下溢
  # BF16计算模式下通常使用默认值1.0，FP16模式下可能需要设置更大的值（如2^15或2^16）
  # 该参数控制损失函数值的缩放，影响梯度更新的数值稳定性
  scale_sense: 1.0

  # use_clip_grad: 是否启用梯度裁剪，需要配合max_grad_norm使用
  # 设置为True时启用梯度裁剪功能，当梯度范数超过max_grad_norm阈值时进行裁剪，防止梯度爆炸
  use_clip_grad: True

  # max_grad_norm: 梯度裁剪的最大范数阈值，当梯度范数超过该值时进行裁剪
  max_grad_norm: 1.0

  # gradient_accumulation_steps: 梯度累积步数，用于模拟更大的有效批次大小
  # 设置为N时，会累积N个micro_batch的梯度后再进行一次参数更新，等价于将batch_size扩大N倍
  # 该参数可以在内存受限的情况下实现更大的全局批次大小训练
  gradient_accumulation_steps: 1

  # stop_step: 自定义训练停止步数，注意需要同步调整学习率调度
  # 设定停止训练的步数 --> 注意学习率部分的计算是否需要随之变化曲率
  stop_step: null

  # resume_training: 是否启用断点续训功能
  # 断点续训（权重或数据集恢复开关）
  resume_training: False

  # data_skip_steps: 手动设定期望跳过数据集的步数，在断点续训或者数据退火等场景使用
  # 为None时，由续训的权重信息自动跳过已训练的数据
  data_skip_steps: null

  # ignore_data_skip: 是否忽略数据跳过设置
  # 设置为True时忽略已跳过的数据步数，从头开始训练，常用于增量数据续训场景
  # 设置为False时，自动根据续训的权重信息跳过已训练的数据，从断点处继续训练
  ignore_data_skip: False

  # use_skip_data_by_global_norm: 是否启用基于global_norm的数据跳过功能
  # 设置为True时，根据global_norm_spike自动跳过异常数据，以保障训练稳定性
  # 配合global_norm_spike_threshold和global_norm_spike_count_threshold使用
  # 当检测到norm异常时，会在线自动跳过该数据样本，不更新权重状态（包括优化器和梯度）
  use_skip_data_by_global_norm: False

  # use_fast_process_recovery_by_global_norm: 是否启用基于global_norm的快速恢复功能
  # 该功能与use_skip_data_by_global_norm功能互斥，不能同时开启
  # 设置为True时，当global_norm超过global_norm_spike_threshold阈值时，会打印精度告警并停止训练
  # 系统会抛出TREError并进行进程级重启训练进程，以保障训练模型的精度
  use_fast_process_recovery_by_global_norm: False

  # global_norm_spike_threshold: global_norm异常阈值，用于检测梯度异常
  global_norm_spike_threshold: 1.0

  # global_norm_spike_count_threshold: global_norm异常计数阈值，超过该次数触发相应处理
  global_norm_spike_count_threshold: 10

  # use_checkpoint_health_monitor: 是否启用检查点健康监控，监测embedding层的local norm状态
  # 设置为True时，会监控权重中embedding层的local norm状态
  # 当embedding_local_norm超过embedding_local_norm_threshold阈值时，该权重会被标记为非健康状态
  # 在恢复训练时，非健康的权重不会被使用，以保障训练稳定性和模型质量
  use_checkpoint_health_monitor: False

  # embedding_local_norm_threshold: embedding local norm异常阈值，超过该值的权重被视为不健康
  # 该值的设定通常需要提起训练模型，观察训练日志中正常的norm值以选择出一个合适的阈值
  embedding_local_norm_threshold: 1.0
```

### 学习策略配置

```yaml
# Learning rate scheduler configuration
# 学习率调度器配置，用于控制训练过程中学习率的变化策略
lr_schedule: # 学习策略1 - 带WarmUp的恒定学习率调度器
  # type: 学习率调度器类型，ConstantWarmUpLR恒定学习率调度器
  type: ConstantWarmUpLR

  # learning_rate: 基础学习率值，训练过程中使用的学习率
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1
```

```yaml
lr_schedule: # 学习策略2 - 线性衰减学习率调度器配置
  # type: 学习率调度器类型，LinearWithWarmUpLR线性衰减学习率调度器
  # 该调度器在预热阶段线性增长学习率，然后在训练过程中线性衰减学习率
  type: LinearWithWarmUpLR

  # learning_rate: 基础学习率值，预热阶段结束后的最大学习率
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1
```

```yaml
lr_schedule: # 学习策略3 - 余弦退火重启学习率调度器配置
  # type: 学习率调度器类型，CosineWithRestartsAndWarmUpLR余弦退火重启学习率调度器
  # 该调度器在预热阶段线性增长学习率，然后在训练过程中按照余弦函数衰减学习率
  type: CosineWithRestartsAndWarmUpLR

  # learning_rate: 基础学习率值，预热阶段结束后的初始学习率
  # 余弦退火会以此为起点，逐渐衰减到接近0的学习率
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1

  # num_cycles: 余弦退火周期数，控制学习率衰减的周期次数
  # 0.5表示在总训练步数内完成半个余弦周期的衰减，即从最大值衰减到最小值
  num_cycles: 0.5

  # lr_end: 最终学习率值，余弦退火衰减的最终学习率
  # 学习率会从learning_rate衰减到该值，通常设置为0或一个很小的正数
  lr_end: 0.

  # decay_steps: 衰减步数，指定学习率衰减的步数
  # 当该值不为null时，默认学习率衰减至total_steps，超出decay_steps的训练步数，学习率将保持不变
  decay_steps: null

  # decay_ratio: 衰减比例，控制学习率衰减占总步数的比例
  # 0表示不使用衰减比例，数值范围为[0, 1]
  # 不为0时，会覆盖decay_steps的设置，且默认decay_steps=total_steps*decay_ratio
  decay_ratio: 0.
```

```yaml
lr_schedule: # 学习策略4 - 带重启的余弦退火学习率调度器配置
  # type: 学习率调度器类型，CosineWithRestartsAndWarmUpLR带重启的余弦退火学习率调度器
  # 该调度器在预热阶段线性增长学习率，然后在训练过程中按照余弦函数衰减学习率，并支持学习率重启机制
  type: CosineWithRestartsAndWarmUpLR

  # learning_rate: 基础学习率值，预热阶段结束后的初始学习率
  # 余弦退火会以此为起点，逐渐衰减到接近0的学习率
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1

  # num_cycles: 余弦退火周期数，控制学习率衰减的周期次数
  # 设置为1.0表示完成一个完整的余弦周期（从最大值衰减到最小值再回升到最大值）
  # 设置为0.5表示仅完成半个余弦周期（从最大值衰减到最小值）
  num_cycles: 1

  # lr_end: 最终学习率值，余弦退火衰减的最终学习率
  # 学习率会从learning_rate衰减到该值，通常设置为0或一个很小的正数
  lr_end: 0.

  # decay_steps: 衰减步数，指定学习率衰减的步数
  # 当该值不为null时，默认学习率衰减至total_steps，超出decay_steps的训练步数，学习率将保持不变
  decay_steps: null
```

```yaml
lr_schedule: # 学习策略5 - 多项式衰减学习率调度器配置
  # type: 学习率调度器类型，PolynomialWithWarmUpLR多项式衰减学习率调度器
  # 该调度器在预热阶段线性增长学习率，然后在训练过程中按照多项式函数衰减学习率
  type: PolynomialWithWarmUpLR

  # learning_rate: 基础学习率值，预热阶段结束后的初始学习率
  # 多项式衰减会以此为起点，按照指定的幂次逐渐衰减到lr_end
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1

  # power: 多项式衰减的幂次，控制学习率衰减的曲线形状
  # 当power=1.0时为线性衰减，power>1.0时衰减曲线更陡峭，power<1.0时衰减曲线更平缓
  power: 1.0

  # lr_end: 最终学习率值，多项式衰减的最终学习率
  # 学习率会从learning_rate按照多项式函数衰减到该值
  lr_end: 0.

  # decay_steps: 衰减步数，指定学习率衰减的步数
  # 当该值不为null时，默认学习率衰减至total_steps，超出decay_steps的训练步数，学习率将保持不变
  decay_steps: null
```

```yaml
lr_schedule: # 学习策略6 - 余弦退火重启学习率调度器配置
  # type: 学习率调度器类型，CosineAnnealingWarmRestarts余弦退火重启学习率调度器
  # 该调度器按照余弦函数周期性地衰减学习率，并在每个周期结束时重启学习率
  type: CosineAnnealingWarmRestarts

  # base_lr: 基础学习率值，每个重启周期的初始学习率
  # 余弦退火会以此为起点，逐渐衰减到eta_min
  base_lr: 1.e-6

  # t_0: 第一个重启周期的步数，控制第一个学习率周期的长度
  # 当训练步数达到t_0的倍数时，学习率会重启到base_lr，需要根据算法需求人工进行设置
  t_0 : 10

  # t_mult: 周期倍数因子，控制后续重启周期长度的倍数
  # 当t_mult=1时，所有周期长度相同；当t_mult>1时，后续周期会逐渐变长
  t_mult : 1.

  # eta_min: 最小学习率值，余弦退火衰减的最终学习率
  # 学习率会在每个周期内从base_lr衰减到该值
  eta_min : null
```

```yaml
lr_schedule: # 学习策略7 - 预热稳定衰减学习率调度器配置
  # type: 学习率调度器类型，WarmUpStableDecayLR预热稳定衰减学习率调度器
  # 该调度器在预热阶段线性增长学习率，然后在稳定阶段保持学习率不变，最后按照多项式函数衰减学习率
  type: WarmUpStableDecayLR

  # learning_rate: 基础学习率值，预热阶段结束后的稳定学习率
  # 在稳定阶段会保持该学习率不变，直到开始衰减阶段
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1

  # lr_end: 最终学习率值，多项式衰减的最终学习率
  # 学习率会从learning_rate按照多项式函数衰减到该值
  lr_end: 1e-7

  # decay_start_steps: 衰减开始步数，指定学习率开始衰减的训练步数
  # 当该值不为null时，学习率会在达到该步数后开始衰减
  decay_start_steps: null

  # decay_start_ratio: 衰减开始比例，控制学习率开始衰减的步数占总训练步数的比例
  # 当该值不为null时，学习率会在total_steps*decay_start_ratio步数后开始衰减, 覆盖decay_start_steps的设置
  decay_start_ratio: null
```

```yaml
lr_schedule: # 学习策略8 - 恒定学习率带冷却调度器配置
  # type: 学习率调度器类型，ConstantWithCoolDownLR恒定学习率带冷却调度器
  # 该调度器在预热阶段线性增长学习率，然后在训练过程中保持恒定学习率，最后进入冷却阶段逐渐降低学习率
  type: ConstantWithCoolDownLR

  # learning_rate: 基础学习率值，预热阶段结束后的恒定学习率
  # 在稳定阶段会保持该学习率不变，直到进入冷却阶段
  learning_rate: 1.e-6

  # warmup_lr_init: 学习率预热初始值，预热阶段的起始学习率
  # 预热阶段会从该值线性增长到learning_rate
  warmup_lr_init: 0.

  # warmup_ratio: warmup比例，控制预热阶段占总训练步数的比例
  # 0表示不使用学习率预热，直接使用恒定学习率，数值范围为[0, 1]
  # 不为0时，会覆盖warmup_steps的设置, 且warmup_steps=total_steps*warmup_ratio
  warmup_ratio: 0.

  # warmup_steps: 预热步数，直接指定预热阶段的学习率预热步数
  # 当该值不为null时，会覆盖warmup_ratio的设置
  warmup_steps: null

  # total_steps: 总训练步数，-1表示使用默认的数据集大小计算出的总步数
  # 设置为正整数时，将使用指定的步数作为训练总步数，如果设置了stop_step训练提前退出功能，建议手动调整该参数
  total_steps: -1

  # num_cycles: 余弦退火周期数，控制学习率衰减的周期次数
  # 0.5表示在总训练步数内完成半个余弦周期的衰减，即从最大值衰减到最小值
  num_cycles: 0.5

  # lr_end1: 第一阶段最终学习率值，在恒定阶段结束时的学习率
  # 学习率会从learning_rate保持到该值
  lr_end1: 0.

  # lr_end2: 第二阶段最终学习率值，在冷却阶段结束时的学习率，该值通常需满足lr_end2<=lr_end1
  # 为None时与lr_end1相等，不为0时，学习率会从lr_end1衰减到该值
  lr_end2: null

  # keep_steps: 保持步数，控制学习率保持恒定的步数
  # 在预热结束后，学习率会保持learning_rate值keep_steps步数
  keep_steps: 0

  # final_steps: 最终步数，指定冷却阶段的步数
  # 当该值不为null时，会在训练的最后final_steps步进入冷却阶段，学习率逐渐降低至lr_end1 or lr_end2
  final_steps: null

  # decay_steps: 衰减步数，指定学习率衰减的步数
  # 当该值不为null时，默认学习率衰减至total_steps，超出decay_steps的训练步数，学习率将保持不变
  decay_steps: null

  # decay_ratio: 衰减比例，控制学习率衰减占总步数的比例
  # 0表示不使用衰减比例，数值范围为[0, 1]
  # 不为0时，会覆盖decay_steps的设置，且默认decay_steps=total_steps*decay_ratio
  decay_ratio: 0.
```

### 优化器配置

```yaml
# Optimizer configuration
# 优化器配置，用于指定训练过程中使用的优化器类型及其相关参数
optimizer: # 优化器1
  # type: 优化器类型
  type: AdamW

  # betas: AdamW优化器的beta参数，控制一阶和二阶动量的衰减率
  # 第一个值(0.9)是梯度动量参数，第二个值(0.95)是平方梯度动量参数
  betas: [0.9, 0.95]

  # eps: AdamW优化器的epsilon参数，用于数值稳定性的小常数
  # 防止除零错误，在分母中加入的小值
  eps: 1.e-8

  # weight_decay: 权重衰减系数，用于L2正则化
  # 控制模型复杂度，防止过拟合，0.0表示不使用权重衰减
  weight_decay: 0.0

  # use_fused: 是否使用融合优化器操作
  # 设置为True时启用 fused kernel 优化，可以提高训练性能，但需要硬件和框架支持
  use_fused: False

  # amsgrad: 是否使用 AMSGrad 变体优化算法，需要use_fused=True使能
  # 设置为True时启用 AMSGrad 优化，是 AdamW 的改进版本，理论上收敛性更好
  amsgrad: False

  # maximize: 是否最大化目标函数，需要use_fused=True使能
  # 设置为True时优化器会尝试最大化损失函数，通常用于特定的强化学习等场景
  # 设置为False时执行标准的最小化优化（默认行为）
  maximize: False

  # swap: 是否启用优化器内存卸载到CPU
  # 设置为True时启用优化器状态的内存交换机制，用于节省显存，适用于大规模模型训练
  swap: False
```

```yaml
optimizer: # 优化器2
  # type: 优化器类型，PmaAdamW优化器
  # Pre-trained Model Average（PMA）权重合并是指在训练过程中，
  # 根据选择 Exponential Moving Average（EMA）算法或 Simple Moving Average（SMA）算法对权重进行融合合并，从而提升模型训练的效果。
  type: PmaAdamW

  # betas: AdamW优化器的beta参数，控制一阶和二阶动量的衰减率
  # 第一个值(0.9)是梯度动量参数，第二个值(0.95)是平方梯度动量参数
  betas: [0.9, 0.95]

  # eps: AdamW优化器的epsilon参数，用于数值稳定性的小常数
  # 防止除零错误，在分母中加入的小值
  eps: 1.e-8

  # weight_decay: 权重衰减系数，用于L2正则化
  # 控制模型复杂度，防止过拟合，0.0表示不使用权重衰减
  weight_decay: 0.0

  # use_fused: 是否使用融合优化器操作
  # 设置为True时启用 fused kernel优化，可以提高训练性能
  use_fused: False

  # amsgrad: 是否使用 AMSGrad 变体优化算法，需要use_fused=True使能
  # 设置为True时启用 AMSGrad 优化，是 AdamW 的改进版本，理论上收敛性更好
  amsgrad: False

  # maximize: 是否最大化目标函数，需要use_fused=True使能
  # 设置为True时优化器会尝试最大化损失函数，通常用于特定的强化学习等场景
  # 设置为False时执行标准的最小化优化（默认行为）
  maximize: False

  # swap: 是否启用优化器内存卸载到CPU
  # 设置为True时启用优化器状态的内存交换机制，用于节省显存，适用于大规模模型训练
  swap: False

  # fused_num: 融合操作的数量参数，控制融合优化器中参与融合的参数组数量
  fused_num: 10

  # interleave_step: 交错步数，控制优化器在训练过程中的交错更新频率
  interleave_step: 1000

  # fused_algo: 融合算法类型，指定使用的融合优化算法
  # 'ema'表示使用指数移动平均算法进行融合优化, 'sma'表示使用简单移动平均算法进行融合优化
  fused_algo: 'ema'

  # ema_alpha: EMA算法的alpha参数，控制指数移动平均的衰减率， fused_algo='ema'时生效
  # 值越小，历史信息的权重越大；值越大，新信息的权重越大
  ema_alpha: 0.2
```

### 数据集

#### 预训练数据集

```yaml
# Dataset configuration
train_dataset:
  # data_loader: 数据加载器配置，指定数据加载的方式和数据集信息
  data_loader:
    # type: 数据加载器类型，BlendedMegatronDatasetDataLoader用于混合多个Megatron格式数据集
    type: BlendedMegatronDatasetDataLoader

    # sizes: 数据集样本数量配置，格式为[train_size, test_size, eval_size]
    sizes:
      - 1000 # 训练集数据样本数，指定训练集包含的样本总数
      - 0    # 测试集数据样本数，当前配置为0表示不使用测试集
      - 0    # 评测集数据样本数，当前配置为0表示不使用评测集

    # config: GPTDataset配置项，定义数据集的具体参数
    config:
      # seq_length: 数据集返回数据的序列长度，指定每个样本的token序列长度
      seq_length: 8192

      # eod_mask_loss: 是否在计算loss时屏蔽EOD(End of Document)位置的损失
      # 设置为True时，EOD位置的token不会参与损失计算，避免影响模型学习效果
      eod_mask_loss: True

      # reset_position_ids: 是否重置position_ids
      # 设置为True时，会在每个新文档开始时重置位置编码，确保位置信息正确
      reset_position_ids: True

      # create_attention_mask: 是否创建attention_mask
      # 设置为True时会生成注意力掩码，用于控制模型在自注意力计算中哪些位置可以被关注
      create_attention_mask: True

      # reset_attention_mask: 是否重置attention_mask
      # 设置为True时会在特定条件下(如遇到EOD)重置注意力掩码，确保模型正确处理序列边界
      reset_attention_mask: True

      # create_compressed_eod_mask: 是否返回压缩后的attention_mask，用于节省内存
      create_compressed_eod_mask: False

      # eod_pad_length: 设置压缩后attention_mask的长度，当create_compressed_eod_mask为True时生效
      eod_pad_length: 128

      # eod: 数据集中eod(End of Document)的token id，用于标识文档结束，需要和训练模型中Tokenizer的eod id一致
      eod: 0

      # pad: 数据集中pad(填充)的token id，用于序列长度对齐，需要和训练模型中Tokenizer的pad id一致
      pad: 1

      # data_path: Megatron数据集采样比例以及路径，定义多个数据集的混合配置
      data_path:
        - '0.3'                          # 数据集1的占比，占总数据的30%
        - "/path/megatron_data1"         # 数据集1的bin文件路径（去除.bin后缀的完整文件名）
        - '0.7'                          # 数据集2的占比，占总数据的70%
        - "/path/megatron_data2"         # 数据集2的bin文件路径（去除.bin后缀的完整文件名）

  # input_columns: 输入列名，指定数据集中包含的字段名称
  # 套件内部会根据不同的场景自动生成，如果用户自定义DataLoader并手动指定，则会优先使用用户自定义的
  input_columns: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # construct_args_key: 构造参数键名，指定传给模型构造函数的参数名称
  # 套件内部会根据不同的场景自动生成，如果用户自定义网络入参，则会优先使用用户自定义的
  construct_args_key: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # drop_remainder: 是否丢弃最后一个不完整的batch，设置为True时会丢弃不足batch大小的数据
  drop_remainder: True

  # num_parallel_workers: 并行数据处理工作线程数，控制数据预处理的并行度
  num_parallel_workers: 8

  # python_multiprocessing: 是否启用Python多进程，设置为True时使用多进程处理数据
  python_multiprocessing: False

  # numa_enable: 是否启用NUMA优化，用于提升多核CPU环境下的数据处理性能
  numa_enable: False

  # prefetch_size: 数据预取大小，控制预加载到内存中的数据batch数量
  # 设置较大的值可以提高数据加载效率，减少训练等待时间，但会增加内存占用
  # 通常建议设置为1-4之间，根据可用内存和数据集大小进行调整
  prefetch_size: 1
```

#### 微调数据集

##### 在线加载数据集

```yaml
# Dataset configuration
train_dataset:
  # data_loader: 数据加载器配置，定义如何加载和处理训练数据
  data_loader:
    # type: 数据加载器类型，HFDataLoader用于加载HuggingFace格式的数据集
    type: HFDataLoader

    # load_func: 数据集加载函数名，指定使用哪个函数来加载数据集，此处复用HuggingFace Datasets的load_dataset函数
    load_func: 'load_dataset'

    # path: 数据集路径或名称，指定要加载的HuggingFace数据集ID 或者 已经下载好的数据集的格式指定
    path: "llm-wizard/alpaca-gpt4-data-zh" # HuggingFace数据集ID，自动下载该数据集
    # path: 'json' # 数据集路径或格式类型，当加载本地文件时可指定为'json'、'csv'等格式，配合data_files使用

    # data_files: 数据文件路径，指定要加载的本地数据文件的完整路径
    # 支持单个文件路径或文件路径列表，用于加载本地存储的数据集文件，此处代表的是已经离线下载好的Hugging Face数据集
    # data_files: '/path/alpaca-gpt4-data.json'

    # create_attention_mask: 是否创建注意力掩码，用于控制模型在自注意力计算中哪些位置可以被关注
    create_attention_mask: True

    # create_compressed_eod_mask: 是否创建压缩的EOD掩码，用于节省内存
    create_compressed_eod_mask: False

    # compressed_eod_mask_length: 压缩EOD掩码的长度，当create_compressed_eod_mask为True时生效
    compressed_eod_mask_length: 128

    # use_broadcast_data: 是否使用广播数据传输，在分布式训练中用于数据分发，分布式场景必须打开
    use_broadcast_data: True

    # shuffle: 是否对数据集进行随机打乱
    shuffle: False

    # handler: 数据处理管道，定义数据预处理的步骤序列
    handler:
      # take数据集采样操作：限制数据集大小，只取前n个样本
      # 通常用于快速测试或调试，正式训练时建议删除此配置项以使用完整数据集
      - type: take
        # n: 要保留的样本数量
        n: 2000

      # AlpacaInstructDataHandler操作：处理Alpaca格式的指令数据
      - type: AlpacaInstructDataHandler
        # seq_length: 序列长度，指定处理后每个样本的token序列长度
        seq_length: 4096
        # padding: 是否进行填充操作
        padding: False
        # tokenizer: 分词器配置
        tokenizer:
          # trust_remote_code: 是否信任远程代码，允许执行远程定义的分词器逻辑
          trust_remote_code: True
          # padding_side: 填充方向，'right'表示在右侧填充
          padding_side: 'right'

      # PackingHandler操作：将多个短序列打包成一个长序列以提高训练效率
      # 微调场景推荐使用，可以有效提高GPU利用率，减少padding浪费
      # 如果不需要序列打包功能，删除此配置项即可
      - type: PackingHandler
        # seq_length: 打包后的序列长度
        seq_length: 4096
        # pack_strategy: 打包策略，'pack'表示使用打包算法将多个短序列组合成长序列
        pack_strategy: 'pack'

  # input_columns: 输入列名，指定数据集中包含的字段名称，输入列名，指定数据集中包含的字段名称, 套件内部会根据不同的场景自动生成，如果用户自定义DataLoader并手动指定，则会优先使用用户自定义的
  input_columns: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # construct_args_key: 构造参数键名，指定传给模型构造函数的参数名称，输入列名，指定数据集中包含的字段名称, 套件内部会根据不同的场景自动生成，如果用户自定义网络入参，则会优先使用用户自定义的
  construct_args_key: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # drop_remainder: 是否丢弃最后一个不完整的batch，设置为True时会丢弃不足batch大小的数据
  drop_remainder: True

  # num_parallel_workers: 并行数据处理工作线程数，控制数据预处理的并行度
  num_parallel_workers: 8

  # python_multiprocessing: 是否启用Python多进程，设置为True时使用多进程处理数据
  python_multiprocessing: False

  # numa_enable: 是否启用NUMA优化，用于提升多核CPU环境下的数据处理性能
  numa_enable: False

  # prefetch_size: 数据预取大小，控制预加载到内存中的数据batch数量
  # 设置较大的值可以提高数据加载效率，减少训练等待时间，但会增加内存占用
  # 通常建议设置为1-4之间，根据可用内存和数据集大小进行调整
  prefetch_size: 1
```

##### 离线加载数据集

```yaml
# Dataset configuration
train_dataset:
  # data_loader: 数据加载器配置，定义如何加载和处理训练数据
  data_loader:
    # type: 数据加载器类型，HFDataLoader用于加载HuggingFace格式的数据集
    type: HFDataLoader

    # load_func: 数据集加载函数名，指定使用哪个函数来加载数据集
    # 'load_from_disk'表示从磁盘加载已处理好的数据集，通常用于加载HuggingFace Dataset保存下来处理好的本地数据集
    load_func: 'load_from_disk'

    # dataset_path: 数据集路径，指定已处理数据集在本地磁盘上的存储目录
    dataset_path: '/path/processed_dataset'

    # create_attention_mask: 是否创建注意力掩码，用于控制模型在自注意力计算中哪些位置可以被关注
    create_attention_mask: True

    # create_compressed_eod_mask: 是否创建压缩的EOD掩码，用于节省内存
    create_compressed_eod_mask: False

    # compressed_eod_mask_length: 压缩EOD掩码的长度，当create_compressed_eod_mask为True时生效
    compressed_eod_mask_length: 128

    # use_broadcast_data: 是否使用广播数据传输，在分布式训练中用于数据分发，分布式场景必须打开
    use_broadcast_data: True

    # shuffle: 是否对数据集进行随机打乱
    shuffle: False

    # handler: 数据处理管道，定义数据预处理的步骤序列
    handler:

      # AlpacaInstructDataHandler操作：处理Alpaca格式的指令数据
      - type: AlpacaInstructDataHandler
        # seq_length: 序列长度，指定处理后每个样本的token序列长度
        seq_length: 4096
        # padding: 是否进行填充操作
        padding: False
        # tokenizer: 分词器配置
        tokenizer:
          # trust_remote_code: 是否信任远程代码，允许执行远程定义的分词器逻辑
          trust_remote_code: True
          # padding_side: 填充方向，'right'表示在右侧填充
          padding_side: 'right'

      # PackingHandler操作：将多个短序列打包成一个长序列以提高训练效率
      # 微调场景推荐使用，可以有效提高GPU利用率，减少padding浪费
      # 如果不需要序列打包功能，删除此配置项即可
      - type: PackingHandler
        # seq_length: 打包后的序列长度
        seq_length: 4096
        # pack_strategy: 打包策略，'pack'表示使用打包算法将多个短序列组合成长序列
        pack_strategy: 'pack'

  # input_columns: 输入列名，指定数据集中包含的字段名称，输入列名，指定数据集中包含的字段名称, 套件内部会根据不同的场景自动生成，如果用户自定义DataLoader并手动指定，则会优先使用用户自定义的
  input_columns: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # construct_args_key: 构造参数键名，指定传给模型构造函数的参数名称，输入列名，指定数据集中包含的字段名称, 套件内部会根据不同的场景自动生成，如果用户自定义网络入参，则会优先使用用户自定义的
  construct_args_key: [ "input_ids", "labels", "loss_mask", "position_ids", "attention_mask" ]

  # drop_remainder: 是否丢弃最后一个不完整的batch，设置为True时会丢弃不足batch大小的数据
  drop_remainder: True

  # num_parallel_workers: 并行数据处理工作线程数，控制数据预处理的并行度
  num_parallel_workers: 8

  # python_multiprocessing: 是否启用Python多进程，设置为True时使用多进程处理数据
  python_multiprocessing: False

  # numa_enable: 是否启用NUMA优化，用于提升多核CPU环境下的数据处理性能
  numa_enable: False

  # prefetch_size: 数据预取大小，控制预加载到内存中的数据batch数量
  # 设置较大的值可以提高数据加载效率，减少训练等待时间，但会增加内存占用
  # 通常建议设置为1-4之间，根据可用内存和数据集大小进行调整
  prefetch_size: 1
```

### 并行配置

```yaml
# Parallel configuration --> 直接使用ModelParallelConfig中定义使能的配置
# 并行配置，用于定义模型训练过程中的各种并行策略
distribute_parallel_config:
  # data_parallel_size: 数据并行大小，控制在不同设备上复制相同模型的并行度
  # null表示自动根据TP、PP、CP的设置和总设备数计算得到
  data_parallel_size: null

  # tensor_model_parallel_size: 张量模型并行大小，将模型的权重分割到多个设备上
  # 1表示不使用张量并行
  tensor_model_parallel_size: 1  # Number of model parallel

  # pipeline_model_parallel_size: 流水线模型并行大小，将模型的不同层分配到不同设备上
  # 1表示不使用流水线并行，需要配合set_auto_parallel_context使用
  pipeline_model_parallel_size: 1  # Number of pipeline parallel  --> set_auto_parallel_context

  # context_parallel_size: 上下文并行大小，用于处理长序列的并行计算
  context_parallel_size: 1

  # cp_comm_type: 上下文并行通信类型
  # 支持ulysses_cp->all_to_all/colossalai_cp->all_gather两种通信方式
  cp_comm_type: all_gather  # 支持ulysses_cp->all_to_all/colossalai_cp->all_gather

  # expert_model_parallel_size: 专家并行大小，用于MoE（Mixture of Experts）模型
  # 1表示不使用专家并行
  expert_model_parallel_size: 1 # 专家并行

  # sequence_parallel: 是否启用序列并行
  # False表示不启用序列并行
  sequence_parallel: False  # Whether to enable sequence parallelism

  # pipeline_parallel_config: 流水线并行配置
  pipeline_parallel_config:
    # pipeline_interleave: 是否启用流水线交错
    # False表示不启用流水线交错
    pipeline_interleave: False

    # pipeline_scheduler: 流水线调度器类型，pipeline_interleave=True生效
    # 验证支持范围（默认行为是1f1b），支持gpipe/1f1b/zero_bubble_v等调度策略
    pipeline_scheduler: "1f1b"

    # virtual_pipeline_model_parallel_size: 虚拟流水线模型并行大小
    # 只有1f1b模式生效
    virtual_pipeline_model_parallel_size: 1 # 只有1f1b模式生效

    # pipeline_stage_offset: 流水线阶段偏移量
    # pp stage的偏移量。控制不同stage的内存负载
    pipeline_stage_offset: 0

  # optimizer_parallel_config: 优化器并行配置
  optimizer_parallel_config:
    # enable_parallel_optimizer: 是否启用并行优化器
    # False表示不启用优化器并行
    enable_parallel_optimizer: False

    # optimizer_level: 优化器并行级别
    # level1表示优化器并行的级别
    optimizer_level: level1

    # optimizer_weight_shard_size: 优化器权重分片大小
    # -1表示自动分配
    optimizer_weight_shard_size: -1

  # gradient_aggregation_group: 梯度聚合组大小
  # 1表示梯度聚合的组大小
  gradient_aggregation_group: 1

  # micro_batch_interleave_num: 微批次交错数量
  # When model_parallel > 1, setting micro_batch_interleave_num to 2 may accelerate the training process.
  # 当模型并行>1时，设置micro_batch_interleave_num为2可能加速训练过程
  micro_batch_interleave_num: 1
```

### 重计算配置

```yaml
# Recomputation configuration
# 重计算配置，用于节省显存但会增加计算时间的训练优化技术
recompute_config:
  # recompute: 是否启用重计算以节省内存
  # 设置为True时启用重计算，可以显著减少显存占用，但会增加计算时间
  recompute: True # bool\list\tuple

  # select_recompute: 是否启用选择性重计算
  # 设置为True时启用选择性重计算，只对部分层或者指定的部分层进行重计算
  select_recompute: False # bool\list

  # select_recompute_exclude: 是否排除某些层的选择性重计算
  # 设置为True时排除指定层或指定对应层不进行选择性重计算
  select_recompute_exclude: False # bool\list

  # select_comm_recompute_exclude: 是否排除通信操作的重计算
  # 设置为True时排除通信操作不进行重计算或者指定通信算子不进行重计算
  select_comm_recompute_exclude: False # bool\list

  # parallel_optimizer_comm_recompute: 是否重计算优化器并行通信
  # 设置为True时对优化器并行通信进行重计算
  parallel_optimizer_comm_recompute: False

  # mp_comm_recompute: 是否重计算模型并行通信
  # 设置为True时对模型并行通信进行重计算，通常用于节省显存
  mp_comm_recompute: True
```

### CPU OffLoad配置

```yaml
# CPU OffLoad配置(Swap Configuration)，用于控制模型训练过程中的内存卸载机制
swap_config:
  # swap: 是否启用交换内存功能
  # 设置为True时启用内存交换，将部分模型参数交换到CPU内存以节省显存
  swap: False

  # default_prefetch: 默认预取数量
  # 控制在交换模式下预取到网络的参数的数量
  default_prefetch: 1

  # layer_swap: 层内存卸载配置
  # null表示不进行特定层的内存卸载配置，可配置为List[Dict]类型来指定特定层的内存卸载策略
  layer_swap: null

  # op_swap: 操作交换配置
  # null表示不进行特定操作的内存卸载配置，可配置为List[Dict]类型来指定特定操作的内存卸载策略
  op_swap: null
```

### 训练监控配置

```yaml
# monitor_config: 训练监控配置，用于设置训练过程中的各种监控指标和日志记录选项
monitor_config:
    # dump_path: 监控数据转储路径，指定监控数据保存的目录
    dump_path: "./dump"

    # local_loss: 本地损失监控配置，指定损失值的记录方式
    # 支持'log'(日志记录)和'tensorboard'(TensorBoard可视化)两种方式
    local_loss: ['log', 'tensorboard']

    # device_local_loss: 设备本地损失监控配置，指定设备上损失值的记录方式
    # 支持'log'(日志记录)和'tensorboard'(TensorBoard可视化)两种方式
    device_local_loss: ['log', 'tensorboard']

    # device_local_norm: 设备本地范数监控配置，指定设备上梯度范数的记录方式
    # 支持'log'(日志记录)和'tensorboard'(TensorBoard可视化)两种方式
    device_local_norm: ['log', 'tensorboard']

    # local_norm: 本地范数监控配置，指定要记录的参数范数
    # null表示不记录任何范数，List[str]表示记录指定参数的范数
    local_norm: null # null/List[str] null代表不记录

    # optimizer_params_state: 优化器参数状态监控配置，指定要记录的优化器参数状态
    # null表示不记录任何优化器参数状态，List[str]表示记录指定优化器参数的状态
    optimizer_params_state: null # null/List[str] null代表不记录

    # net_weight_params_state: 网络权重参数状态监控配置，指定要记录的网络权重参数状态
    # null表示不记录任何网络权重参数状态，List[str]表示记录指定网络权重参数的状态
    net_weight_params_state: null # null/List[str] null代表不记录

    # target_parameters: 目标参数监控配置，指定要监控的参数模式
    # 使用正则表达式匹配参数名称，仅在local_norm、optimizer_params_state开启时生效
    target_parameters: [".*"] # 监控target_parameters指定的参数，仅支持在local_norm、optimizer_params_state开启情形下生效

    # target_parameters_invert: 目标参数监控取反配置，控制是否排除指定参数的监控
    # 设置为False时监控target_parameters指定的参数，在target_parameters不为None或不为[]时生效
    target_parameters_invert: False # 不监控target_parameters指定的参数，在target_parameters不为None或不为[]时生效

    # embedding_local_norm: Embedding层本地范数监控开关
    # 设置为True时专门监控Embedding层的本地范数，推荐使用local_norm配合target_parameters替换该功能
    embedding_local_norm: False # bool 仅用于打印embedding_local_norm，推荐可以使用local_norm配合target_parameters替换该功能

    # step_interval: 步数间隔，控制日志打印或监控信息落盘的频率
    step_interval: 1 # 日志打印或监控信息落盘的频率
```

### TensorBoard可视化配置

```yaml
# tensorboard: TensorBoard可视化配置，用于控制训练过程中的指标可视化记录
tensorboard:
    # tensorboard_on: 是否启用TensorBoard记录功能
    # 设置为True时启用TensorBoard记录，将训练指标写入TensorBoard日志文件
    tensorboard_on: False

    # tensorboard_dir: TensorBoard日志文件存储目录
    # 指定TensorBoard日志文件的保存路径
    tensorboard_dir: './tensorboard'

    # tensorboard_queue_size: TensorBoard队列大小
    # 控制TensorBoard事件写入队列的大小，用于缓冲日志数据
    tensorboard_queue_size: 10

    # log_loss_scale_to_tensorboard: 是否将损失缩放因子记录到TensorBoard
    # 设置为True时会将损失缩放因子的变化记录到TensorBoard中
    log_loss_scale_to_tensorboard: False

    # log_timers_to_tensorboard: 是否将计时器信息记录到TensorBoard
    # 设置为True时会将训练过程中的时间统计信息记录到TensorBoard中
    log_timers_to_tensorboard: False

    # log_expert_load_to_tensorboard: 是否将专家负载信息记录到TensorBoard
    # 设置为True时会将MoE模型中专家负载均衡信息记录到TensorBoard中
    log_expert_load_to_tensorboard: False
```

### Profile配置

```yaml
# 训练profile: 性能分析配置，用于对训练过程进行性能分析和瓶颈定位
profile:
  # profile_on: 是否启用性能分析功能
  # 设置为True时启用性能分析，收集训练过程中的性能数据
  profile_on: False

  # profile_output: 性能分析结果输出路径
  # 指定性能分析结果的保存目录，null表示使用默认路径
  profile_output: null

  # profiler_level: 性能分析级别
  # 控制性能分析的详细程度，数值越大分析越详细
  profiler_level: 1

  # profile_start_step: 性能分析开始步数
  # 指定从第几步开始进行性能分析
  profile_start_step: 1

  # profile_stop_step: 性能分析结束步数
  # 指定在第几步停止性能分析
  profile_stop_step: 10

  # init_start_profile: 是否在初始化阶段就开始性能分析
  # 设置为True时在训练初始化阶段就开始收集性能数据
  init_start_profile: False

  # profile_rank_ids: 指定进行性能分析的设备ID列表
  # null表示对所有设备进行分析，List[int]表示只对指定设备进行分析
  profile_rank_ids: null # List[int]

  # profile_pipeline: 是否分析流水线并行性能
  # 设置为True时收集流水线并行相关的性能数据
  profile_pipeline: False

  # profile_communication: 是否分析通信性能
  # 设置为True时收集设备间通信相关的性能数据
  profile_communication: False

  # profile_memory: 是否分析内存使用情况
  # 设置为True时收集内存使用相关的性能数据
  profile_memory: True

  # with_stack: 是否收集调用栈信息
  # 设置为True时收集函数调用栈信息，用于详细定位性能瓶颈
  with_stack: False

  # data_simplification: 是否简化数据
  # 设置为True时对收集的性能数据进行简化处理
  data_simplification: False

  # mstx: 是否启用MSTX分析
  # 设置为True时启用MSTX(MindSpore Tensor eXpression)性能分析
  mstx: False

  # use_llm_token_profile: 是否启用LLM文本数据token分布分析
  # 设置为True时收集LLM训练中文本token的分布情况，需配合llm_token_profile_config使用
  use_llm_token_profile: False # LLM文本数据token分布采集，配合llm_token_profile_config使用

  # llm_token_profile_config: LLM token分析配置
  # null表示使用默认配置，用于控制LLM文本数据token分布采集的具体参数
  llm_token_profile_config: null
```

### 运行环境配置

#### Context

```yaml
# MindSpore context initialization configuration, reference: https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore/mindspore.set_context.html
# MindSpore上下文初始化配置，用于设置运行时环境参数
context:
  # mode: 运行模式，0表示图模式(Graph Mode)
  # 训练仅支持图模式执行
  mode: 0

  # device_id: 设备ID，指定要使用的硬件设备编号
  device_id: 0

  # device_target: 目标设备类型，指定运行的目标硬件平台
  # 仅支持"Ascend"昇腾设备
  device_target: "Ascend"  # Target device to run (only supports "Ascend")

  # max_device_memory: 设备最大内存限制，指定设备可用的最大内存容量
  # 不同的昇腾设备内存大小不同，请根据实际情况设置
  # 单设备32GB通常设置为<=29GB，单设备64GB通常设置为<=59GB，需要预留部分内存给系统使用
  max_device_memory: "58GB"  # Maximum memory available for the device

  # mempool_block_size: 内存池块大小，控制内存分配的块大小
  # 用于设置内存池中每个内存块的大小，较大的块可以减少内存碎片但可能增加内存使用
  # 最大可设置为max_device_memory同等大小，最大程度复用碎片内存整理
  mempool_block_size: "1GB"

  # memory_optimize_level: 内存优化级别，控制内存优化的程度
  # "O0"表示基础内存优化级别，不进行额外的内存优化
  # "O1"表示进行内存整理优化，可以减少内存碎片，提升内存利用率
  memory_optimize_level: "O0"

  # jit_config: 全局JIT编译配置，用于控制编译时优化
  jit_config:
    # jit_level: 编译优化级别，控制编译优化的程度
    # O0: 除必要影响功能的优化外，其他优化均关闭, O1: 使能常用优化和自动算子融合优化。
    jit_level: "O0"  # Compilation optimization level

  # ascend_config: 昇腾硬件平台特定参数配置
  ascend_config:  # Parameters specific to the Ascend hardware platform
    # parallel_speed_up_json_path: 并行加速JSON文件路径
    # 指向并行优化配置的JSON文件路径，可以参考官方文档:
    # https://gitee.com/mindspore/mindspore/blob/master/config/parallel_speed_up.json
    parallel_speed_up_json_path: "./configs/model_path/parallel_speed_up.json"  # Path to the parallel speedup JSON file
```

#### Parallel Context

```yaml
# Parallel context configuration
# 并行上下文配置，用于设置分布式训练的并行策略和通信模式
parallel:
  # parallel_mode: 并行模式，控制训练过程中的并行策略
  # 0--data parallel(数据并行); 1--semi-auto parallel(半自动并行); 2--auto parallel(自动并行)
  # 仅支持0、1、2三种模式, 不论模型大小，并行模式下优先使用semi-auto parallel模式
  # auto parallel模式需配合search_mode="sharding_propagation"使用
  parallel_mode: 1

  # enable_alltoall: 是否启用AllToAll通信操作
  # 设置为True时在并行通信中启用AllToAll通信算子
  enable_alltoall: True

  # search_mode: 并行策略搜索模式
  # "sharding_propagation"表示使用全自动的并行策略搜索模式
  search_mode: "sharding_propagation"
```

### 模型配置

## 推理配置说明

### 基础配置

```yaml
# output_dir: 推理输出目录，MindFormers Logger日志的推理文件
# 建议使用绝对路径或相对于项目根目录的相对路径
output_dir: './output'

# run_mode: 运行模式，指定当前任务的执行模式
# 可选值: 'predict'(推理预测模式)
# 不同模式下会加载不同的组件和执行不同的训练/推理流程
run_mode: 'predict'

# use_parallel: 是否启用并行推理功能
# 设置为True时启用分布式并行推理(包括数据并行、张量并行等)
# 设置为False时使用单卡推理模式，适用于调试或小规模推理场景
use_parallel: True

# predict_batch_size: 推理批次大小
# 指定每次推理处理的样本数量
predict_batch_size: 1

# pretrained_model_dir: HuggingFace模型文件目录，用于直接读取HF的模型配置、词表、权重等内容
# 指定预训练模型的本地路径，支持从HuggingFace格式的模型文件加载权重和配置
# 路径应包含config.json、tokenizer.json、model.safetensors等标准HF模型文件
pretrained_model_dir: '/path/hf_dir'

# load_checkpoint: 加载检查点路径
# 指定模型权重文件的路径，空字符串表示不加载，pretrained_model_dir有效则会优先使用pretrained_model_dir下的权重
# 当仅当load_checkpoint == “not_load_any_ckpt”时, 不会加载任何权重，包括pretrained_model_dir下的
# “not_load_any_ckpt”字段场景通常用于pretrained_model_dir有效的情况下，复用其下除权重之外的文件
load_checkpoint: ''

# infer_seed: 推理时的种子数
# 用于确保推理过程的可复现性
infer_seed: 1234

# infer_precision_sync: 推理确定性计算开关
# True表示启用确定性计算保证结果一致，False表示非确定性计算
infer_precision_sync: False

# context: MindSpore上下文初始化配置
# 显示写出来支持的context配置，推理场景下能支持的内容
context:
  # mode: 运行模式
  # 0表示图模式(Graph Mode)，1表示动态图模式(Pynative Mode)
  mode: 0 #0--Graph Mode; 1--Pynative Mode

  # max_device_memory: 设备最大内存限制，指定设备可用的最大内存容量
  # 不同的昇腾设备内存大小不同，请根据实际情况设置
  # 单设备32GB通常设置为<=29GB，单设备64GB通常设置为<=59GB，需要预留部分内存给系统使用
  max_device_memory: "59GB"  # Maximum memory available for the device

  affinity_cpu_list: None

# parallel: 并行上下文配置
parallel:
  # parallel_mode: 并行模式
  # "MANUAL_PARALLEL"表示手动并行模式，LLM推理默认走手切模式
  parallel_mode: "MANUAL_PARALLEL"

  # enable_alltoall: 是否启用AllToAll通信
  # False表示不启用AllToAll通信操作，通常配合MOE模型专家并行使用
  enable_alltoall: False

# Trainer配置，用于指定训练器类型和任务相关信息
trainer:
  # type: 训练器类型，使用新的LLMTrainer简化推理流程
  # LLMTrainer是专门为大语言模型设计的Trainer，提供简化的推理逻辑和优化
  type: LLMTrainer # 新配置格式 新Trainer流程 简化逻辑

  # task_name: 任务名称，用户可自定义，如qwen3、glm4等
  # 用于标识当前训练任务的名称，字符内容没有限制，建议使用模型名称便于识别和管理
  task_name: 'llm_model'  # qwen3 glm4等，用户可自己定义，字符内容没有限制
```

### 并行配置

```yaml
distribute_parallel_config:
  # data_parallel_size: 数据并行大小，控制在不同设备上复制相同模型的并行度
  # null表示自动根据TP的设置和总设备数计算得到
  data_parallel_size: null

  # tensor_model_parallel_size: 张量模型并行大小，将模型的权重分割到多个设备上
  # 1表示不使用张量并行
  tensor_model_parallel_size: 1  # Number of model parallel

  # pipeline_model_parallel_size: 流水线模型并行大小，将模型的不同层分配到不同设备上
  # 1表示不使用流水线并行，需要配合set_auto_parallel_context使用
  pipeline_model_parallel_size: 1  # Number of pipeline parallel  --> set_auto_parallel_context

  # expert_model_parallel_size: 专家并行大小，用于MoE（Mixture of Experts）模型
  # 1表示不使用专家并行
  expert_model_parallel_size: 1 # 专家并行
```

### 模型配置

