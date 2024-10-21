mindformers.experimental.parallel_core.pynative.distributed.DistributedDataParallel
===================================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.distributed.DistributedDataParallel(config, ddp_config, module, disable_bucketing=False)

    分布式数据并行包装器（DistributedDataParallel wrapper）。分布式数据并行会为参数和梯度分配连续的内存缓冲区。它还支持梯度反向传播计算和通信。当启用重叠时，参数和梯度将被分解成桶，这是在数据并行组中进行all-reduce/reduce-scatter通信的单位。

    参数：
        - **config** (TrainingConfig) - 包含训练相关配置的TrainingConfig对象。
        - **ddp_config** (DistributedDataParallelConfig) - 包含分布式数据并行相关配置的DistributedDataParallelConfig对象。
        - **module** (Module) - 要被DDP包装的模块。
        - **disable_bucketing** (bool) - 禁用桶化，这意味着所有参数和梯度将被分配到一个桶中。默认值：``False``。

    输出：
        用DistributedDataParallel包装的模型。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst