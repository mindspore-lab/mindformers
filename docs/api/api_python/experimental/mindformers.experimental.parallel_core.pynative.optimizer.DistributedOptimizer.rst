mindformers.experimental.parallel_core.pynative.optimizer.DistributedOptimizer
==============================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.optimizer.DistributedOptimizer(optimizer, config, grad_scaler, init_state_fn, per_model_buffers, data_parallel_group)

    这个类构建了在优化器并行性中由这个数据并行等级负责的参数范围与它们在缓冲区、分片桶、集成桶和集成参数中的分片范围之间的映射。这个映射是将模型参数索引与主参数分片索引之间进行转换所必需的。这个类还根据参数分片信息更新非并行优化器属性。

    参数：
        - **optimizer** (mindspore.mint.optim) - 非并行优化器。
        - **config** (dict) - 包含优化器相关配置的OptimizerConfig对象。
        - **grad_scaler** (GradScaler) - 梯度缩放。当它为None时，不会对梯度使用缩放器。
        - **init_state_fn** (Function) - 用于初始化优化器的状态参数的函数。
        - **per_model_buffers** (List) - 所有模型块的缓冲区列表。
        - **data_parallel_group** (str) - 数据并行组名称。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst