mindformers.core.MFLossMonitor
==============================

.. py:class:: mindformers.core.MFLossMonitor(learning_rate: Optional[Union[float, LearningRateSchedule]] = None, per_print_times: int = 1, micro_batch_num: int = 1, micro_batch_interleave_num: int = 1, origin_epochs: int = None, dataset_size: int = None, initial_epoch: int = 0, initial_step: int = 0, global_batch_size: int = 0, gradient_accumulation_steps: int = 1)

    监控训练过程中loss等相关参数的回调函数。

    参数：
        - **learning_rate** (Union[float, LearningRateSchedule], optional) - 学习率调度器。默认值： ``None`` 。
        - **per_print_times** (int) - 每多少次step打印日志信息。默认值： ``1`` 。
        - **micro_batch_num** (int) - 流水线并行时设置的MicroBatch大小。默认值： ``None`` 。
        - **micro_batch_interleave_num** (int) - interleaved pipeline流水线并行时设置的MicroBatch大小。默认值： ``1`` 。
        - **origin_epochs** (int) - 训练的epoch数量。默认值： ``None`` 。
        - **dataset_size** (int) - 训练的数据集数量。默认值： ``None`` 。
        - **initial_epoch** (int) - 训练开始的epoch数。默认值： ``0`` 。
        - **global_batch_size** (int) - 总BatchSize大小。默认值： ``0`` 。
        - **device_num** (int) - 设备数量。默认值： ``0`` 。