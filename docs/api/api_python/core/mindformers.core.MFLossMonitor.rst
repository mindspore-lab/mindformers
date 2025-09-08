mindformers.core.MFLossMonitor
==============================

.. py:class:: mindformers.core.MFLossMonitor(learning_rate=None, per_print_times=1, micro_batch_num=1, micro_batch_interleave_num=1, origin_epochs=None, dataset_size=None, initial_epoch=0, initial_step=0, global_batch_size=0, gradient_accumulation_steps=1, check_for_nan_in_loss_and_grad=False, calculate_per_token_loss=False)

    监控训练过程中loss等相关参数的回调函数。

    参数：
        - **learning_rate** (Union[float, LearningRateSchedule], 可选) - 学习率调度器。默认值： ``None`` 。
        - **per_print_times** (int, 可选) - 每多少次step打印日志信息。默认值： ``1`` 。
        - **micro_batch_num** (int, 可选) - 流水线并行时设置的MicroBatch大小。默认值： ``1`` 。
        - **micro_batch_interleave_num** (int, 可选) - interleaved pipeline流水线并行时设置的MicroBatch大小。默认值： ``1`` 。
        - **origin_epochs** (int, 可选) - 训练的epoch数量。默认值： ``None`` 。
        - **dataset_size** (int, 可选) - 训练的数据集数量。默认值： ``None`` 。
        - **initial_epoch** (int, 可选) - 训练开始的epoch数。默认值： ``0`` 。
        - **initial_step** (int, 可选) - 训练开始的step数。默认值： ``0`` 。
        - **global_batch_size** (int, 可选) - 总BatchSize大小。默认值： ``0`` 。
        - **gradient_accumulation_steps** (int, 可选) - 梯度累加步数。默认值： ``1`` 。
        - **check_for_nan_in_loss_and_grad** (bool, 可选) - 是否检查损失和梯度存在Nan。默认值： ``False`` 。
        - **calculate_per_token_loss** (bool, 可选) - 是否计算每个token的loss。默认值： ``False`` 。
        - **print_separate_loss** (bool, 可选) - 是否分开打印loss。默认值： ``False`` 。
