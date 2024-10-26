mindformers.core.MFLossMonitor
==============================

.. py:class:: mindformers.core.MFLossMonitor(learning_rate: Optional[Union[float, LearningRateSchedule]] = None, per_print_times: int = 1, micro_batch_num: int = 1, micro_batch_interleave_num: int = 1, origin_epochs: int = None, dataset_size: int = None, initial_epoch: int = 0, initial_step: int = 0, global_batch_size: int = 0, gradient_accumulation_steps: int = 1, enable_tensorboard: bool = False, tensorboard_path: str = None, check_for_nan_in_loss_and_grad: bool = False)

    监控训练过程中loss等相关参数的回调函数。

    参数：
        - **learning_rate** (Union[float, LearningRateSchedule], optional) - 学习率调度器。默认值： ``None`` 。
        - **per_print_times** (int) - 每多少次step打印日志信息。默认值： ``1`` 。
        - **micro_batch_num** (int) - 流水线并行时设置的MicroBatch大小。默认值： ``None`` 。
        - **micro_batch_interleave_num** (int) - interleaved pipeline流水线并行时设置的MicroBatch大小。默认值： ``1`` 。
        - **origin_epochs** (int) - 训练的epoch数量。默认值： ``None`` 。
        - **dataset_size** (int) - 训练的数据集数量。默认值： ``None`` 。
        - **initial_epoch** (int) - 训练开始的epoch数。默认值： ``0`` 。
        - **initial_step** (int) - 训练开始的step数。默认值： ``0`` 。
        - **global_batch_size** (int) - 总BatchSize大小。默认值： ``0`` 。
        - **gradient_accumulation_steps** (int) - 梯度累加步数。默认值： ``1`` 。
        - **enable_tensorboard** (bool) - 是否在训练中启用 TensorBoard 日志记录。默认值： ``False`` 。
        - **tensorboard_path** (str) - 如果启用了 TensorBoard，则该路径指定TensorBoard 日志文件保存的路径。如果设置为 None，则使用默认输出路径（例如 './output'）。默认值：None。
        - **check_for_nan_in_loss_and_grad** (bool) - 是否检查损失和梯度存在Nan。默认值： ``False`` 。
