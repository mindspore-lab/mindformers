mindformers.wrapper.MFPipelineWithLossScaleCell
===============================================

.. py:class:: mindformers.wrapper.MFPipelineWithLossScaleCell(network, optimizer, use_clip_grad=True, max_grad_norm=1.0, scale_sense=1.0, micro_batch_num=1, local_norm=False, **kwargs)

    为MindFormers的单步训练单元扩充流水线并行的损失缩放功能。

    参数：
        - **network** (Cell) - 训练网络，已包含损失函数。
        - **optimizer** (Optimizer) - 用于更新权重的优化器。
        - **use_clip_grad** (bool, 可选) - 是否使用梯度裁剪功能。默认值： ``True`` 。
        - **max_grad_norm** (float, 可选) - 最大梯度约束值。默认值： ``1.0`` 。
        - **scale_sense** (Union[Tensor, Cell], 可选) - 用于损失缩放的 Cell 实例或 Tensor。默认值： ``1.0`` 。
        - **micro_batch_num** (int, 可选) - 流水线并行的微批次数。默认值： ``1`` 。
        - **local_norm** (bool, 可选) - 是否计算局部范数。默认值： ``False`` 。
        - **kwargs** (Any) - 其他参数。

    输入：
        - **(\*inputs)** (Tuple(Tensor)) - 形状为 :math:`(N, \ldots)` 的输入张量元组。

    输出：
        3个张量的元组，包括损失值、溢出标志和当前的损失缩放值：

        - **loss** (Tensor) -  损失值（标量）。
        - **overflow** (Tensor) -  是否发生溢出（布尔值）。
        - **loss scale** (Tensor) -  损失缩放值，形状为 :math:`()` 或 :math:`(1,)`。

    异常：
        - **TypeError** - 如果 `scale_sense` 既不是 Cell 也不是 Tensor。
        - **ValueError** - 如果 `scale_sense` 的形状既不是 `(1,)` 也不是 `()`。
        - **ValueError** - 如果并行模式不是 [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL] 之一。
