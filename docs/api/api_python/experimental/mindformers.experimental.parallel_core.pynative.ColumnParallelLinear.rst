mindformers.experimental.parallel_core.pynative.ColumnParallelLinear
=====================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.ColumnParallelLinear(input_size, output_size, *, config, init_method, bias=True, gather_output=False, stride=1, keep_master_weight_for_test=False, skip_bias_add=False, skip_weight_param_allocation=False, embedding_activation_buffer=None, grad_output_buffer=None, is_expert=False, tp_comm_buffer_name=None, disable_grad_reduce=False, bias_init=Zero(), param_init_dtype=None, compute_dtype=None, transpose_b=True)

    稠密线性层计算，在权重的第二个维度按照张量并行大小进行切分，并行计算。

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias}

    其中， :math:`inputs` 代表输入张量， :math:`\text{weight}` 代表该线性层的权重矩阵， :math:`\text{bias}` 代表该线性层的偏置向量（当且仅当 `has_bias` 为 ``True`` 时会参与计算）。

    参数：
        - **input_size** (int) - 输入空间的通道数。
        - **output_size** (int) - 输出空间的通道数。
        - **config** (dict) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **init_method** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练参数的初始化方式。若传入值类型为字符串，则对应 `initializer` 的函数名。
        - **bias** (bool) - 指定该层是否使用偏置向量。默认值： ``True`` 。
        - **gather_output** (bool) - 指定是否要在每个张量并行进程计算完成后进行聚合。默认值： ``False`` 。
        - **stride** (int) - 描述线性层计算的步长。默认值： ``1`` 。
        - **keep_master_weight_for_test** (bool) - 用于测试，正常使用时应当设置为 ``False`` 。该参数会返回用于初始化的主权重。默认值： ``False`` 。
        - **skip_bias_add** (bool) - 如果为 ``True`` ，则计算式不会加上偏置，而是会将偏置返回用于后续的融合计算。默认值： ``False`` 。
        - **skip_weight_param_allocation** (bool) - 指定是否跳过权重参数初始化。当设置为 ``True`` 时，需往前向接口传入一个权重张量。默认值： ``False`` 。
        - **embedding_activation_buffer** (Tensor) - 该缓冲区存放了最后一个流水线阶段的最后一个线性嵌入层的输入激活值。默认值： ``None`` 。
        - **grad_output_buffer** (Tensor) - 该缓冲区存放了最后一个流水线阶段的最后一个线性嵌入层的输出梯度。默认值： ``None`` 。
        - **is_expert** (bool) - 指定该线性层是否为专家。默认值： ``False`` 。
        - **tp_comm_buffer_name** (str) - 通信缓冲区在非Transformer引擎模块中不会被使用。默认值： ``None`` 。
        - **disable_grad_reduce** (bool) - 如果设置为 ``True`` ，将不会使能跨张量并行进程的输出梯度聚合。默认值： ``False`` 。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的偏置参数初始化方法。若传入值类型为字符串，则对应 `initializer` 的函数名。默认值： ``Zero()`` 。
        - **param_init_dtype** (dtype.Number) - 参数初始化类型。默认值： ``None`` 。
        - **compute_dtype** (dtype.Number) - 计算类型。默认值： ``None`` 。
        - **transpose_b** (bool) - 指定是否将权重参数初始化为转置矩阵。默认值： ``True`` 。

    输入：
        - **input\_** (Tensor) - 形状为 :math:`(*, in\_channels)` 的输入张量。接口参数中的 `input_size` 需与 :math:`in\_channels` 一致。
        - **weight** (Tensor) - 形状为 :math:`(in\_channels, out\_channels)` 的张量。接口参数中的 `input_size` 需与 :math:`in\_channels` 一致。接口参数中的 `output_size` 需与 :math:`out\_channels` 一致。

    输出：
        形状为 :math:`(*, out\_channels)` 的张量。接口参数中的 `output_size` 需与 :math:`out\_channels` 一致。

    异常：
        - **ValueError** - `skip_weight_param_allocation=True` 但是权重张量并未传给前向函数。
        - **NotImplementedError** - 当前不支持 `stride > 1` 。
        - **NotImplementedError** - 当前不支持 `keep_master_weight_for_test=True` 。
        - **NotImplementedError** - 当前不支持 `embedding_activation_buffer` 。
        - **NotImplementedError** - 当前不支持 `grad_output_buffer` 。
        - **NotImplementedError** - 当前不支持 `tp_comm_buffer_name` 。
        - **NotImplementedError** - 当前不支持 `disable_grad_reduce=True` 。
        - **NotImplementedError** - 当前不支持 `config.parallel_config.use_cpu_initialization` 。
        - **RuntimeError** - 使用了 `zero3` 优化器并行，但是未初始化数据并行通信。
        - **RuntimeError** - `allreduce_dgrad` 和 `sequence_parallel` 不能同时使能。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst