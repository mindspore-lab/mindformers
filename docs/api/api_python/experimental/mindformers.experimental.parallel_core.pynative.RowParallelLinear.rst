mindformers.experimental.parallel_core.pynative.RowParallelLinear
=====================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.RowParallelLinear(input_size, output_size, *, config, init_method, bias, input_is_parallel, skip_bias_add=True, stride=1, keep_master_weight_for_test=False, is_expert=False, tp_comm_buffer_name=None, bias_init=Zero(), param_init_dtype=None, compute_dtype=None, transpose_b=True)

    稠密线性层计算，在权重的第一个维度按照张量并行大小进行切分，并行计算。
    该层实现的计算公式为：

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias}
    
    其中， :math:`inputs` 代表输入张量， :math:`\text{weight}` 代表该线性层的权重矩阵， :math:`\text{bias}` 代表该线性层的偏置向量（当且仅当 `has_bias` 为 ``True`` 时会参与计算）。
    
    参数：
        - **input_size** (int) - 输入空间的通道数。
        - **output_size** (int) - 输出空间的通道数。
        - **config** (dict) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **init_method** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练参数的初始化方式。若传入值类型为字符串，则对应 `initializer` 的函数名。
        - **bias** (bool) - 指定该层是否使用偏置向量。默认值： ``True`` 。
        - **input_is_parallel** (bool) - 指定输入张量是否已经按照张量并行策略进行了切分，若是，那么我们就不用再进行切分了。
        - **skip_bias_add** (bool) - 如果为 ``True`` ，则计算式不会加上偏置，而是会将偏置返回用于后续的融合计算。默认值： ``False`` 。
        - **stride** (int) - 描述线性层计算的步长。默认值： ``1`` 。
        - **keep_master_weight_for_test** (bool) - 用于测试，正常使用时应当设置为 ``False`` 。该参数会返回用于初始化的主权重。默认值： ``False`` 。
        - **is_expert** (bool) - 指定该线性层是否为专家。默认值： ``False`` 。
        - **tp_comm_buffer_name** (str) - 通信缓冲区在非Transformer引擎模块中不会被使用。默认值： ``None`` 。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的偏置参数初始化方法。若传入值类型为字符串，则对应 `initializer` 的函数名。默认值： ``Zero()`` 。
        - **param_init_dtype** (dtype.Number) - 参数初始化类型。默认值： ``None`` 。
        - **compute_dtype** (dtype.Number) - 计算类型。默认值： ``None`` 。
        - **transpose_b** (bool) - 指定是否将权重参数初始化为转置矩阵。默认值： ``True`` 。

    输入：
        - **input\_** (Tensor) - 形状为 :math:`(*, in\_channels)` 的输入张量。接口参数中的 `input_size` 需与 :math:`in\_channels` 一致。

    输出：
        形状为 :math:`(*, out\_channels)` 的张量。接口参数中的 `output_size` 需与 :math:`out\_channels` 一致。

    异常：
        - **ValueError** - 当 `input_is_parallel` 为 ``False`` 时， `sequence_parallel` 应当为 ``False`` ，但是被设置为了 ``True`` 。
        - **ValueError** - 当 `explicit_expert_comm` 为 ``True`` 时， `skip_bias_add` 应当为 ``True`` ，但是被设置为了 ``False`` 。
        - **NotImplementedError** - 当前不支持 `stride > 1` 。
        - **NotImplementedError** - 当前不支持 `keep_master_weight_for_test=True` 。
        - **NotImplementedError** - 当前不支持 `tp_comm_buffer_name` 。
        - **RuntimeError** - 使用了 `zero3` 优化器并行，但是未初始化数据并行通信。
        - **RuntimeError** - 为了使能 `sequence_arallel` ， `input_is_parallel` 必须为 ``True`` 但是得到了 ``False`` 。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst