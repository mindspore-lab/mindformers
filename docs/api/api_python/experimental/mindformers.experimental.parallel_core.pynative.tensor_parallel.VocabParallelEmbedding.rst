mindformers.experimental.parallel_core.pynative.tensor_parallel.VocabParallelEmbedding
======================================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.tensor_parallel.VocabParallelEmbedding(num_embeddings, embedding_dim, *, init_method, reduce_scatter_embeddings=False, config, param_init_dtype=None)

    在词汇维度并行的嵌入计算。

    参数：
        - **num_embeddings** (int) - 词表大小。
        - **embedding_dim** (int) - 隐含层大小。
        - **init_method** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练参数的初始化方式。若传入值类型为字符串，则对应 `initializer` 的函数名。
        - **reduce_scatter_embeddings** (bool) - 指定在嵌入查询后是否要执行ReduceScatter操作。默认值： ``False`` 。
        - **config** (dict) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **param_init_dtype** (dtype.Number) - 参数初始化类型。默认值： ``None`` 。

    输入：
        - **input\_** (Tensor) - 形状为 :math:`(B, S)` 或 :math:`(S, B)` 的输入张量。

    输出：
        形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量，与输入张量对应。

    异常：
        - **ValueError** - 词表大小无法被张量并行数整除。
        - **NotImplementedError** - 当前不支持 `config.parallel_config.deterministic_mode` 。
        - **NotImplementedError** - 当前不支持 `config.parallel_config.use_cpu_initialization` 。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst
        
        该样例需要在4卡环境下运行。