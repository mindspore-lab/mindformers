mindformers.experimental.parallel_core.pynative.transformer.ParallelLMLogits
============================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.transformer.ParallelLMLogits(config, bias=False, compute_dtype=None)

    得到vocab中每一个token的logits。

    参数:
        - **config** (dict) - 并行配置。
        - **bias** (bool) - 指定模型是否使用偏置向量。 默认: ``False``。
        - **compute_dtype** (dtype.Number) - 计算类型。 默认: ``None``。

    输入:
        - **input_** (Tensor) - 隐藏状态的张量。
        - **word_embedding_table** (Parameter) - 从嵌入层通过的权重矩阵。
        - **parallel_output** (bool) - 指定是否返回各张量并行权重上的并行输出。默认: True。
        - **bias** (Tensor) - 可训练的偏置参数。

    输出:
        - **logits_parallel** (Tensor) - 如果在 ParallelLMLogits 中设置 parallel_output 为 True，则每个张量并行等级上的输出将是一个并行的logits张量，
          否则，输出将是一个收集所有并行输出的logits张量。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst