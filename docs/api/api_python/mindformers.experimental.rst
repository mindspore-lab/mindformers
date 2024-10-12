mindformers.experimental
=========================

实验模块，包含图、推理和并行核心。

并行核心
---------------------

.. mscnautosummary::
    :toctree: experimental
    :nosignatures:

    mindformers.experimental.parallel_core.pynative.tensor_parallel.ColumnParallelLinear
    mindformers.experimental.parallel_core.pynative.tensor_parallel.RowParallelLinear
    mindformers.experimental.parallel_core.pynative.tensor_parallel.VocabParallelEmbedding
    mindformers.experimental.parallel_core.pynative.tensor_parallel.VocabParallelCrossEntropy
    mindformers.experimental.parallel_core.pynative.transformer.TransformerLanguageModel
    mindformers.experimental.parallel_core.pynative.transformer.ParallelMLP
    mindformers.experimental.parallel_core.pynative.transformer.ParallelAttention
    mindformers.experimental.parallel_core.pynative.transformer.ParallelTransformerLayer
    mindformers.experimental.parallel_core.pynative.transformer.ParallelTransformer
    mindformers.experimental.parallel_core.pynative.transformer.ParallelLMLogits
    mindformers.experimental.parallel_core.pynative.transformer.RotaryEmbedding
    mindformers.experimental.parallel_core.pynative.transformer.moe.MoELayer