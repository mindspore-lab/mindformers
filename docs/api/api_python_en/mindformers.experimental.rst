mindformers.experimental
=========================

Experimental module, including graph, infer and parallel core features.

Parallel Core
---------------------

.. autosummary::
    :toctree: experimental
    :nosignatures:
    :template: classtemplate.rst

    mindformers.experimental.parallel_core.pynative.distributed.DistributedDataParallel
    mindformers.experimental.parallel_core.pynative.optimizer.DistributedOptimizer
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