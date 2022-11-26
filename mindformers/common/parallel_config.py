"""Parallel Config Init."""
from mindspore.nn.transformer.moe import default_moe_config, MoEConfig
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


def build_parallel_config(config):
    """Build context config."""
    if config.moe_config:
        config.moe_config = MoEConfig(**config.moe_config)
    else:
        config.moe_config = default_moe_config
    if config.recompute_config:
        config.recompute_config = TransformerRecomputeConfig(**config.recompute_config)
    else:
        config.recompute_config = default_recompute_config
    if config.parallel_config:
        config.parallel_config = TransformerOpParallelConfig(recompute=config.recompute_config,
                                                             **config.parallel_config)
    else:
        config.parallel_config = default_parallel_config
