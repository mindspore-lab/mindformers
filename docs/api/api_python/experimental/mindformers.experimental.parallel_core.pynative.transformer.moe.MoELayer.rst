mindformers.experimental.parallel_core.pynative.transformer.moe.MoELayer
========================================================================

.. py:class:: mindformers.experimental.parallel_core.pynative.transformer.moe.MoELayer(config: TransformerConfig, submodules=None, layer_number: int = None)

    专家层。

    参数:
        - **config** (TransformerConfig) - transformer 模型的配置对象。
        - **submodules** - 预留参数，目前未使用。
        - **layer_number** - 预留参数，目前未使用。

    输入:
        - **hidden_states** (Tensor) - 本地专家的输入隐藏状态张量。

    输出:
        两个张量的元组。

        - **output** (Tensor) - 本地专家的输出。
        - **mlp_bias** (Tensor) - 目前未使用。

    异常:
        - **ValueError** - 如果ep_world_size小于或等于 0，则引发 ValueError 异常。
        - **ValueError** - 如果num_experts % ep_world_size不等于 0，则引发 ValueError 异常。
        - **ValueError** - 如果local_expert_indices的元素大于或等于num_experts，则引发 ValueError 异常。
        - **ValueError** - 如果moe_config.moe_token_dispatcher_type不是 “alltoall”，则引发 ValueError 异常。
        - **ValueError** - 如果self.training为真且get_tensor_model_parallel_world_size()大于 1，并且self.sp不为真，则引发 ValueError 异常。

    样例：

    .. note::
        .. include:: ./mindformers.experimental.parallel_core.pynative.comm_note.rst