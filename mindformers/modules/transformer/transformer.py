# Copyright 2021-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Note:
    Transformer Networks. This is interface that is subject to change or deletion.
"""
from __future__ import absolute_import

from enum import Enum
from typing import Union
import numpy as np

import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import Zero
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore._checkparam import args_type_check

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindformers.modules.transformer.op_parallel_config import _PipeLineConfig, OpParallelConfig, \
    _Config, MoEParallelConfig

from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.utils import is_pynative

__all__ = [
    "LowerTriangularMaskWithDynamic",
    "TransformerOpParallelConfig",
    "EmbeddingOpParallelConfig",
    "TransformerRecomputeConfig",
    "TransformerSwapConfig"]


class EmbeddingOpParallelConfig(_Config):
    r"""
        The parallel config of :class:`VocabEmbedding`
        for the setting data parallel or model parallel for the embedding table.

        Args:
            data_parallel(int): The data parallel way. The input data will be sliced into n parts for embedding layer
                according to this value. Default: 1.
            model_parallel(int): The model parallel way. The embedding table parameters
                will be sliced at 0-th axis according to the model parallel way. Default: 1.
            vocab_emb_dp(bool): Shard embedding in model parallel or data parallel. If True, the embedding lookup
                will be a data parallel style training and model_parallel value will be ignored.  If false, the
                embedding table will be sharded into n parts at the 0-th dimension row slice of the embedding table,
                where the n is the model parallel way determined by this parameter. Default: True

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import EmbeddingOpParallelConfig
            >>> config=EmbeddingOpParallelConfig(data_parallel=1, model_parallel=1, vocab_emb_dp=True)
    """

    def __init__(self, data_parallel=1, model_parallel=1, context_parallel=1,
                 use_seq_parallel=False, select_recompute=False,
                 vocab_emb_dp=True):
        self._dp_mp_config = OpParallelConfig(data_parallel=data_parallel,
                                              use_seq_parallel=use_seq_parallel,
                                              context_parallel=context_parallel,
                                              model_parallel=model_parallel,
                                              select_recompute=select_recompute)
        Validator.check_bool(vocab_emb_dp, "vocab_emb_dp")
        self.vocab_emb_dp = vocab_emb_dp
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute

    @property
    def data_parallel(self):
        return self._dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._dp_mp_config.data_parallel = value

    @property
    def model_parallel(self):
        return self._dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._dp_mp_config.model_parallel = value

    @property
    def context_parallel(self):
        return self._dp_mp_config.context_parallel

    @context_parallel.setter
    def context_parallel(self, value):
        self._dp_mp_config.context_parallel = value

    @property
    def vocab_emb_dp(self):
        return self._vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        Validator.check_bool(value, "vocab_emb_dp")
        self._vocab_emb_dp = value

    @property
    def dp_mp_config(self):
        return self._dp_mp_config

    def __eq__(self, other) -> bool:
        return isinstance(other, EmbeddingOpParallelConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        """
        Compare the configuration dictionary of the current object with the default configuration dictionary,
        identify the differences between the two, and store these differences in a new dictionary called res-dict
        """
        config_dict = self.to_dict()
        default_dict = EmbeddingOpParallelConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'context_parallel': self.context_parallel,
            'model_parallel': self.model_parallel,
            'select_recompute': self.select_recompute,
            'use_seq_parallel': self.use_seq_parallel,
            'vocab_emb_dp': self.vocab_emb_dp
        }
        return config_dict


class TransformerSwapConfig(_Config):
    r"""
        TransformerSwapConfig for the setting offload attributes for encoder/decoder layers.

        Args:
            swap (bool): Enable offload of the transformer block or not. Default: False.
            layer_swap (list or dict): Configuration for layer swapping. Each item in the list specifies
                the `backward_prefetch` value for a specific layer. Default: None.
            op_swap (list or dict): Configuration for operator swapping. Each item in the list specifies
                the `backward_prefetch` value for operators matching a specific pattern. Default: None.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> from mindformers.modules.transformer import TransformerSwapConfig
            >>> layer_swap_config = [{"backward_prefetch": 2, "layers": [0, 1]}]
            >>> op_swap_config = [{"op_name": "matmul", "backward_prefetch": True}]
            >>> swap_config = TransformerSwapConfig(swap=True, layer_swap=layer_swap_config, op_swap=op_swap_config)
    """

    def __init__(self, layer_swap=None, op_swap=None, swap=False, default_prefetch=1):
        Validator.check_bool(swap, "swap")
        self.backward_prefetch = 'backward_prefetch'
        self.layers = 'layers'
        self._swap = swap
        self._default_prefetch = default_prefetch
        self._layer_swap, self._op_swap = self._initialize_swap(layer_swap, op_swap)

    def _initialize_swap(self, layer_swap, op_swap):
        """Initialize the swap configuration."""
        if layer_swap is None and op_swap is None:
            op_swap_initialized = {}
            op_swap_initialized[r'.*\.flash_attention'] = [{
                'backward_prefetch': self._default_prefetch,
                'layers': True
            }]
            return [], op_swap_initialized
        layer_swap_initialized = self._initialize_layer_swap(layer_swap)
        op_swap_initialized = self._initialize_op_swap(op_swap)
        return layer_swap_initialized, op_swap_initialized

    def _initialize_layer_swap(self, layer_swap):
        """Initializes and validates the layer swap configuration."""
        if layer_swap is None:
            return []
        if not isinstance(layer_swap, (list, dict)):
            raise ValueError("layer_swap must be a list or dict")
        if isinstance(layer_swap, dict):
            layer_swap = [layer_swap]
        if self._validate_layers_consistency(layer_swap):
            return [{"backward_prefetch": layer_swap[0][self.backward_prefetch], "layers": True}]
        return layer_swap

    def _initialize_op_swap(self, op_swap):
        """Initializes and validates the operation swap configuration."""
        if op_swap is None:
            return []
        if not isinstance(op_swap, (list, dict)):
            raise ValueError("op_swap must be a list or dict")
        if isinstance(op_swap, dict):
            op_swap = [op_swap]
        op_swap_dict = self.op_swap_to_dict(op_swap)
        for k, v in op_swap_dict.items():
            if self._validate_layers_consistency(v, mode=f'op_swap: {k}'):
                op_swap_dict[k] = [{"backward_prefetch": v[0][self.backward_prefetch], "layers": True}]
        return op_swap_dict

    def _validate_layers_consistency(self, layer_swap, mode='layer_swap'):
        """
        Validates the consistency of the layers configuration.
        Raise ValueError if prefetch values and layers are conflict.
        """
        prev_backward_prefetch = None
        has_boolean_layers = False
        has_different_prefetch = False
        for i, item in enumerate(layer_swap):
            if prev_backward_prefetch is not None and prev_backward_prefetch != item.get(self.backward_prefetch):
                has_different_prefetch = True
            has_boolean_layers = self._validate_layer_type(item.get(self.layers), has_boolean_layers,
                                                           mode) or has_boolean_layers
            if has_different_prefetch and has_boolean_layers:
                raise ValueError(
                    f"Invalid {mode} configuration at index {i}: {item}. Inconsistent 'backward_prefetch' values.")
            prev_backward_prefetch = item[self.backward_prefetch]
        return has_boolean_layers

    def _validate_layer_type(self, layers, has_boolean_layers, mode='layer_swap'):
        """Validates the type of the layers configuration."""
        if not isinstance(layers, (list, tuple, bool)):
            raise ValueError(f"Invalid {mode} configuration: {layers}. Expected 'layers' to be a list, tuple, or bool.")
        if isinstance(layers, (list, tuple)):
            if not self._is_list_of_list_of_ints(layers) and not self._is_list_or_tuple_of_types(layers):
                raise ValueError(
                    f"Invalid {mode} configuration: {layers}. \
                        Expected 'layers' to be a list of ints or list of lists of ints.")
        if isinstance(layers, bool):
            has_boolean_layers = layers
        return has_boolean_layers

    def op_swap_to_dict(self, op_swap):
        """Converts the operation swap configuration to a dictionary."""
        dic = {}
        for i, item in enumerate(op_swap):
            if not isinstance(item['op_name'], (str, list, tuple)):
                raise ValueError(
                    f"Invalid op_swap configuration at index {i}: {item}. \
                        'op_name' must be a string, list, or tuple.")
            if isinstance(item['op_name'], (list, tuple)):
                if not self._is_list_or_tuple_of_types(item['op_name'], str):
                    raise ValueError(
                        f"Invalid op_swap configuration at index {i}: {item}. \
                            'op_name' list must contain only strings.")
                for key in item['op_name']:
                    self._add_to_dict(dic, key, item)
            else:
                self._add_to_dict(dic, item['op_name'], item)
        return dic

    def _add_to_dict(self, dic, key, item):
        """Adds an operation swap configuration to the dictionary."""
        if key in dic:
            dic[key].append(
                {
                    'layers': item.get(self.layers),
                    'backward_prefetch': item.get(self.backward_prefetch)
                }
            )
        else:
            dic[key] = [
                {
                    'layers': item.get(self.layers),
                    'backward_prefetch': item.get(self.backward_prefetch)
                }
            ]
        return dic

    def _is_list_or_tuple_of_types(self, obj, types=int) -> bool:
        """check obj of list[types]/tuple[types]"""
        if isinstance(obj, (list, tuple)):
            if types == int and any(isinstance(item, bool) for item in obj):
                return False
            return all(isinstance(item, types) for item in obj)
        return False

    def _is_list_of_list_of_ints(self, obj) -> bool:
        """check obj of list[list[int]]"""
        return isinstance(obj, (list, tuple)) and all(self._is_list_or_tuple_of_types(item, int) for item in obj)

    @property
    def swap(self):
        return self._swap

    @swap.setter
    def swap(self, value):
        Validator.check_bool(value, "swap")
        self._swap = value

    @property
    def default_prefetch(self):
        return self._default_prefetch

    @default_prefetch.setter
    def default_prefetch(self, value):
        self._default_prefetch = value

    @property
    def layer_swap(self):
        return self._layer_swap

    @layer_swap.setter
    def layer_swap(self, value):
        self._layer_swap = value

    @property
    def op_swap(self):
        return self._op_swap

    @op_swap.setter
    def op_swap(self, value):
        self._op_swap = value

    def __eq__(self, other) -> bool:
        return isinstance(other, TransformerSwapConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = TransformerRecomputeConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        config_dict = {
            "layer_swap": self._layer_swap,
            "default_prefetch": self._default_prefetch,
            "op_swap": self._op_swap,
            "swap": self._swap
        }
        return config_dict


class TransformerRecomputeConfig(_Config):
    r"""
        TransformerRecomputeConfig for the setting recompute attributes for encoder/decoder layers.

        Args:
            recompute (bool): Enable recomputation of the transformer block or not. Default: False.
            parallel_optimizer_comm_recompute (bool): Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
            mp_comm_recompute (bool): Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            recompute_slice_activation (bool): Slice the cell output which would remains in memory. Default: False.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import TransformerRecomputeConfig
            >>> config=TransformerRecomputeConfig(recompute=True, parallel_optimizer_comm_recompute=True, \
            ...                                   mp_comm_recompute=True, recompute_slice_activation=True)
    """

    def __init__(self, recompute=False, select_recompute=False,
                 parallel_optimizer_comm_recompute=False, select_comm_recompute=False,
                 mp_comm_recompute=True, recompute_slice_activation=False,
                 select_recompute_exclude=False, select_comm_recompute_exclude=False):
        Validator.check_bool(parallel_optimizer_comm_recompute, "parallel_optimizer_comm_recompute")
        Validator.check_bool(mp_comm_recompute, "mp_comm_recompute")
        Validator.check_bool(recompute_slice_activation, "recompute_slice_activation")
        self._recompute = recompute
        self._select_recompute = select_recompute
        self._select_comm_recompute = select_comm_recompute
        self._parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute
        self._mp_comm_recompute = mp_comm_recompute
        self._recompute_slice_activation = recompute_slice_activation
        self._select_recompute_exclude = select_recompute_exclude
        self._select_comm_recompute_exclude = select_comm_recompute_exclude

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        Validator.check_bool(value, "recompute")
        self._recompute = value

    @property
    def select_recompute(self):
        return self._select_recompute

    @property
    def select_comm_recompute(self):
        return self._select_comm_recompute

    @select_recompute.setter
    def select_recompute(self, value):
        Validator.check_bool(value, "select_recompute")
        self._select_recompute = value

    @select_comm_recompute.setter
    def select_comm_recompute(self, value):
        Validator.check_bool(value, "select_comm_recompute")
        self._select_comm_recompute = value

    @property
    def select_recompute_exclude(self):
        return self._select_recompute_exclude

    @property
    def select_comm_recompute_exclude(self):
        return self._select_comm_recompute_exclude

    @property
    def parallel_optimizer_comm_recompute(self):
        return self._parallel_optimizer_comm_recompute

    @select_recompute_exclude.setter
    def select_recompute_exclude(self, value):
        Validator.check_bool(value, "select_recompute_exclude")
        self._select_recompute_exclude = value

    @select_comm_recompute_exclude.setter
    def select_comm_recompute_exclude(self, value):
        Validator.check_bool(value, "select_comm_recompute_exclude")
        self._select_comm_recompute_exclude = value

    @parallel_optimizer_comm_recompute.setter
    def parallel_optimizer_comm_recompute(self, value):
        Validator.check_bool(value, "parallel_optimizer_comm_recompute")
        self._parallel_optimizer_comm_recompute = value

    @property
    def mp_comm_recompute(self):
        return self._mp_comm_recompute

    @mp_comm_recompute.setter
    def mp_comm_recompute(self, value):
        Validator.check_bool(value, "mp_comm_recompute")
        self._mp_comm_recompute = value

    @property
    def recompute_slice_activation(self):
        return self._recompute_slice_activation

    @recompute_slice_activation.setter
    def recompute_slice_activation(self, value):
        Validator.check_bool(value, "recompute_slice_activation")
        self._recompute_slice_activation = value

    def __eq__(self, other) -> bool:
        return isinstance(other, TransformerRecomputeConfig) and (self.to_dict() == other.to_dict())

    def to_diff_dict(self):
        config_dict = self.to_dict()
        default_dict = TransformerRecomputeConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        return res_dict

    def to_dict(self):
        """return config in dict format"""
        config_dict = {
            "recompute": self._recompute,
            "select_recompute": self._select_recompute,
            "parallel_optimizer_comm_recompute": self._parallel_optimizer_comm_recompute,
            "mp_comm_recompute": self._mp_comm_recompute,
            "recompute_slice_activation": self._recompute_slice_activation,
            "select_recompute_exclude": self._select_recompute_exclude,
            "select_comm_recompute_exclude": self._select_comm_recompute_exclude,
            "select_comm_recompute": self._select_comm_recompute,
        }
        return config_dict


class ContextParallelAlgo(Enum):
    """context parallel algorithm type.

    Args:
        Enum (str): chosses context parallel type
    """
    COLOSSALAI_CP = "colossalai_cp"
    ULYSSES_CP = "ulysses_cp"
    HYBRID_CP = "hybrid_cp"


default_transformer_swap_config = TransformerSwapConfig()
default_transformer_recompute_config = TransformerRecomputeConfig()


class TransformerOpParallelConfig(_Config):
    r"""
        TransformerOpParallelConfig for setting parallel configuration, such as the data parallel and model parallel.

        Note:
            Except the recompute argument, other arguments will **not** be effective when the user doesn't set
            auto_parallel_context to `SEMI_AUTO_PARALLEL` or `AUTO_PARALLEL`.
            The micro_batch_num must be greater than or equal to pipeline_stage when training.
            The data_parallel\*model_parallel \*pipeline_stage must be equal or less equal to the device. When setting
            the pipeline stage and optimizer_shard, the config will overwrite the auto_parallel_context. When given the
            8 devices and the data_parallel is 1 and model_parallel is 1, the calculation will be repeated on each
            device.

        Args:
            data_parallel (int): The data parallel way. The input data will be sliced into n parts for each layer
                according to the data parallel way. Default: 1.
            model_parallel (int): The model parallel way. The parameters of dense layers in MultiheadAttention and
                FeedForward layer will be sliced according to the model parallel way. Default: 1.
            expert_parallel (int): The expert parallel way. This is effective only when MoE (Mixture of Experts)
                is applied. This value specifies the number of partitions to split the experts into.
            pipeline_stage (int): The number of the pipeline stage. Should be a positive value. Default: 1.
            micro_batch_num (int): The micro size of the batches for the pipeline training. Default: 1.
            optimizer_shard (bool): *optimizer_shard is deprecated from MindFormers r0.7. It will not have any effect.
                It will be removed in the future version. Using parallel.enable_parallel_optimizer instead.*
            gradient_aggregation_group (int): The fusion group size of the optimizer state sharding. Default: 4.
            recompute (Union[TransformerRecomputeConfig, bool]): The configuration of recomputation for
                the transformer block. Default: An instance of TransformerRecomputeConfig with default values.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. Default: True.
            context_parallel_algo (str): Which type of context parallel algorithm to use. Supports `colossalai_cp`,
                `ulysses_cp` and `hybrid_cp`. Only takes effect when context_parallel > 1. Default: `colossalai_cp`
            ulysses_degree_in_cp (int): When using hybrid_cp, how many cp should be used for ulysses. context_parallel
                should be divisible by it. Only takes effect when `hybrid_cp` algorithm is chosen. Default: 1

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> from mindformers.modules.transformer import TransformerRecomputeConfig
            >>> recompute_config=TransformerRecomputeConfig(recompute=True, parallel_optimizer_comm_recompute=True, \
            ...                                             mp_comm_recompute=True, recompute_slice_activation=True)
            >>> config=TransformerOpParallelConfig(data_parallel=1, model_parallel=1, recompute=recompute_config)
    """

    @args_type_check(recompute=(TransformerRecomputeConfig, dict))
    def __init__(self, data_parallel=1, model_parallel=1, context_parallel=1,
                 expert_parallel=1, pipeline_stage=1, micro_batch_num=1, seq_split_num=1,
                 recompute: Union[TransformerRecomputeConfig, dict] = default_transformer_recompute_config,
                 use_seq_parallel=False, optimizer_shard=None, gradient_aggregation_group=4, vocab_emb_dp=True,
                 context_parallel_algo: str = "colossalai_cp", ulysses_degree_in_cp=1, mem_coeff=0.1,
                 swap: Union[TransformerSwapConfig, dict] = default_transformer_swap_config):
        if isinstance(recompute, dict):
            recompute = TransformerRecomputeConfig(**recompute)
        if isinstance(swap, dict):
            swap = TransformerSwapConfig(**swap)
        self.swap = swap
        self.recompute = recompute
        self.select_recompute = recompute.select_recompute
        self.use_seq_parallel = use_seq_parallel
        self.context_parallel_algo = ContextParallelAlgo(context_parallel_algo)
        self.ulysses_degree_in_cp = ulysses_degree_in_cp
        self.mem_coeff = mem_coeff
        self.optimizer_shard = optimizer_shard
        self.gradient_aggregation_group = gradient_aggregation_group
        self._embed_dp_mp_config = EmbeddingOpParallelConfig(
            data_parallel=data_parallel, model_parallel=model_parallel, context_parallel=context_parallel,
            vocab_emb_dp=vocab_emb_dp, use_seq_parallel=use_seq_parallel,
            select_recompute=recompute.select_recompute)
        self._pp_config = _PipeLineConfig(pipeline_stage=pipeline_stage, micro_batch_num=micro_batch_num,
                                          seq_split_num=seq_split_num)
        self._moe_config = MoEParallelConfig(
            data_parallel=data_parallel, model_parallel=model_parallel, context_parallel=context_parallel,
            select_recompute=recompute.select_recompute,
            expert_parallel=expert_parallel, use_seq_parallel=use_seq_parallel)
        self._check_context_parallel()

    def __eq__(self, other) -> bool:
        return isinstance(other, TransformerOpParallelConfig) and (self.to_dict() == other.to_dict())

    def _check_context_parallel(self):
        """check whether context parallel config is valid.

        Raises:
            ValueError: in hybrid_cp algorithm, context_parallel should be divisible by ulysses_degree_in_cp
        """
        if self.context_parallel == 1:
            if self.context_parallel_algo != ContextParallelAlgo.COLOSSALAI_CP:
                logger.warning(f"context_parallel_algo {self.context_parallel_algo.value} will not take effect "
                               "when context_parallel == 1.")
            if self.ulysses_degree_in_cp > 1:
                logger.warning(f"ulysses_degree_in_cp {self.ulysses_degree_in_cp} will not take effect "
                               "when context_parallel == 1.")
            return

        # here context parallel > 1
        if self.context_parallel_algo != ContextParallelAlgo.HYBRID_CP and self.ulysses_degree_in_cp > 1:
            logger.warning(f"ulysses_degree_in_cp {self.ulysses_degree_in_cp} will not take effect when "
                           f"context_parallel_algo {self.context_parallel_algo.value} is not `hybrid_cp`.")
        if (self.context_parallel_algo == ContextParallelAlgo.HYBRID_CP and
                self.context_parallel % self.ulysses_degree_in_cp != 0):
            raise ValueError(f"When using hybrid_cp algorithm, context_parallel {self.context_parallel} "
                             f"should be divisible by ulysses_degree_in_cp {self.ulysses_degree_in_cp}. "
                             "Please check your `ulysses_degree_in_cp`.")

    def get_ulysses_cp_num(self):
        """get ulysses context parallel num under this config.

        Returns:
            int: ulysses degrees.
        """
        if self.context_parallel == 1:
            return 1
        if self.context_parallel_algo == ContextParallelAlgo.COLOSSALAI_CP:
            return 1
        if self.context_parallel_algo == ContextParallelAlgo.ULYSSES_CP:
            return self.context_parallel
        # hybird
        return self.ulysses_degree_in_cp

    def to_diff_dict(self):
        """
        Compare the configuration dictionary of the current object with the default configuration dictionary,
        identify the differences between the two, and store these differences in a new dictionary called res-dict
        """
        config_dict = self.to_dict()
        default_dict = default_transformer_config.to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        if "recompute" in res_dict:
            res_dict["recompute"] = self.recompute.to_diff_dict()
        return res_dict

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'context_parallel': self.context_parallel,
            'model_parallel': self.model_parallel,
            'expert_parallel': self.expert_parallel,
            'pipeline_stage': self.pipeline_stage,
            'micro_batch_num': self.micro_batch_num,
            'seq_split_num': self.seq_split_num,
            'use_seq_parallel': self.use_seq_parallel,
            'optimizer_shard': self.optimizer_shard,
            'gradient_aggregation_group': self.gradient_aggregation_group,
            'vocab_emb_dp': self.vocab_emb_dp,
            'recompute': self.recompute.to_dict(),
            'context_parallel_algo': self.context_parallel_algo.value,
            'ulysses_degree_in_cp': self.ulysses_degree_in_cp,
            'mem_coeff': self.mem_coeff,
        }
        return config_dict

    @property
    def recompute(self):
        return self._recompute

    @recompute.setter
    def recompute(self, value):
        if not isinstance(value, TransformerRecomputeConfig) and not isinstance(value, bool):
            raise TypeError(f"recompute must be a TransformerRecomputeConfig/bool, but got {type(value).__name__}.")
        if isinstance(value, bool):
            logger.warning("TransformerRecomputeConfig is recommended as the recompute configuration type.")
        self._recompute = value

    @property
    def vocab_emb_dp(self):
        return self._embed_dp_mp_config.vocab_emb_dp

    @vocab_emb_dp.setter
    def vocab_emb_dp(self, value):
        self._embed_dp_mp_config.vocab_emb_dp = value

    @property
    def gradient_aggregation_group(self):
        return self._gradient_aggregation_group

    @gradient_aggregation_group.setter
    def gradient_aggregation_group(self, value):
        Validator.check_positive_int(value, "gradient_aggregation_group")
        self._gradient_aggregation_group = value

    @property
    def micro_batch_num(self):
        return self._pp_config.micro_batch_num

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        self._pp_config.micro_batch_num = value

    @property
    def seq_split_num(self):
        return self._pp_config.seq_split_num

    @seq_split_num.setter
    def seq_split_num(self, value):
        self._pp_config.seq_split_num = value

    @property
    def model_parallel(self):
        return self._embed_dp_mp_config.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        self._embed_dp_mp_config.model_parallel = value
        self._moe_config.model_parallel = value

    @property
    def context_parallel(self):
        return self._embed_dp_mp_config.context_parallel

    @context_parallel.setter
    def context_parallel(self, value):
        self._embed_dp_mp_config.context_parallel = value
        self._moe_config.context_parallel = value

    @property
    def data_parallel(self):
        return self._embed_dp_mp_config.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        self._embed_dp_mp_config.data_parallel = value
        self._moe_config.data_parallel = value

    @property
    def expert_parallel(self):
        return self._moe_config.expert_parallel

    @expert_parallel.setter
    def expert_parallel(self, value):
        self._moe_config.expert_parallel = value

    @property
    def pipeline_stage(self):
        return self._pp_config.pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        self._pp_config.pipeline_stage = value

    @property
    def optimizer_shard(self):
        return self._optimizer_shard

    @optimizer_shard.setter
    def optimizer_shard(self, value):
        self._optimizer_shard = value
        if value:
            logger.warning("\"parallel_config.optimizer_shard\" is deprecated from MindFormers r0.7. It will not have "
                           "any effect. Please use \"parallel.enable_parallel_optimizer\" to turn on or off the "
                           "optimizer parallel.")

    @property
    def embedding_dp_mp_config(self):
        return self._embed_dp_mp_config

    @property
    def dp_mp_config(self):
        return self._embed_dp_mp_config.dp_mp_config

    @property
    def moe_parallel_config(self):
        return self._moe_config


default_transformer_config = TransformerOpParallelConfig()


class LowerTriangularMaskWithDynamic(Cell):
    r"""
            Get the Strictly Lower triangular matrix from the input_ids.
    """

    @_LogActionOnce(m_logger=logger, key='AttentionMask',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    def __init__(self, seq_length, batch_size=1, compute_type=mstype.float16,
                 is_dynamic=False, pad_token_id=0, use_flash_attention=False, use_attn_mask_compression=False,
                 use_past=False, seq_split_num=1, chunk_prefill=False):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        self.is_pynative = is_pynative()
        self.chunk_prefill = chunk_prefill
        if use_past:
            if chunk_prefill:
                self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                                  dtype=compute_type)
            else:
                if not self.use_flash_attention:
                    self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                                      dtype=compute_type)
                elif self.is_dynamic:
                    mask_coeff = 1.0 if compute_type is mstype.bfloat16 else -10000.0
                    self.lower_triangle_mask = Tensor(
                        np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff, dtype=compute_type
                    )
                    self.hard_mask = Tensor([0], dtype=compute_type).reshape(1, 1)
                else:
                    self.lower_triangle_mask = None
        else:
            if use_attn_mask_compression:
                if seq_length < 2048:
                    raise ValueError("seq_length should be larger than 2048 when use mask_compression")
                self.lower_triangle_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
            else:
                self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                                  dtype=compute_type)
            self.hard_mask = Tensor([0], dtype=compute_type).reshape(1, 1)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.bmm = P.BatchMatMul()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.mul_post = P.Mul()
        self.expand_dim_post = P.ExpandDims()
        # seq pp

        self.gather = P.Gather()
        self.seq_split_num = seq_split_num
        self.seq_pipe = seq_split_num > 1
        if self.seq_pipe:
            self.mask_cache = Parameter(Tensor(shape=(batch_size, seq_length), dtype=mstype.float32, init=Zero()),
                                        name="mask_cache", requires_grad=False, parallel_optimizer=False)
            mask_mask = np.zeros((batch_size, seq_length), dtype=np.int32)
            self.seq_seg_len = seq_length // self.seq_split_num
            for s in range(self.seq_split_num):
                mask_mask[:, s * self.seq_seg_len: (s + 1) * self.seq_seg_len] = s
            self.mask_mask = Tensor(mask_mask)
            self.add_mask = P.Add()
            self.tile_mask = P.Tile()
            self.assign_mask = P.Assign()
            self.mul_mask = P.Mul()
            self.equal_mask = P.Equal()
            np_range = np.arange(seq_length // self.seq_split_num)
            self.seq_seg_range = Tensor(np_range, dtype=mstype.int32)
            self.seq_seg_len = Tensor(seq_length // self.seq_split_num, dtype=mstype.int32)
            self.add_seq = P.Add()

    def construct(self, tokens=None, masks=None, seq_chunk=None):
        """Forward process of the CausalMask"""
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right

        lower_triangle_mask = self.lower_triangle_mask
        if self.is_pynative or self.is_dynamic:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
        lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        if self.seq_pipe:
            seq_seg_range = self.add_seq(self.seq_seg_range, self.seq_seg_len * seq_chunk)
            attention_mask_chunk = self.gather(lower_triangle, seq_seg_range, 1)
            mask_mask = self.cast(self.equal_mask(self.mask_mask, seq_chunk), self.dtype)
            input_mask = self.tile_mask(input_mask, (1, self.seq_split_num))
            input_mask = self.mul_mask(input_mask, mask_mask)
            input_mask_update = self.add_mask(input_mask, self.mask_cache)
            mask_update = self.assign_mask(self.mask_cache, input_mask_update)
            mask_reshape = self.reshape(input_mask_update, (bs, 1, seq_len * self.seq_split_num))
            mask_reshape = F.depend(mask_reshape, mask_update)
            attention_mask = self.mul(mask_reshape, attention_mask_chunk)
            attention_mask = self.sub(self.one, attention_mask)
            attention_mask = self.expand_dim_post(attention_mask, 1)
            attention_mask = self.cast(attention_mask, mstype.uint8)
            return attention_mask
        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
        else:
            attention_mask = self.mul_post(attention_mask, self.multiply_data)
        return attention_mask

    def prefill(self):
        return self.lower_triangle_mask

    def chunk_masks(self, seq_range):
        masks = self.gather(self.lower_triangle_mask, seq_range, 0)
        return 1 - masks

    def gen_attention_mask(self, is_prefill):
        if is_prefill:
            attention_mask = self.lower_triangle_mask
        else:
            attention_mask = self.hard_mask
        return attention_mask

    def shard(self, parallel_config):
        """sharding for LowerTriangularMaskWithDynamic"""
        dp = parallel_config.data_parallel
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.not_equal.shard(((dp, 1), ()))
            self.bmm.shard(((dp, 1, 1), (dp, 1, 1)))
            self.expand_dim.shard(((1, 1),))
            self.expand_dim_post.shard(((dp, 1, 1),))
        else:
            self.not_equal.shard(((dp, 1), ()))
            self.bmm.shard(((dp, 1, 1), (dp, 1, 1)))
            self.expand_dim.shard(((1, 1),))
            self.mul.shard(((dp, 1, 1), (1, 1, 1)))
            self.less_equal.shard(((1, 1, 1), (1, 1, 1)))
            self.sub.shard(((1,), (dp, 1, 1)))
            self.mul_post.shard(((dp, 1, 1, 1), (1,)))
            self.expand_dim_post.shard(((dp, 1, 1),))
        if self.seq_pipe:
            self.add_seq.shard(((1,), ()))
            self.gather.shard(((1, 1, 1), (1,)))
            self.add_mask.shard(((dp, 1), (dp, 1)))
            self.tile_mask.shard(((dp, 1),))
            self.assign_mask.shard(((dp, 1), (dp, 1)))
            self.mul_mask.shard(((dp, 1), (dp, 1)))
            self.equal_mask.shard(((dp, 1), ()))
        if self.chunk_prefill:
            self.gather.shard(((1, 1), (1, 1)))
