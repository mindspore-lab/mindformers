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
import math
from typing import Union
import numpy as np

import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Zero
from mindspore import nn
from mindspore import context
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

try:
    from mindspore.ops.operations.nn_ops import PromptFlashAttention
except ImportError:
    PROMPTFLASHATTENTION_VALID = False
    logger.warning("PromptFlashAttention is not available.")
else:
    PROMPTFLASHATTENTION_VALID = True
    logger.info("PromptFlashAttention is available.")

from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.layers import LayerNorm, Linear, \
    _args_type_validator_check, _valid_type_checks, _valid_value_checks, \
    _check_past_none_input_none, _check_input_dtype
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, _PipeLineConfig, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig
from mindformers.modules.transformer.moe import default_moe_config, MoE, _check_moe_config
from mindformers.version_control import get_dropout, choose_flash_attention_dtype, \
    check_valid_flash_attention

from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.logger import logger as log
from mindformers.tools.utils import is_pynative

__all__ = [
    "AttentionMask",
    "AttentionMaskHF",
    "LowerTriangularMaskWithDynamic",
    "VocabEmbedding",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "Transformer",
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
            return [dict(backward_prefetch=layer_swap[0][self.backward_prefetch], layers=True)]
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
                op_swap_dict[k] = [dict(backward_prefetch=v[0][self.backward_prefetch], layers=True)]
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
                dict(
                    layers=item.get(self.layers),
                    backward_prefetch=item.get(self.backward_prefetch)
                )
            )
        else:
            dic[key] = [
                dict(
                    layers=item.get(self.layers),
                    backward_prefetch=item.get(self.backward_prefetch)
                )
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
        }
        return config_dict


class ContextParallelAlgo(Enum):
    """context parallel algorithm type.

    Args:
        Enum (str): chosses context parallel type
    """
    colossalai_cp = "colossalai_cp"
    ulysses_cp = "ulysses_cp"
    hybrid_cp = "hybrid_cp"


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
            if self.context_parallel_algo != ContextParallelAlgo.colossalai_cp:
                logger.warning(f"context_parallel_algo {self.context_parallel_algo.value} will not take effect "
                               "when context_parallel == 1.")
            if self.ulysses_degree_in_cp > 1:
                logger.warning(f"ulysses_degree_in_cp {self.ulysses_degree_in_cp} will not take effect "
                               "when context_parallel == 1.")
            return

        # here context parallel > 1
        if self.context_parallel_algo != ContextParallelAlgo.hybrid_cp and self.ulysses_degree_in_cp > 1:
            logger.warning(f"ulysses_degree_in_cp {self.ulysses_degree_in_cp} will not take effect when "
                           f"context_parallel_algo {self.context_parallel_algo.value} is not `hybrid_cp`.")
        if (self.context_parallel_algo == ContextParallelAlgo.hybrid_cp and
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
        if self.context_parallel_algo == ContextParallelAlgo.colossalai_cp:
            return 1
        if self.context_parallel_algo == ContextParallelAlgo.ulysses_cp:
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
            logger.warning(f"TransformerRecomputeConfig is recommended as the recompute configuration type.")
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
default_embedding_parallel_config = EmbeddingOpParallelConfig()


class FeedForward(Cell):
    r"""
        The multilayer perceptron with two linear layers with dropout applied at final output. The first linear
        will project the input dimension from hidden_size to ffn_hidden_size. The second linear will project the
        dimension from ffn_hidden_size to hidden_size. The first linear is sharded on the relative dimension,
        and the second linear is sharded on the output dimension. The overview process can be:

        .. math::
            Dropout((xW_1+b_1)W_2 + b_2)

        where the :math:`W_1, W_2, b_1` and :math:`b_2` are trainable parameters.

        Args:
            hidden_size (int): The dimension of the inputs.
            ffn_hidden_size (int): The intermediate hidden size.
            dropout_rate (float): The dropout rate for the second linear's output.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see examples. Default: gelu.
            expert_num (int): The number of experts used in Linear. For the case expert_num > 1, BatchMatMul is used
                and the first dimension in BatchMatMul indicate expert_num. Default: 1.
            expert_group_size (int): The number of tokens in each data parallel group. Default: None. This parameter is
                effective only when in AUTO_PARALLEL mode, and NOT SHARDING_PROPAGATION.
            param_init_type (dtype.Number): The parameter initialization type. Should be mstype.float32 or
                mstype.float16. Default: mstype.float32.
            parallel_config (OpParallelConfig, MoEParallelConfig): The config of parallel setting, see
                `OpParallelConfig` or `MoEParallelConfig`. When MoE is applied, MoEParallelConfig is effective,
                otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            TypeError: `hidden_act` is not a string or nn.Cell.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.
            ValueError: `ffn_hidden_size` is not a multiple of the model parallel way.
            ValueError: `hidden_size` is not a multiple of the model parallel way.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import FeedForward
            >>> from mindspore import dtype as mstype
            >>> from mindspore import Tensor, nn
            >>> import mindspore.ops as ops
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
            >>> # Example 2 using custom hidden activation
            >>> class MyActivationNoShard(nn.Cell):
            ...     def __init__(self):
            ...         super(MyActivationNoShard, self).__init__()
            ...         self.add = ops.Add()
            ...     def construct(self, x):
            ...         return self.add(x, 0.1)
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1,
            ...                     hidden_act=MyActivationNoShard)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
            >>> # Example 3 using custom hidden activation with activation_shard
            >>> # If user wantss to run on the SEMI/AUTO parallel mode, the custom activation must provide
            >>> # a class function named activation_shard. It accepts the argument parallel_config (OpParallelConfig,
            >>> # MoEParallelConfig) and set the shard for the primitives used in the construct.
            >>> class MyActivationWithShard(nn.Cell):
            ...     def __init__(self):
            ...         super(MyActivationWithShard, self).__init__()
            ...         self.add = ops.Add()
            ...     def construct(self, x):
            ...         return self.add(x, 0.1)
            ...     def activation_shard(self, parallel_config):
            ...         self.add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), ()))
            >>>
            >>> model = FeedForward(hidden_size=15, ffn_hidden_size=30, dropout_rate=0.1,
            ...                     hidden_act=MyActivationWithShard)
            >>> tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> output = model(tensor)
            >>> print(output.shape)
            (2, 20, 15)
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                dropout_rate=Validator.check_non_negative_float,
                                param_init_type=_valid_value_checks([mstype.float32, mstype.bfloat16, mstype.float16],
                                                                    "FeedForward"),
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.bfloat16, mstype.float16],
                                                                  "FeedForward"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "FeedForward"))
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 expert_num=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config,
                 compute_dtype=mstype.float16):
        super(FeedForward, self).__init__()
        self.dtype = compute_dtype
        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))
            input_size = hidden_size
            output_size = ffn_hidden_size

            # Project to ffn_hidden_size
            self.mapping = Linear(in_channels=input_size,
                                  out_channels=output_size,
                                  activation=hidden_act,
                                  transpose_b=False,
                                  expert_num=expert_num,
                                  expert_group_size=expert_group_size,
                                  outer_batch=dp,
                                  param_init_type=param_init_type,
                                  compute_dtype=compute_dtype)

            # Project back to hidden_size
            self.projection = Linear(in_channels=output_size,
                                     out_channels=input_size,
                                     transpose_b=False,
                                     expert_num=expert_num,
                                     expert_group_size=expert_group_size,
                                     outer_batch=dp,
                                     param_init_type=param_init_type,
                                     compute_dtype=compute_dtype)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = get_dropout(dropout_rate)
            self.dropout_3d = get_dropout(dropout_rate)
            self.dropout_4d = get_dropout(dropout_rate)
            self.cast = P.Cast()
        else:
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))
            input_size = hidden_size
            output_size = ffn_hidden_size

            # Project to ffn_hidden_size
            self.mapping = Linear(in_channels=input_size,
                                  out_channels=output_size,
                                  activation=hidden_act,
                                  transpose_b=False,
                                  expert_num=expert_num,
                                  expert_group_size=expert_group_size,
                                  outer_batch=dp,
                                  param_init_type=param_init_type,
                                  compute_dtype=compute_dtype)

            if expert_num > 1:
                self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                                   strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)),
                                   strategy_activation=((dp, ep, 1, mp),))
            else:
                self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                                   strategy_bias=((dp, mp), (mp,)),
                                   strategy_activation=((dp, mp),))
            # Project back to hidden_size
            self.projection = Linear(in_channels=output_size,
                                     out_channels=input_size,
                                     transpose_b=False,
                                     expert_num=expert_num,
                                     expert_group_size=expert_group_size,
                                     outer_batch=dp,
                                     param_init_type=param_init_type,
                                     compute_dtype=compute_dtype)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                      strategy_bias=((dp, ep, 1, 1), (1, ep, 1, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                                      strategy_bias=((dp, 1), (1,)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = get_dropout(dropout_rate)
            self.dropout_3d = get_dropout(dropout_rate)
            self.dropout_4d = get_dropout(dropout_rate)
            self.dropout.dropout.shard(((dp, 1),))
            self.dropout_3d.dropout.shard(((dp, 1, 1),))
            self.dropout_4d.dropout.shard(((dp, ep, 1, 1),))
            self.cast = P.Cast()

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        x = self.cast(x, self.dtype)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        else:
            output = self.dropout_4d(output)
        return output


class AttentionMask(Cell):
    r"""
        Get the Lower triangular matrix from the input mask. The input mask is a 2D tensor (batch_size, seq_length)
        with 1 and 0, where 1 indicates the current position is a valid token, otherwise not.

        Args:
            seq_length(int): The sequence length of the input tensor.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                               an instance of `OpParallelConfig` with default args.

        Inputs:
            - **input_mask** (Tensor) - The mask indicating whether each position is a valid input with
              (batch_size, seq_length).

        Outputs:
            Tensor. The attention mask matrix with shape (batch_size, seq_length, seq_length).

        Raises:
            TypeError: `seq_length` is not an integer.
            ValueError: `seq_length` is not a positive value.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import AttentionMask
            >>> from mindspore import Tensor
            >>> mask = AttentionMask(seq_length=4)
            >>> mask_array = np.array([[1, 1, 1, 0]], np.float32)
            >>> inputs = Tensor(mask_array)
            >>> res = mask(inputs)
            >>> print(res)
            [[[1. 0. 0. 0]
              [1. 1. 0. 0]
              [1. 1. 1. 0]
              [0. 0. 0. 0]]]
    """

    @_LogActionOnce(m_logger=logger, key='AttentionMask',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(seq_length=Validator.check_positive_int,
                                parallel_config=_valid_type_checks([OpParallelConfig], "AttentionMask"))
    def __init__(self, seq_length, parallel_config=default_dpmp_config, compute_dtype=mstype.float16):
        super(AttentionMask, self).__init__()
        self.seq_length = seq_length
        self.compute_dtype = compute_dtype
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(seq_length, seq_length))
        # Default lower triangle mask matrix
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((parallel_config.data_parallel, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        """Forward process of the AttentionMask"""
        _check_input_dtype(F.dtype(input_mask), "input_mask",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        input_mask = P.Cast()(self.not_equal(input_mask, 0), self.compute_dtype)
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(attention_mask, lower_traiangle)
        return attention_mask


class AttentionMaskHF(Cell):
    r"""
        Get the Lower triangular matrix from the input mask. The input mask is a 2D tensor (batch_size, seq_length).

        Args:
            seq_length(int): The sequence length of the input tensor.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                                               an instance of `OpParallelConfig` with default args.

        Inputs:
            - **input_mask** (Tensor) - The mask indicating whether each position is a valid input with
              (batch_size, seq_length).

        Outputs:
            Tensor. The attention mask matrix with shape (batch_size, seq_length, seq_length).

        Raises:
            TypeError: `seq_length` is not an integer.
            ValueError: `seq_length` is not a positive value.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import AttentionMaskHF
            >>> from mindspore import Tensor
            >>> mask = AttentionMaskHF(seq_length=4)
            >>> mask_array = np.array([[1, 1, 1, 0]], np.float32)
            >>> inputs = Tensor(mask_array)
            >>> res = mask(inputs)
            >>> print(res)
            [[[1. 0. 0. 0]
              [1. 1. 0. 0]
              [1. 1. 1. 0]
              [1. 1. 1. 1]]]
    """

    @_LogActionOnce(m_logger=logger, key='AttentionMaskHF',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(seq_length=Validator.check_positive_int,
                                parallel_config=_valid_type_checks([OpParallelConfig], "AttentionMaskHF"))
    def __init__(self, seq_length, parallel_config=default_dpmp_config, compute_dtype=mstype.float16):
        super(AttentionMaskHF, self).__init__()
        self.seq_length = seq_length
        self.compute_dtype = compute_dtype
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(seq_length, seq_length))
        # Default lower triangle mask matrix
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((parallel_config.data_parallel, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        """Forward process of the AttentionMaskHF"""
        _check_input_dtype(F.dtype(input_mask), "input_mask",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        input_mask = P.Cast()(P.OnesLike()(input_mask), self.compute_dtype)
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)
        return attention_mask


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


class VocabEmbedding(Cell):
    """
        The embedding lookup table from the 0-th dim of the parameter table. When the parallel_config.vocab_emb_dp is
        True and in the `AUTO_PARALLEL` mode, the embedding lookup will be trained by the data parallel way, as the
        parameters will be repeated on each device. If false, the embedding table will be sharded into n parts at
        the 0-th dimension of the embedding table, where the n is the model parallel way determined by
        `parallel_config.model_parallel` (EmbeddingOpParallelConfig).

        Note:
            When `AUTO_PARALLEL` or `SEMI_AUTO_PARALLEL` mode is enabled, this layer support only 2-d dimension inputs,
            as the shard is designed for 2d inputs.

        Args:
            vocab_size (int): Size of the dictionary of embeddings.
            embedding_size (int): The size of each embedding vector.
            parallel_config (EmbeddingOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.

        Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

        Outputs:
            Tuple, a tuple contains (`output`, `embedding_table`)

            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
            - **embedding_table** (Tensor) - The embedding table with shape (vocab_size, embedding_size).

        Raises:
            ValueError: If the parallel_config.vocab_emb_dp is True, the vocab size is not a multiple of
                parallel_config.model_parallel
            ValueError: `vocab_size` is not a positive value.
            ValueError: `embedding_size` is not a positive value.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import VocabEmbedding
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> model = VocabEmbedding(vocab_size=30, embedding_size=30)
            >>> tensor = Tensor(np.ones((20, 15)), mstype.int32)
            >>> output, table = model(tensor)
            >>> print(output.shape)
            (20, 15, 30)
            >>> print(table.shape)
            (30, 30)
    """

    @_LogActionOnce(m_logger=logger, key='VocabEmbedding',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(vocab_size=Validator.check_positive_int,
                                embedding_size=Validator.check_positive_int,
                                parallel_config=_valid_type_checks([EmbeddingOpParallelConfig], "VocabEmbedding"))
    def __init__(self, vocab_size, embedding_size, parallel_config=default_embedding_parallel_config,
                 param_init='normal', param_init_type=mstype.float32):
        super(VocabEmbedding, self).__init__()
        _check_config(parallel_config)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_table = Parameter(
            initializer(param_init, [self.vocab_size, self.embedding_size], param_init_type),
            name='embedding_table', parallel_optimizer=False)

        if parallel_config.vocab_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (parallel_config.data_parallel, 1)))
            logger.info(f"Using {parallel_config.data_parallel} data parallel for the embedding lookup.")
        else:
            if self.vocab_size % parallel_config.model_parallel != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of parallel_config.model_parallel {parallel_config.model_parallel}.")
            self.gather = P.Gather().shard(((parallel_config.model_parallel, 1), (parallel_config.data_parallel, 1)))
            logger.info(f"Using {parallel_config.data_parallel} data parallel and {parallel_config.model_parallel} "
                        f"model parallel for the embedding lookup.")

    def construct(self, input_ids):
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32], self.cls_name)
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()


class MultiHeadAttention(Cell):
    r"""
        This is an implementation of multihead attention in the paper `Attention is all you need
        <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
        key and value vector with target length, the attention will be performed as the following

        .. math::
               MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

        where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

        if query, key and value tensor is same, then it will be self attention.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            src_seq_length(int): The sequence length of the query vector.
            tgt_seq_length(int): The sequence length of the key and value vector.
            hidden_size(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            compute_dtype(dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            softmax_compute_type(dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            param_init_type(dtype.Number): The parameter initialization type of the module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **query_tensor** (Tensor) - The query vector with shape (batch_size, src_seq_length, hidden_size) or
              (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **key_tensor** (Tensor) - The key vector with shape (batch_size, tgt_seq_length, hidden_size) or
              (batch_size * tgt_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **value_tensor** (Tensor) - The value vector with shape (batch_size, tgt_seq_length, hidden_size) or
              (batch_size * tgt_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
              Otherwise, must be (batch_size, 1, hidden_size)
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
              matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
              in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, size_per_head, tgt_seq_length).
              The past calculated key vector. Used for incremental prediction when the use_past is True.
              Default None.
            - **value_past** (Tensor) - Float16 tensor with shape
              (batch_size, num_heads, tgt_seq_length, size_per_head).
              The past calculated value vector. Used for incremental prediction when the use_past is True.
              Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
              shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
              if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, tgt_seq_length),
              (batch_size, num_heads, tgt_seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindformers.modules.transformer import MultiHeadAttention
            >>> from mindspore import dtype as mstype
            >>> from mindspore import Tensor
            >>> model = MultiHeadAttention(batch_size=None, hidden_size=15, src_seq_length=20, tgt_seq_length=20,
            ...                            num_heads=3)
            >>> from_tensor = Tensor(np.ones((2, 20, 15)), mstype.float32)
            >>> to_tensor = Tensor(np.ones((2, 20, 15)), mstype.float16)
            >>> attention_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask)
            >>> print(attn_out.shape)
            (2, 20, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
            >>> # When use use_past=True, it includes two steps to implement the incremental prediction.
            >>> # Step 1: set is_first_iteration=True, and input the full sequence length's state.
            >>> # We need to prepare the memory parameters for saving key and value states firstly.
            >>> model = MultiHeadAttention(batch_size=2, hidden_size=15, src_seq_length=20, tgt_seq_length=20,
            ...                            num_heads=3, use_past=True)
            >>> key_past = Tensor(np.zeros(shape=(2, 3, 5, 20)), mstype.float16)
            >>> value_past = Tensor(np.zeros(shape=(2, 3, 20, 5)), mstype.float16)
            >>> batch_valid_length = Tensor(np.ones((2,)), mstype.int32)
            >>> # Set is_first_iteration=True to generate the full memory states
            >>> model.add_flags_recursive(is_first_iteration=True)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past,
            ...                        batch_valid_length)
            >>> print(attn_out.shape)
            (2, 20, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
            >>> from_tensor = Tensor(np.ones((2, 1, 15)), mstype.float32)
            >>> to_tensor = Tensor(np.ones((2, 1, 15)), mstype.float16)
            >>> attention_mask = Tensor(np.ones((2, 1, 20)), mstype.float16)
            >>> # Step 2: set is_first_iteration=False, and pass the single word to run the prediction rather than the
            >>> # full sequence.
            >>> model.add_flags_recursive(is_first_iteration=False)
            >>> attn_out, past = model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past,
            ...                        batch_valid_length)
            >>> print(attn_out.shape)
            (2, 1, 15)
            >>> print(past[0].shape)
            (2, 3, 5, 20)
            >>> print(past[1].shape)
            (2, 3, 20, 5)
    """

    @_LogActionOnce(m_logger=logger, key='MultiHeadAttention',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "MultiHeadAttention"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "MultiHeadAttention"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "MultiHeadAttention"),
                                parallel_config=_valid_type_checks([OpParallelConfig],
                                                                   "MultiHeadAttention"),
                                use_past=Validator.check_bool,
                                use_flash_attention=Validator.check_bool,
                                use_prompt_flash_attention=Validator.check_bool)
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config,
                 use_flash_attention=False,
                 use_prompt_flash_attention=False):
        super(MultiHeadAttention, self).__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            self.reshape = P.Reshape()
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul()
            self.real_div = P.RealDiv()
            self.sub = P.Sub()
            self.sub_sa = P.Sub()
            self.mul = P.Mul()
            self.add = P.Add()
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.use_past = use_past
            self.dropout = get_dropout(hidden_dropout_rate)
            self.prob_dropout = get_dropout(attention_dropout_rate)
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_cast = P.Cast()
            self.softmax_reshape = P.Reshape()
            self.expand_dims = P.ExpandDims()

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the num_heads is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(num_heads, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_channels=hidden_size,
                                     out_channels=hidden_size,
                                     transpose_b=False,
                                     compute_dtype=compute_dtype,
                                     param_init_type=param_init_type)
            self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                                  strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                                   (parallel_config.model_parallel, 1)))
            self.projection.bias.parallel_optimizer = False
            self.transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
            self.merger_head_transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.reshape = P.Reshape()
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            self.batch_matmul = P.BatchMatMul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.real_div = P.RealDiv().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
            self.sub_sa = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.sub = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.mul = P.Mul().shard(
                ((parallel_config.data_parallel, 1, 1, 1), (1,)))
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.use_past = use_past
            self.dropout = get_dropout(hidden_dropout_rate)
            self.prob_dropout = get_dropout(attention_dropout_rate)
            self.dropout.dropout.shard(((parallel_config.data_parallel, 1),))
            self.prob_dropout.dropout.shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax = nn.Softmax().to_float(softmax_compute_type)
            self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
            self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
            self.softmax_cast = P.Cast()
            self.softmax_reshape = P.Reshape()
            self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1),
                                               (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1),
                                               (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))

            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
            self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1),
                                               (parallel_config.model_parallel, 1)),
                              strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                             (parallel_config.model_parallel,)))
            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

            if parallel_config.use_seq_parallel:
                self.dropout.dropout.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                self.projection.shard(
                    strategy_bias=((parallel_config.data_parallel * parallel_config.model_parallel, 1), (1,)),
                    strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                     (parallel_config.model_parallel, 1)),
                    out_strategy_matmul=((parallel_config.data_parallel * parallel_config.model_parallel, 1),))

        self.use_flash_attention = use_flash_attention
        self.use_prompt_flash_attention = use_prompt_flash_attention

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(
                head_num=num_heads,
                pre_tokens=65536,
                next_tokens=0,
                keep_prob=1. - attention_dropout_rate,
                scale_value=1. / math.sqrt(self.size_per_head),
                input_layout="BNSD",
                sparse_mode=0,
                use_attention_mask=True,
            )
            self.flash_attention.shard(parallel_config)
            self.sub = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))

            self.one = Tensor([1.0], dtype=compute_dtype)

        if self.use_prompt_flash_attention:
            self.prompt_flash_attention = PromptFlashAttention(num_heads=num_heads,
                                                               scale_value=1.0 / (math.sqrt(self.size_per_head)),
                                                               pre_tokens=self.src_seq_length,
                                                               next_tokens=0,
                                                               input_layout='BNSD',
                                                               num_key_value_heads=0)
            self.prompt_flash_attention.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                               (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                               (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                               (parallel_config.data_parallel, 1, 1, 1)))
            self.sub_pfa = P.Sub().shard(
                ((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.one = Tensor([1.0], dtype=compute_dtype)

        if parallel_config.select_recompute:
            # recompute is used in _attn, usually with sequence parallel
            self.batch_matmul.recompute()
            self.sub.recompute()
            self.add.recompute()
            self.merger_head_transpose.recompute()
            self.softmax_reshape.recompute()
            self.prob_dropout.recompute()
            self.softmax_cast.recompute()
            self.softmax.softmax.recompute()
            self.softmax_3d.recompute()

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = F.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = F.dtype(query_tensor)
        query_tensor = F.cast(query_tensor, self.dtype)
        key_tensor = F.cast(key_tensor, self.dtype)
        value_tensor = F.cast(value_tensor, self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # FA shape: [bs, num_heads, seq_length, size_per_head] or [bs, num_heads, size_per_head, seq_length]
        if not self.training:
            do_different_shape = (self.use_flash_attention
                                  or self.use_prompt_flash_attention)
        else:
            do_different_shape = self.use_flash_attention

        key_transpose_shape = (0, 2, 1, 3) if do_different_shape else (0, 2, 3, 1)
        key = self.transpose(
            F.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            key_transpose_shape)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        # pfa and fas use 4d mask
        if attention_mask is not None and len(F.shape(attention_mask)) == 3 and \
                (self.use_flash_attention or not self.use_flash_attention or
                 ((not self.training) and self.use_prompt_flash_attention)):
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                expand_axis = 3 if do_different_shape else 2
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, expand_axis))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = batch_valid_length - 1
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                if do_different_shape:
                    multiples = (1, 1, self.seq_length, 1)
                    expand_axis = 3
                else:
                    multiples = (1, 1, 1, self.seq_length)
                    expand_axis = 2
                current_key = self.mul1(self.tile(key, multiples),
                                        self.expand_dims(valid_length_vector, expand_axis))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = F.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        if not self.training and self.use_prompt_flash_attention:
            if self.use_past and not self.is_first_iteration:
                key = self.transpose(key, (0, 1, 3, 2))
                attention = self._attn(query, key, value, attention_mask)
            else:
                query, key, attention_mask = self._pfa_ifa_data_preprocess(query, key, attention_mask,
                                                                           batch_valid_length)
                attention = self.prompt_flash_attention(query, key, value, attention_mask,
                                                        None, None, None, None, None, None, None, None)
                attention = self._merge_heads(attention)
        elif self.use_flash_attention:
            attention_mask = self.cast(attention_mask, mstype.uint8)
            attention = self._flash_attn(query, key, value, attention_mask)
        else:
            attention = self._attn(query, key, value, attention_mask)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        output = F.cast(output, ori_dtype)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        r"""Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if len(F.shape(query)) == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return F.shape(query)[0] // self.src_seq_length
        return F.shape(query)[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _pfa_ifa_data_preprocess(self, query, key, attention_mask, batch_valid_length):
        r"""Return processed q, k and attention mask"""
        if self.use_past and not self.is_first_iteration:
            # Get the precise position index
            current_index = batch_valid_length.squeeze(0)
            index = self.sub1(F.cast(current_index, mstype.int32), 1)
            index = F.reshape(index, (-1, 1, 1))
            # Calculate the attention_mask matrix via the position index
            attention_mask = F.cast(self.tensor_le(self.range, index), P.DType()(query))
            attention_mask = self.expand_dims(attention_mask, 2)

        attention_mask = self.sub_pfa(
            P.Cast()(self.one, P.DType()(query)),
            P.Cast()(attention_mask, P.DType()(query)))

        return query, key, attention_mask

    def _check_inputs(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                      value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(query_tensor), "query_tensor",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(F.dtype(key_tensor), "key_tensor",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        _check_input_dtype(F.dtype(value_tensor), "value_tensor",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(F.dtype(attention_mask), "attention_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16],
                               self.cls_name)

        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(F.dtype(key_past), "key_past", [mstype.float16, mstype.bfloat16], self.cls_name)
            _check_input_dtype(F.dtype(value_past), "value_past", [mstype.float16, mstype.bfloat16], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                self.softmax_reshape(attention_scores, (shape[0], -1, shape[-1])))
            attention_probs = self.softmax_reshape(attention_probs, shape)
        return attention_probs

    def _flash_attn(self, query, key, value, attention_mask):
        """
        flash attention
        """
        if attention_mask is not None:
            attention_mask_dtype = choose_flash_attention_dtype()
            attention_mask = self.sub(
                P.Cast()(self.one, attention_mask_dtype),
                P.Cast()(attention_mask, attention_mask_dtype))

        weighted_values = self.flash_attention(query, key, value, attention_mask)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        attention_scores = self.softmax_cast(score, self.softmax_dtype)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum(F.cast(self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                                (F.shape(query)[0], 1, 1,
                                                                                 self.seq_length),
                                                                                (1, 1, 1, 1)),
                                                                     0), mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(F.cast(current_index, mstype.int32), 1)
                index = F.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = F.cast(self.tensor_le(self.range, index), mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub_sa(
                P.Cast()(F.tuple_to_array((1.0,)), P.DType()(attention_scores)),
                P.Cast()(attention_mask, P.DType()(attention_scores)))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class TransformerEncoderLayer(Cell):
    r"""
        Transformer Encoder Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            num_heads(int): The number of the heads.
            seq_length(int): The input sequence length.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            compute_dtype(dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, seq_length),
              (batch_size, num_heads, seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import TransformerEncoderLayer
            >>> from mindspore import Tensor
            >>> model = TransformerEncoderLayer(batch_size=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
            ...                                 num_heads=2)
            >>> encoder_input_value = Tensor(np.ones((2, 16, 8)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 16, 16)), mstype.float16)
            >>> output, past = model(encoder_input_value, encoder_input_mask)
            >>> print(output.shape)
            (2, 16, 8)
            >>> print(past[0].shape)
            (2, 2, 4, 16)
            >>> print(past[1].shape)
            (2, 2, 16, 4)
            >>> # When use use_past=True, it includes two steps to implement the incremental prediction.
            >>> # Step 1: set is_first_iteration=True, and input the full sequence length's state.
            >>> batch_valid_length = Tensor(np.ones((2,)), mstype.int32)
            >>> init_reset = Tensor([True], mstype.bool_)
            >>> # Set is_first_iteration=True to generate the full memory states
            >>> model = TransformerEncoderLayer(batch_size=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
            ...                                 num_heads=2, use_past=True)
            >>> model.add_flags_recursive(is_first_iteration=True)
            >>> hidden, past = model(encoder_input_value, encoder_input_mask, init_reset, batch_valid_length)
            >>> print(hidden.shape)
            (2, 16, 8)
            >>> print(past[0].shape)
            (2, 2, 4, 16)
            >>> print(past[1].shape)
            (2, 2, 16, 4)
            >>> encoder_input_value = Tensor(np.ones((2, 1, 8)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 1, 16)), mstype.float16)
            >>> init_reset = Tensor([False], mstype.bool_)
            >>> # Step 2: set is_first_iteration=False, and pass the single word to run the prediction rather than
            >>> # the full sequence.
            >>> model.add_flags_recursive(is_first_iteration=False)
            >>> hidden, past = model(encoder_input_value, encoder_input_mask, init_reset, batch_valid_length)
            >>> print(hidden.shape)
            (2, 1, 8)
            >>> print(past[0].shape)
            (2, 2, 4, 16)
            >>> print(past[1].shape)
            (2, 2, 16, 4)
    """

    @_LogActionOnce(m_logger=logger, key='TransformerEncoderLayer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "TransformerEncoderLayer"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "TransformerEncoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "TransformerEncoderLayer"),
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                  "TransformerEncoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "TransformerEncoderLayer"),
                                use_past=Validator.check_bool,
                                use_flash_attention=Validator.check_bool,
                                use_prompt_flash_attention=Validator.check_bool)
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config,
                 use_flash_attention=False,
                 use_prompt_flash_attention=False,
                 compute_dtype=mstype.float16):
        super(TransformerEncoderLayer, self).__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                    .format(ffn_hidden_size, parallel_config.model_parallel))
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = MultiHeadAttention(batch_size=batch_size,
                                                src_seq_length=seq_length,
                                                tgt_seq_length=seq_length,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                parallel_config=attention_parallel_config,
                                                use_flash_attention=use_flash_attention,
                                                use_prompt_flash_attention=use_prompt_flash_attention)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          compute_dtype=compute_dtype,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                    .format(ffn_hidden_size, parallel_config.model_parallel))
            # flash attention / prompt flash attention / incre flash attention version validation
            if use_flash_attention and not check_valid_flash_attention(fa_type="FlashAttention"):
                use_flash_attention = False
                log.info("Current MindSpore do not support flash attention, please upgrade to 2.2.0 or higher")
            if use_prompt_flash_attention and \
                    not check_valid_flash_attention(PROMPTFLASHATTENTION_VALID, "PromptFlashAttention"):
                use_prompt_flash_attention = False
                log.info("Current MindSpore or device do not support prompt flash attention, "
                         "please upgrade to 2.2.0 or higher or use 910B to run pfa")

            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = MultiHeadAttention(batch_size=batch_size,
                                                src_seq_length=seq_length,
                                                tgt_seq_length=seq_length,
                                                hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                parallel_config=attention_parallel_config,
                                                use_flash_attention=use_flash_attention,
                                                use_prompt_flash_attention=use_prompt_flash_attention)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          compute_dtype=compute_dtype,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                if use_prompt_flash_attention or use_flash_attention:
                    self.key_shape = (batch_size, num_heads, seq_length, size_per_head)
                else:
                    self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

            if parallel_config.use_seq_parallel:
                self.add.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),
                                (parallel_config.data_parallel * parallel_config.model_parallel, 1)))
                self.layernorm1.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                self.layernorm2.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                if parallel_config.recompute.select_recompute:
                    # 此处会消耗较大内存，开启后会损失一部分计算性能
                    self.layernorm2.layer_norm.recompute()
                if not self.use_moe:
                    self.output.projection.shard(
                        strategy_bias=((parallel_config.data_parallel * parallel_config.model_parallel, 1), (1,)),
                        strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel, 1)),
                        out_strategy_matmul=((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                    self.output.dropout.dropout.shard(
                        ((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = F.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = F.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)

        if self.use_moe:
            return output, layer_present, aux_loss
        return output, layer_present

    def _check_input(self, x, input_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if input_mask is not None:
            _check_input_dtype(F.dtype(input_mask), "input_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True


class TransformerDecoderLayer(Cell):
    r"""
        Transformer Decoder Layer. This is an implementation of the single layer of the transformer
        decoder layer, including self-attention, cross attention and feedward layer. When the encoder_output is None,
        the cross attention will not be effective.

        Args:
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            num_heads(int): The number of the heads.
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            src_seq_length(int): The input source sequence length.
            tgt_seq_length(int): The input target sequence length.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            use_past(bool): Use the past state to compute, used for incremental prediction. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be dtype.float32 or dtype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **hidden_stats** (Tensor) - The input tensor with shape [batch_size, tgt_seq_length, hidden_size] or
              [batch_size * tgt_seq_length, hidden_size].
            - **decoder_mask** (Tensor) - The attention mask for decoder with shape [batch_size, src_seq_length,
              seq_length] or None. None means there will be no mask in softmax computation in self attention.
            - **encoder_output** (Tensor) - The output of the encoder with shape [batch_size, seq_length, hidden_size]
              or [batch_size * seq_length, hidden_size].
              Note this args can not be passed by None when the net is in outermost layer. Default None.
            - **memory_mask** (Tensor) - The memory mask of the cross attention with shape [batch, tgt_seq_length,
              src_seq_length] where tgt_seq_length is the length of the decoder. The user can also pass None. None
              means there will be no mask in softmax computation in cross attention. Default None.
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - The output logit of this layer. The shape is [batch, seq_length, hidden_size] or
              [batch * seq_length, hidden_size].
            - **layer_present** (Tuple) - A tuple, where each tuple is the tensor of the projected key and value
              vector in self attention with shape ((batch_size, num_heads, size_per_head, tgt_seq_length),
              (batch_size, num_heads, tgt_seq_length, size_per_head), and of the projected key and value vector
              in cross attention with shape  (batch_size, num_heads, size_per_head, src_seq_length),
              (batch_size, num_heads, src_seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import TransformerDecoderLayer
            >>> from mindspore import Tensor
            >>> model = TransformerDecoderLayer(batch_size=2, hidden_size=64, ffn_hidden_size=64, num_heads=2,
            ...                                 src_seq_length=20, tgt_seq_length=10)
            >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
            >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
            >>> decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
            >>> memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
            >>> output, past = model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
            >>> print(output.shape)
            (2, 10, 64)
            >>> print(past[0].shape)
            (2, 2, 32, 10)
            >>> print(past[1].shape)
            (2, 2, 10, 32)
            >>> print(past[2].shape)
            (2, 2, 32, 20)
            >>> print(past[3].shape)
            (2, 2, 20, 32)
    """

    @_LogActionOnce(m_logger=logger, key='TransformerDecoderLayer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "TransformerDecoderLayer"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "TransformerDecoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "TransformerDecoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "TransformerDecoderLayer"),
                                use_past=Validator.check_bool)
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 use_past=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config):
        super(TransformerDecoderLayer, self).__init__()
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        config_to_attention = parallel_config.dpmp if self.use_moe else parallel_config
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'TransformerDecoderLayer', the class variable 'num_heads' must be divisibled by "
                                 "'parallel_config.model_parallel', but got the num_heads is {} and "
                                 "parallel_config.model_parallel is {}.".format(num_heads,
                                                                                parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "'parallel_config.model_parallel', but got the hidden_size is {} and "
                    "parallel_config.model_parallel is {}."
                    .format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError("For 'TransformerDecoderLayer', the class variable 'ffn_hidden_size' must be "
                                 "divisibled by 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                                 "and parallel_config.model_parallel is {}."
                                 .format(ffn_hidden_size, parallel_config.model_parallel))
            if use_past:
                raise ValueError(f"The {self.cls_name} does not support use_past=True.")
            self.batch_size = batch_size
            self.use_past = use_past
            self.softmax_compute_type = softmax_compute_type

            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            self.hidden_size = hidden_size

            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.attention = MultiHeadAttention(hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                batch_size=batch_size,
                                                src_seq_length=tgt_seq_length,
                                                tgt_seq_length=tgt_seq_length,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                use_past=use_past,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                parallel_config=config_to_attention)

            # Cross attention with the output of encoder as memory tensor
            self.cross_attention = MultiHeadAttention(hidden_size=hidden_size,
                                                      num_heads=num_heads,
                                                      batch_size=batch_size,
                                                      src_seq_length=tgt_seq_length,
                                                      tgt_seq_length=src_seq_length,
                                                      hidden_dropout_rate=hidden_dropout_rate,
                                                      attention_dropout_rate=attention_dropout_rate,
                                                      softmax_compute_type=softmax_compute_type,
                                                      use_past=use_past,
                                                      param_init_type=param_init_type,
                                                      parallel_config=config_to_attention)
            self.cross_attention_layernorm = LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type)

            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=hidden_act,
                                          param_init_type=param_init_type,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add()
            self.add_3d = P.Add()
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None
            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, tgt_seq_length)
                self.value_shape = (batch_size, num_heads, tgt_seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError("For 'TransformerDecoderLayer', the class variable 'num_heads' must be divisibled by "
                                 "'parallel_config.model_parallel', but got the num_heads is {} and "
                                 "parallel_config.model_parallel is {}.".format(num_heads,
                                                                                parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerDecoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "'parallel_config.model_parallel', but got the hidden_size is {} and "
                    "parallel_config.model_parallel is {}."
                    .format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError("For 'TransformerDecoderLayer', the class variable 'ffn_hidden_size' must be "
                                 "divisibled by 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                                 "and parallel_config.model_parallel is {}."
                                 .format(ffn_hidden_size, parallel_config.model_parallel))
            if use_past:
                raise ValueError(f"The {self.cls_name} does not support use_past=True.")
            self.batch_size = batch_size
            self.use_past = use_past
            self.softmax_compute_type = softmax_compute_type

            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            self.hidden_size = hidden_size

            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))
            self.attention = MultiHeadAttention(hidden_size=hidden_size,
                                                num_heads=num_heads,
                                                batch_size=batch_size,
                                                src_seq_length=tgt_seq_length,
                                                tgt_seq_length=tgt_seq_length,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                attention_dropout_rate=attention_dropout_rate,
                                                use_past=use_past,
                                                softmax_compute_type=softmax_compute_type,
                                                param_init_type=param_init_type,
                                                parallel_config=config_to_attention)

            # Cross attention with the output of encoder as memory tensor
            self.cross_attention = MultiHeadAttention(hidden_size=hidden_size,
                                                      num_heads=num_heads,
                                                      batch_size=batch_size,
                                                      src_seq_length=tgt_seq_length,
                                                      tgt_seq_length=src_seq_length,
                                                      hidden_dropout_rate=hidden_dropout_rate,
                                                      attention_dropout_rate=attention_dropout_rate,
                                                      softmax_compute_type=softmax_compute_type,
                                                      use_past=use_past,
                                                      param_init_type=param_init_type,
                                                      parallel_config=config_to_attention)
            self.cross_attention_layernorm = LayerNorm((hidden_size,)).to_float(
                layernorm_compute_type)
            self.cross_attention_layernorm.shard(((parallel_config.data_parallel, 1),))

            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=hidden_act,
                                          param_init_type=param_init_type,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None
            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, tgt_seq_length)
                self.value_shape = (batch_size, num_heads, tgt_seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

            if parallel_config.use_seq_parallel:
                self.add.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),
                                (parallel_config.data_parallel * parallel_config.model_parallel, 1)))
                self.layernorm1.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                self.layernorm2.shard(((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                if parallel_config.recompute.select_recompute:
                    # 此处会消耗较大内存，开启后会损失一部分计算性能
                    self.layernorm2.layer_norm.recompute()
                if not self.use_moe:
                    self.output.projection.shard(
                        strategy_bias=((parallel_config.data_parallel * parallel_config.model_parallel, 1), (1,)),
                        strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel, 1)),
                        out_strategy_matmul=((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
                    self.output.dropout.dropout.shard(
                        ((parallel_config.data_parallel * parallel_config.model_parallel, 1),))
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, hidden_stats,
                  decoder_mask,
                  encoder_output=None,
                  memory_mask=None,
                  init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(hidden_stats, decoder_mask, encoder_output, memory_mask, init_reset, batch_valid_length)
        # the returned shape is [bs, seq_length, embedding_size] or [bs * seq_length, embedding_size]
        hidden_shape = F.shape(hidden_stats)
        hidden_stats = F.reshape(hidden_stats, (-1, hidden_shape[-1]))
        input_x = self.layernorm1(hidden_stats)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None
        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, decoder_mask, self.key_past,
                                                  self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(hidden_stats, attention)

        middle_output = None
        if encoder_output is not None:
            middle_output = self.cross_attention_layernorm(x)
            middle_output = F.cast(middle_output, self.dtype)
            encoder_output = F.cast(encoder_output, self.dtype)
            cross_attn_output, cross_layer_present = self.cross_attention(middle_output, encoder_output,
                                                                          encoder_output,
                                                                          memory_mask, self.key_past,
                                                                          self.value_past, batch_valid_length)
            layer_present += cross_layer_present
            if self.post_layernorm_residual:
                x = self.add(middle_output, cross_attn_output)
            else:
                x = self.add(x, cross_attn_output)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(hidden_shape) == 3:
            output_x = P.Reshape()(output_x, hidden_shape)
            mlp_logit = P.Reshape()(mlp_logit, hidden_shape)
            x = P.Reshape()(x, hidden_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, hidden_shape)

        if self.use_moe:
            return output, layer_present, aux_loss
        return output, layer_present

    def _check_input(self, hidden_states, attention_mask, encoder_output, memory_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(hidden_states), "hidden_states",
                           [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(F.dtype(attention_mask), "attention_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16],
                               self.cls_name)
        if encoder_output is not None:
            _check_input_dtype(F.dtype(encoder_output), "encoder_output",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)
        if memory_mask is not None:
            _check_input_dtype(F.dtype(memory_mask), "memory_mask",
                               [mstype.float32, mstype.float16, mstype.bfloat16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)


def _get_lambda_func(total_layer=None):
    r"""
    A wrapper function of specifying pipeline stage and gradient aggregation fusion. If the total layer
    is not None, for example, set in the transformer model, the pipeline stage setting function will be
    `(layer_id + 0) // (total_layers / parallel_config.pipeline_stage)` for the encoder and,
    `(layer_id + offset) //
    (total_layers / parallel_config.pipeline_stage)` for the decoder, where `offset` is the layers in the encoder.
    """

    def _set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
        r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs an offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
        """
        # override the layers
        if total_layer:
            layers = total_layer
        # Used for the pipeline's stages setting
        if layers < parallel_config.pipeline_stage:
            raise ValueError(f"layers {layers} must be larger than pipeline stage {parallel_config.pipeline_stage}")

        pp_dis = max(layers // parallel_config.pipeline_stage, 1)
        # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
        pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
        network.pipeline_stage = pp_id

        # Used for optimizer's fusion tag
        dis = max(layers // parallel_config.gradient_aggregation_group, 1)
        network.set_comm_fusion((layer_id + offset) // dis + 1)
        # Used for enabling recomputation of the block
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute and not parallel_config.select_recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute and not parallel_config.recompute.select_recompute:
                paralel_op_comm_compute = parallel_config.recompute.parallel_optimizer_comm_recompute
                network.recompute(parallel_optimizer_comm_recompute=paralel_op_comm_compute,
                                  mp_comm_recompute=parallel_config.recompute.mp_comm_recompute,
                                  recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    return _set_parallel_configure_for_layer


class TransformerEncoder(Cell):
    r"""
        Transformer Encoder module with multi-layer stacked of `TransformerEncoderLayer`, including multihead self
        attention and feedforward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            num_layers(int): The layers of the `TransformerEncoderLayer`
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            seq_length(int): The seq_length of the input tensor.
            num_heads(int): The number of the heads.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default: 0.1.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default: mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default: mstype.float32.
            lambda_func(function): A function can determine the fusion index,
                pipeline stages and recompute attribute. If the
                user wants to determine the pipeline stage and gradient aggregation fusion, the user can pass a
                function that accepts `network`, `layer_id`, `offset`, `parallel_config`, `layers`. The `network(Cell)`
                represents the transformer block, `layer_id(int)` means the layer index for the current module, counts
                from zero, `offset(int)` means the layer_index needs an offset, if there are other modules in the net.
                The default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.
                Default: None.
            offset(int): The initial layer index for the `encoder`. Used for setting the fusion id and stage id, to not
                overlap with the encoder layer. Default 0.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default: False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
                an instance of `TransformerOpParallelConfig` with default args.

        Inputs:
            - **hidden_states** (Tensor) - Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size].
            - **attention_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).
            - **layer_present** (Tuple) - A tuple with size of num_layers, where each tuple contains the Tensor the
              projected key and value vector with shape ((batch_size, num_heads, size_per_head, seq_length),
              and (batch_size, num_heads, seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import TransformerEncoder
            >>> from mindspore import Tensor
            >>> model = TransformerEncoder(batch_size=2, num_layers=2, hidden_size=8, ffn_hidden_size=64,
            ...                            seq_length=16, num_heads=2)
            >>> encoder_input_value = Tensor(np.ones((2, 16, 8)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 16, 16)), mstype.float16)
            >>> output, past = model(encoder_input_value, encoder_input_mask)
            >>> print(output.shape)
            (2, 16, 8)
            >>> print(len(past))
            2
            >>> print(past[0][0].shape)
            (2, 2, 4, 16)
            >>> print(past[0][1].shape)
            (2, 2, 16, 4)
            >>> # When use use_past=True, it includes two steps to implement the incremental prediction.
            >>> # Step 1: set is_first_iteration=True, and input the full sequence length's state.
            >>> batch_valid_length = Tensor(np.ones((2,)), mstype.int32)
            >>> init_reset = Tensor([True], mstype.bool_)
            >>> # Set is_first_iteration=True to generate the full memory states
            >>> model = TransformerEncoder(batch_size=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
            ...                            num_heads=2, num_layers=2, use_past=True)
            >>> model.add_flags_recursive(is_first_iteration=True)
            >>> hidden, past = model(encoder_input_value, encoder_input_mask, init_reset, batch_valid_length)
            >>> print(hidden.shape)
            (2, 16, 8)
            >>> print(past[0][0].shape)
            (2, 2, 4, 16)
            >>> print(past[0][1].shape)
            (2, 2, 16, 4)
            >>> encoder_input_value = Tensor(np.ones((2, 1, 8)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 1, 16)), mstype.float16)
            >>> init_reset = Tensor([False], mstype.bool_)
            >>> # Step 2: set is_first_iteration=False, and pass the single word to run the prediction rather than
            >>> # the full sequence.
            >>> model.add_flags_recursive(is_first_iteration=False)
            >>> hidden, past = model(encoder_input_value, encoder_input_mask, init_reset, batch_valid_length)
            >>> print(hidden.shape)
            (2, 1, 8)
            >>> print(past[0][0].shape)
            (2, 2, 4, 16)
            >>> print(past[0][1].shape)
            (2, 2, 16, 4)
    """

    @_LogActionOnce(m_logger=logger, key='TransformerEncoder',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                seq_length=Validator.check_positive_int,
                                num_layers=Validator.check_positive_int,
                                offset=Validator.check_non_negative_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "TransformerEncoder"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "TransformerEncoder"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "TransformerEncoder"),
                                parallel_config=_valid_type_checks([TransformerOpParallelConfig],
                                                                   "TransformerEncoder"),
                                use_past=Validator.check_bool)
    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super(TransformerEncoder, self).__init__()
        _check_config(parallel_config)
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = TransformerEncoderLayer(hidden_size=hidden_size,
                                                batch_size=batch_size,
                                                ffn_hidden_size=ffn_hidden_size,
                                                seq_length=seq_length,
                                                attention_dropout_rate=attention_dropout_rate,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                layernorm_compute_type=layernorm_compute_type,
                                                softmax_compute_type=softmax_compute_type,
                                                num_heads=num_heads,
                                                hidden_act=hidden_act,
                                                post_layernorm_residual=post_layernorm_residual,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                moe_config=moe_config,
                                                parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = TransformerEncoderLayer(hidden_size=hidden_size,
                                                batch_size=batch_size,
                                                ffn_hidden_size=ffn_hidden_size,
                                                seq_length=seq_length,
                                                attention_dropout_rate=attention_dropout_rate,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                layernorm_compute_type=layernorm_compute_type,
                                                softmax_compute_type=softmax_compute_type,
                                                num_heads=num_heads,
                                                hidden_act=hidden_act,
                                                post_layernorm_residual=post_layernorm_residual,
                                                param_init_type=param_init_type,
                                                use_past=use_past,
                                                moe_config=moe_config,
                                                parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None):
        """forward process"""
        present_layer = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, present, aux_loss = self.blocks[i](hidden_states,
                                                                  attention_mask,
                                                                  init_reset,
                                                                  batch_valid_length)
                present_layer = present_layer + (present,)
                accum_loss = self.add(accum_loss, aux_loss)
            return hidden_states, present_layer, accum_loss

        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer


class TransformerDecoder(Cell):
    r"""
        Transformer Decoder module with multi-layer stacked of `TransformerDecoderLayer`, including multihead self
        attention, cross attention and feedforward layer.

        Args:
            num_layers(int): The layers of the `TransformerDecoderLayer`.
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            src_seq_length(int): The input source sequence length.
            tgt_seq_length(int): The input target sequence length.
            num_heads(int): The number of the heads.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            lambda_func(function): A function can determine the fusion index,
                pipeline stages and recompute attribute. If the
                user wants to determine the pipeline stage and gradient aggregation fusion, the user can pass a
                function that accepts `network`, `layer_id`, `offset`, `parallel_config`, `layers`. The `network(Cell)`
                represents the transformer block, `layer_id(int)` means the layer index for the current module, counts
                from zero, `offset(int)` means the layer_index needs an offset, if there are other modules in the net.
                The default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.
                Default: None.
            use_past(bool): Use the past state to compute, used for incremental prediction. Default False.
            offset(int): The initial layer index for the `decoder`. Used for setting the fusion id and stage id, to not
                overlap with the encoder layer. Default 0.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
                an instance of `TransformerOpParallelConfig` with default args.

        Inputs:
            - **hidden_stats** (Tensor) - The input tensor with shape [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size]
            - **attention_mask** (Tensor) - The attention mask for decoder with shape
              [batch_size, seq_length, seq_length] or None. None means there will be no mask in softmax
              computation in self attention.
            - **encoder_output** (Tensor) - The output of the encoder with shape [batch_size, seq_length, hidden_size]
              or [batch_size * seq_length, hidden_size]. Note this args can not be passed by None when the net is in
              outermost layer. Default None.
            - **memory_mask** (Tensor) - The memory mask of the cross attention with shape [batch, tgt_seq_length,
              src_seq_length] where tgt_seq_length is the length of the decoder. The user can also pass None. None
              means there will be no mask in softmax computation in cross attention. Default None.
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - The output logit of this layer. The shape is [batch, tgt_seq_length, hidden_size] or
              [batch * tgt_seq_length, hidden_size]
            - **layer_present** (Tuple) - A tuple with size of num_layers, where each tuple is the tensor of the
              projected key and value vector in self attention with shape ((batch_size, num_heads, size_per_head,
              tgt_seq_length), (batch_size, num_heads, tgt_seq_length, size_per_head), and of the projected key
              and value vector in cross attention with shape  (batch_size, num_heads, size_per_head, src_seq_length),
              (batch_size, num_heads, src_seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import TransformerDecoder
            >>> from mindspore import Tensor
            >>> model = TransformerDecoder(batch_size=2, num_layers=1, hidden_size=64, ffn_hidden_size=64,
            ...                            num_heads=2, src_seq_length=20, tgt_seq_length=10)
            >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
            >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
            >>> decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
            >>> memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
            >>> output, past = model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)
            >>> print(output.shape)
            (2, 10, 64)
            >>> print(len(past))
            1
            >>> print(past[0][0].shape)
            (2, 2, 32, 10)
            >>> print(past[0][1].shape)
            (2, 2, 10, 32)
            >>> print(past[0][2].shape)
            (2, 2, 32, 20)
            >>> print(past[0][3].shape)
            (2, 2, 20, 32)
    """

    @_LogActionOnce(m_logger=logger, key='TransformerDecoder',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                num_layers=Validator.check_positive_int,
                                tgt_seq_length=Validator.check_positive_int,
                                offset=Validator.check_non_negative_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "TransformerDecoder"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "TransformerDecoder"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16, mstype.bfloat16],
                                                                    "TransformerDecoder"),
                                parallel_config=_valid_type_checks([TransformerOpParallelConfig],
                                                                   "TransformerDecoder"),
                                use_past=Validator.check_bool)
    def __init__(self,
                 num_layers,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 src_seq_length,
                 tgt_seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 lambda_func=None,
                 use_past=False,
                 offset=0,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super(TransformerDecoder, self).__init__()
        _check_moe_config(moe_config, parallel_config)
        _check_config(parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()

            for i in range(num_layers):
                block = TransformerDecoderLayer(hidden_size=hidden_size,
                                                batch_size=batch_size,
                                                ffn_hidden_size=ffn_hidden_size,
                                                src_seq_length=src_seq_length,
                                                tgt_seq_length=tgt_seq_length,
                                                attention_dropout_rate=attention_dropout_rate,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                num_heads=num_heads,
                                                layernorm_compute_type=layernorm_compute_type,
                                                softmax_compute_type=softmax_compute_type,
                                                hidden_act=hidden_act,
                                                use_past=use_past,
                                                param_init_type=param_init_type,
                                                post_layernorm_residual=post_layernorm_residual,
                                                moe_config=moe_config,
                                                parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)

                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = TransformerDecoderLayer(hidden_size=hidden_size,
                                                batch_size=batch_size,
                                                ffn_hidden_size=ffn_hidden_size,
                                                src_seq_length=src_seq_length,
                                                tgt_seq_length=tgt_seq_length,
                                                attention_dropout_rate=attention_dropout_rate,
                                                hidden_dropout_rate=hidden_dropout_rate,
                                                num_heads=num_heads,
                                                layernorm_compute_type=layernorm_compute_type,
                                                softmax_compute_type=softmax_compute_type,
                                                hidden_act=hidden_act,
                                                use_past=use_past,
                                                param_init_type=param_init_type,
                                                post_layernorm_residual=post_layernorm_residual,
                                                moe_config=moe_config,
                                                parallel_config=config_to_layer)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)

                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, hidden_states, attention_mask, encoder_output=None, memory_mask=None,
                  init_reset=True, batch_valid_length=None):
        """forward process"""
        present_layer = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, present, aux_loss = self.blocks[i](hidden_states,
                                                                  attention_mask,
                                                                  encoder_output,
                                                                  memory_mask,
                                                                  init_reset,
                                                                  batch_valid_length)
                present_layer = present_layer + (present,)
                accum_loss = self.add(accum_loss, aux_loss)
            return hidden_states, present_layer, accum_loss

        # Loop through each self-attention layer
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask,
                                                    encoder_output,
                                                    memory_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer


class Transformer(Cell):
    r"""
        Transformer module including encoder and decoder. The difference with the original implements is the module use
        the residual addition before the layer normalization. And the default hidden act is `gelu`.
        The details can be found in `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_.

        Note:
            This is an experimental interface that is subject to change or deletion.

        Args:
            hidden_size(int): The hidden size of the input.
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            src_seq_length(int): The seq_length of the encoder's input tensor.
            tgt_seq_length(int): The seq_length of the decoder's input tensor.
            encoder_layers(int): The layers of the `TransformerEncoderLayer`. Default 3.
            decoder_layers(int): The layers of the `TransformerDecoderLayer`. Default 3.
            num_heads(int): The number of the heads. Default: 2.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be dtype.float32 or dtype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be dtype.float32 or dtype.float16. Default dtype.float32.
            lambda_func: A function can determine the fusion index, pipeline stages and recompute attribute. If the user
                wants to determine the pipeline stage and gradient aggregation fusion, the user can pass a function
                that accepts `network`, `layer_id`, `offset`, `parallel_config`, `layers`. The `network(Cell)`
                represents the transformer block, `layer_id(int)` means the layer index for the current module, counts
                from zero, `offset(int)` means the layer_index needs an offset, if there are other modules in the net.
                The default setting for the pipeline is: `(layer_id + offset) // ((encoder_layers + decoder_layers)
                / pipeline_stage)`. Default None.
            use_past(bool): Use the past state to compute, used for incremental prediction. Default False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
                an instance of `TransformerOpParallelConfig` with default args.

        Inputs:
            - **encoder_inputs** (Tensor) - The input tensor with shape [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size].
            - **encoder_masks** (Tensor) - The attention mask for decoder with shape
              [batch_size, seq_length, seq_length] or None. None means there will be no mask in softmax computation
              in self attention of the encoder module.
            - **decoder_inputs** (Tensor) - The output of the encoder with shape [batch_size, seq_length, hidden_size]
              or [batch_size * seq_length, hidden_size], this should be none if the decoder layer is 0.
            - **decoder_masks** (Tensor) - The attention mask for decoder with shape
              [batch_size, seq_length, seq_length] or None. None means there will be no mask in softmax computation
              in self attention of the decoder module.
            - **memory_mask** (Tensor) - The memory mask of the cross attention with shape [batch, tgt_seq_length,
              src_seq_length]
              where tgt_seq_length is the length of the decoder. The output of the encoder with shape [batch_size,
              seq_length, hidden_size], this should be none if the decoder layer is 0 or the user wants no mask.
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `encoder_layer_present`, `decoder_layer_present`, `accum_loss`)

            - **output** (Tensor) - If there is only encoder, the output logit of the encoder layer. The shape is
              [batch, src_seq_length, hidden_size] or [batch * src_seq_length, hidden_size], if there are encoder and
              decoders, the output is from the decoder layer. The shape is [batch, tgt_seq_length, hidden_size] or
              [batch * tgt_seq_length, hidden_size].
            - **encoder_layer_present** (Tuple) - A tuple with size of num_layers, where each tuple is the tensor the
              projected key and value vector in self attention with shape ((batch_size, num_heads, size_per_head,
              src_seq_length), (batch_size, num_heads, src_seq_length, size_per_head)).
            - **decoder_layer_present** (Tuple) - A tuple with size of num_layers, where each tuple is the tensor
              of the projected key and value vector in self attention with shape ((batch_size, num_heads, size_per_head,
              tgt_seq_length), (batch_size, num_heads, tgt_seq_length, size_per_head)), and the
              projected key and value vector in cross attention with shape
              ((batch_size, num_heads, size_per_head, src_seq_length),
              (batch_size, num_heads, src_seq_length, size_per_head)). If the decoder is not set, the
              returned value will be None.
            - **accum_loss** (Tensor) - A Tensor indicates an auxiliary loss to minimize the mean square of the data
              part routed to each expert, and only returned if the number of experts is greater than 1.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import dtype as mstype
            >>> from mindformers.modules.transformer import Transformer
            >>> from mindspore import Tensor
            >>> model = Transformer(batch_size=2, encoder_layers=1, decoder_layers=2, hidden_size=64,
            ...                     ffn_hidden_size=64, src_seq_length=20, tgt_seq_length=10)
            >>> encoder_input_value = Tensor(np.ones((2, 20, 64)), mstype.float32)
            >>> encoder_input_mask = Tensor(np.ones((2, 20, 20)), mstype.float16)
            >>> decoder_input_value = Tensor(np.ones((2, 10, 64)), mstype.float32)
            >>> decoder_input_mask = Tensor(np.ones((2, 10, 10)), mstype.float16)
            >>> memory_mask = Tensor(np.ones((2, 10, 20)), mstype.float16)
            >>> output, en_past, de_past = model(encoder_input_value, encoder_input_mask, decoder_input_value,
            ...                                  decoder_input_mask, memory_mask)
            >>> print(output.shape)
            (2, 10, 64)
            >>> print(len(en_past))
            1
            >>> print(len(de_past))
            2
            >>> print(en_past[0][0].shape)
            (2, 2, 32, 20)
            >>> print(en_past[0][1].shape)
            (2, 2, 20, 32)
            >>> print(de_past[0][0].shape)
            (2, 2, 32, 10)
            >>> print(de_past[0][1].shape)
            (2, 2, 10, 32)
            >>> print(de_past[0][2].shape)
            (2, 2, 32, 20)
            >>> print(de_past[0][3].shape)
            (2, 2, 20, 32)
    """

    @_LogActionOnce(m_logger=logger, key='Transformer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                encoder_layers=Validator.check_positive_int,
                                decoder_layers=Validator.check_non_negative_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32,
                                                                            mstype.float16, mstype.bfloat16],
                                                                           "Transformer"),
                                softmax_compute_type=_valid_value_checks([mstype.float32,
                                                                          mstype.float16, mstype.bfloat16],
                                                                         "Transformer"),
                                param_init_type=_valid_value_checks([mstype.float32,
                                                                     mstype.float16, mstype.bfloat16], "Transformer"),
                                parallel_config=_valid_type_checks([TransformerOpParallelConfig], "Transformer"),
                                use_past=Validator.check_bool)
    def __init__(self,
                 hidden_size,
                 batch_size,
                 ffn_hidden_size,
                 src_seq_length,
                 tgt_seq_length,
                 encoder_layers=3,
                 decoder_layers=3,
                 num_heads=2,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config):
        super(Transformer, self).__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            if encoder_layers <= 0 < decoder_layers:
                raise ValueError(f"Transformer doest support encoder layer {encoder_layers} and decoder"
                                 f"layer {decoder_layers}, please use TransformerDecoder")
            if encoder_layers > 0 and decoder_layers > 0 and use_past:
                raise ValueError(f"The {self.cls_name} with encoder and decoder does not support use_past=True.")
            # The shard setting of Transformer is set within the TransformerEncoderLayer
            if not lambda_func:
                lambda_func = _get_lambda_func(total_layer=encoder_layers + decoder_layers)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoder(num_layers=encoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  seq_length=src_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  param_init_type=param_init_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
            else:
                self.encoder = None

            # Offset is needed as the encoder has consumed some flags.
            # so the decoder need to increase the flags based on the encoder layer
            self.decoder = None
            if decoder_layers > 0:
                self.decoder = TransformerDecoder(num_layers=decoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  src_seq_length=src_seq_length,
                                                  tgt_seq_length=tgt_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  param_init_type=param_init_type,
                                                  offset=encoder_layers,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.use_past = use_past
            if encoder_layers <= 0 < decoder_layers:
                raise ValueError(f"Transformer doest support encoder layer {encoder_layers} and decoder"
                                 f"layer {decoder_layers}, please use TransformerDecoder")
            if encoder_layers > 0 and decoder_layers > 0 and use_past:
                raise ValueError(f"The {self.cls_name} with encoder and decoder does not support use_past=True.")
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            # The shard setting of Transformer is set within the TransformerEncoderLayer
            if not lambda_func:
                lambda_func = _get_lambda_func(total_layer=encoder_layers + decoder_layers)
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoder(num_layers=encoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  seq_length=src_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  param_init_type=param_init_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
            else:
                self.encoder = None

            # Offset is needed as the encoder has consumed some flags.
            # so the decoder need to increase the flags based on the encoder layer
            self.decoder = None
            if decoder_layers > 0:
                self.decoder = TransformerDecoder(num_layers=decoder_layers,
                                                  batch_size=batch_size,
                                                  hidden_size=hidden_size,
                                                  ffn_hidden_size=ffn_hidden_size,
                                                  num_heads=num_heads,
                                                  src_seq_length=src_seq_length,
                                                  tgt_seq_length=tgt_seq_length,
                                                  attention_dropout_rate=attention_dropout_rate,
                                                  hidden_dropout_rate=hidden_dropout_rate,
                                                  hidden_act=hidden_act,
                                                  post_layernorm_residual=post_layernorm_residual,
                                                  layernorm_compute_type=layernorm_compute_type,
                                                  softmax_compute_type=softmax_compute_type,
                                                  lambda_func=lambda_func,
                                                  use_past=use_past,
                                                  param_init_type=param_init_type,
                                                  offset=encoder_layers,
                                                  moe_config=moe_config,
                                                  parallel_config=parallel_config)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, encoder_inputs,
                  encoder_masks,
                  decoder_inputs=None,
                  decoder_masks=None,
                  memory_mask=None,
                  init_reset=True,
                  batch_valid_length=None):
        """forward process"""
        encoder_output = None
        output = None
        encoder_layer_present = None
        decoder_layer_present = None
        accum_loss = self.aux_loss
        if self.encoder is not None:
            if self.use_moe:
                encoder_output, encoder_layer_present, encoder_aux_loss = self.encoder(encoder_inputs, encoder_masks,
                                                                                       init_reset, batch_valid_length)
                accum_loss = self.add(accum_loss, encoder_aux_loss)
            else:
                encoder_output, encoder_layer_present = self.encoder(encoder_inputs, encoder_masks, init_reset,
                                                                     batch_valid_length)
            output = encoder_output

        if self.decoder is not None:
            # decoder mask should be created outside the model
            if self.use_moe:
                decoder_output, decoder_layer_present, decoder_aux_loss = self.decoder(decoder_inputs, decoder_masks,
                                                                                       encoder_output, memory_mask,
                                                                                       init_reset, batch_valid_length)
                accum_loss = self.add(accum_loss, decoder_aux_loss)
            else:
                decoder_output, decoder_layer_present = self.decoder(decoder_inputs,
                                                                     decoder_masks,
                                                                     encoder_output,
                                                                     memory_mask, init_reset,
                                                                     batch_valid_length)
            output = decoder_output
        if self.use_moe:
            return output, encoder_layer_present, decoder_layer_present, accum_loss
        return output, encoder_layer_present, decoder_layer_present
