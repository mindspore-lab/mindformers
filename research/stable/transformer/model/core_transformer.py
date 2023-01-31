# Copyright 2022 Huawei Technologies Co., Ltd
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

import math
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import nn
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config
from mindspore.nn.transformer.transformer import TransformerEncoderLayer, TransformerDecoderLayer
from mindspore.nn.transformer.transformer import EmbeddingOpParallelConfig, TransformerRecomputeConfig, TransformerOpParallelConfig
from mindspore.nn.transformer.transformer import FeedForward, _get_lambda_func
from mindspore.nn.transformer.transformer import TransformerEncoder, TransformerDecoder, Transformer
from research.ntlb.transformer.model.moe import default_moe_config, MoE


__all__ = [
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayerM",
    "TransformerDecoderLayerM",
    "Crtransformer"]

default_transformer_recompute_config = TransformerRecomputeConfig()
default_transformer_config = TransformerOpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()


class TransformerEncoderLayerM(TransformerEncoderLayer):
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
                 parallel_config=default_dpmp_config):
        super(TransformerEncoderLayerM, self).__init__()
        self.use_moe = (moe_config.expert_num > 1)
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
        

class TransformerDecoderLayerM(TransformerDecoderLayer):
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
        super(TransformerDecoderLayerM, self).__init__()
        self.use_moe = (moe_config.expert_num > 1)
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


class TransformerEncoderM(TransformerEncoder):
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
        super(TransformerEncoderM, self).__init__()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            # _check_config(parallel_config)
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            # parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
            for i in range(num_layers):
                block = TransformerEncoderLayerM(hidden_size=hidden_size,
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
                                                parallel_config=parallel_config_args)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            # _check_config(parallel_config)
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            # parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
            for i in range(num_layers):
                block = TransformerEncoderLayerM(hidden_size=hidden_size,
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
                                                parallel_config=parallel_config.moe_parallel_config if self.use_moe
                                                else parallel_config.dp_mp_config)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")


class TransformerDecoderM(TransformerDecoder):
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
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            # _check_config(parallel_config)

            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            # parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
            for i in range(num_layers):
                block = TransformerDecoderLayerM(hidden_size=hidden_size,
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
                                                parallel_config=parallel_config_args)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)

                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            # _check_config(parallel_config)

            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            # parallel_config_args = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
            for i in range(num_layers):
                block = TransformerDecoderLayerM(hidden_size=hidden_size,
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
                                                parallel_config=parallel_config.moe_parallel_config if self.use_moe
                                                else parallel_config.dp_mp_config)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)

                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")


class Crtransformer(Transformer):
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
        super(Crtransformer, self).__init__()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            # _check_config(parallel_config)
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
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoderM(num_layers=encoder_layers,
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
                self.decoder = TransformerDecoderM(num_layers=decoder_layers,
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
            # _check_config(parallel_config)
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
            # _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            if encoder_layers > 0:
                self.encoder = TransformerEncoderM(num_layers=encoder_layers,
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
                self.decoder = TransformerDecoderM(num_layers=decoder_layers,
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