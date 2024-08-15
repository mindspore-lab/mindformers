# Copyright 2024 Huawei Technologies Co., Ltd
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
# This file was refer to project:
# https://huggingface.co/Qwen/Qwen-VL
# ============================================================================
"""CogVLM2 models' APIs."""
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import nn, Tensor, Parameter
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.models.build_model import build_network
from mindformers.models.utils import convert_mstype
from mindformers.modules.activation import GELU, SiLU
from mindformers.modules.layers import Linear, LayerNorm
from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from .cogvlm2_config import CogVLM2Config
from ..multi_modal.base_model import BaseXModalToTextModel

__all__ = ['CogVLM2ForCausalLM']


class GLU(nn.Cell):
    """
    The implementation of MLP module with GELU function.
    """
    def __init__(self, in_features, hidden_size, intermediate_size,
                 compute_dtype=ms.float16, param_init_type=ms.float16):
        super().__init__()

        self.linear_proj = Linear(in_features, hidden_size, has_bias=False)
        self.norm1 = LayerNorm(hidden_size)
        self.act1 = GELU(approximate=False)
        self.act2 = SiLU()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size, has_bias=False,
                                    compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.gate_proj = Linear(hidden_size, intermediate_size, has_bias=False,
                                compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size, has_bias=False,
                                    compute_dtype=compute_dtype, param_init_type=param_init_type)

        self.mul = P.Mul()

    def construct(self, x):
        """Visual GLU Forward."""
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.mul(self.act2(self.gate_proj(x)), self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class VisionMLPAdapter(nn.Cell):
    """
    The implementation of visual-text adapter module for cogvlm2 model.
    """
    def __init__(self, vision_grid_size, vision_hidden_size, text_hidden_size, text_intermediate_size,
                 compute_dtype=ms.float16, param_init_type=ms.float16):
        super().__init__()

        self.grid_size = vision_grid_size

        self.linear_proj = GLU(in_features=vision_hidden_size,
                               hidden_size=text_hidden_size,
                               intermediate_size=text_intermediate_size,
                               compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.conv = nn.Conv2d(in_channels=vision_hidden_size, out_channels=vision_hidden_size,
                              kernel_size=2, stride=2, dtype=param_init_type, has_bias=True)
        self.boi = Parameter(ops.zeros((1, 1, text_hidden_size), dtype=param_init_type))
        self.eoi = Parameter(ops.zeros((1, 1, text_hidden_size), dtype=param_init_type))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.flatten = nn.Flatten(start_dim=2)
        self.transpose = P.Transpose()
        self.broadcast_to = ops.broadcast_to
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """VisionMLPAdapter Forward."""
        bs, _, hidden_size = self.shape(x)
        x = self.reshape(x, (bs, self.grid_size, self.grid_size, hidden_size))
        x = self.transpose(x, (0, 3, 1, 2))
        x = self.conv(x)
        x = self.flatten(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.linear_proj(x)
        boi = self.broadcast_to(self.boi, (bs, -1, -1))
        eoi = self.broadcast_to(self.eoi, (bs, -1, -1))
        x = self.concat((boi, x, eoi))
        return x


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class CogVLM2ForCausalLM(BaseXModalToTextModel):
    """
    Provide CogVLM2 training loss or logits through network.

    Args:
        config (CogVLM2Config): The config of CogVLM2 model.
    """

    # @lazy_inline
    def __init__(self, config: CogVLM2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.is_video = config.is_video
        self.vision_config = config.vision_model.model_config
        self.llm_config = config.llm_model.model_config

        self.image_size = self.vision_config.image_size
        self.image_patch_size = self.vision_config.patch_size
        self.image_grid_size = int(self.image_size / self.image_patch_size)

        self.vision_encoder = build_network(config.vision_model)
        self.llm_model = build_network(config.llm_model)
        self.mlp_adapter = VisionMLPAdapter(
            self.image_grid_size, vision_hidden_size=self.vision_config.hidden_size,
            text_hidden_size=self.llm_config.hidden_size,
            text_intermediate_size=self.llm_config.intermediate_size,
            compute_dtype=convert_mstype(self.vision_config.compute_dtype),
            param_init_type=convert_mstype(self.vision_config.param_init_type)
        )

        self.image_start_id = self.config.image_start_id
        self.image_pad_id = self.config.image_pad_id
        self.num_queries = self.config.num_queries

        self.use_past = self.config.use_past
        self.is_first_iteration = True
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.ignore_token_id = ms.Tensor(config.ignore_token_id, mstype.int32)

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.ones = P.Ones()
        self.concat = P.Concat(axis=1)
        self.gather_nd = P.GatherNd()
        self.expand_dims = P.ExpandDims()

        parallel_config = config.parallel_config
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.slice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.masked_fill = P.MaskedFill().shard(
            ((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1), ()))

        self.gather = P.Gather().shard(((1, 1, 1), ()))
        self.equal = P.Equal().shard(((parallel_config.data_parallel, 1), ()))

        self.batch_idx = None

    def generate_batch_idx(self, batch_size):
        """Generate batched index."""
        if self.batch_idx is None:
            self.batch_idx = Tensor([[i] for i in range(batch_size)], dtype=mstype.int32)

    # pylint: disable=W0613
    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        """Update model kwargs before generate."""
        batch_size, _ = input_ids.shape
        self.generate_batch_idx(batch_size)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation in inference."""
        is_first_iteration = self.is_first_iteration
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
            is_first_iteration = True
        batch_size, _ = input_ids.shape

        position_ids = kwargs.pop("position_ids")
        if is_first_iteration or not self.use_past:
            images = kwargs.pop("images")
            video_context_pos = kwargs.pop("video_context_pos")
        else:
            # generate data not used in incremental predict
            img_shape = (batch_size * 24, 3, self.image_size, self.image_size)
            images = self.ones(img_shape, ms.float32)
            video_context_pos = self.ones((batch_size * 24, self.config.num_queries, 2), mstype.int32)

        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "images": images,
            "video_context_pos": video_context_pos,
            "position_ids": position_ids
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Prepare inputs for predict layout."""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq_length = F.shape(input_ids)

        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((bs, 3, self.image_size, self.image_size)), ms.float32)

        if "img_pos" in kwargs:
            img_pos = Tensor(kwargs.get("img_pos"))
        else:
            img_pos = Tensor(np.random.randint(0, self.num_queries, (bs, self.num_queries, 2)), ms.int32)

        if 'position_ids' in kwargs:
            position_ids = Tensor(kwargs.get("position_ids"))
        else:
            position_ids = Tensor(np.tile(np.arange(seq_length), bs).reshape((bs, seq_length)), ms.int32)

        self.generate_batch_idx(bs)
        slot_mapping = Tensor(np.ones(shape=tuple([bs])), mstype.int32)
        return input_ids, images, img_pos, None, None, position_ids, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self, **kwargs):
        """Set dynamic inputs for model."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
        dynamic_video_context_pos = Tensor(shape=[None, None, None], dtype=mstype.int32)
        dynamic_position_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_video_context_pos, None, None,
                        dynamic_position_ids, None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        self.llm_model.set_dynamic_inputs()
        logger.info("Set dynamic inputs for CogVLM2")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.llm_model.add_flags_custom(is_first_iteration=is_first_iteration)

    def kvcache(self, layer_idx):
        """Get kvcache from llm with input layer index."""
        return self.llm_model.kvcache(layer_idx)

    def gat_batched_position(self, batch_valid_length, position_ids):
        """Get batched position_ids when use dynamic inputs."""
        batch_size, _ = F.shape(position_ids)
        position_idx = self.concat((self.batch_idx, self.reshape(batch_valid_length, (batch_size, -1))))
        position_ids = self.reshape(self.gather_nd(position_ids, position_idx), (batch_size, -1))
        return position_ids

    def construct(self, input_ids, images, video_context_pos: Tensor = None, labels=None,
                  input_position=None, position_ids=None, attention_mask=None, init_reset=None, batch_valid_length=None,
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None):
        """Forward of CogVLM2"""
        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.stride_slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
            if labels is None:
                pad_input_ids_pos = self.equal(input_ids, self.pad_token_id)
                labels = self.masked_fill(input_ids, pad_input_ids_pos, self.ignore_token_id)
                pad_label_pos = self.equal(labels, self.pad_token_id)
                labels = self.masked_fill(labels, pad_label_pos, self.ignore_token_id)
        else:
            tokens = input_ids

        input_embeds = self.llm_model.to_embeddings(tokens)

        if attention_mask is None:
            attention_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

        if self.is_first_iteration or self.training:
            if images.ndim == 5:
                images_shape = self.shape(images)
                stack_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
                images = self.reshape(images, stack_shape)

            image_embeds = self.vision_encoder(images)
            image_embeds = self.mlp_adapter(image_embeds)
            input_embeds = self.update_modal_to_text(image_embeds, input_embeds, video_context_pos)

        if self.is_first_iteration:
            position_ids = self.slice(position_ids, (0, 0), (bs, seq_len), (1, 1))
        else:
            position_ids = self.gat_batched_position(batch_valid_length, position_ids)

        return self.llm_model(
            input_ids=None,
            labels=labels,
            input_position=input_position,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_embeds=input_embeds,
            init_reset=init_reset,
            batch_valid_length=batch_valid_length,
            batch_index=batch_index,
            zactivate_len=zactivate_len,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )