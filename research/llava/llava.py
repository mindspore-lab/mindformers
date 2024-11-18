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
# ============================================================================
"""
Llava Model
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from llava_config import LlavaConfig
from mindformers import BaseXModalToTextModel
from mindformers.models.build_model import build_network
from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import lazy_inline
from mindformers.modules.layers import Linear
from mindformers.tools import logger


class LlavaAdapter(nn.Cell):
    r"""
    Module to resize the image dimension into the llm model dimension
    Args:
        config (LlavaConfig): The config of Llava model.
    """

    def __init__(self, config: LlavaConfig):
        super(LlavaAdapter, self).__init__(config)

        self.adapter = Linear(
            in_channels=config.vision_config.model_config.hidden_size,
            out_channels=config.text_config.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )
        self.activation_func = P.GeLU()
        self.adapter_2 = Linear(
            in_channels=config.text_config.model_config.hidden_size,
            out_channels=config.text_config.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type
        )

    def construct(self, x):
        """adapter forward method"""
        output = self.adapter(x)
        output = self.activation_func(output)
        output = self.adapter_2(output)
        return output

    def shard(self, parallel_config):
        """distributed method"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.adapter.shard(((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.adapter_2.shard(((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))
        self.activation_func.shard(((dp, 1, mp),))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlavaVlm(BaseXModalToTextModel):
    """
    Provide Llava1.5 training loss or logits through network.

    Args:
        config (LlavaConfig): The config of Llava model.
    """

    @lazy_inline
    def __init__(self, config: LlavaConfig, **kwargs):
        super(LlavaVlm, self).__init__(config, **kwargs)
        self.config = config if config is not None else LlavaConfig()

        self.language_model = build_network(self.config.text_config)
        self.vision_encoder = build_network(self.config.vision_config)
        self.adapter = LlavaAdapter(self.config)

        self.vision_encoder.pipeline_stage = 0
        self.adapter.pipeline_stage = 0

        self.is_first_iteration = True
        self.use_past = config.use_past
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.image_size = self.config.vision_config.model_config.image_size
        self.num_queries = self.config.vision_config.model_config.num_queries
        self.ignore_token_id = Tensor(config.ignore_token_id, mstype.int32)

        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.slice = P.StridedSlice()
        self.ones = P.Ones()

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.adapter.shard(config.parallel_config)
            self.slice.shard(((dp, 1),))

        self.freeze_component()

    def freeze_component(self):
        """freeze parameters of model according to different command"""
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_encoder.trainable_params():
                param.requires_grad = False

        if self.config.freeze_llm:
            logger.info("freeze llm model")
            for param in self.language_model.trainable_params():
                param.requires_grad = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation in inference"""
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
        images = kwargs.pop("images")
        image_context_pos = kwargs.pop("image_context_pos", None)
        if image_context_pos is not None and not isinstance(image_context_pos, ms.Tensor):
            image_context_pos = ms.Tensor(image_context_pos, mstype.int32)

        return {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "images": images,
            "image_context_pos": image_context_pos
        }

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """prepare inputs for predict layout"""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq = input_ids.shape
        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((bs, 3, self.image_size, self.image_size)), ms.float32)

        if "image_context_pos" in kwargs:
            image_context_pos = Tensor(kwargs.get("image_context_pos"))
        else:
            image_context_pos = Tensor(np.random.randint(0, self.num_queries, (bs, self.num_queries, 2)), ms.int32)

        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, images, image_context_pos, None, None, None, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self):
        """set inputs when is_dynamic=True"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
        dynamic_img_pos = Tensor(shape=[None, self.num_queries, 2], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_img_pos, None, None, None,
                        None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        logger.info("Set dynamic inputs for LLava-1.5")

    def construct(self, input_ids, images, image_context_pos: Tensor = None, labels=None, input_position=None,
                  position_ids=None, attention_mask=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        r"""
        LlavaVlm forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            images(Tensor): the image tensor with datatype float32, Tensor of shape :math:
            `(batch, 3, image_resolution, image_resolution)`
            image_context_pos(Tensor): the position index of the image in final input embedding. Tensor of shape :math
            `(batch, num_queries, 2)`
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): the input embedding Tensor of shape :math:`(batch, seq\_length, hidden_size)`.
                Default None.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
            labels = self.slice(labels, (0, 1), (bs, seq_len), (1, 1))
        else:
            tokens = input_ids
        input_embeds = self.language_model.to_embeddings(tokens)

        if self.is_first_iteration:
            if images is not None and images.ndim == 5:
                images_shape = self.shape(images)
                new_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
                images = self.reshape(images, new_shape)
            image_embeds = self.vision_encoder(images)

            image_embeds = self.adapter(image_embeds)  # 1 576 4096
            input_embeds = self.update_modal_to_text(image_embeds, input_embeds, image_context_pos)

        return self.language_model(
            input_ids=tokens,
            input_embeds=input_embeds,
            labels=labels,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            input_position=input_position,
            init_reset=init_reset,
            position_ids=position_ids,
            attention_mask=attention_mask,
            batch_index=batch_index,
            zactivate_len=zactivate_len
        )

    def kvcache(self, layer_idx):
        key_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
