# Copyright 2023 Huawei Technologies Co., Ltd
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
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""
BLIP2 Base Model, contains Blip2Base, ViTModelForBlip2,
as well as itm computing procedures.
"""
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_model import BaseModel
from mindformers.models.blip2.qformer import BertLMHeadModel
from mindformers.models.vit.vit import ViTModel, ViTConfig
from mindformers.modules.layers import LayerNorm


class ViTModelForBlip2(ViTModel):
    """
    ViTModel For Blip2 Models, loading a pretrained weight.
    forward will return the penultimate output.
    """
    _support_list = MindFormerBook.get_config_support_list()['vit']

    def __init__(self, config: ViTConfig):
        super(ViTModelForBlip2, self).__init__(config)
        self.load_checkpoint(config)

    def construct(self, image):
        return self.construct_without_pool(image)


class Blip2Base(BaseModel):
    """
    BLIP2 BaseModel, all BLIP2 models inherit this class.
    """
    _support_list = MindFormerBook.get_model_support_list()['blip2']

    def init_qformer(self):
        """
        Init qformer for blip2 model

        Raises:
            ValueError: qformer config wrong

        Returns:
            qformer, query_tokens
        """
        qformer_config = self.config.qformer_config
        qformer_config.parallel_config = self.config.parallel_config
        qformer = BertLMHeadModel(qformer_config)
        if qformer is None:
            raise ValueError("qformer configuration is wrong. \
            please check 'qformer_config' is set in Blip2Config")
        query_tokens = ms.Parameter(initializer(
            Normal(mean=0.0, sigma=qformer_config.initializer_range),
            [1, qformer_config.query_length, qformer_config.hidden_size]))
        return qformer, query_tokens

    def init_vision_encoder(self):
        """
        init vision encoder for blip2 model

        Raises:
            ValueError: vit config wrong

        Returns:
            visual_encoder, ln_vision
        """
        vision_config = self.config.vision_config
        if vision_config is not None:
            visual_encoder = ViTModelForBlip2(vision_config)
        if visual_encoder is None:
            raise ValueError("visual_encoder configuration is wrong. \
            please check 'vision_config' is set in Blip2Config")
        for block in visual_encoder.blocks:
            mapping = block.output.mapping
            if mapping.activation_flag and isinstance(mapping.activation, nn.GELU):
                mapping.activation = nn.GELU(approximate=False)

        ln_vision = LayerNorm(visual_encoder.config.embed_dim)
        return visual_encoder, ln_vision


class ImageTextEmbeddingConcat(nn.Cell):
    """
    Layer to concat image embedding and text embedding
    """
    def __init__(self, pad_token_id):
        super().__init__()
        self.concat_2d = P.Concat(axis=1)
        self.concat_3d = P.Concat(axis=1)
        self.not_equal = P.NotEqual()
        self.ones = P.Ones()
        self.cast = P.Cast()

        self.pad_token_id = pad_token_id

    def construct(self, image_embeddings: ms.Tensor, text_embeddings: ms.Tensor, text_input_ids: ms.Tensor):
        text_embeddings = self.cast(text_embeddings, mstype.float32)
        text_embeddings_atts = self.cast(self.not_equal(text_input_ids, self.pad_token_id), mstype.float32)

        image_embeddings_atts = self.ones(image_embeddings.shape[:-1], mstype.float32)

        concat_embeds = self.concat_3d([image_embeddings, text_embeddings])
        concat_attention_mask = self.concat_2d([image_embeddings_atts, text_embeddings_atts])
        return concat_embeds, concat_attention_mask
