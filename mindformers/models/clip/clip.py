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
ClipModel
"""

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal, initializer
from mindspore import Parameter, Tensor
import mindspore.ops as ops

from ...mindformer_book import MindFormerBook
from ..base_model import BaseModel
from .clip_modules import VisionTransformer, Transformer, LayerNorm
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ClipModel(BaseModel):
    """
    ClipModel.
    The supported model name could be selected from ClipModel.show_support_list().

    Args:
        config (ClipConfig): the config of clip model.
    """
    _support_list = MindFormerBook.get_model_support_list()['clip']

    def __init__(self, config):
        super(ClipModel, self).__init__(config)
        self.dtype = self.get_dtype(config.dtype)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction="mean", sparse=True)

        self.max_position_embeddings = config.text_config.max_position_embeddings
        vision_heads = config.vision_config.hidden_size // config.ratio
        self.visual = VisionTransformer(
            input_resolution=config.vision_config.image_size,
            patch_size=config.vision_config.patch_size,
            width=config.vision_config.hidden_size,
            layers=config.vision_config.num_hidden_layers,
            heads=vision_heads,
            output_dim=config.projection_dim,
            dtype=self.dtype
        )

        transformer_heads = config.text_config.hidden_size // config.ratio
        self.transformer = Transformer(
            width=config.text_config.hidden_size,
            layers=config.text_config.num_hidden_layers,
            heads=transformer_heads,
            dtype=self.dtype,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = \
            nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size,
                         embedding_table=Normal(mean=0.0, sigma=0.02))
        self.positional_embedding = Parameter(initializer(
            Normal(mean=0.0, sigma=0.01), [config.text_config.max_position_embeddings,
                                           config.text_config.hidden_size]))
        self.ln_final = LayerNorm([config.text_config.hidden_size])

        self.text_projection = Parameter(initializer(
            Normal(mean=0.0, sigma=config.text_config.hidden_size ** -0.5),
            [config.text_config.hidden_size, config.projection_dim], ms.float32))
        self.logit_scale = Parameter(Tensor(np.log(1 / 0.07)).astype(ms.float32))
        self.exp = ops.Exp()

        self._load_checkpoint(config)

    def get_dtype(self, dtype):
        """get_dtype"""
        if dtype == "float16":
            return ms.float16
        if dtype == "float32":
            return ms.float32
        raise TypeError("unsupported data type.")

    def construct(self, image, text):
        """
        construct

        Args:
            image (tensor): a image tensor processed by feature extractor
            text (tensor): a text id tensor processed by tokenizer

        Returns:
            if self.trainining:
                logits_per_image: similarity between image and text
                logits_per_text: similarity between text and image
            else:
                loss: constructive language image pretraining loss
        """
        image_features = self.get_image_features(image)
        text_features = self. get_text_features(text)

        image_features = image_features / image_features.norm(1, keep_dims=True)
        text_features = text_features / text_features.norm(1, keep_dims=True)
        logit_scale = self.exp(self.logit_scale)

        if not self.training:
            logits_per_image = ops.matmul(logit_scale * image_features, text_features.T)
            logits_per_text = logits_per_image.T

            return logits_per_image, logits_per_text

        logits = ops.matmul(logit_scale * image_features, text_features.T)
        batch_size, _ = F.shape(logits)

        labels = ms.Tensor(np.arange(batch_size))

        images_loss = self.cross_entropy(logits, labels)
        texts_loss = self.cross_entropy(logits.T, labels)
        loss = (images_loss + texts_loss) / 2
        return loss

    def build_attention_mask(self):
        """buil_attention_mask"""
        mask = np.ones((self.max_position_embeddings, self.max_position_embeddings))
        mask = np.triu(mask * float("-inf"), k=1)
        return Tensor(mask).astype(self.dtype)

    def get_image_features(self, image):
        """
        get_image_features

        Args:
            image (tensor): a image tensor processed by feature extractor

        Returns:
            image feature
        """
        image = image.astype(self.dtype)
        return self.visual(image)

    def get_text_features(self, text):
        """
        get_text_features

        Args:
            text (tensor): a text id tensor processed by tokenizer

        Returns:
            text feature
        """
        text_ = self.token_embedding(text).astype(self.dtype)
        text_ = ops.Add()(text_, self.positional_embedding).astype(self.dtype)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.transformer(text_)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.ln_final(text_).astype(self.dtype)

        text_ = ops.matmul(
            text_[ms.numpy.arange(text_.shape[0]), text.argmax(-1)], self.text_projection)
        return text_
