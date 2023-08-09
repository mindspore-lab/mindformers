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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal

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
            vision_config.parallel_config = self.config.parallel_config
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

    def append_text_outputs(self, text_input, text_feats, text_ids):
        """append_text_outputs

        Args:
            text_input: text_input, can be:
                        1) 2-dimension: 1 image 1 caption
                        2) 3-dimension: 1 image multiple captions
            text_feats: output to append
            text_ids: output to append
        """
        if text_input.ndim == 3:
            for input_ids in text_input:
                text_embed, _ = self.forward_text(input_ids)
                text_feat = ops.L2Normalize(axis=1, epsilon=1e-12)(self.text_proj(text_embed[:, 0, :]))
                text_feats.append(text_feat)
                text_ids.append(input_ids)
        else:
            text_embed, _ = self.forward_text(text_input)
            text_feat = ops.L2Normalize(axis=1, epsilon=1e-12)(self.text_proj(text_embed[:, 0, :]))
            text_feats.append(text_feat)
            text_ids.append(text_input)

    def append_image_outputs(self, image_input, image_feats, vit_outputs):
        """append_image_outputs

        Args:
            image_input: image_input, can be:
                        1) 4-dimension: 1 caption 1 image
                        2) 5-dimension: 1 caption multiple images
            image_feats: output to append
            vit_outputs: output to append
        """
        if image_input.ndim == 5:
            for image in image_input:
                image_embed, vit_output = self.forward_image(image)
                image_embed = self.vision_proj(image_embed)
                image_feat = ops.L2Normalize(axis=-1, epsilon=1e-12)(image_embed)
                image_feats.append(image_feat)
                vit_outputs.append(vit_output)
        else:
            image_embed, vit_output = self.forward_image(image_input)
            image_embed = self.vision_proj(image_embed)
            image_feat = ops.L2Normalize(axis=-1, epsilon=1e-12)(image_embed)
            image_feats.append(image_feat)
            vit_outputs.append(vit_output)

    def prepare_inputs_for_itm_eval(self, dataloader):
        """
        prepare inputs of BLIP-2 for image_to_text_reatrieval task.

        Args:
            dataloader (GeneratorDataset): image-caption pair dataloader

        Returns:
            image_feats, text_feats, vit_outputs, text_ids, text_atts
        """
        image_feats = []
        text_feats = []
        vit_outputs = []
        text_ids = []
        for image_input, text_input in dataloader:
            self.append_text_outputs(text_input, text_feats, text_ids)
            self.append_image_outputs(image_input, image_feats, vit_outputs)

        image_feats = ops.concat(image_feats, axis=0)
        text_feats = ops.concat(text_feats, axis=0)
        vit_outputs = ops.concat(vit_outputs, axis=0)
        text_ids = ops.concat(text_ids, axis=0)

        return (
            image_feats,
            text_feats,
            vit_outputs,
            text_ids,
        )

    def compute_extra_itm(self, extra_args, i, k_test, topk_idx, i2t=True):
        """ compute extra_itm score, for those model have
            its own itm computing method, like blip-2.

        Args:
            extra_args (tuple): required args for this method.
            i (int): index in the sim_matrix
            k_test (int): k_test number
            topk_idx (Tensor): topk_idx
            i2t (bool, optional): is image-to-text or text-to-image matching.

        Returns:
            _type_: extra itm score
        """
        vit_outputs, text_ids = extra_args
        if i2t:
            image_inputs = ms.numpy.tile(vit_outputs[i], (k_test, 1, 1))
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=text_ids[topk_idx],
                vit_computed=True
            ).astype(ms.float32)
        else:
            image_inputs = vit_outputs[topk_idx]
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=ms.numpy.tile(text_ids[i], (k_test, 1)),
                vit_computed=True).astype(ms.float32)

        return score
