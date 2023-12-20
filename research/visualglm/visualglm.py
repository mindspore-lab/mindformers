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
# ============================================================================
"""visualglm model implementation."""

import os
import time

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.ops import operations as P

from mindformers.modules.layers import Linear
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from visualglm_base import VisualGLMBase
from visualglm_config import VisualGLMConfig
from visualglm_text_generation_pipeline import register_pipeline_task

__all__ = ['VisualGLMModel', 'VisualGLMImageToTextGeneration']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class VisualGLMModel(VisualGLMBase):
    """
    visualglm with llm model.
    """

    def __init__(self, config: VisualGLMConfig, **kwargs):
        super(VisualGLMModel, self).__init__(config, **kwargs)
        self.batch_size = None
        self.config = config if config is not None else VisualGLMConfig()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder()
        if config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.visual_encoder.trainable_params():
                param.requires_grad = False
            self.visual_encoder.set_train(False)

        self.qformer, self.query_tokens = self.init_qformer()

        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if config.text_config.seq_length < config.max_txt_len + config.qformer_config.query_length:
            raise ValueError(
                f"seq_length should be greater than sum of max_text_len and num_query_token "
                f"{config.max_txt_len + config.qformer_config.query_length:d}, but got "
                f"{config.text_config.seq_length:d}")

        self.llm_model = self.init_llm()

        if config.freeze_text:
            logger.info("freeze llm model")
            for param in self.llm_model.trainable_params():
                param.requires_grad = False
            self.llm_model.set_train(False)

        dp = config.parallel_config.data_parallel

        micro_batch_interleave_num = config.micro_batch_interleave_num

        self.init_batch_size(dp, micro_batch_interleave_num)

        self.llm_proj = Linear(in_channels=self.config.qformer_config.hidden_size,
                               out_channels=self.config.text_config.hidden_size,
                               param_init_type=config.dtype,
                               compute_dtype=config.compute_dtype)

        pp = config.parallel_config.pipeline_stage
        if pp > 1:
            self.visual_encoder.pipeline_stage = 0
            self.qformer.pipeline_stage = 0
            self.llm_proj.pipeline_stage = 0

        if config.checkpoint_name_or_path:
            logger.info(
                "load blip2 first stage pretrained model for vision model and qformer, checkpoint_name_or_path: "
                "%s. pretrained llm model: %s", config.checkpoint_name_or_path,
                config.text_config.checkpoint_name_or_path)
            self.load_checkpoint(config)
        else:
            self.init_checkpoint(config)

        self.eos_token_id = config.text_config.eos_token_id
        self.pad_token_id = config.text_config.pad_token_id
        self.ignore_token_id = config.text_config.ignore_token_id
        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt
        self.prompt_length = config.prompt_length

        self.broadcast_to = P.BroadcastTo((self.batch_size,
                                           self.config.qformer_config.query_length,
                                           self.config.qformer_config.hidden_size)).shard(((1, 1, 1),))
        self.fill = P.Fill().shard(((dp, 1),))
        self.masked_fill = P.MaskedFill().shard(((dp, 1), ()))
        self.ones = P.Ones().shard(((dp, 1),))
        self.concat_2d = P.Concat(axis=1).shard(((dp, 1), (dp, 1)))
        self.concat_3d = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.cast = P.Cast()

        register_pipeline_task()

    def init_batch_size(self, dp, micro_batch_interleave_num):
        """
        init batch size
        :param dp: data parallel config
        :param micro_batch_interleave_num: micro batch interleave num
        """
        batch_size = self.config.batch_size
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            full_batch = ms.get_auto_parallel_context("full_batch")
            if full_batch:
                self.batch_size = batch_size * dp * micro_batch_interleave_num
            else:
                card_num = int(os.getenv('RANK_SIZE', '1'))
                self.batch_size = int(card_num * batch_size / micro_batch_interleave_num)
        else:
            self.batch_size = batch_size

    @staticmethod
    def init_checkpoint(config):
        """ init checkpoint """
        if config.vision_config.checkpoint_name_or_path:
            vision_checkpoint = config.vision_config.checkpoint_name_or_path
        else:
            vision_checkpoint = 'not configured'
        if config.text_config.checkpoint_name_or_path:
            text_checkpoint = config.text_config.checkpoint_name_or_path
        else:
            text_checkpoint = 'not configured'
        if config.qformer_config.checkpoint_name_or_path:
            qformer_checkpoint = config.qformer_config.checkpoint_name_or_path
        else:
            qformer_checkpoint = 'not configured'
        logger.info("training blip2 second stage, pretrained vision model: %s, pretrained llm model: %s, "
                    "pretrained qformer: %s", vision_checkpoint, text_checkpoint, qformer_checkpoint)

    # pylint: disable=W0221
    def construct(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor):
        """
        model entrance function
        :param image: image tensor
        :param pre_input_ids: text input id
        :param post_input_ids: post input id
        :return: loss value
        """
        del post_input_ids
        projected_qformer_output = self.forward_qformer_and_proj(image)
        projected_qformer_output_atts = self.ones(projected_qformer_output.shape[:-1], mstype.float32)

        batch_size, seq_length = pre_input_ids.shape
        tokens = self.slice(pre_input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))

        targets = ops.masked_fill(pre_input_ids, pre_input_ids == self.pad_token_id, self.ignore_token_id)

        if self.prompt:
            prompt_label = self.ignore_token_id * self.ones((batch_size, self.prompt_length), mstype.int32)
            targets = self.concat_2d([prompt_label, targets[:, self.prompt_length:]])

        image_label_shape = (projected_qformer_output_atts.shape[0], projected_qformer_output_atts.shape[1] - 1)
        empty_targets = self.fill(mstype.int32, image_label_shape, self.ignore_token_id)
        targets = self.concat_2d([empty_targets, targets])  # [batch_size, 1, query_size + max_txt_length]

        # [batch_size, max_txt_length, llm_hidden_size]
        text_inputs_embeds = self.llm_model.model.tok_embeddings(tokens)
        text_inputs_embeds = self.cast(text_inputs_embeds, mstype.float32)
        text_inputs_atts = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

        # [batch_size, query_size + max_txt_length, llm_hidden_size]
        llm_inputs_embeds = self.concat_3d([projected_qformer_output, text_inputs_embeds])
        # [batch_size, query_size + max_txt_length]
        llm_inputs_attention_mask = self.concat_2d([projected_qformer_output_atts, text_inputs_atts])

        loss = self.llm_model(
            input_embeddings=llm_inputs_embeds,
            labels=targets,
            attention_mask=llm_inputs_attention_mask
        )
        return loss

    def forward_qformer_and_proj(self, image: ms.Tensor):
        """forward the image tensor to the qformer, then project the output to adapt the dimension"""

        image_embeds = self.visual_encoder(image)
        image_embeds = self.ln_vision(image_embeds)  # [batch_size, vit_seq_length, vit_encoder_hidden_width]
        image_atts = self.ones(image_embeds.shape[:-1], mstype.float32)  # [batch_size, vit_seq_length]

        query_tokens = self.broadcast_to(self.query_tokens)  # [batch_size, query_size, qformer_hidden_size]

        query_output = self.qformer.bert(query_embeds=query_tokens,
                                         encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts,
                                         use_cache=True)

        # [batch_size, query_size, qformer_hidden_size] -> [batch_size, query_size, llm_hidden_size]
        return self.llm_proj(query_output[0])


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class VisualGLMImageToTextGeneration(VisualGLMModel):
    """
    VisualGLMImageToTextGeneration rely on Blip2Llm, used for image to text genearation.

    Args:
        config (VisualGLMConfig): The config of Blip2ImageToTextGeneration.

    Examples:
        >>> from mindformers import Blip2ImageToTextGeneration
        >>> model = Blip2ImageToTextGeneration.from_pretrained('itt_blip2_stage2_vit_g_llama_7b')
        >>> type(model)
        <class 'mindformers.models.blip2.blip2_llama.Blip2ImageToTextGeneration'>
    """

    def __init__(self, config: VisualGLMConfig, **kwargs):
        super(VisualGLMImageToTextGeneration, self).__init__(config, **kwargs)

        self.llm_model.set_train(False)
        self.one_prefix = ops.Ones()
        self.expand_dims = P.ExpandDims()

        self.query_length = self.config.qformer_config.query_length

    def construct(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor):
        if len(pre_input_ids.shape) == 1:
            pre_input_ids = self.expand_dims(pre_input_ids, 0)

        if len(post_input_ids.shape) == 1:
            post_input_ids = self.expand_dims(post_input_ids, 0)

        batch_size = image.shape[0]
        prefix_ones = self.one_prefix((batch_size, self.query_length), mstype.int32)

        extend_text_input_ids = self.concat_2d([pre_input_ids, prefix_ones, post_input_ids])
        projected_qformer_output = self.forward_qformer_and_proj(image)
        return extend_text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids

    def generate_text_for_image(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor, **kwargs):
        """generate text for image by calling llm generate"""
        start = time.time()
        text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids = self(image, pre_input_ids,
                                                                                       post_input_ids)
        gene_start = time.time()
        output_ids = self.llm_model.generate(input_ids=text_input_ids.asnumpy(),
                                             image_embeds=projected_qformer_output,
                                             pre_input_ids=pre_input_ids.asnumpy(),
                                             post_input_ids=post_input_ids.asnumpy(),
                                             **kwargs)
        end = time.time()
        generate_len = sum(len(output_ids[i]) - text_input_ids[i].asnumpy().tolist().index(130004)
                           for i in range(len(output_ids)))
        print("---------------------------------")
        print(f"visualglm generate time: {end - gene_start},"
              f" total time: {end - start}, generate_len: {generate_len}",
              f" speed_no_tokenizer: {generate_len / (end - start)} tokens/s")
        return output_ids
