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
"""Blip2 second stage pretrain and infer with Llama model"""
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import ops, Tensor
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.blip2.blip2 import Blip2Base
from mindformers.models.blip2.blip2_config import Blip2Config
from mindformers.models.llama import LlamaConfig
from mindformers.models.llama.llama import LlamaForCausalLM, LlamaModel
from mindformers.modules.layers import Linear
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['Blip2Llama', 'Blip2ImageToTextGeneration']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaModelForBlip2(LlamaModel):
    r"""
        Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`],
        it inherits from LlamaModel and is used to adapt blip2.
        Args:
            config(LlamaConfig): the config of network

        Inputs:
            input_ids: the tokenized inputs with datatype int32

        Returns:
            output: Tensor, the output of llama decoder layer
        """

    def __init__(self, config):
        super().__init__(config)

    # pylint: disable=W0221
    def construct(self, input_embeddings: Tensor,
                  input_attention_masks: Tensor,
                  input_position=None,
                  init_reset=True,
                  batch_valid_length=None):

        bs, seq_len, _ = input_embeddings.shape
        if self.use_past:
            if not isinstance(init_reset, Tensor):
                init_reset = Tensor([init_reset], mstype.bool_)
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bs, 1), mstype.int32)

        if self.is_first_iteration:
            freqs_cis = (self.tile(self.reshape(self.freqs_cos, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.tile(self.reshape(self.freqs_sin, (1, 1, seq_len, -1)), (bs, 1, 1, 1)),
                         self.swap_mask)
            mask = self.get_attention_mask(input_attention_masks)
            # mask: [bs, seq, seq]
        else:
            cur_pos = batch_valid_length - 1
            valid_length = self.reshape(cur_pos, (-1, 1, 1))
            freqs_cis = (self.reshape(self.gather_past(self.freqs_cos, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.reshape(self.gather_past(self.freqs_sin, cur_pos, 0), (bs, 1, seq_len, -1)),
                         self.swap_mask)
            mask = self.cast(self.le_past(self.range, valid_length), self.dtype)
            # mask: [bs, 1, 1]

        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dims(mask, 1)
            mask = self.mul_mask(mask, self.multiply_data)

        h = input_embeddings
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            h, _ = self.layers[i](h, freqs_cis, mask,
                                  init_reset=init_reset, batch_valid_length=batch_valid_length)
        output = self.norm_out(h)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlamaForBlip2(LlamaForCausalLM):
    r"""
        Provide llama inference loss or logits through network for blip2.
        Args:
            config (LlamaConfig): The config of llama model.`

        Inputs:
            input_embeddings(Tensor): the embeddings of inputs, shape :math:`(batch, seq\_length, hidden\_size)`
            input_ids(Tensor): the input ids of inputs, shape: math:`(batch, seq\_length)`
            label_ids(Tensor): Labels for computing loss. Tensor of shape :math:`(batch, seq\_length)`
            attention_mask(Tensor): attention_mask, used by model.construct

        Returns:
            Tensor, the loss of the network.

        Examples:
            >>> from mindformers.models.llama import LlamaConfig
            >>> from mindformers.models.blip2.blip2_llama import LlamaForBlip2
            >>> config = LlamaConfig(batch_size=1)
            >>> network = LlamaForBlip2(config=config)
        """

    def __init__(self, config: LlamaConfig = None):
        checkpoint_name_or_path = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = ""

        super().__init__(config)
        self.model = LlamaModelForBlip2(config)

        self.config.checkpoint_name_or_path = checkpoint_name_or_path
        self.load_checkpoint(config)

        self.concat_2d = P.Concat(axis=1)
        self.concat_3d = P.Concat(axis=1)
        self.not_equal = P.NotEqual()
        self.ones = P.Ones()
        self.cast = P.Cast()

        self.is_first_iteration = True

    # pylint: disable=W0221
    def construct(self, input_embeddings=None,
                  input_ids=None,
                  labels=None,
                  input_position=None,
                  attention_mask=None,
                  init_reset=True,
                  batch_valid_length=None):
        """LlamaForBlip2 forward."""
        if input_embeddings is None and input_ids is not None:  # for incremental infer
            input_embeddings = self.model.tok_embeddings(input_ids)

        batch_size, seq_length, _ = input_embeddings.shape

        output = self.model(input_embeddings=input_embeddings,
                            input_attention_masks=attention_mask,
                            input_position=input_position,
                            init_reset=init_reset,
                            batch_valid_length=batch_valid_length)
        logits = self.lm_head(output)
        logits = self.cast(logits, mstype.float32)

        if labels is None:
            # inference
            logits = self.reshape(logits, (batch_size, seq_length, -1))
            return logits, attention_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))

        labels = self.reshape(labels, (-1,))
        label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
        attention_mask = self.reshape(attention_mask, (-1,))
        input_mask = self.mul(attention_mask, label_mask)
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        if self.is_first_iteration or not self.config.use_past:
            image_embeddings = kwargs.pop("image_embeds")
            image_embeddings_atts = self.ones(image_embeddings.shape[:-1], mstype.float32)

            image_embeddings_length = image_embeddings.shape[1]
            text_input_ids = Tensor(input_ids[:, image_embeddings_length:], mstype.int32)
            text_embeddings = self.model.tok_embeddings(text_input_ids)
            text_embeddings = self.cast(text_embeddings, mstype.float32)
            text_embeddings_atts = self.cast(self.not_equal(text_input_ids, self.pad_token_id), mstype.float32)

            llama_inputs_embeds = self.concat_3d([image_embeddings, text_embeddings])
            llama_inputs_attention_mask = self.concat_2d([image_embeddings_atts, text_embeddings_atts])
            return {
                "input_ids": Tensor(input_ids, mstype.int32),
                "input_embeddings": llama_inputs_embeds,
                "attention_mask": llama_inputs_attention_mask
            }
        return {
            "input_ids": Tensor(input_ids, mstype.int32),
            "input_embeddings": None,
            "attention_mask": None
        }


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Blip2Llama(Blip2Base):
    """
    BLIP2 Llama model
    """
    _support_list = MindFormerBook.get_model_support_list()['blip2']['2-stg']

    def __init__(self, config: Blip2Config, **kwargs):
        super(Blip2Llama, self).__init__(config, **kwargs)
        self.config = config if config is not None else Blip2Config()

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
                "seq_length should be greater than sum of max_text_len and num_query_token %d, but got %d" %
                (config.max_txt_len + config.qformer_config.query_length, config.text_config.seq_length))

        self.llama_model = LlamaForBlip2(config.text_config)

        if config.freeze_text:
            logger.info("freeze llm model")
            for param in self.llama_model.trainable_params():
                param.requires_grad = False
            self.llama_model.set_train(False)

        dp = config.parallel_config.data_parallel

        self.llama_proj = Linear(in_channels=self.config.qformer_config.hidden_size,
                                 out_channels=self.config.text_config.hidden_size,
                                 param_init_type=config.dtype,
                                 compute_dtype=config.compute_dtype)

        if config.checkpoint_name_or_path:
            logger.info(
                "load blip2 first stage pretrained model for vision model and qformer, checkpoint_name_or_path: "
                "%s. pretrained llm model: %s", config.checkpoint_name_or_path,
                config.text_config.checkpoint_name_or_path)
            self.load_checkpoint(config)
        else:
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

        self.eos_token_id = config.text_config.eos_token_id
        self.pad_token_id = config.text_config.pad_token_id
        self.ignore_token_id = config.text_config.ignore_token_id
        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt
        self.prompt_length = config.prompt_length

        self.broadcast_to = P.BroadcastTo((dp * self.config.batch_size,
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

    # pylint: disable=W0221
    def construct(self, image: ms.Tensor, text_input_ids: ms.Tensor):
        """Blip2Llama forward."""
        projected_qformer_output = self.forward_qformer_and_proj(image)
        projected_qformer_output_atts = self.ones(projected_qformer_output.shape[:-1], mstype.float32)

        batch_size, seq_length = text_input_ids.shape
        tokens = self.slice(text_input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))

        text_label_ids = self.slice(text_input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        targets = ops.masked_fill(text_label_ids, text_label_ids == self.pad_token_id, self.ignore_token_id)

        if self.prompt:
            prompt_label = self.ignore_token_id * self.ones((batch_size, self.prompt_length), mstype.int32)
            targets = self.concat_2d([prompt_label, targets[:, self.prompt_length:]])

        empty_targets = self.fill(mstype.int32, projected_qformer_output_atts.shape, self.ignore_token_id)
        targets = self.concat_2d([empty_targets, targets])  # [batch_size, 1, query_size + max_txt_length]

        # [batch_size, max_txt_length, llama_hidden_size]
        text_inputs_embeds = self.llama_model.model.tok_embeddings(tokens)
        text_inputs_embeds = self.cast(text_inputs_embeds, mstype.float32)
        text_inputs_atts = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)

        # [batch_size, query_size + max_txt_length, llama_hidden_size]
        llama_inputs_embeds = self.concat_3d([projected_qformer_output, text_inputs_embeds])
        # [batch_size, query_size + max_txt_length]
        llama_inputs_attention_mask = self.concat_2d([projected_qformer_output_atts, text_inputs_atts])

        loss = self.llama_model(
            input_embeddings=llama_inputs_embeds,
            labels=targets,
            attention_mask=llama_inputs_attention_mask
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
                                         use_cache=True,
                                         return_dict=True)

        # [batch_size, query_size, qformer_hidden_size] -> [batch_size, query_size, llama_hidden_size]
        return self.llama_proj(query_output[0])


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Blip2ImageToTextGeneration(Blip2Llama):
    """
    Blip2ImageToTextGeneration rely on Blip2Llama, used for image to text genearation.
    Usage:
        >>> from mindformers import AutoModel
        >>> model_type = 'itt_blip2_stage2_vit_g_llama_7b'
        >>> model = AutoModel.from_pretrained(model_type)
    """
    _support_list = MindFormerBook.get_model_support_list()['itt']['blip2']

    def __init__(self, config: Blip2Config, **kwargs):
        super(Blip2ImageToTextGeneration, self).__init__(config, **kwargs)

        self.llama_model.set_train(False)
        self.one_prefix = ops.Ones()

    def generate_text_for_image(self, image: ms.Tensor, prompt_input_ids: ms.Tensor):
        batch_size = image.shape[0]
        prefix_ones = self.one_prefix((batch_size, self.config.qformer_config.query_length), mstype.int32)

        text_input_ids = self.concat_2d([prefix_ones, prompt_input_ids])
        projected_qformer_output = self.forward_qformer_and_proj(image)

        output_ids = self.llama_model.generate(input_ids=text_input_ids.asnumpy(),
                                               image_embeds=projected_qformer_output)
        return output_ids
