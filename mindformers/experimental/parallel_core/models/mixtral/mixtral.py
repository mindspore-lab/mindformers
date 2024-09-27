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
"""Mixtral models' APIs."""

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import mint

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits, TransformerLanguageModel
from mindformers.experimental.parallel_core.pynative.transformer.module import Module


__all__ = [
    "MixtralModel",
]


# pylint: disable=W0613
class MixtralModel(Module):
    r"""
    Mixtral Model
    Args:
        config (TransformerConfig): the config of network;
        num_tokentypes (int): if > 0, using tokentypes embedding. Default: 0;
        parallel_output: (bool), Specifies whether return paralleled output on each tensor parallel rank. Default: True;
        pre_process (bool) when using pipeline parallel, indicate whether it's the first stage. Default: True,
        post_process (bool) when using pipeline parallel, indicate whether it's the last stage. Default: True,
        loss_func: loss function

    Returns:
        output (Tensor): mixtral loss or hidden states

    Examples:
    ```python
    def model_provider_func(pre_process=True, post_process=True):
        ''' get mixtral model '''
        loss = get_loss_func(config.training_config)
        network = MixtralModel(
            model_config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
            )
        return network

    network = get_model(model_provider_func, parallel_config)
    ```
    """

    def __init__(
            self,
            config: TransformerConfig,
            num_tokentypes: int = 0,
            parallel_output: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
            loss_func=None,
            **kwargs):
        super(MixtralModel, self).__init__()
        self.config = config
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.pad_token = config.dataset_config.pad_token
        self.compute_dtype = config.compute_dtype

        self.language_model = TransformerLanguageModel(
            config,
            encoder_attn_mask_type=None,
            num_tokentypes=num_tokentypes,
            pre_process=self.pre_process,
            post_process=self.post_process
            )
        if self.post_process:
            self.head = ParallelLMLogits(
                config=config,
                bias=False,
                compute_dtype=config.compute_dtype
                )

            self.loss = loss_func

        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(
            self, input_ids: ms.Tensor,
            labels: ms.Tensor = None,
            attention_mask: ms.Tensor = None,
            tokentype_ids: ms.Tensor = None,
            inference_params: ms.Tensor = None,
            loss_mask: ms.Tensor = None):
        """
        Forward of mixtral model.

        Args:
            input_ids (Tensor): the tokenized inputs with datatype int32
            attention_mask (Tensor):
        Returns:
            output (Tensor): the output of mixtral decoderlayer
        """
                # ensure `input_ids` and `labels` shape are [bs, seq]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(dim=0)
        elif labels is None:
            labels = input_ids

        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        position_ids = None

        if loss_mask is None:
            if self.pad_token is None:
                raise RuntimeError("If 'pad_token' is not pass into model, the 'loss_mask' must be not None.")
            loss_mask = mint.ne(input_ids, self.pad_token).astype(self.compute_dtype)

        hidden_states = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            )
        # pylint: disable=R1705
        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            # pylint: disable=E1121
            loss = self.post_language_model_processing(
                hidden_states,
                labels,
                logit_weights,
                loss_mask
                )
            return loss
        else:
            return hidden_states

    def post_language_model_processing(
            self,
            lm_output,
            labels,
            logit_weights,
            loss_mask):
        """define post language model process"""
        logits = self.head(lm_output, logit_weights, self.parallel_output)

        logits = logits.reshape(-1, logits.shape[-1]).to(mstype.float32)
        labels = labels.reshape(-1,).to(mstype.int32)

        loss = self.loss(logits, labels, loss_mask)

        return loss
