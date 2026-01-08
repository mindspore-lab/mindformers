# Copyright 2026 Huawei Technologies Co., Ltd
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
""" For language model """
__all__ = [
    "LanguageModelEmbedding",
]

from typing import Literal

from mindspore import nn, ops, mint
from mindformers.pynative.layers.dropout import Dropout
from mindformers.pynative.base_models.common.embeddings.vocab_embedding import VocabEmbedding
from mindformers.parallel_core.transformer_config import TransformerConfig


class LanguageModelEmbedding(nn.Cell):
    """
    A embedding layer contain word embediing, position embedding and tokentypes embedding.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding.
        position_embedding_type (Literal['learned_absolute', 'rope', 'none']):
            Type of positional encoding to use. Defaults to 'learned_absolute'.
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.

    Inputs:
        input_ids: input ids
        position_ids: position ids
        tokentype_ids: tokentype ids

    Outputs:
        embeddings: the embedding output

    Supported Platforms:
        Ascend
    """

    def __init__(
            self,
            config: TransformerConfig,
            vocab_size: int,
            max_sequence_length: int,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            num_tokentypes: int = 0,
    ):
        super().__init__()

        self.compute_dtype = config.compute_dtype
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method

        # Word embedding
        self.word_embeddings = VocabEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
            init_method=self.init_method)

        # Position embedding
        self.add_position_embedding = position_embedding_type == 'learned_absolute'
        if self.add_position_embedding:
            self.position_embeddings = VocabEmbedding(
                num_embeddings=max_sequence_length,
                embedding_dim=config.hidden_size,
                init_method=self.init_method)
        else:
            self.position_embeddings = None

        # Tokentypes embedding
        if num_tokentypes > 0:
            self.tokentype_embeddings = VocabEmbedding(
                num_embeddings=num_tokentypes,
                embedding_dim=config.hidden_size,
                init_method=self.init_method)
        else:
            self.tokentype_embeddings = None

        # Operations
        self.add_pe = mint.add
        self.add_te = mint.add
        self.transpose = mint.transpose
        self.cast = ops.cast

        # Embedding dropout
        self.embedding_dropout_prob = config.hidden_dropout
        self.embedding_dropout = Dropout(self.embedding_dropout_prob)

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int, optional): The token type ids. Used when args.bert_binary_head is
                set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        words_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = self.add_pe(words_embeddings, position_embeddings)
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            if self.tokentype_embeddings is None:
                raise RuntimeError("Embedding layer got 'tokentype_ids' input, "
                                   "but 'tokentype_embeddings' layer is not initialized")
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids)
            embeddings = self.add_te(embeddings, tokentype_embedding)
        else:
            if self.tokentype_embeddings is not None:
                raise RuntimeError("The 'tokentype_ids' input for Embedding layer is None, "
                                   "but 'tokentype_embeddings' layer is initialized")

        # Data format change to avoid explicit transposes : [b s h] --> [s b h].
        embeddings = self.transpose(embeddings, 0, 1)

        # Dropout
        if self.embedding_dropout_prob > 0:
            embeddings = self.embedding_dropout(embeddings)

        embeddings = self.cast(embeddings, self.compute_dtype)

        return embeddings
