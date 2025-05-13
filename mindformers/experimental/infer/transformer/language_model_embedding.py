# Copyright 2025 Huawei Technologies Co., Ltd
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
Language model embedding for transformer.
"""
from mindspore import nn, Tensor

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.tensor_parallel import VocabParallelEmbedding


__all__ = [
    'LanguageModelEmbedding',
]


class LanguageModelEmbedding(nn.Cell):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        add_position_embedding (bool): Add a position embedding, which is not currently supported
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head, currently not supported
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region, which is not currently supported
    """

    def __init__(
            self,
            config: TransformerConfig,
            vocab_size: int,
            max_sequence_length: int,
            add_position_embedding: bool = False,
            num_tokentypes: int = None,
            scatter_to_sequence_parallel: bool = False,
    ):
        super(LanguageModelEmbedding, self).__init__()
        if add_position_embedding:
            raise ValueError("`add_position_embedding` is not supported.")
        if num_tokentypes is not None:
            raise ValueError("`num_tokentypes` is not supported.")
        if scatter_to_sequence_parallel:
            raise ValueError("`scatter_to_sequence_parallel` is not supported.")

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length

        # Word embeddings
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            config=config,
            init_method="normal",
            init_type=self.config.embedding_init_type,
        )

    def construct(self, input_ids: Tensor) -> Tensor:
        """
        Forward of LanguageModelEmbedding.

        Args:
            input_ids (Tensor): The input tokens

        Returns:
            Tensor: The output embeddings
        """
        embeddings = self.word_embeddings(input_ids)
        return embeddings
