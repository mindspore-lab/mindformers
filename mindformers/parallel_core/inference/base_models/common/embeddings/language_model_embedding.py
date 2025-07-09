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
__all__ = [
    "LanguageModelEmbedding",
]

from typing import Literal, Optional

from mindspore import nn, Tensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.layers import VocabParallelEmbedding
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs


class LanguageModelEmbedding(nn.Cell):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        position_embedding_type (str): The type of position embedding. Defaults to none.
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head, currently not supported
        scatter_to_sequence_parallel (bool): Set False to disable scatter of embedding
            across sequence parallel region, which is not currently supported
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.


    Inputs:
        - **input_ids** (Tensor) - The tensor of input ids

    Outputs:
        - **embeddings** (Tensor) - The output embeddings

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            vocab_size: int,
            max_sequence_length: int,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            num_tokentypes: int = None,
            scatter_to_sequence_parallel: bool = False,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
    ):
        super(LanguageModelEmbedding, self).__init__()
        if num_tokentypes is not None:
            # Parameter used by tokentypes embedding is not supported
            raise NotImplementedError("For LanguageModelEmbedding, `num_tokentypes` is not supported.")
        if scatter_to_sequence_parallel:
            raise NotImplementedError("For LanguageModelEmbedding, `scatter_to_sequence_parallel` is not supported.")

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'

        # Word embeddings
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            config=config,
            init_method=self.config.init_method,
            tp_group=model_comm_pgs.tp
        )

        # Note: There is no need to use position embedding in mcore inference

        # Note: There is no need to use tokentypes embedding in mcore inference

    def construct(self, input_ids: Tensor) -> Tensor:
        """Forward of LanguageModelEmbedding."""
        embeddings = self.word_embeddings(input_ids)
        return embeddings
