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
    "VocabEmbedding",
]

from typing import Callable

import mindspore._checkparam as Validator
from mindspore import nn, mint
from mindspore.common import dtype
from mindspore.common.parameter import Parameter


class VocabEmbedding(nn.Cell):
    """Vocab Embedding.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        init_method (Callable): The initialization method.
    """
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            init_method: Callable,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = mint.nn.functional.embedding
        self.reshape = mint.reshape

        self.weight = Parameter(init_method([self.num_embeddings, self.embedding_dim]), name="weight")

    def construct(self, input_):
        """
        Forward of vocab embedding.

        input_: (B, S)
        weight: (V, H)
        output: (B, S, H)
        """
        Validator.check_type_name("input_ids", input_.dtype, [dtype.int32, dtype.int64], self.cls_name)

        bs, seq_len = input_.shape
        _, hidden = self.weight.shape
        input_ = self.reshape(input_, (bs * seq_len,))
        masked_input = input_

        output = self.embedding(masked_input, self.weight)
        output = self.reshape(output, (bs, -1, hidden))

        return output
