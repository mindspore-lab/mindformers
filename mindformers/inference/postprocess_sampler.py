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
"""postprocess sampler"""
import mindspore as ms
import mindspore.ops as P
import mindspore.nn as nn


class Sampler(nn.Cell):
    r"""
    the postprocess sampler currently support using lite infer to do temperature, top_k and repetition_penalty.

    Args:
        logits (float):
            prediction_scores.
        temperature (float):
            The value used to modulate the next token probabilities.
        top_k (`int`, *optional*, defaults to 1):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty (float, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        input_ids(tensor, *optional*, defaults to None):
            Indices of input sequence tokens in the vocabulary.
    """
    def __init__(self):
        super(Sampler, self).__init__()

        self.divide = P.Div()
        self.topk = P.TopK(sorted=True)
        self.softmax = P.Softmax(-1)
        self.index_fill = P.IndexFill()

    def construct(self, logits, temperature, top_k=1, repetition_penalty=1.0, input_ids=None):
        """sampler forward"""
        if repetition_penalty != 1.0 and input_ids is not None:
            score = ms.numpy.take_along_axis(logits, input_ids, 1)
            negative_index = score < 0
            positive_index = score >= 0
            score[negative_index] = score[negative_index] * repetition_penalty
            score[positive_index] = score[positive_index] / repetition_penalty
            logits = self.index_fill(logits, 1, input_ids, score)

        logits = self.divide(logits, temperature)
        p_tmp, p_args = self.topk(logits, top_k)
        p = self.softmax(p_tmp)
        return p, p_args
