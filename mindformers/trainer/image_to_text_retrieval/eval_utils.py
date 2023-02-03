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
"""Image-to-text Retrieval Trainer Utils."""
import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype


class LateSimilarity(nn.Cell):
    """Late Similarity Class"""
    def __init__(self):
        super(LateSimilarity, self).__init__()
        self.matmul = ops.MatMul(transpose_b=True)
        self.concat = ops.Concat(1)
        self.topk = ops.TopK(sorted=True)
        self.equal = ops.Equal()
        self.cast = ops.Cast()

    def construct(self, rep1, rep2):
        batch_size1, n_token1, feat_dim = rep1.shape
        _, n_token2, _ = rep2.shape
        rep1 = rep1.reshape(-1, feat_dim)
        rep2 = rep2.reshape(-1, feat_dim)
        out = self.matmul(rep1, rep2)
        out = out.reshape(batch_size1, n_token1, -1, n_token2)
        out = out.max(3)
        out = out.mean(1)
        return out


def late_similarity(rep1, rep2):
    """Compute late similarity between rep1 and rep2"""
    bs, _, _ = rep2.shape
    matrix = LateSimilarity()
    chunk_size = 128
    num_shards = math.ceil(bs // 128)
    result = []
    for i in range(num_shards):
        rep2_seg = rep2[chunk_size * i: chunk_size * (i + 1)]
        result_seg = matrix(rep1, rep2_seg)
        result.append(result_seg)
    result = ops.Concat(1)(result)
    return result


def get_metrics_ms(image_features: np.ndarray, text_features: np.ndarray):
    """get evaluation metric"""
    image_features = Tensor(image_features, dtype=mstype.float32)
    text_features = Tensor(text_features, dtype=mstype.float32)
    metrics = {}

    logits_per_image = late_similarity(image_features, text_features)
    logits_per_text = late_similarity(text_features, image_features)

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = Tensor(np.arange(len(text_features)).reshape((-1, 1)), mindspore.int32)

    for name, logit in logits.items():
        _, ranking = ops.Sort(descending=True)(logit)
        condition = ops.equal(ranking, ground_truth)
        condition = condition.asnumpy()
        preds = np.nonzero(condition)[1]
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics
