# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindFormer Self-Define Loss."""

from mindspore import nn, Tensor
from mindspore.nn.transformer.loss import CrossEntropyLoss
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class SoftTargetCrossEntropy(LossBase):
    """SoftTargetCrossEntropy for MixUp Augment."""

    def __init__(self, parallel_config=None):
        super(SoftTargetCrossEntropy, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.mean_ops = P.ReduceMean(keep_dims=False).shard(((1,),))
        self.sum_ops = P.ReduceSum(keep_dims=False).shard(((dp, 1),))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.mul1d = P.Mul().shard(((dp, 1), ()))
        self.log_softmax = P.LogSoftmax().shard(((dp, 1),))

    def construct(self, logit, label):
        logit = P.Cast()(logit, mstype.float32)
        label = P.Cast()(label, mstype.float32)
        logit_softmax = self.log_softmax(logit)
        neg_target = self.mul1d(label, -1)
        soft_target = self.mul(neg_target, logit_softmax)
        loss = self.sum_ops(soft_target, -1)
        return self.mean_ops(loss)


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class MSELoss(nn.Cell):
    """MSELoss for parallel."""
    def __init__(self, norm_pixel_loss=True, parallel_config=None):
        super(MSELoss, self).__init__()
        if parallel_config is not None:
            dp = parallel_config.data_parallel
        else:
            dp = 1
        self.add_loss = P.Add().shard(((dp, 1, 1), ()))
        self.sub = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.divide = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
        self.pow = P.Pow().shard(((dp, 1, 1), ()))
        self.divide1 = P.RealDiv().shard(((), ()))
        self.divide2 = P.RealDiv().shard(((dp, 1, 1), ()))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.cast = P.Cast()
        self.mean1 = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.mean2 = P.ReduceMean().shard(((dp, 1, 1),))
        self.mul = P.Mul().shard(((dp, 1), (dp, 1)))
        self.sum = P.ReduceSum().shard(((dp, 1,),))
        self.sum2 = P.ReduceSum(keep_dims=True).shard(((dp, 1, 1),))
        self.norm_pixel_loss = norm_pixel_loss

    def construct(self, pred, target, mask):
        """mse loss construct."""
        pred = self.cast(pred, mstype.float32)
        target = self.cast(target, mstype.float32)
        mask = self.cast(mask, mstype.float32)
        if self.norm_pixel_loss:
            mean = self.mean1(target, -1)
            var = self.variance(target)
            var = self.add_loss(var, 1e-6)
            std = self.pow(var, 0.5)
            sub = self.sub(target, mean)
            target = self.divide(sub, std)
        res = self.sub(pred, target)
        recon_loss = self.square(res)
        recon_loss = self.mean2(recon_loss, -1)
        loss_mask = self.mul(recon_loss, mask)
        loss_sum = self.sum(loss_mask)
        mask_sum = self.sum(mask)
        loss = self.divide1(loss_sum, mask_sum)
        return loss

    def variance(self, x):
        """get variance."""
        axis = (x.ndim - 1,)
        x_mean = self.mean1(x, axis)
        x_sub = self.sub(x, x_mean)
        x_pow = self.pow(x_sub, 2)
        x_sum = self.sum2(x_pow, axis)
        x_var = self.divide2(x_sum, x.shape[-1])
        return x_var


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class InfoNceLoss(nn.Cell):
    """InfoNceLoss for parallel."""
    def __init__(self, temperature=0.1, batch_size=64, n_views=2, parallel_config=None):
        super(InfoNceLoss, self).__init__()
        if parallel_config:
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
        else:
            dp = 1
            mp = 1

        self.batch_size = batch_size // 2
        self.temperature = temperature
        self.n_views = n_views
        self.norm = P.L2Normalize(axis=-1).shard(((dp, 1),))
        self.matmul = P.MatMul(transpose_b=True).shard(((dp, 1), (mp, 1)))
        parallel_config.model_parallel = 1
        self.cross_entropy = CrossEntropyLoss(parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.gather = P.GatherNd().shard(((1, 1), (1, 1)))
        self.cat = P.Concat(axis=2).shard(((1, 1, 1), (1, 1, 1)))

        self.pos_mask = Tensor(
            [[i, j]
             for i in range(self.batch_size * self.n_views)
             for j in range(self.batch_size * self.n_views)
             if j % self.batch_size == i % self.batch_size and j != i], mstype.int32)
        self.neg_mask = Tensor(
            [[i, j]
             for i in range(self.batch_size * self.n_views)
             for j in range(self.batch_size * self.n_views)
             if j % self.batch_size != i % self.batch_size], mstype.int32)

        self.ones_like = P.OnesLike().shard(((dp,),))
        self.zeros = P.Zeros().shard(((dp,),))
        self.real_div = P.RealDiv().shard(((dp, 1), ()))
        self.expand_dim = P.ExpandDims().shard(((dp, 1),))

    def construct(self, features):
        """InfoNceLoss construct."""
        b = self.batch_size
        n = self.n_views
        features = self.reshape(features, (b * n, -1))
        # [ B * N, B * N ]
        features = self.norm(features)
        # [ B * N, E ]
        similarity_matrix = self.matmul(features, features)
        # [ B * N, E ] * [ E, B * N ] = [ B * N, B * N ]

        pos = self.gather(similarity_matrix, self.pos_mask)
        # [ B * N, N - 1 ]
        neg = self.gather(similarity_matrix, self.neg_mask)
        # [ B * N, (B - 1) * N ]

        pos = self.reshape(pos, (b * n, -1))
        neg = self.reshape(neg, (b * n, -1))
        pos = self.expand_dim(pos, 0)
        neg = self.expand_dim(neg, 0)
        logits = self.cat((pos, neg))
        logits = self.reshape(logits, (logits.shape[1], -1))

        labels = self.zeros(logits.shape[0], mstype.int32)
        logits = self.real_div(logits, self.temperature)
        input_mask = self.ones_like(labels)
        input_mask = self.cast(input_mask, mstype.float32)
        return self.cross_entropy(logits, labels, input_mask)


@MindFormerRegister.register(MindFormerModuleType.LOSS)
class L1Loss(LossBase):
    """L1Loss for parallel."""
    def __init__(self, reduction='mean', parallel_config=None):
        super(L1Loss, self).__init__()

        if parallel_config:
            dp = parallel_config.data_parallel
        else:
            dp = 1

        self.abs = P.Abs().shard(((dp, 1, 1, 1),))
        self.sub = P.Sub().shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))

        self.mul = P.Mul().shard(((), (dp, 1, 1, 1)))
        self.reduce_mean = P.ReduceMean().shard(((dp, 1, 1, 1),))
        self.reduce_sum = P.ReduceSum().shard(((dp, 1, 1, 1),))
        self.cast = P.Cast()

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

    def get_loss(self, x, weights=1.0):
        """get loss."""
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, logits, labels):
        """L1Loss construct."""
        x_sub = self.sub(logits, labels)
        x = self.abs(x_sub)
        return self.get_loss(x)
