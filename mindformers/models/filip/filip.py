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

"""
FilipModel
"""
import math
import os

import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore import nn, Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_group_size, get_rank, create_group
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops.primitive import constexpr

from ...mindformer_book import MindFormerBook
from ..base_model import BaseModel
from .filip_modules import VisualTransformer, TextTransformer
from ...tools.register import MindFormerRegister, MindFormerModuleType


@constexpr
def int_num(num):
    return int(num)


class SoftCrossEntropyLoss(nn.Cell):
    """
    SoftCrossEntropyLoss.
    """
    def __init__(self, smooth=0.1):
        super(SoftCrossEntropyLoss, self).__init__()
        self.smooth = smooth
        self.expand_op = ops.ExpandDims()
        self.concat_op = ops.Concat(1)
        self.ones = ops.Ones()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.logsoftmax = nn.LogSoftmax()
        self.onehot = ops.OneHot()

    def construct(self, logit, target):
        target_onehot = self.onehot(target, logit.shape[1], self.on_value, self.off_value)
        target_onehot = (1 - self.smooth) * target_onehot + self.smooth / logit.shape[1]
        log_probs = self.logsoftmax(logit)
        loss = (- target_onehot * log_probs).mean(0).sum()
        return loss


class FilipImgProcess(nn.Cell):
    """
    FilipImgProcess, norm the input image features.
    """
    def __init__(self):
        super(FilipImgProcess, self).__init__()
        self.image_norm = nn.Norm(axis=-1, keep_dims=True)

    def construct(self, image_features):
        image_features = image_features[:, 1:, :]
        image_features = image_features / self.image_norm(image_features)
        return image_features


class FilipTextProcess(nn.Cell):
    """
    FilipTextProcess, norm the input text features and get text padding mask.
    """
    def __init__(self):
        super(FilipTextProcess, self).__init__()
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)
        self.expand_dims = ops.ExpandDims()

    def construct(self, text_features, text):
        text_features = text_features / self.text_norm(text_features)
        text_pad_mask = text > 0
        text_pad_mask_expand = self.expand_dims(text_pad_mask, -1)
        text_features = text_features * text_pad_mask_expand
        return text_features, text_pad_mask


class FilipTopKFeature(nn.Cell):
    """
    FilipTopKFeature, used in filip loss.
    """
    def __init__(self, top_token_percent, local_group_size):
        super(FilipTopKFeature, self).__init__()
        self.percent_i, self.percent_t = top_token_percent
        self.local_group_size = local_group_size

        self.rank_size = int(os.getenv("RANK_SIZE", '1'))
        if self.rank_size > 1:
            rank_id = get_rank()
            sub_group = rank_id // local_group_size
            group = "{}-{}".format(sub_group * local_group_size, (sub_group + 1) * local_group_size)
            rank_ids = list(range(sub_group * local_group_size, (sub_group + 1) * local_group_size))
            create_group(group, rank_ids)
            self.local_gather = ops.AllGather(group)

        self.matmul = ops.MatMul(transpose_b=True)
        self.topk = ops.TopK(sorted=True)
        self.expand_dim = ops.ExpandDims()

    def construct(self, image_feature, text_feature, text_pad_mask):
        """construct of FilipTopKFeature"""

        if self.rank_size > 1:
            text_local_gather = self.local_gather(text_feature)
            image_local_gather = self.local_gather(image_feature)
        else:
            text_local_gather = text_feature
            image_local_gather = image_feature

        # get image features
        batch_size, n_token1, feat_dim = image_feature.shape
        output = self.matmul(image_feature.reshape((-1, feat_dim)), text_local_gather.reshape((-1, feat_dim)))
        token_rep = output.reshape((batch_size, n_token1, -1)).max(2)
        max_token_idx = self.topk(token_rep, int_num(n_token1 * self.percent_i))[1]
        max_token_idx = ops.stop_gradient(max_token_idx)
        bs_index = nn.Range(batch_size)()
        bs_index = self.expand_dim(bs_index, 1)
        image_feature = image_feature[bs_index, max_token_idx]
        # get text features
        batch_size, n_token1, feat_dim = text_feature.shape
        output = self.matmul(text_feature.reshape((-1, feat_dim)), image_local_gather.reshape((-1, feat_dim)))
        token_rep = output.reshape((batch_size, n_token1, -1)).max(2)
        max_token_idx = self.topk(token_rep, int_num(n_token1 * self.percent_i))[1]
        max_token_idx = ops.stop_gradient(max_token_idx)
        bs_index = nn.Range(batch_size)()
        bs_index = self.expand_dim(bs_index, 1)
        text_feature = text_feature[bs_index, max_token_idx]
        text_pad_mask = text_pad_mask[bs_index, max_token_idx]
        return image_feature, text_feature, text_pad_mask


class FilipLogit(nn.Cell):
    """
    FilipLogit, compute logit.
    """
    def __init__(self, use_mask_flag=False):
        super(FilipLogit, self).__init__()
        self.use_mask_flag = use_mask_flag
        self.matmul = ops.MatMul(transpose_b=True)
        logit_value = math.exp(3.8665097)
        self.logit_scale = Parameter(Tensor(np.log(logit_value), dtype=ms.float32))
        self.exp = ops.Exp()
        self.cast = ops.Cast()

    def construct(self, rep1, rep2, mask=None):
        """construct of FilipLogit"""
        logit_scale = self.exp(self.logit_scale)
        batch_size1, n_token1, feat_dim = rep1.shape
        _, n_token2, _ = rep2.shape
        rep1 = rep1.reshape((-1, feat_dim))
        rep2 = rep2.reshape((-1, feat_dim))
        out = self.matmul(rep1, rep2)
        out = out.reshape((batch_size1, n_token1, -1, n_token2)).max(3)
        if self.use_mask_flag:
            out = out.sum(1)
            mask = self.cast(mask, ms.float32)
            mask_sum = mask.sum(axis=1, keepdims=True).clip(min=1.0, max=None)
            logits = out / mask_sum
        else:
            logits = out.mean(1)
        logits = logit_scale * logits
        logits = logits.clip(min=-100, max=100)
        return logits


class FilipGather(nn.Cell):
    """
    FilipGather, input image features, text features, and text padding mask, return logits.
    """
    def __init__(self, top_token_percent, local_group_size):
        super(FilipGather, self).__init__()
        self.top_k_feature = FilipTopKFeature(top_token_percent, local_group_size)
        self.get_img_logits = FilipLogit()
        self.get_txt_logits = FilipLogit(True)
        self.rank_size = int(os.getenv("RANK_SIZE", '1'))
        if self.rank_size > 1:
            self.all_gather = ops.AllGather()

    def construct(self, image_features, text_features, text_pad_mask):
        """
        construct of FilipGather
        """
        if self.rank_size > 1:
            image_gather = self.all_gather(image_features)
            text_gather = self.all_gather(text_features)
        else:
            image_gather = image_features
            text_gather = text_features

        logits_per_image = self.get_img_logits(image_features, text_gather)
        logits_per_text = self.get_txt_logits(text_features, image_gather, text_pad_mask)
        return logits_per_image, logits_per_text


class FilipLossCell(nn.Cell):
    """
    FilipLossCell.
    """
    def __init__(self,
                 label_smooth=0.1,
                 local_group_size=1,
                 top_token_percent=(1, 1),
                 ):
        super(FilipLossCell, self).__init__()
        self.local_group_size = local_group_size
        self.top_token_percent = top_token_percent
        self.process_img_features = FilipImgProcess()
        self.process_text_features = FilipTextProcess()

        self.rank_size = int(os.getenv("RANK_SIZE", '1'))
        if self.rank_size > 1:
            self.rank = get_rank()
        else:
            self.rank = 0

        self.two = Tensor(2.0, dtype=mstype.float32)
        # self.all_gather = ops.AllGather()
        self.equal = ops.Equal()
        self.cast = ops.Cast()
        self.logsoftmax = nn.LogSoftmax()
        self.loss = SoftCrossEntropyLoss(label_smooth)
        self.gather_and_compute = FilipGather(top_token_percent, local_group_size)

    def construct(self, image_features, text_features, text):
        """construct of FilipLossCell"""
        image_features = self.process_img_features(image_features)
        text_features, text_pad_mask = self.process_text_features(text_features, text)
        batch_size = image_features.shape[0]
        logits_per_image, logits_per_text = self.gather_and_compute(
            image_features, text_features, text_pad_mask
        )

        target = nn.Range(self.rank * batch_size, (self.rank + 1) * batch_size)()
        target = F.cast(target, ms.int32)
        loss = (self.loss(logits_per_image, target) + self.loss(logits_per_text, target)) / self.two

        return loss

@MindFormerRegister.register(MindFormerModuleType.MODELS)
class FilipModel(BaseModel):
    """
    FilipModel.
    The supported model name could be selected from FilipModel.show_support_list().

    Args:
        config (FilipConfig): the config of filip model.
    """
    _support_list = MindFormerBook.get_model_support_list()['filip']

    def __init__(self, config, is_training=True):
        super(FilipModel, self).__init__(config)
        self.image_encoder = VisualTransformer(config=config)
        self.text_encoder = TextTransformer(config=config)
        self.image_norm = nn.Norm(axis=-1, keep_dims=True)
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)
        self.is_training = is_training
        self.loss = FilipLossCell()
        self.cast = ops.Cast()
        self.load_checkpoint(config)

    def get_image_feature(self, image):
        image_features = self.image_encoder(image)
        return image_features

    def get_text_feature(self, text):
        text_features = self.text_encoder(text)
        return text_features

    def construct(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        if not self.is_training:
            return image_features, text_features

        total_loss = self.loss(image_features, text_features, text)
        total_loss = self.cast(total_loss, mstype.float32)
        return total_loss


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = ops.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                     F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """Construct the trainer of Bert."""
    return grad * reciprocal(scale)

_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    """Construct the trainer of Bert."""
    return grad_overflow(grad)

@MindFormerRegister.register(MindFormerModuleType.WRAPPER, alias="FilipTrainOneStepWithLossScaleWrapper")
class FilipTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    FilipTrainOneStepWithLossScaleCell wrapper
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(FilipTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = ops.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self, image, text, sens=None):
        """
        construct of FilipTrainOneStepWithLossScaleCell
        """
        weights = self.weights
        loss = self.network(image, text)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(image, text, self.cast(scaling_sens, mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
        else:
            print(">>>>overflow")
        return loss, cond, scaling_sens
