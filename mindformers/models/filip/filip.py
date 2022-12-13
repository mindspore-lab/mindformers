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

"""
FilipModel
"""

import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore import nn, Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, create_group
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
        self.scatter = ops.ScatterNd()
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
        text_local_gather = self.local_gather(text_feature)
        image_local_gather = self.local_gather(image_feature)
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
        logit_value = 3.8665097
        self.logit_scale = Parameter(Tensor(logit_value, dtype=ms.float32))
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
        self.all_gather = ops.AllGather()

    def construct(self, image_features, text_features, text_pad_mask):
        image_gather = self.all_gather(image_features)
        text_gather = self.all_gather(text_features)
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
        self.rank = get_rank()
        self.two = Tensor(2.0, dtype=mstype.float32)
        self.all_gather = ops.AllGather()
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

    def __init__(self, config):
        super(FilipModel, self).__init__(config)
        self.image_encoder = VisualTransformer(config=config)
        self.text_encoder = TextTransformer(config=config)
        self.image_norm = nn.Norm(axis=-1, keep_dims=True)
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)

    def get_image_feature(self, image):
        image_features = self.image_encoder(image)
        return image_features

    def get_text_feature(self, text):
        text_features = self.text_encoder(text)
        return text_features

    def construct(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        return image_features, text_features
