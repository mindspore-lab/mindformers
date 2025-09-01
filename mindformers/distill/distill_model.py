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
"""Distill model for llm model."""
from mindspore import Tensor, nn, jit
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore._checkparam import args_type_check

from mindformers.core.context.build_context import is_legacy_model
from mindformers.models.utils import lazy_inline
from mindformers.distill.distill_config import DistillationConfig
from mindformers.distill.distill_loss import LogitsKLLoss, LossBalancer
from mindformers.utils.parameter_register import parameter_register


class DistillationModel(nn.Cell):
    """
    DPOModel define direct preference optimization model for LLM model.
    Args:
        config(DistillationConfig): .
        student_model: pretrained model for DPO.
        teacher_model: pretrained model for DPO.
    """
    @lazy_inline
    @args_type_check(config=(dict, DistillationConfig))
    def __init__(self, config: DistillationConfig, student_model, teacher_model):
        super().__init__(auto_prefix=True)
        self.is_legacy = is_legacy_model()
        self.config = student_model.config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.kd_loss = LogitsKLLoss(temperature=config.temperature)
        self.skip_lm_loss = config.skip_lm_loss
        self.loss_balancer = LossBalancer(kd_loss_scale=config.kd_loss_scale,
                                          skip_original_loss=config.skip_lm_loss)
        self.freeze_teacher()
        self.print_separate_loss = config.print_separate_loss and not self.is_legacy and not self.skip_lm_loss
        if self.print_separate_loss:
            self.init_separate_loss()

    def freeze_teacher(self):
        """freeze teacher model"""
        for param in self.teacher_model.trainable_params():
            param.requires_grad = False
        self.teacher_model.add_flags_recursive(freeze=True)

    def is_mtp_model(self):
        """Check whether the student model is a mtp model."""
        if not self.is_legacy and not self.skip_lm_loss:
            return self.student_model.is_mtp_model()
        return False

    def is_moe_model(self):
        """Check whether the student model is a moe mcore model."""
        if not self.is_legacy and not self.skip_lm_loss:
            return self.student_model.is_moe_model()
        return False

    def init_separate_loss(self):
        """init separate loss from registered parameters"""
        self.kl_loss_parameter = parameter_register.register("kl_loss", Tensor([0], mstype.float32))
        self.aux_loss_parameter = parameter_register.register("aux_loss", Tensor([0], mstype.float32))
        self.mtp_loss_parameter = parameter_register.register("mtp_loss", Tensor([0], mstype.float32))
        self.lm_loss_parameter = parameter_register.register("lm_loss", Tensor([0], mstype.float32))

    @jit(fullgraph=True)
    def construct(self, *args, **kwargs):
        """construct for distillation training"""
        if self.is_legacy:
            logit_t = self.teacher_model(*args, **kwargs)
            loss_s, logit_s, loss_mask = self.student_model(*args, **kwargs)
            kl_loss = self.kd_loss(logit_s, logit_t, loss_mask)
            return self.loss_balancer(loss_s, kl_loss)

        logit_t = self.teacher_model(*args, **kwargs)
        logit_s, loss_mask, lm_loss, mtp_loss, extra_loss = self.student_model(*args, **kwargs)
        if self.skip_lm_loss:
            loss_s = Tensor([0], mstype.float32)
        else:
            loss_s = lm_loss + mtp_loss + extra_loss
        kl_loss = self.kd_loss(logit_s, logit_t, loss_mask)
        if self.print_separate_loss:
            F.assign_add(self.kl_loss_parameter, kl_loss)
            F.assign_add(self.lm_loss_parameter, lm_loss)
            F.assign_add(self.mtp_loss_parameter, mtp_loss)
            F.assign_add(self.aux_loss_parameter, extra_loss)
        return self.loss_balancer(loss_s, kl_loss)
