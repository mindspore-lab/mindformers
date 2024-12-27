# Copyright 2024 Huawei Technologies Co., Ltd
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
"""DPO model for llm model."""
from typing import Union
import copy

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.reinforcement_learning.rl_config import DPOConfig


class DPOModel(PreTrainedModel):
    """
    DPOModel define direct preference optimization model for LLM model.
    Args:
        config(DPOConfig): DPO config,define direct preference optimization algorithm.
        base_model(PreTrainedModel): pretrained model for DPO.
    """

    @args_type_check(config=(dict, DPOConfig))
    def __init__(self, config: Union[dict, DPOConfig], base_model: PreTrainedModel):
        super().__init__(base_model.config, auto_prefix=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.sub_batch_valid_len = P.Sub()

        config = base_model.config
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.input_sliced_sig = config.input_sliced_sig
        self.dpo_model = base_model
        self.use_past = config.use_past
        if config.parallel_config.vocab_emb_dp or (config.vocab_size % mp != 0):
            self.dpo_loss = DPOLoss(config)
        else:
            loss_parallel_config = copy.deepcopy(config)
            # total mp
            loss_parallel_config.parallel_config.model_parallel = dp * mp
            loss_parallel_config.parallel_config.data_parallel = 1
            if dp >= 32 and dp % 8 == 0:  # For large scale training
                loss_parallel_config.parallel_config.model_parallel = 8
                loss_parallel_config.parallel_config.data_parallel = dp * mp // 8
            self.dpo_loss = DPOLoss(loss_parallel_config)

        self.alpha = config.rl_config.dpo_alpha
        self.beta = config.rl_config.dpo_beta

    def update_model_kwargs_before_generate(self, input_ids, model_kwargs: dict):
        return self.dpo_model.update_model_kwargs_before_generate(input_ids, model_kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.dpo_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        return self.dpo_model.prepare_inputs_for_predict_layout(input_ids, **kwargs)

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        return self.dpo_model.slice_incremental_inputs(model_inputs, current_index)

    def set_dynamic_inputs(self, **kwargs):
        return self.dpo_model.set_dynamic_inputs(**kwargs)

    def add_flags_custom(self, is_first_iteration):
        return self.dpo_model.add_flags_custom(is_first_iteration)

    def to_embeddings(self, tokens):
        return self.dpo_model.to_embeddings(tokens)

    def convert_name(self, weight_name):
        return self.dpo_model.convert_name(weight_name)

    def convert_weight_dict(self, source_dict, **kwargs):
        return self.dpo_model.convert_weight_dict(source_dict, **kwargs)

    def convert_map_dict(self, source_dict, **kwargs):
        return self.dpo_model.convert_map_dict(source_dict, **kwargs)

    def construct(self, chosen_input_ids, chosen_labels=None, chosen_attention_mask=None, chosen_loss_mask=None,
                  chosen_ref_logps=None, rejected_input_ids=None, rejected_attention_mask=None, rejected_labels=None,
                  rejected_loss_mask=None, rejected_ref_logps=None,
                  input_ids=None, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        """ Construct of DPO model """

        input_ids = ops.concat((chosen_input_ids, rejected_input_ids), axis=0)
        labels = ops.concat((chosen_labels, rejected_labels), axis=0)
        if chosen_attention_mask is not None:
            attention_mask = ops.concat((chosen_attention_mask, rejected_attention_mask), axis=0)

        bsz, ori_seqlen = self.shape(input_ids)
        tokens = input_ids
        if not self.input_sliced_sig:
            chosen_loss_mask = self.slice(chosen_loss_mask, (0, 1), (bsz, ori_seqlen), (1, 1))
            rejected_loss_mask = self.slice(rejected_loss_mask, (0, 1), (bsz, ori_seqlen), (1, 1))

        logits = self.dpo_model(input_ids=input_ids,
                                labels=labels,
                                input_position=input_position,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                input_embeds=input_embeds,
                                init_reset=init_reset,
                                batch_valid_length=batch_valid_length,
                                batch_index=batch_index,
                                zactivate_len=zactivate_len,
                                block_tables=block_tables,
                                slot_mapping=slot_mapping)

        if not self.input_sliced_sig:
            labels = self.slice(labels, (0, 1), (bsz, ori_seqlen), (1, 1))
            tokens = self.slice(input_ids, (0, 0), (bsz, ori_seqlen - 1), (1, 1))

        if logits.ndim <= 2:
            logits = self.reshape(logits, (bsz, tokens.shape[1], logits.shape[-1]))
        policy_logits = self.cast(logits, mstype.float32)
        dpo_loss, sft_loss = self.dpo_loss(policy_logits, labels, chosen_loss_mask, rejected_loss_mask,
                                           chosen_ref_logps.reshape((-1,)), rejected_ref_logps.reshape((-1,)))
        return self.alpha * dpo_loss + self.beta * sft_loss


class DPOLoss(nn.Cell):
    """ DPO Loss function """
    def __init__(self, config):
        super(DPOLoss, self).__init__()
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.gatherd = P.GatherD().shard(((dp, mp, 1), (dp, mp, 1)))
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum(keep_dims=False).shard(((dp, mp),))
        self.slice = P.StridedSlice()
        self.slice_ind = P.StridedSlice()
        self.slice_mask = P.StridedSlice()
        self.mul = P.Mul().shard(((dp, mp), (dp, mp)))
        self.sub = P.Sub().shard(((dp, mp), (dp, mp)))
        self.log_softmax = P.LogSoftmax().shard(((dp, mp, 1),))
        self.squeeze = P.Squeeze(-1).shard(((dp, mp, 1),))
        self.expand = P.ExpandDims().shard(((dp, mp),))
        self.label_pad_token_id = config.pad_token_id
        self.average_log_prob = True
        self.reference_free = False
        self.log_sigmoid = nn.LogSigmoid()
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        # for cal reward
        self.beta = 0.2
        self.enable_force_redistribute = True
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL):
            self.enable_force_redistribute = True
            self.add = P.Add().shard(((dp, mp), ())).add_prim_attr("keep_alive", True)
            self.add_label = P.Add().shard(((dp,), ())).add_prim_attr("keep_alive", True)

    def _get_batch_logps(self, logits, labels, loss_mask=None):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities.
                    Label tokens with value of label_pad_token_id are ignored.
            Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log
            probabilities of the given labels under the given logits.
        """
        if loss_mask is None:
            loss_mask = self.not_equal(labels, self.label_pad_token_id)
        # [bs, seq_len] -> [bs, seq_len]
        labels = self.mul(labels, loss_mask)
        # [bs, seq_len, vocab_size]
        log_probs = self.log_softmax(logits)
        # [bs, seq_len] -> [bs, seq_len, 1]
        index = self.expand(labels, -1)
        index = self.cast(index, mstype.int32)
        # [bs, seq_len, 1]
        per_token_logps = self.gatherd(log_probs, -1, index)
        # [bs, seq_len, 1] -> [bs, seq_len]
        per_token_logps = self.squeeze(per_token_logps)
        if self.average_log_prob:
            return self.reduce_sum(self.mul(per_token_logps, loss_mask), -1) / self.reduce_sum(loss_mask, -1)
        return self.reduce_sum(self.mul(per_token_logps, loss_mask), -1)

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_mask):
        """
        Compute dpo loss of the given labels under the given logits.

        Args:
            policy_chosen_logps: Log probabilities of the chosen actions by the policy.
            policy_rejected_logps: Log probabilities of the rejected actions by the policy.
            ref_chosen_logps: Log probabilities of the chosen actions by the reference model.
            ref_rejected_logps: Log probabilities of the rejected actions by the reference model.
            loss_mask: Mask to indicate which elements to consider for loss computation.

        Returns:
            losses: Computed losses.
            chosen_rewards: Rewards for chosen actions.
            rejected_rewards: Rewards for rejected actions.
            policy_chosen_logps_avg: Average log probabilities of chosen actions by the policy.
        """
        bs, seq_len = loss_mask.shape
        if self.average_log_prob:
            # already computed, just use
            policy_chosen_logps_avg = policy_chosen_logps
        else:
            chosen_loss_mask = self.slice_mask(loss_mask, (0, 0), (bs // 2, seq_len), (1, 1))
            chosen_valid_len = self.reduce_sum(chosen_loss_mask, -1)
            policy_chosen_logps_avg = policy_chosen_logps / chosen_valid_len

        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps

        if self.reference_free:
            ref_log_ratios = 0
        logits = policy_log_ratios - ref_log_ratios

        losses = -self.log_sigmoid(self.beta * logits)  # if logits is very large, the losses would be nan
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        return losses, chosen_rewards, rejected_rewards, policy_chosen_logps_avg

    def construct(self, policy_logits, policy_labels, chosen_loss_mask, rejected_loss_mask, ref_chosen_logps,
                  ref_rejected_logps):
        """ Construct of DPO loss """
        loss_mask = ops.concat((chosen_loss_mask, rejected_loss_mask), axis=0)
        all_logps = self._get_batch_logps(policy_logits, policy_labels, loss_mask)
        bs = all_logps.shape[0] // 2  # a sample has two bs responses (chosen and rejected)
        policy_chosen_logps = self.slice_ind(all_logps, (0,), (bs,), (1,))
        policy_rejected_logps = self.slice_ind(all_logps, (bs,), (2 * bs,), (1,))

        if self.average_log_prob:
            ref_chosen_logps = ref_chosen_logps / self.reduce_sum(chosen_loss_mask, -1)
            ref_rejected_logps = ref_rejected_logps / self.reduce_sum(rejected_loss_mask, -1)

        dpo_loss, chosen_rewards, rejected_rewards, policy_chosen_logps_avg = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_mask
        )
        sft_loss = -policy_chosen_logps_avg
        if self.phase == "train":
            return dpo_loss, sft_loss
        return dpo_loss, sft_loss, chosen_rewards, rejected_rewards
