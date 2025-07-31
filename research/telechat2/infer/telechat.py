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
"""Telechat models' APIs."""
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, ops, mint, mutable
from mindspore.communication._comm_helper import _is_initialized as mindspore_comm_has_init

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules import Linear
from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode, is_pynative
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel, is_initialized
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.models.utils import jit

from research.telechat2.infer.utils import convert_model_config
from research.telechat2.infer.layers import ColumnParallelLinear
from research.telechat2.infer.telechat_transformers import TelechatParallelTransformer
from research.telechat2.telechat_config import TelechatConfig


__all__ = ["ParallelTelechatForCausalLM"]


class TelechatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TelechatConfig
    base_model_prefix = "telechat"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class ParallelTelechatForCausalLM(TelechatPreTrainedModel):
    r"""
    Provide telechat training loss or logits through network.

    Args:
        config (TelechatConfig): The config of llama model.

    Returns:
        output: Tensor, the output of llama decoderlayer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=True)
        self.config = convert_model_config(config)
        self.config.out_proj_has_bias = True
        if not is_initialized() and mindspore_comm_has_init():
            initialize_model_parallel(tensor_model_parallel_size=self.config.parallel_config.model_parallel,
                                      order='tp-dp')
        if is_initialized():
            model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp', 'dp'])
        else:
            model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.is_pynative = is_pynative()

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.slice = ops.StridedSlice()
        self.not_equal = ops.NotEqual()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.ones = ops.Ones()
        self.gather = ops.Gather(1) if self.is_pynative else ops.Gather()
        self.sub_batch_valid_len = ops.Sub()
        self.model = TelechatParallelTransformer(config=config, model_comm_pgs=model_comm_pgs)
        if config.parallel_config.vocab_emb_dp:
            self.lm_head = Linear(
                in_channels=config.hidden_size,
                out_channels=config.vocab_size,
                weight_init="normal",
                has_bias=False,
                param_init_type=config.param_init_type,
                compute_dtype=config.compute_dtype
            )
        else:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                config=config.parallel_config,
                bias=False,
                gather_output=True,
                param_init_type=config.param_init_dtype,
                compute_dtype=config.compute_dtype,
                tp_group=model_comm_pgs.tp,
            )

        self.load_checkpoint(config)
        self.predict_run_mode = get_predict_run_mode()

        self.use_past = config.use_past
        self.npu_mem_size = config.npu_mem_size if hasattr(config, "npu_mem_size") else 2
        self.return_hidden_states = config.return_hidden_states

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Telechat model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, slot_mapping, prefix_keys_values

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        prepare inputs for generation.
        A model class needs to define a `prepare_inputs_for_generation` method
        in order to use `.generate()`

        """
        model_inputs = {"input_ids": Tensor.from_numpy(input_ids.astype(np.int32))}
        batch_valid_length = kwargs.get("valid_length_each_example")
        prefill = kwargs.get("prefill")

        if self.is_pynative:
            model_inputs = {}
            if self.config.is_dynamic and "origin_inputs" in kwargs and self.use_past:
                input_ids = kwargs["origin_inputs"]
            model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        else:
            if self.config.is_dynamic:
                if prefill and "origin_inputs" in kwargs:
                    origin_inputs = kwargs["origin_inputs"]
                    slot_mapping = kwargs.get("slot_mapping")
                    model_inputs = self._prepare_inputs_for_prefill_flatten(origin_inputs,
                                                                            batch_valid_length,
                                                                            slot_mapping,
                                                                            model_inputs)
        position_ids = batch_valid_length - 1
        model_inputs["position_ids"] = ms.Tensor(position_ids, dtype=ms.int32).reshape(-1)

        if not prefill:
            q_seq_lens = np.ones(batch_valid_length.shape, dtype=np.int32).reshape(-1)
        else:
            q_seq_lens = batch_valid_length.astype(np.int32).reshape(-1)
        model_inputs["q_seq_lens"] = Tensor.from_numpy(q_seq_lens)

        model_inputs["attention_mask"] = self.model.casual_mask.gen_attention_mask(prefill)
        return model_inputs

    def set_dynamic_inputs(self, **kwargs):
        """Set dynamic input for telechat."""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_position_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.config.compute_dtype)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)

        def get_input():
            if self.npu_mem_size > 0:
                return None
            cache_list = []
            for _ in self.model.layers:
                cache_list.append(Tensor(shape=[None, None, None, None], dtype=self.config.compute_dtype))
            return mutable(cache_list)

        key_cache = get_input()
        value_cache = get_input()
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values, None, key_cache, value_cache)
        else:
            self.set_inputs(dynamic_input_ids, None, None, dynamic_position_ids, dynamic_attention_mask, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None, None, dynamic_q_seq_lens, key_cache, value_cache)
        logger.info("Set dynamic input for telechat.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    @jit
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                  q_seq_lens=None, key_cache=None, value_cache=None):
        """
        Forward of telechat model.
        """
        bsz, _ = self.shape(input_ids)
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
            else:
                batch_valid_length = self.reshape(batch_valid_length, (-1,))
        output = self.model(input_ids, batch_valid_length, batch_index, zactivate_len, block_tables,
                            slot_mapping, prefix_keys_values, position_ids=position_ids, attention_mask=attention_mask,
                            q_seq_lens=q_seq_lens, key_cache=key_cache, value_cache=value_cache)
        if self.return_hidden_states:
            output = self.reshape(output, (-1, output.shape[-1]))
            return output
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            if not self.is_pynative:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        logits = self.cast(logits, mstype.float32)
        if self.predict_run_mode:
            return self.reshape(logits, (-1, logits.shape[-1]))
        input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
        return logits, input_ids, input_mask

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
