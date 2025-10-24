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
"""GPT Transformer language model"""
from typing import Dict, Iterable, Optional, Set, Tuple, Literal
import numpy as np

from mindspore import nn, ops, mint, Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
from mindformers.parallel_core.inference.utils import divide, generate_padding_index
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.parallel_core.inference.weights_utils import (default_weight_loader, make_expert_params_mapping,
                                                               make_expert_params_mapping_with_expert_dim)
from mindformers.parallel_core.inference.parallel_state import is_pipeline_last_stage
from mindformers.parallel_core.process_group_config import get_model_comm_pgs
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_pynative


class GPTModel(nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        vocab_size (int): Vocabulary size.
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding.
        pre_process (bool, optional): Set to true if you need to compute embedings.
            Currently only supports setting to True. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits. Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Whether to use FP16 for cross entropy,
            does not support setting to True currently. Default: False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks.
            Default: True.
        share_embeddings_and_output_weights (bool, optional): Whether to share input/output embeddings,
            does not support setting to True currently. Default: False.
        position_embedding_type (Literal['learned_absolute', 'rope', 'llama3', 'yarn', 'none'], optional):
            Type of positional embedding to use. Default: 'learned_absolute'.
        rotary_percent (float, optional): Percentage of dimensions to apply rotary embeddings. Default: 1.0.
        rotary_base (int, optional): Base value for rotary embeddings. Default: 10000.
        rope_scaling (bool, optional): Whether to use rope scaling. Default: False.
        seq_len_interpolation_factor (float, optional): Sequence length interpolation factor. Default: None.
        mtp_block_spec (ModuleSpec, optional): Specification for MTP blocks,
            does not support to set currently. Default: None.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.

    Inputs:
        - **input_ids** (Tensor) - Input token ids
        - **positions** (Tensor, optional) - Token positions
        - **batch_valid_length** (Tensor, optional) - Valid length of each sequence in batch
        - **context_lens_tensor** (Tensor, optional) - Context lengths tensor
        - **q_seq_lens** (Tensor, optional) - Query sequence lengths
        - **block_tables** (Tensor, optional) - Block tables for KV cache
        - **slot_mapping** (Tensor, optional) - Slot mapping for KV cache
        - **attention_mask** (Tensor, optional) - Tensor of attention mask
        - **attn_metadata** (dict, optional) - Additional attention metadata
        - **attn_padding_idx** (Tensor) - Indices mapping positions in attention output sequence to
            original token positions, used for padding attention output to fixed size.
        - **attn_unpadding_idx** (Tensor) - Indices mapping valid tokens in padded attention output sequence to
            their original positions, used for removing padding in attention output.
        - **ffn_padding_idx** (Tensor) - Indices mapping positions in MoE output sequence to
            flattened valid token positions, used for padding MoE output to fixed size.
        - **ffn_unpadding_idx** (Tensor) - Indices mapping valid tokens in padded MoE output sequence to
            their original positions, used for removing padding in MoE output.
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **output** (Tensor) - return hidden states after decoder when no post-processing

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'llama3', 'yarn', "partial_rope", 'none'] = \
                                    'none',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_block_spec: Optional[ModuleSpec] = None,
            model_comm_pgs: Optional[ModelCommProcessGroups] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.check_support(fp16_lm_cross_entropy, rope_scaling)
        self.config = config
        self.quant_config = quant_config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.mtp_block_spec: ModuleSpec = mtp_block_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = getattr(config, "post_process", post_process)
        self.parallel_output = parallel_output
        self.compute_dtype = self.config.compute_dtype
        self.max_position_embeddings = max_sequence_length
        self.head_dim = self.get_head_dim()
        self.rotary_percent = rotary_percent
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        if model_comm_pgs is None:
            model_comm_pgs = get_model_comm_pgs()
        self.model_comm_pgs = model_comm_pgs
        self.tp = model_comm_pgs.tp
        self.tp_group_size = self.tp.size
        self.tp_rank = self.tp.rank
        self.expert_map = self._determine_expert_map()
        self.is_prefill = True
        self.is_chunked = False
        self.return_hidden_states = False  # For serving, return hidden_states early and skip output_layer
        self.is_mtp_model = self.mtp_block_spec is not None
        self.is_pynative = is_pynative()
        self.move_lens_to_cpu = True

        self.position_embedding_type = position_embedding_type if position_embedding_type != "none" else \
            getattr(self.config, 'position_embedding_type')

        self.rotary_base = self.config.rotary_base if hasattr(self.config, 'rotary_base') else rotary_base

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                model_comm_pgs=model_comm_pgs,
            )

        self.rotary_pos_emb = get_rope(
            config,
            hidden_dim=self.head_dim,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rotary_dtype=self.config.rotary_dtype,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            position_embedding_type=self.position_embedding_type,
            original_max_position_embeddings=self.max_position_embeddings,
        )

        # Transformer
        if not self.is_mtp_model:
            self.decoder = TransformerBlock(
                config=self.config,
                spec=transformer_layer_spec,
                post_process=is_pipeline_last_stage(),
                model_comm_pgs=model_comm_pgs,
                quant_config=quant_config,
            )
        else:
            self.decoder = build_module(self.mtp_block_spec, config=config,
                                        model_comm_pgs=model_comm_pgs, quant_config=quant_config)

        # Output
        if self.post_process:
            self.output_layer = ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                bias=False,
                gather_output=self.parallel_output,
                compute_dtype=self.config.compute_dtype,
                tp_group=model_comm_pgs.tp,
            )
        if share_embeddings_and_output_weights:
            self.output_layer.weight = self.embedding.word_embeddings.weight

        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.depend = P.Depend()

        self.set_modules({"model": self})

    def check_support(self, fp16_lm_cross_entropy, rope_scaling):
        """Check support for GPTModel."""
        if fp16_lm_cross_entropy:
            raise NotImplementedError("For GPTModel, `fp16_lm_cross_entropy` is not supported")
        if rope_scaling:
            raise NotImplementedError("For GPTModel, `rope_scaling` is not supported. "
                                      "Please use `rope_type` to control the selection of extrapolation algorithm.")

    def set_modules(self, model_dicts: Dict[str, nn.Cell]):
        self.modules_dict = model_dicts

    def pre_gather_func(self, output, context_lens_tensor, seq_lens_tensor):
        """Pre gather operation in infer mode."""
        if self.is_prefill:
            q_seq_lens_tensor = mint.sub(seq_lens_tensor, context_lens_tensor)
            gather_index = mint.sub(mint.cumsum(q_seq_lens_tensor, 0), 1)
            output = self.gather(output, gather_index, 0)
        return output

    def update_padding_index_to_inputs(self, model_inputs: Dict):
        r"""
        Update the model input and add the related parameters of padding_index.
        """
        tp_group_size = self.model_comm_pgs.tp.size
        dp_group_size = self.model_comm_pgs.dp.size
        ep_group_size = self.model_comm_pgs.moe_ep.size
        q_seq_lens = model_inputs.get("q_seq_lens", None)

        if dp_group_size == 1 or q_seq_lens is None or (dp_group_size == ep_group_size and tp_group_size == 1):
            return model_inputs

        (
            attn_padding_idx,
            attn_unpadding_idx,
            ffn_padding_idx,
            ffn_unpadding_idx,
        ) = generate_padding_index(q_seq_lens)

        model_inputs.update({
            "attn_padding_idx": attn_padding_idx,
            "attn_unpadding_idx": attn_unpadding_idx,
            "ffn_padding_idx": ffn_padding_idx,
            "ffn_unpadding_idx": ffn_unpadding_idx,
        })
        return model_inputs

    def construct(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                  context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """ Construct function of GPTModel. """

        # Generate cos and sin for RoPE.
        if self.is_prefill and not self.is_chunked:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_decode(positions)

        # current aclgraph not support moveto in graph
        # add moveto config for model to control
        if self.move_lens_to_cpu:
            if self.is_pynative:  # ops.move_to not support pynative mode
                batch_valid_length_cpu = batch_valid_length.move_to("CPU")
                q_seq_lens_cpu = q_seq_lens.move_to("CPU")
                context_lens_tensor_cpu = context_lens_tensor.move_to("CPU")
            else:
                batch_valid_length_cpu = ops.move_to(batch_valid_length, "CPU")
                q_seq_lens_cpu = ops.move_to(q_seq_lens, "CPU")
                context_lens_tensor_cpu = ops.move_to(context_lens_tensor, "CPU")

            # embedding contains the allreduce ops. Adding the depend ops ensures that the move_to ops is
            # launched before the allreduce, reducing the sync waiting time when the move_to ops launched.
            input_ids = self.depend(input_ids, q_seq_lens_cpu)
            input_ids = self.depend(input_ids, batch_valid_length_cpu)
            input_ids = self.depend(input_ids, context_lens_tensor_cpu)
        else:
            batch_valid_length_cpu = None
            q_seq_lens_cpu = None
            context_lens_tensor_cpu = None

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.cast(self.embedding(input_ids), self.compute_dtype)
        else:
            if hidden_states is None:
                raise ValueError("When pre_process is False, hidden_states must be provided.")
            decoder_input = self.cast(hidden_states, self.compute_dtype)

        hidden_states = (decoder_input,) if not self.is_mtp_model else (decoder_input, hidden_states)
        # Run decoder.
        hidden_states = self.decoder(
            *hidden_states,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache,
            q_seq_lens_cpu=q_seq_lens_cpu,
            batch_valid_length_cpu=batch_valid_length_cpu,
            context_lens_tensor_cpu=context_lens_tensor_cpu,
        )

        # Return hidden states.
        if not self.post_process or self.return_hidden_states:
            return hidden_states

        output = self.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)
        # Return logits.
        logits = self.output_layer(output)
        logits = self.cast(logits.squeeze(0), mstype.float32)
        return logits

    def get_params_dict(self):
        params_dict = {}
        for _, module in self.modules_dict.items():
            module_params = module.parameters_dict()
            for param_name, param in module_params.items():
                params_dict[param_name] = param

        return params_dict

    def _determine_expert_map(self):
        """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks
        """
        ep_group_size = self.model_comm_pgs.moe_ep.size
        if ep_group_size == 1:
            return None

        num_local_experts = self.config.num_moe_experts // ep_group_size
        ep_rank = self.model_comm_pgs.moe_ep.rank
        # Create a list of size num_experts filled with -1
        expert_map = np.full((self.config.num_moe_experts,), -1, dtype=np.int32)
        # Create an expert map for the local experts
        expert_map[ep_rank * num_local_experts: (ep_rank + 1) * num_local_experts] = \
            np.arange(0, num_local_experts, dtype=np.int32)
        return expert_map

    def map_global_expert_id_to_local_expert_id(self, global_expert_id: int) -> int:
        r"""
        Map global expert_id to local ID on current ep_rank.
        Each rank get base number of experts.

        Args:
            global_expert_id (int): Global expert ID

        Returns:
            local_expert_id (int): Local expert ID, returns -1 if not belonging to current rank
        """
        if self.expert_map is None:
            return global_expert_id
        return self.expert_map[global_expert_id].item()

    def generate_expert_mapping(self, expert_params_mapping, has_num_experts_dim, num_experts):
        """
        Generate expert parameter mapping configuration.

        Args:
            expert_params_mapping: Expert parameter mapping configuration, if None or empty it will be auto-generated
            has_num_experts_dim (bool): Whether it has expert dimension
            num_experts (int): Number of experts, required when has_num_experts_dim is False

        Returns:
            Expert parameter mapping.
        """
        if not expert_params_mapping:
            if has_num_experts_dim:
                expert_params_mapping = make_expert_params_mapping_with_expert_dim(
                    ckpt_gate_proj_name="gating",
                    ckpt_down_proj_name="linear_fc2",
                    ckpt_up_proj_name="hidden"
                )
            else:
                expert_params_mapping = make_expert_params_mapping(
                    ckpt_gate_proj_name="gating",
                    ckpt_down_proj_name="linear_fc2",
                    ckpt_up_proj_name="hidden",
                    num_experts=num_experts
                )
        return expert_params_mapping

    def load_default_param(self, loaded_params, loaded_weight, name, params_dict, is_hf_weight):
        """
        Load default parameters into the model.

        Args:
            loaded_params: Set of already loaded parameters, used to record processed parameter names
            loaded_weight: Weight data to be loaded
            name: Parameter name
            params_dict: Parameter dictionary containing model parameter objects
            is_hf_weight: Boolean flag indicating whether the weight is from HuggingFace format
        """
        if name in params_dict:
            if '.weight1' in name or '.weight2' in name:
                num_experts = self.config.num_moe_experts
                weight = {}
                if '.weight1' in name:
                    weight = loaded_weight[:].reshape(num_experts, self.config.hidden_size, -1)
                if '.weight2' in name:
                    weight = loaded_weight[:].reshape(num_experts, self.config.moe_ffn_hidden_size, -1)

                for expert_id in range(num_experts):
                    expert_id = self.map_global_expert_id_to_local_expert_id(expert_id)
                    loaded_weight = weight[expert_id]
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=None, expert_id=expert_id)
                    loaded_params.add(name)
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if '.linear_fc1.' in name:
                    weight_loader(param, loaded_weight, is_hf_weight=is_hf_weight)
                else:
                    weight_loader(param, loaded_weight)
                loaded_params.add(name)

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]], stacked_params_mapping=None, is_hf_weight=True):
        r"""
        The weight is processed in modules, and the weight is cut online and loaded.

        Args:
            weights: An iterable (usually a generator) yielding tuples of (parameter_name, parameter_tensor).
                    The tensor is a sliceable object (like PySafeSlice) for memory efficiency.
            stacked_params_mapping: A list of expert parameter mappings, where each element is a tuple of
                                    (param_name, weight_name, shard_id).
                                    - param_name: Prefix of the model parameter name
                                    - weight_name: Weight name of the logical expert
                                    - shard_id: Shard ID
        """
        loaded_params: Set[str] = set()
        params_dict = self.get_params_dict()

        # Create weight mapping for routed experts in Mixture of Experts (MoE)
        num_experts = self.config.num_moe_experts
        expert_params_mapping = []

        for name, loaded_weight in weights:

            if "weight_scale_inv" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                if '.experts.' in name and '.weight1' not in name and '.weight2' not in name:
                    has_num_experts_dim = (loaded_weight.get_shape()[0] == num_experts
                                           and len(loaded_weight.get_shape()) == 3)
                    expert_params_mapping = self.generate_expert_mapping(expert_params_mapping, has_num_experts_dim,
                                                                         num_experts)
                    for mapping in expert_params_mapping:
                        if has_num_experts_dim:
                            param_name, weight_name, shard_id = mapping
                            expert_ids = range(num_experts)
                        else:
                            param_name, weight_name, expert_id, shard_id = mapping
                            expert_ids = [expert_id]

                        if weight_name not in name:
                            continue

                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            continue

                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in expert_ids:
                            expert_id = self.map_global_expert_id_to_local_expert_id(expert_id)
                            loaded_weight = loaded_weight[expert_id] if has_num_experts_dim else loaded_weight
                            weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                            loaded_params.add(name)
                            break
                else:
                    self.load_default_param(loaded_params, loaded_weight, name, params_dict, is_hf_weight)

        network_not_load = set(params_dict.keys()) - loaded_params
        logger.warning(f'These parameters are not loaded in the network: {network_not_load}')
        return loaded_params

    def get_head_dim(self):
        """ Get head_dim from model config. """
        if not hasattr(self.config, "qk_pos_emb_head_dim"):
            if hasattr(self.config, "kv_channels"):
                return getattr(self.config, "kv_channels")
            return divide(self.config.hidden_size, self.config.num_attention_heads)
        return self.config.qk_pos_emb_head_dim
