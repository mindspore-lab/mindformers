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

from mindspore import nn, ops, mint, Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
from mindformers.parallel_core.inference.utils import divide, generate_padding_index
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.inference.weights_utils import default_weight_loader, make_expert_params_mapping
from mindformers.tools.logger import logger


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
            position_embedding_type: Literal[
                'learned_absolute', 'rope', 'llama3', 'yarn', 'none'
            ] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_block_spec: Optional[ModuleSpec] = None,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
    ):
        super(GPTModel, self).__init__()
        if not pre_process:
            raise NotImplementedError("For GPTModel, `pre_process` is not supported to set False")
        if fp16_lm_cross_entropy:
            raise NotImplementedError("For GPTModel, `fp16_lm_cross_entropy` is not supported")
        if rope_scaling:
            raise NotImplementedError("For GPTModel, `rope_scaling` is not supported. "
                                      "Please use `rope_type` to control the selection of extrapolation algorithm.")
        if mtp_block_spec:
            raise NotImplementedError("For GPTModel, `mtp_block_spec` is not supported")

        self.config = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = getattr(config, "post_process", post_process)
        self.parallel_output = parallel_output
        self.compute_dtype = self.config.compute_dtype

        self.max_position_embeddings = max_sequence_length
        if not hasattr(config, "qk_pos_emb_head_dim"):
            if hasattr(config, "kv_channels"):
                self.head_dim = getattr(config, "kv_channels")
            else:
                self.head_dim = divide(config.hidden_size, config.num_attention_heads)
        else:
            self.head_dim = config.qk_pos_emb_head_dim
        self.rotary_percent = rotary_percent
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.model_comm_pgs = model_comm_pgs
        self.tp = model_comm_pgs.tp
        self.tp_group_size = self.tp.size
        self.tp_rank = self.tp.rank
        self.is_prefill = True

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base

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
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            model_comm_pgs=model_comm_pgs,
        )

        # Output
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

        self.set_modules({"model": self})

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

    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None,
                  q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """ Construct function of GPTModel. """

        # Generate cos and sin for RoPE.
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = \
                self.rotary_pos_emb.get_cos_sin_for_decode(positions)

        # Decoder embedding.
        decoder_input = self.cast(self.embedding(input_ids), self.compute_dtype)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
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
            value_cache=value_cache
        )

        # Return hidden states.
        if not self.post_process:
            return hidden_states

        output = self.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)
        # Return logits.
        logits = self.output_layer(output)
        logits = self.cast(logits.squeeze(0), mstype.float32)
        return logits

    def get_params_dict(self):
        params_dict = dict()
        for _, module in self.modules_dict.items():
            module_params = module.parameters_dict()
            for param_name, param in module_params.items():
                params_dict[param_name] = param

        return params_dict

    def map_global_expert_id_to_local_expert_id(self, global_expert_id: int) -> int:
        r"""
        Map global expert_id to local ID on current ep_rank.
        Each rank except the last rank get base number of experts, remaining experts go to the last rank.

        Args:
            global_expert_id (int): Global expert ID

        Returns:
            local_expert_id (int): Local expert ID, returns -1 if not belonging to current rank
        """
        if not getattr(self.model_comm_pgs, 'moe_ep', None):
            raise RuntimeError("ep communication domain not found.")
        ep_group_size = self.model_comm_pgs.moe_ep.size
        ep_rank = self.model_comm_pgs.moe_ep.rank
        num_experts = self.config.num_moe_experts
        num_local_experts = num_experts // ep_group_size

        # Check if current ep rank is responsible for this global expert ID
        if ep_rank < ep_group_size - 1:
            start_idx = ep_rank * num_local_experts
            end_idx = start_idx + num_local_experts

            if start_idx <= global_expert_id < end_idx:
                return global_expert_id - start_idx
        else:
            # Last ep rank is responsible for all remaining experts
            start_idx = (ep_group_size - 1) * num_local_experts

            if global_expert_id >= start_idx:
                return global_expert_id - start_idx

        # Not belong to current ep rank
        return -1

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]], stacked_params_mapping=None):
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

        expert_params_mapping = []
        if self.config.num_moe_experts:
            expert_params_mapping = make_expert_params_mapping(
                ckpt_gate_proj_name="gating",
                ckpt_down_proj_name="linear_fc2",
                ckpt_up_proj_name="hidden",
                num_experts=self.config.num_moe_experts)

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
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    expert_id = self.map_global_expert_id_to_local_expert_id(expert_id)
                    if expert_id == -1:
                        continue
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    name = name[:name.rfind('.')]
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                        loaded_params.add(name)
                        break
                else:
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)

        network_not_load = set(params_dict.keys()) - loaded_params
        logger.warning(f'These parameters are not loaded in the network: {network_not_load}')
        return loaded_params
