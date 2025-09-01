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
"""Deepseek-MTP models' APIs."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

from mindspore import Tensor, nn, ops
from mindspore.communication._comm_helper import _is_initialized as mindspore_comm_has_init

from mindformers.models.utils import jit
from mindformers.models.build_config import get_quant_config
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel

from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding
from mindformers.parallel_core.inference.base_models.common.embeddings.rope_utils import get_rope
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin
from mindformers.parallel_core.inference.parallel_state import is_initialized, initialize_model_parallel
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.quantization import QuantizationConfig
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls
from mindformers.parallel_core.inference.utils import update_comm_config

from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.transformer_config import MLATransformerConfig, TransformerConfig
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from .configuration_deepseek_v3 import DeepseekV3Config

@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
    """
    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


class MultiTokenPredictionLayer(nn.Cell):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/abs/2412.19437
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MultiTokenPredictionLayerSubmodules,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.submodules = submodules
        self.dtype = config.compute_dtype

        self.enorm = build_module(
            self.submodules.enorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.eh_proj = build_module(
            self.submodules.eh_proj,
            config.hidden_size * 2,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
        )

        self.transformer = build_module(self.submodules.transformer_layer, config=config,
                                        model_comm_pgs=model_comm_pgs, quant_config=quant_config)

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.concat = ops.Concat(axis=-1)
        self.concat_mp = ops.Concat(axis=-1)
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()


    def construct(self, decoder_input, hidden_states, attention_mask=None, rotary_pos_cos=None,
                  rotary_pos_sin=None, batch_valid_length=None, context_lens_tensor=None, q_seq_lens=None,
                  block_tables=None, slot_mapping=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """ Perform the forward pass through the MTP layer. """

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)

        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = self.concat((decoder_input, hidden_states))
        hidden_states = self.eh_proj(hidden_states)
        hidden_states = self.transformer(
            hidden_states=hidden_states,
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
        output = self.final_layernorm(hidden_states)
        return output


class DeepseekV3MTPModel(GPTModel, nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        mtp_layer_spec (ModuleSpec): Specifies module to use for mtp layers.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.

    Inputs:
        - **input_ids** (Tensor) - Input token ids
        - **hidden_states** (Tensor) - Previous hidden states
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
        - **output** (Tensor) - return hidden states after decoder

    Supported Platforms:
        ``Ascend``
    """

    # pylint: disable=W0231
    def __init__(
            self,
            config: TransformerConfig,
            mtp_layer_spec: ModuleSpec,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        nn.Cell.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.mtp_layer_spec: ModuleSpec = mtp_layer_spec
        self.model_comm_pgs = model_comm_pgs

        self.compute_dtype = self.config.compute_dtype

        if not hasattr(config, "qk_pos_emb_head_dim"):
            self.hidden_dim = getattr(config, "kv_channels", divide(config.hidden_size, config.num_attention_heads))
        else:
            self.hidden_dim = config.qk_pos_emb_head_dim

        self.is_prefill = True

        self.embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            model_comm_pgs=model_comm_pgs,
        )

        self.rotary_pos_emb = get_rope(
            config,
            hidden_dim=self.hidden_dim,
            rotary_percent=1.0,
            rotary_base=config.rotary_base,
            rotary_dtype=config.rotary_dtype,
            position_embedding_type=config.position_embedding_type,
            original_max_position_embeddings=config.max_position_embeddings,
        )

        self.mtp_layer = build_module(self.mtp_layer_spec, config=config,
                                      model_comm_pgs=model_comm_pgs, quant_config=quant_config)

        self.output_layer = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=self.config,
            bias=False,
            gather_output=True,
            compute_dtype=config.compute_dtype,
            tp_group=model_comm_pgs.tp,
        )

        self.cast = ops.Cast()

        self.set_modules({"model": self})

    def set_modules(self, model_dicts: Dict[str, nn.Cell]):
        self.modules_dict = model_dicts

    def construct(self, input_ids, hidden_states=None, positions=None, batch_valid_length=None,
                  context_lens_tensor=None, q_seq_lens=None, block_tables=None, slot_mapping=None,
                  attention_mask=None, attn_metadata=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """ Construct function of GPTModel. """

        # Generate cos and sin for RoPE.
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cos_sin_for_decode(positions)

        decoder_input = self.cast(self.embedding(input_ids), self.compute_dtype)

        key_cache = key_cache[0] if key_cache is not None else None
        value_cache = value_cache[0] if value_cache is not None else None

        hidden_states = self.mtp_layer(
            decoder_input=decoder_input,
            hidden_states=hidden_states,
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

        return hidden_states


class InferenceDeepseekMTPForCausalLM(DeepseekV3PreTrainedModel, InferModelMixin):
    r"""
    Provide Deepseek3 model infer through network.

    Args:
        config (Deepseek3Config): The config of deepseek3 model.

    Returns:
        output: Tensor, the output of qwen3 deepseek3 layer

    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, auto_prefix=False)
        delattr(config, "quantization")    # weight is not quantized

        config.first_k_dense_replace = 0
        config.num_hidden_layers = config.num_nextn_predict_layers

        self.config = config
        config: MLATransformerConfig = convert_to_transformer_config(
            self.config,
            is_mla_model=True,
        )
        self.transformer_config = config
        if not is_initialized() and mindspore_comm_has_init():
            initialize_model_parallel(
                data_parallel_size=config.data_parallel_size,
                tensor_model_parallel_size=config.tensor_model_parallel_size,
                expert_model_parallel_size=config.expert_model_parallel_size,
                order='tp-ep-dp',
            )
        if is_initialized():
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(
                required_groups=['tp', 'moe_ep', 'moe_tp', 'dp', 'tpdp'])
        else:
            self.model_comm_pgs = default_model_comm_pgs

        # update communication-related configuration in TransformerConfig
        config = update_comm_config(config)
        self.quant_config = get_quant_config(self.config.to_dict(), self.weight_mapping)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype

        self.is_prefill = True
        self.need_hidden_states = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None

        self.use_fused_mla = False
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=True,
            qk_layernorm=config.qk_layernorm,
            gated_linear_unit=config.gated_linear_unit,
            multi_latent_attention=config.multi_latent_attention,
            normalization=config.normalization,
            use_flash_attention=config.use_flash_attention,
            qk_l2_norm=False,
            use_alltoall=config.use_alltoall,
            use_fused_mla=config.use_fused_mla,
        )
        mtp_layer_spec = ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=get_norm_cls(config.normalization),
                hnorm=get_norm_cls(config.normalization),
                eh_proj=ColumnParallelLinear,
                transformer_layer=transformer_layer_spec,
                layer_norm=get_norm_cls(config.normalization),
            ),
        )
        self.model = DeepseekV3MTPModel(
            config=config,
            mtp_layer_spec=mtp_layer_spec,
            model_comm_pgs=self.model_comm_pgs,
            quant_config=self.quant_config
        )

        self.weight_mapping = [
            ('model.layers.61.enorm.', 'mtp_layer.enorm.'),
            ('model.layers.61.hnorm.', 'mtp_layer.hnorm.'),
            ('model.layers.61.eh_proj.', 'mtp_layer.eh_proj.'),
            ('model.layers.61.embed_tokens.', 'embedding.word_embeddings.'),
            ('model.layers.61.shared_head.norm.', 'mtp_layer.final_layernorm.'),
            ('model.layers.61.shared_head.head.', 'output_layer.'),
            ('model.layers.61.', 'mtp_layer.transformer.')
        ] + DeepseekV3PreTrainedModel.weight_mapping

    def add_flags_custom_mcore(self, is_prefill):
        r"""
        Add flag to distinguish fa and pa.

        Args:
            is_prefill: flag to distinguish fa and pa.

        Returns:

        """
        self.add_flags(is_prefill=is_prefill)
        self.model.add_flags(is_prefill=is_prefill)
        self.model.mtp_layer.add_flags(is_prefill=is_prefill)
        self.model.mtp_layer.transformer.self_attention.add_flags(is_prefill=is_prefill)
        self.model.mtp_layer.transformer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    def get_mutable_hidden_states(self):
        return Tensor(shape=[None, None], dtype=self.compute_dtype)

    @jit
    def construct(
            self,
            input_ids,
            hidden_states=None,
            positions=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            attention_mask=None,
            attn_metadata=None,
            attn_padding_idx=None,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
            key_cache=None,
            value_cache=None
    ):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            hidden_states: hidden states.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            q_seq_lens: query sequence lengths.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata.
            attn_padding_idx: Indices mapping positions in attention output sequence to original token positions,
                used for padding attention output to fixed size.
            attn_unpadding_idx: Indices mapping valid tokens in padded attention output sequence to
                their original positions, used for removing padding in attention output.
            ffn_padding_idx: Indices mapping positions in MoE output sequence to flattened valid token positions,
                used for padding MoE output to fixed size.
            ffn_unpadding_idx: Indices mapping valid tokens in padded MoE output sequence to their original positions,
                used for removing padding in MoE output.
            key_cache: key cache for incremental inference.
            value_cache: value cache for incremental inference.

        Returns:
            logits: the output logits.

        """
        logits = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        return logits

    def load_weights(self, weights_path=None, weights: Iterable[Tuple[str, Tensor]] = None):
        r"""
        Load weights.

        Args:
            weights_path: The path of weights.

        """
        import os
        if not os.path.isdir(weights_path):
            if not weights:
                raise ValueError(
                    f"Either 'weights_path' or 'weights' is required, "
                    f"but got weights_path={weights_path}, weights={weights}"
                )
            self.model.load_weights(weights)
        else:
            # check mtp files
            weights_files = [
                os.path.join(weights_path, file)
                for file in os.listdir(weights_path)
                if file.endswith(".safetensors") and 'quant' not in file
            ]

            if not weights_files:
                raise ValueError(f"No .safetensors files found in {weights_path}")

            self.model.load_weights(
                self._safetensors_weights_iterator(weights_files),
                self.generate_mapping()
            )
        self.process_weights_after_loading(self.model)
