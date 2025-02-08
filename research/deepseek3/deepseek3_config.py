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
"""DeepSeekV3 Config API."""
from typing import Union

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, \
    TransformerOpParallelConfig, default_moe_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from deepseek2_config import DeepseekV2Config


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class DeepseekV3Config(DeepseekV2Config):
    """DeepSeekV3 config class which defines the model size. It redefines the attention block and experts in moe blocks.
    NOTE: If any nuisances, refer to in-between llama_config and the transformer original deepseek configuration.

    Args:
        batch_size (Optional[int]): batch size for input data, use in predict.
        seq_length (Optional[int]): The sequence length of input_ids, default is 1024.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the BERT model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        multiple_of (Optional[int]): Define SwiGLU hidden layer size multiples, default 256.
        n_kv_heads (Optional[int]): Define multi group head attention heads number, default None.
        ffn_dim_multiplier (Optional[int]): Define ffn layer dim multiples, default None.
        rms_norm_eps (Optional[float]): The epsilon value of the denominator. Default 1e-5.
        bos_token_id (Optional[int]): The id of the *beginning-of-sequence* token.
        eos_token_id (Optional[int]): The id of the *end-of-sequence* token.
        pad_token_id (Optional[int]): The id of the *padding* token.
        ignore_token_id (Optional[int]): The id of the *ignoring* token.
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        rotary_dtype (Optional[str]):
            rope compute dtype, default is "float32".
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float16".
        qkv_has_bias (Optional[bool]):
            Whether the Query, Key, and Value projection has bias.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        pretrain_seqlen(int): The pretrained model seq length, default 2048.
        extend_method(str): The extend method of seq length of inferencem,default None.
        compute_in_2d(bool): Whether compute in 2-dims tensor, default False.
        use_flash_attention(bool): Whether enable flash attention ops, default False.
        use_paged_attention(bool): Whether enable paged attention ops, default False.
        offset(int): Offset of transformer layer when set pipeline stage number.
        use_past_shard(bool): The configuration of kvcache parallel shard, default False.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        block_size (`int`, *optional*, defaults to 16):
            The maximum number of tokens in one block can have when using paged attention.
        num_blocks (`int`, *optional*, defaults to 512):
            The maximum number of blocks when using paged attention.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        mtp_depth (`int`): The depth for multi-token prediction.
        mtp_loss_factor (`float`): The loss factor for multi-token prediction.
        return_extra_loss (`bool`): Whether to use extra loss for moe modules.

        Returns:
            Class, DeepseekV3Config.
    """
    model_type = "deepseekv3"

    def __init__(self,
                 batch_size=1,
                 seq_length=2048,
                 hidden_size=4096,
                 num_layers=30,
                 num_heads=32,
                 n_kv_heads=32,
                 max_position_embeddings=4096,
                 intermediate_size=12288,
                 kv_lora_rank=512,
                 q_lora_rank=1536,
                 qk_rope_head_dim=64,
                 v_head_dim=128,
                 qk_nope_head_dim=128,
                 vocab_size=102400,
                 multiple_of=256,
                 ffn_dim_multiplier=None,
                 rms_norm_eps=0.000001,
                 bos_token_id=1,
                 eos_token_id=2,
                 pad_token_id=0,
                 ignore_token_id=-100,
                 compute_dtype="float16",
                 layernorm_compute_type="float32",
                 softmax_compute_type="float32",
                 theta=10000,
                 rotary_dtype="float32",
                 param_init_type="float16",
                 embedding_init_type=None,
                 qkv_has_bias=False,
                 qkv_concat=False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_past=False,
                 pretrain_seqlen=None,
                 compute_in_2d=None,
                 use_past_shard=None,
                 extend_method="YARN",
                 scaling_factor=None,
                 is_dynamic=False,
                 use_kvcache_op=False,
                 is_flexible_shape=False,
                 use_rope_slice=False,
                 use_flash_attention=False,
                 use_paged_attention=False,
                 use_prompt_flash_attention=False,
                 use_incre_flash_attention=False,
                 fine_grain_interleave=1,
                 pp_interleave_num=1,
                 offset=0,
                 checkpoint_name_or_path="",
                 repetition_penalty=1,
                 max_decode_length=1024,
                 block_size=16,
                 num_blocks=512,
                 top_k=5,
                 top_p=1,
                 do_sample=True,
                 mtp_depth=0,
                 mtp_loss_factor=0.3,
                 return_extra_loss=False,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         seq_length=seq_length,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         n_kv_heads=n_kv_heads,
                         max_position_embeddings=max_position_embeddings,
                         intermediate_size=intermediate_size,
                         kv_lora_rank=kv_lora_rank,
                         q_lora_rank=q_lora_rank,
                         qk_rope_head_dim=qk_rope_head_dim,
                         v_head_dim=v_head_dim,
                         qk_nope_head_dim=qk_nope_head_dim,
                         vocab_size=vocab_size,
                         multiple_of=multiple_of,
                         ffn_dim_multiplier=ffn_dim_multiplier,
                         rms_norm_eps=rms_norm_eps,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id,
                         ignore_token_id=ignore_token_id,
                         compute_dtype=compute_dtype,
                         layernorm_compute_type=layernorm_compute_type,
                         softmax_compute_type=softmax_compute_type,
                         theta=theta,
                         rotary_dtype=rotary_dtype,
                         param_init_type=param_init_type,
                         embedding_init_type=embedding_init_type,
                         qkv_has_bias=qkv_has_bias,
                         qkv_concat=qkv_concat,
                         parallel_config=parallel_config,
                         moe_config=moe_config,
                         use_past=use_past,
                         pretrain_seqlen=pretrain_seqlen,
                         compute_in_2d=compute_in_2d,
                         use_past_shard=use_past_shard,
                         extend_method=extend_method,
                         scaling_factor=scaling_factor,
                         is_dynamic=is_dynamic,
                         use_kvcache_op=use_kvcache_op,
                         is_flexible_shape=is_flexible_shape,
                         use_rope_slice=use_rope_slice,
                         use_flash_attention=use_flash_attention,
                         use_paged_attention=use_paged_attention,
                         use_prompt_flash_attention=use_prompt_flash_attention,
                         use_incre_flash_attention=use_incre_flash_attention,
                         fine_grain_interleave=fine_grain_interleave,
                         pp_interleave_num=pp_interleave_num,
                         offset=offset,
                         checkpoint_name_or_path=checkpoint_name_or_path,
                         repetition_penalty=repetition_penalty,
                         max_decode_length=max_decode_length,
                         block_size=block_size,
                         num_blocks=num_blocks,
                         top_k=top_k,
                         top_p=top_p,
                         do_sample=do_sample,
                         return_extra_loss=return_extra_loss,
                         **kwargs)
        self.mtp_depth = mtp_depth
        self.mtp_loss_factor = mtp_loss_factor
