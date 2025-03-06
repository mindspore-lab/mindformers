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

"""
transform huggingface model to mindspore ckpt.
"""
import argparse
import json
import os
from collections import defaultdict
import math
import multiprocessing
import numpy as np

import mindspore as ms
import torch
from safetensors.torch import load_file

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}

default_config = {
    'num_routed_experts': 256,
    'n_head': 128,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'num_layers': 61,
    'num_nextn_predict_layers': 1,
    'first_k_dense_replace': 3,
    'dtype': ms.bfloat16,
    'use_gemm': False,
    'save_format': "safetensors"
}

infer_config = {
    'num_head': 128,
    'qk_rope_head_dim': 64,
    'qk_nope_head_dim': 128,
    'kv_lora_rank': 512,
    'v_head_dim': 128,
    'rope_dim': 192,  # qk_rope_head_dim + qk_nope_head_dim
    'kv_head_dim': 576,  # kv_lora_rank + qk_rope_head_dim
    'total_layer_num': 62,
    'num_routed_experts': 256
}


def infer_name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.q_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
    weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
    weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
    weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
    weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
    weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
    weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
    weight_name = weight_name.replace('mlp.experts.', 'feed_forward.routed_experts.ffn.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
    weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
    weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                      'feed_forward.routed_experts.router.e_score_correction_bias')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    weight_name = weight_name.replace('model.tok_embeddings.weight', 'model.tok_embeddings.embedding_weight')
    weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
    return weight_name


def infer_trans_rope_weight(weight):
    """process rope routed weight"""
    w1 = weight[..., -infer_config['qk_rope_head_dim']::2, :]
    w2 = weight[..., -infer_config['qk_rope_head_dim'] + 1::2, :]
    weight[..., -infer_config['qk_rope_head_dim']:, :] = np.concatenate([w1, w2], axis=-2)
    return weight


def infer_process_moe_routed_expert_ffn_weight(params_dict, dst_ms_dir, layer, ms_meta):
    """process moe routed expert weight"""
    w1 = []
    w2 = []
    w3 = []

    w1_keys = []
    w2_keys = []
    w3_keys = []
    ffn_dtype = ms.bfloat16
    for index in range(0, infer_config['num_routed_experts']):
        w1_key = f"model.layers.{layer}.mlp.experts.{index}.gate_proj.weight"
        w2_key = f"model.layers.{layer}.mlp.experts.{index}.down_proj.weight"
        w3_key = f"model.layers.{layer}.mlp.experts.{index}.up_proj.weight"
        ffn_dtype = params_dict[w1_key].dtype
        w1.append(params_dict[w1_key].astype(ms.float32).asnumpy())
        w2.append(params_dict[w2_key].astype(ms.float32).asnumpy())
        w3.append(params_dict[w3_key].astype(ms.float32).asnumpy())

        w1_keys.append(w1_key)
        w2_keys.append(w2_key)
        w3_keys.append(w3_key)

    params_w2 = {}
    ffn_w2_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w2.weight"
    params_w2[ffn_w2_key] = ms.Parameter(ms.Tensor(np.stack(w2, axis=0).transpose(0, 2, 1), ffn_dtype), name=ffn_w2_key)
    dst_w2_name = f"model_layer_{layer}_routed_experts_w2.safetensors"
    ms_meta[ffn_w2_key] = dst_w2_name
    w2_dst_path = f"{dst_ms_dir}/{dst_w2_name}"
    ms.save_checkpoint(params_w2, w2_dst_path, format="safetensors")

    params_w1 = {}
    params_w3 = {}
    ffn_w1_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w1.weight"
    ffn_w3_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w3.weight"
    params_w1[ffn_w1_key] = ms.Parameter(ms.Tensor(np.stack(w1, axis=0).transpose(0, 2, 1), ffn_dtype),
                                         name=ffn_w1_key)
    params_w3[ffn_w3_key] = ms.Parameter(ms.Tensor(np.stack(w3, axis=0).transpose(0, 2, 1), ffn_dtype),
                                         name=ffn_w3_key)
    dst_w1_name = f"model_layer_{layer}_routed_experts_w1.safetensors"
    dst_w3_name = f"model_layer_{layer}_routed_experts_w3.safetensors"
    ms_meta[ffn_w1_key] = dst_w1_name
    ms_meta[ffn_w3_key] = dst_w3_name

    w1_dst_path = f"{dst_ms_dir}/{dst_w1_name}"
    w3_dst_path = f"{dst_ms_dir}/{dst_w3_name}"
    ms.save_checkpoint(params_w1, w1_dst_path, format="safetensors")
    ms.save_checkpoint(params_w3, w3_dst_path, format="safetensors")

    for index in range(0, infer_config['num_routed_experts']):
        params_dict.pop(w1_keys[index])
        params_dict.pop(w2_keys[index])
        params_dict.pop(w3_keys[index])


def infer_process_moe_shared_expert_ffn_weight(trans_params):
    """process moe shared expert weight"""
    params_dict, ms_param, layer, ms_meta, dst_name = trans_params
    w1_key = f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
    w2_key = f"model.layers.{layer}.mlp.shared_experts.down_proj.weight"
    w3_key = f"model.layers.{layer}.mlp.shared_experts.up_proj.weight"

    ffn_w2_key = f"model.layers.{layer}.feed_forward.shared_experts.w2.weight"
    ms_param[ffn_w2_key] = params_dict[w2_key]
    ms_meta[ffn_w2_key] = dst_name

    ffn_w1_key = f"model.layers.{layer}.feed_forward.shared_experts.w1.weight"
    ffn_w3_key = f"model.layers.{layer}.feed_forward.shared_experts.w3.weight"

    ms_param[ffn_w1_key] = params_dict[w1_key]
    ms_param[ffn_w3_key] = params_dict[w3_key]

    ms_meta[ffn_w1_key] = dst_name
    ms_meta[ffn_w3_key] = dst_name

    params_dict.pop(w1_key)
    params_dict.pop(w2_key)
    params_dict.pop(w3_key)


def infer_process_dense_ffn_weight(trans_params):
    """process dense ffn weight"""
    params_dict, ms_param, layer, ms_meta, dst_name = trans_params
    w1_key = f"model.layers.{layer}.mlp.gate_proj.weight"
    w2_key = f"model.layers.{layer}.mlp.down_proj.weight"
    w3_key = f"model.layers.{layer}.mlp.up_proj.weight"

    w1 = params_dict[w1_key]
    w2 = params_dict[w2_key]
    w3 = params_dict[w3_key]

    w2_key_new = f"model.layers.{layer}.feed_forward.w2.weight"
    ms_param[w2_key_new] = ms.Parameter(ms.Tensor(w2, w2.dtype), name=w2_key_new)
    ms_meta[w2_key_new] = dst_name

    w1_key_new = f"model.layers.{layer}.feed_forward.w1.weight"
    w3_key_new = f"model.layers.{layer}.feed_forward.w3.weight"
    ms_param[w1_key_new] = ms.Parameter(ms.Tensor(w1, w1.dtype), name=w1_key_new)
    ms_param[w3_key_new] = ms.Parameter(ms.Tensor(w3, w3.dtype), name=w3_key_new)
    ms_meta[w3_key_new] = dst_name
    ms_meta[w3_key_new] = dst_name

    params_dict.pop(w1_key)
    params_dict.pop(w2_key)
    params_dict.pop(w3_key)


def infer_convert_layer_weight(src_hf_dir, dst_ms_dir, layer, queue):
    """convert single layer weight"""
    print(f"..... start convert layer {layer} .......", flush=True)
    ms_meta = {}
    with open(os.path.join(src_hf_dir, "model.safetensors.index.json"), "r") as fp:
        hf_meta = json.load(fp).get('weight_map')

    safetensor_files = set()
    for param_key, param_path in hf_meta.items():
        if f"model.layers.{layer}." in param_key:
            safetensor_files.add(param_path)

    params_dict = {}
    for ckpt in safetensor_files:
        src_path = f"{src_hf_dir}/{ckpt}"
        p = ms.load_checkpoint(src_path, format="safetensors")
        params_dict.update(p)

    ms_param = {}
    dst_name = f"model_layer_{layer}.safetensors"

    if layer >= 3:
        infer_process_moe_routed_expert_ffn_weight(params_dict, dst_ms_dir, layer, ms_meta)
        infer_process_moe_shared_expert_ffn_weight((params_dict, ms_param, layer, ms_meta, dst_name))
    else:
        infer_process_dense_ffn_weight((params_dict, ms_param, layer, ms_meta, dst_name))

    num_head = infer_config['num_head']
    rope_dim = infer_config['rope_dim']
    kv_head_dim = infer_config['kv_head_dim']
    qk_nope_head_dim = infer_config['qk_nope_head_dim']
    v_head_dim = infer_config['v_head_dim']

    for key, value in params_dict.items():  # pylint: disable=redefined-outer-name
        if not key.startswith(f"model.layers.{layer}."):
            continue
        value = params_dict[key]
        dtype = params_dict[key].dtype
        ms_key = infer_name_replace(key)

        # l2q_proj
        if "attention.l2q_proj.weight" in ms_key:
            value = value.astype(np.float32).asnumpy()
            value = value.reshape(num_head, rope_dim, -1)
            weight = infer_trans_rope_weight(value)
            weight = weight.reshape(num_head * rope_dim, -1)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        # kv2l
        elif "attention.kv2l.weight" in ms_key:
            value = value.astype(np.float32).asnumpy()
            value = value.reshape(kv_head_dim, -1)
            weight = infer_trans_rope_weight(value)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        # .attention.lkv2kv.
        elif ".attention.lkv2kv." in ms_key:
            value = value.astype(np.float32).asnumpy()
            lkv2kv_head = qk_nope_head_dim + v_head_dim
            value = value.reshape(num_head, lkv2kv_head, -1)
            value_k_nope, value_v = value[:, :qk_nope_head_dim, :], value[:, qk_nope_head_dim:, :]
            value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
            value_v = value_v.reshape(-1, value_v.shape[-1])
            name_k_nope = ms_key.replace(".attention.lkv2kv.", ".attention.lkv2kv_k_nope.")
            name_v = ms_key.replace(".attention.lkv2kv.", ".attention.lkv2kv_v.")
            ms_param[name_k_nope] = ms.Parameter(ms.Tensor(value_k_nope, dtype), name=name_k_nope)
            ms_param[name_v] = ms.Parameter(ms.Tensor(value_v, dtype), name=name_v)
            ms_meta[name_k_nope] = dst_name
            ms_meta[name_v] = dst_name
            continue
        else:
            ms_param[ms_key] = ms.Parameter(ms.Tensor(value, dtype), name=ms_key)
        ms_meta[ms_key] = dst_name
    dst_path = os.path.join(dst_ms_dir, dst_name)
    ms.save_checkpoint(ms_param, dst_path, format="safetensors")
    queue.put(ms_meta)
    print(f"..... end convert layer {layer} .......", flush=True)


def infer_convert_outer_weight(src_hf_dir, dst_ms_dir, ms_meta, param_json):
    """convert weight not in model"""
    with open(f"{src_hf_dir}/{param_json}", "r") as fp:
        hf_meta = json.load(fp)['weight_map']

    safetensor_files = set()
    for param_key, param_path in hf_meta.items():
        if "model.layers." not in param_key:
            safetensor_files.add(param_path)

    params_dict = {}
    for ckpt in safetensor_files:
        src_path = f"{src_hf_dir}/{ckpt}"
        p = ms.load_checkpoint(src_path, format="safetensors")
        params_dict.update(p)

    ms_param = {}
    dst_name = "model.safetensors"
    for key, value in params_dict.items():  # pylint: disable=redefined-outer-name
        if "model.layers." in key:
            continue
        value = params_dict[key]
        ms_key = infer_name_replace(key)
        ms_param[ms_key] = ms.Parameter(ms.Tensor(value), name=ms_key)
        ms_meta[ms_key] = dst_name
    dst_path = f"{dst_ms_dir}/{dst_name}"
    ms.save_checkpoint(ms_param, dst_path, format="safetensors")


def infer_quant_name_replace(weight_name: str):
    """replace quant net weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')

    # attn is pertensor quant
    weight_name = weight_name.replace('.deq_scale', '._layer.matmul.dequant_scale')
    weight_name = weight_name.replace('.quant_bias', '._layer.matmul.quant_bias')
    weight_name = weight_name.replace('.input_scale', '.quant_op.input_scale')
    weight_name = weight_name.replace('.input_offset', '.quant_op.input_zp')

    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.q_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
    weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
    weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
    weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
    weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')

    # mlp is pertoken quant
    weight_name = weight_name.replace('.weight_scale', '.matmul.weight_scale')
    weight_name = weight_name.replace('.weight_offset', '.matmul.weight_offset')

    weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1._layer.')
    weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2._layer.')
    weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3._layer.')
    weight_name = weight_name.replace('mlp.experts.', 'feed_forward.routed_experts.ffn.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1._layer.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2._layer.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3._layer.')
    weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
    weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                      'feed_forward.routed_experts.router.e_score_correction_bias')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    weight_name = weight_name.replace('model.tok_embeddings.weight', 'model.tok_embeddings.embedding_weight')
    weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
    return weight_name


def routed_expert_ffn_no_concat(quant_trans_params):
    '''routed_expert_ffn_no_concat'''
    ms_meta, layer, w1, w1_offset, w1_scale, w3, w3_offset, w3_scale, dst_ms_dir, \
    ffn_dtype, ffn_offset_dtype, ffn_scale_dtype = quant_trans_params
    params_w1 = {}
    params_w3 = {}
    ffn_w1_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w1._layer.weight"
    ffn_w1_offset_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_offset"
    ffn_w1_scale_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_scale"

    ffn_w3_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w3._layer.weight"
    ffn_w3_offset_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_offset"
    ffn_w3_scale_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_scale"

    params_w1[ffn_w1_key] = ms.Parameter(ms.Tensor(np.stack(w1, axis=0).transpose(0, 2, 1), ffn_dtype),
                                         name=ffn_w1_key)
    params_w1[ffn_w1_offset_key] = ms.Parameter(ms.Tensor(np.stack(w1_offset, axis=0), ffn_offset_dtype),
                                                name=ffn_w1_offset_key)
    params_w1[ffn_w1_scale_key] = ms.Parameter(ms.Tensor(np.stack(w1_scale, axis=0), ffn_scale_dtype),
                                               name=ffn_w1_scale_key)

    params_w3[ffn_w3_key] = ms.Parameter(ms.Tensor(np.stack(w3, axis=0).transpose(0, 2, 1), ffn_dtype),
                                         name=ffn_w3_key)
    params_w3[ffn_w3_offset_key] = ms.Parameter(ms.Tensor(np.stack(w3_offset, axis=0), ffn_offset_dtype),
                                                name=ffn_w3_offset_key)
    params_w3[ffn_w3_scale_key] = ms.Parameter(ms.Tensor(np.stack(w3_scale, axis=0), ffn_scale_dtype),
                                               name=ffn_w3_scale_key)

    dst_w1_name = f"model_layer_{layer}_routed_experts_w1.safetensors"
    dst_w3_name = f"model_layer_{layer}_routed_experts_w3.safetensors"
    ms_meta[ffn_w1_key] = dst_w1_name
    ms_meta[ffn_w3_key] = dst_w3_name
    ms_meta[ffn_w1_scale_key] = dst_w1_name
    ms_meta[ffn_w3_scale_key] = dst_w3_name
    ms_meta[ffn_w1_offset_key] = dst_w1_name
    ms_meta[ffn_w3_offset_key] = dst_w3_name

    w1_dst_path = f"{dst_ms_dir}/{dst_w1_name}"
    w3_dst_path = f"{dst_ms_dir}/{dst_w3_name}"
    ms.save_checkpoint(params_w1, w1_dst_path, format="safetensors")
    ms.save_checkpoint(params_w3, w3_dst_path, format="safetensors")


def infer_quant_param_moe_pop(quant_trans_parms):
    """infer_quant_param_moe_pop"""
    params_dict, w1_keys, w2_keys, w3_keys, w1_offset_keys, w2_offset_keys, w3_offset_keys, \
    w1_scale_keys, w2_scale_keys, w3_scale_keys = quant_trans_parms
    for index in range(0, infer_config['num_routed_experts']):
        params_dict.pop(w1_keys[index])
        params_dict.pop(w2_keys[index])
        params_dict.pop(w3_keys[index])
        params_dict.pop(w1_offset_keys[index])
        params_dict.pop(w2_offset_keys[index])
        params_dict.pop(w3_offset_keys[index])
        params_dict.pop(w1_scale_keys[index])
        params_dict.pop(w2_scale_keys[index])
        params_dict.pop(w3_scale_keys[index])


def infer_quant_process_moe_routed_expert_ffn_weight(params_dict, dst_ms_dir, layer, ms_meta):
    """infer_quant_process_moe_routed_expert_ffn_weight"""
    w1, w1_offset, w1_scale, w2, w2_offset, w2_scale, w3, w3_offset, w3_scale = [], [], [], [], [], [], [], [], []
    w1_keys, w1_offset_keys, w1_scale_keys, w2_keys, w2_offset_keys, w2_scale_keys = [], [], [], [], [], []
    w3_keys, w3_offset_keys, w3_scale_keys = [], [], []

    ffn_dtype = ms.int8
    for index in range(0, infer_config['num_routed_experts']):
        w1_key = f"model.layers.{layer}.mlp.experts.{index}.gate_proj.weight"
        w1_offset_key = w1_key + "_offset"
        w1_scale_key = w1_key + "_scale"
        w2_key = f"model.layers.{layer}.mlp.experts.{index}.down_proj.weight"
        w2_offset_key = w2_key + "_offset"
        w2_scale_key = w2_key + "_scale"
        w3_key = f"model.layers.{layer}.mlp.experts.{index}.up_proj.weight"
        w3_offset_key = w3_key + "_offset"
        w3_scale_key = w3_key + "_scale"

        ffn_dtype = params_dict[w1_key].dtype
        ffn_offset_dtype = params_dict[w1_offset_key].dtype
        ffn_scale_dtype = params_dict[w1_scale_key].dtype

        w1.append(params_dict[w1_key].astype(ms.int8).asnumpy())
        w1_offset.append(np.squeeze(params_dict[w1_offset_key].astype(ms.bfloat16).asnumpy(), axis=-1))
        w1_scale.append(np.squeeze(params_dict[w1_scale_key].astype(ms.bfloat16).asnumpy(), axis=-1))

        w2.append(params_dict[w2_key].astype(ms.int8).asnumpy())
        w2_offset.append(np.squeeze(params_dict[w2_offset_key].astype(ms.bfloat16).asnumpy(), axis=-1))
        w2_scale.append(np.squeeze(params_dict[w2_scale_key].astype(ms.bfloat16).asnumpy(), axis=-1))

        w3.append(params_dict[w3_key].astype(ms.int8).asnumpy())
        w3_offset.append(np.squeeze(params_dict[w3_offset_key].astype(ms.bfloat16).asnumpy(), axis=-1))
        w3_scale.append(np.squeeze(params_dict[w3_scale_key].astype(ms.bfloat16).asnumpy(), axis=-1))

        w1_keys.append(w1_key)
        w2_keys.append(w2_key)
        w3_keys.append(w3_key)

        w1_offset_keys.append(w1_offset_key)
        w2_offset_keys.append(w2_offset_key)
        w3_offset_keys.append(w3_offset_key)

        w1_scale_keys.append(w1_scale_key)
        w2_scale_keys.append(w2_scale_key)
        w3_scale_keys.append(w3_scale_key)

    params_w2 = {}
    ffn_w2_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w2._layer.weight"
    ffn_w2_offset_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w2._layer.matmul.weight_offset"
    ffn_w2_scale_key = f"model.layers.{layer}.feed_forward.routed_experts.ffn.w2._layer.matmul.weight_scale"
    params_w2[ffn_w2_key] = ms.Parameter(ms.Tensor(np.stack(w2, axis=0).transpose(0, 2, 1), ffn_dtype),
                                         name=ffn_w2_key)
    params_w2[ffn_w2_offset_key] = ms.Parameter(ms.Tensor(np.stack(w2_offset, axis=0), ffn_offset_dtype),
                                                name=ffn_w2_offset_key)
    params_w2[ffn_w2_scale_key] = ms.Parameter(ms.Tensor(np.stack(w2_scale, axis=0), ffn_scale_dtype),
                                               name=ffn_w2_scale_key)

    dst_w2_name = f"model_layer_{layer}_routed_experts_w2.safetensors"
    ms_meta[ffn_w2_key] = dst_w2_name
    ms_meta[ffn_w2_offset_key] = dst_w2_name
    ms_meta[ffn_w2_scale_key] = dst_w2_name
    w2_dst_path = f"{dst_ms_dir}/{dst_w2_name}"
    ms.save_checkpoint(params_w2, w2_dst_path, format="safetensors")

    routed_expert_ffn_no_concat((ms_meta, layer, w1, w1_offset, w1_scale, w3, w3_offset, w3_scale, dst_ms_dir,
                                 ffn_dtype, ffn_offset_dtype, ffn_scale_dtype))

    infer_quant_param_moe_pop((params_dict, w1_keys, w2_keys, w3_keys, w1_offset_keys, w2_offset_keys, w3_offset_keys,
                               w1_scale_keys, w2_scale_keys, w3_scale_keys))


def infer_quant_process_moe_shared_expert_ffn_weight(quant_trans_params):
    """infer_quant_process_moe_shared_expert_ffn_weight"""
    params_dict, ms_param, layer, ms_meta, dst_name = quant_trans_params
    w1_key = f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
    w2_key = f"model.layers.{layer}.mlp.shared_experts.down_proj.weight"
    w3_key = f"model.layers.{layer}.mlp.shared_experts.up_proj.weight"
    w1_scale_key = f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight_scale"
    w2_scale_key = f"model.layers.{layer}.mlp.shared_experts.down_proj.weight_scale"
    w3_scale_key = f"model.layers.{layer}.mlp.shared_experts.up_proj.weight_scale"
    w1_offset_key = f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight_offset"
    w2_offset_key = f"model.layers.{layer}.mlp.shared_experts.down_proj.weight_offset"
    w3_offset_key = f"model.layers.{layer}.mlp.shared_experts.up_proj.weight_offset"

    ffn_w2_key = f"model.layers.{layer}.feed_forward.shared_experts.w2._layer.weight"
    ffn_w2_scale_key = f"model.layers.{layer}.feed_forward.shared_experts.w2._layer.matmul.weight_scale"
    ffn_w2_offset_key = f"model.layers.{layer}.feed_forward.shared_experts.w2._layer.matmul.weight_offset"

    ms_param[ffn_w2_key] = params_dict[w2_key]
    ms_meta[ffn_w2_key] = dst_name
    ms_param[ffn_w2_scale_key] = ms.Parameter(params_dict[w2_scale_key].squeeze(axis=-1), ffn_w2_scale_key)
    ms_meta[ffn_w2_scale_key] = dst_name
    ms_param[ffn_w2_offset_key] = ms.Parameter(params_dict[w2_offset_key].squeeze(axis=-1), ffn_w2_offset_key)
    ms_meta[ffn_w2_offset_key] = dst_name

    ffn_w1_key = f"model.layers.{layer}.feed_forward.shared_experts.w1._layer.weight"
    ffn_w1_scale_key = f"model.layers.{layer}.feed_forward.shared_experts.w1._layer.matmul.weight_scale"
    ffn_w1_offset_key = f"model.layers.{layer}.feed_forward.shared_experts.w1._layer.matmul.weight_offset"
    ffn_w3_key = f"model.layers.{layer}.feed_forward.shared_experts.w3._layer.weight"
    ffn_w3_scale_key = f"model.layers.{layer}.feed_forward.shared_experts.w3._layer.matmul.weight_scale"
    ffn_w3_offset_key = f"model.layers.{layer}.feed_forward.shared_experts.w3._layer.matmul.weight_offset"

    ms_param[ffn_w1_key] = params_dict[w1_key]
    ms_param[ffn_w3_key] = params_dict[w3_key]
    w1_scale = params_dict[w1_scale_key].squeeze(axis=-1)
    w3_scale = params_dict[w3_scale_key].squeeze(axis=-1)
    w1_offset = params_dict[w1_offset_key].squeeze(axis=-1)
    w3_offset = params_dict[w3_offset_key].squeeze(axis=-1)
    ms_param[ffn_w1_scale_key] = ms.Parameter(ms.Tensor(w1_scale, w1_scale.dtype), name=ffn_w1_scale_key)
    ms_param[ffn_w3_scale_key] = ms.Parameter(ms.Tensor(w3_scale, w3_scale.dtype), name=ffn_w3_scale_key)
    ms_param[ffn_w1_offset_key] = ms.Parameter(ms.Tensor(w1_offset, w1_offset.dtype), name=ffn_w1_offset_key)
    ms_param[ffn_w3_offset_key] = ms.Parameter(ms.Tensor(w3_offset, w3_offset.dtype), name=ffn_w3_offset_key)

    ms_meta[ffn_w1_key] = dst_name
    ms_meta[ffn_w3_key] = dst_name
    ms_meta[ffn_w1_scale_key] = dst_name
    ms_meta[ffn_w3_scale_key] = dst_name
    ms_meta[ffn_w1_offset_key] = dst_name
    ms_meta[ffn_w3_offset_key] = dst_name

    params_dict.pop(w1_key)
    params_dict.pop(w2_key)
    params_dict.pop(w3_key)
    params_dict.pop(w1_scale_key)
    params_dict.pop(w2_scale_key)
    params_dict.pop(w3_scale_key)
    params_dict.pop(w1_offset_key)
    params_dict.pop(w2_offset_key)
    params_dict.pop(w3_offset_key)


def infer_quant_process_dense_ffn_weight(quant_trans_params):
    """infer_quant_process_dense_ffn_weight"""
    params_dict, ms_param, layer, ms_meta, dst_name = quant_trans_params
    w1_key = f"model.layers.{layer}.mlp.gate_proj.weight"
    w2_key = f"model.layers.{layer}.mlp.down_proj.weight"
    w3_key = f"model.layers.{layer}.mlp.up_proj.weight"
    w1_scale_key = f"model.layers.{layer}.mlp.gate_proj.weight_scale"
    w2_scale_key = f"model.layers.{layer}.mlp.down_proj.weight_scale"
    w3_scale_key = f"model.layers.{layer}.mlp.up_proj.weight_scale"
    w1_offset_key = f"model.layers.{layer}.mlp.gate_proj.weight_offset"
    w2_offset_key = f"model.layers.{layer}.mlp.down_proj.weight_offset"
    w3_offset_key = f"model.layers.{layer}.mlp.up_proj.weight_offset"

    w1 = params_dict[w1_key]
    w2 = params_dict[w2_key]
    w3 = params_dict[w3_key]
    w1_scale = params_dict[w1_scale_key].squeeze(axis=-1)
    w2_scale = params_dict[w2_scale_key].squeeze(axis=-1)
    w3_scale = params_dict[w3_scale_key].squeeze(axis=-1)
    w1_offset = params_dict[w1_offset_key].squeeze(axis=-1)
    w2_offset = params_dict[w2_offset_key].squeeze(axis=-1)
    w3_offset = params_dict[w3_offset_key].squeeze(axis=-1)

    w2_key_new = f"model.layers.{layer}.feed_forward.w2._layer.weight"
    w2_key_new_scale = f"model.layers.{layer}.feed_forward.w2._layer.matmul.weight_scale"
    w2_key_new_offset = f"model.layers.{layer}.feed_forward.w2._layer.matmul.weight_offset"
    ms_param[w2_key_new] = ms.Parameter(ms.Tensor(w2, w2.dtype), name=w2_key_new)
    ms_meta[w2_key_new] = dst_name
    ms_param[w2_key_new_scale] = ms.Parameter(ms.Tensor(w2_scale, w2_scale.dtype), name=w2_key_new_scale)
    ms_meta[w2_key_new_scale] = dst_name
    ms_param[w2_key_new_offset] = ms.Parameter(ms.Tensor(w2_offset, w2_offset.dtype), name=w2_key_new_offset)
    ms_meta[w2_key_new_offset] = dst_name

    w1_key_new = f"model.layers.{layer}.feed_forward.w1._layer.weight"
    w3_key_new = f"model.layers.{layer}.feed_forward.w3._layer.weight"
    w1_scale_key_new = f"model.layers.{layer}.feed_forward.w1._layer.matmul.weight_scale"
    w3_scale_key_new = f"model.layers.{layer}.feed_forward.w3._layer.matmul.weight_scale"
    w1_offset_key_new = f"model.layers.{layer}.feed_forward.w1._layer.matmul.weight_offset"
    w3_offset_key_new = f"model.layers.{layer}.feed_forward.w3._layer.matmul.weight_offset"
    ms_param[w1_key_new] = ms.Parameter(ms.Tensor(w1, w1.dtype), name=w1_key_new)
    ms_param[w3_key_new] = ms.Parameter(ms.Tensor(w3, w3.dtype), name=w3_key_new)
    ms_param[w1_scale_key_new] = ms.Parameter(ms.Tensor(w1_scale, w1_scale.dtype), name=w1_scale_key_new)
    ms_param[w3_scale_key_new] = ms.Parameter(ms.Tensor(w3_scale, w3_scale.dtype), name=w3_scale_key_new)
    ms_param[w1_offset_key_new] = ms.Parameter(ms.Tensor(w1_offset, w1_offset.dtype), name=w1_offset_key_new)
    ms_param[w3_offset_key_new] = ms.Parameter(ms.Tensor(w3_offset, w3_offset.dtype), name=w3_offset_key_new)
    ms_meta[w1_key_new] = dst_name
    ms_meta[w3_key_new] = dst_name
    ms_meta[w1_scale_key_new] = dst_name
    ms_meta[w3_scale_key_new] = dst_name
    ms_meta[w1_offset_key_new] = dst_name
    ms_meta[w3_offset_key_new] = dst_name

    params_dict.pop(w1_key)
    params_dict.pop(w2_key)
    params_dict.pop(w3_key)
    params_dict.pop(w1_scale_key)
    params_dict.pop(w2_scale_key)
    params_dict.pop(w3_scale_key)
    params_dict.pop(w1_offset_key)
    params_dict.pop(w2_offset_key)
    params_dict.pop(w3_offset_key)


def infer_convert_qkv2l_concat_weight(param_dict, layer, ms_meta):
    """convert qkv2l concat weight"""
    wq2l_proj_weight_name = f"model.layers.{layer}.attention.q2l_proj._layer.weight"
    wq2l_proj_weight_scale_name = f"model.layers.{layer}.attention.q2l_proj._layer.matmul.dequant_scale"
    wq2l_proj_weight_bias_name = f"model.layers.{layer}.attention.q2l_proj._layer.matmul.quant_bias"
    wkv2l_weight_name = f"model.layers.{layer}.attention.kv2l._layer.weight"
    wkv2l_weight_scale_name = f"model.layers.{layer}.attention.kv2l._layer.matmul.dequant_scale"
    wkv2l_weight_bias_name = f"model.layers.{layer}.attention.kv2l._layer.matmul.quant_bias"
    wqkv2l_weight_name = f"model.layers.{layer}.attention.qkv2l._layer.weight"
    wqkv2l_weight_scale_name = f"model.layers.{layer}.attention.qkv2l._layer.matmul.dequant_scale"
    wqkv2l_weight_bias_name = f"model.layers.{layer}.attention.qkv2l._layer.matmul.quant_bias"
    # concat weight
    wq2l_proj_weight = param_dict[wq2l_proj_weight_name].asnumpy()
    wkv2l_weight = param_dict[wkv2l_weight_name].asnumpy()
    wqkv2l_weight = np.concatenate((wq2l_proj_weight, wkv2l_weight), 0)
    param_dict[wqkv2l_weight_name] = ms.Parameter(wqkv2l_weight, name=wqkv2l_weight_name)
    ms_meta[wqkv2l_weight_name] = f"model_layer_{layer}.safetensors"
    # concat weight scale
    wq2l_proj_weight_scale = param_dict[wq2l_proj_weight_scale_name].asnumpy()
    wkv2l_weight_scale = param_dict[wkv2l_weight_scale_name].asnumpy()
    wqkv2l_weight_scale = np.concatenate((wq2l_proj_weight_scale, wkv2l_weight_scale), 0)
    param_dict[wqkv2l_weight_scale_name] = ms.Parameter(wqkv2l_weight_scale, name=wqkv2l_weight_scale_name)
    ms_meta[wqkv2l_weight_scale_name] = f"model_layer_{layer}.safetensors"
    # concat bias
    wq2l_proj_weight_bias = param_dict[wq2l_proj_weight_bias_name].asnumpy()
    wkv2l_weight_bias = param_dict[wkv2l_weight_bias_name].asnumpy()
    wqkv2l_weight_bias = np.concatenate((wq2l_proj_weight_bias, wkv2l_weight_bias), 0)
    param_dict[wqkv2l_weight_bias_name] = ms.Parameter(wqkv2l_weight_bias, name=wqkv2l_weight_bias_name)
    ms_meta[wqkv2l_weight_bias_name] = f"model_layer_{layer}.safetensors"

    param_dict.pop(wq2l_proj_weight_name)
    param_dict.pop(wq2l_proj_weight_scale_name)
    param_dict.pop(wq2l_proj_weight_bias_name)
    param_dict.pop(wkv2l_weight_name)
    param_dict.pop(wkv2l_weight_scale_name)
    param_dict.pop(wkv2l_weight_bias_name)
    ms_meta.pop(wq2l_proj_weight_name)
    ms_meta.pop(wq2l_proj_weight_scale_name)
    ms_meta.pop(wq2l_proj_weight_bias_name)
    ms_meta.pop(wkv2l_weight_name)
    ms_meta.pop(wkv2l_weight_scale_name)
    ms_meta.pop(wkv2l_weight_bias_name)
    print("concat: {}".format(wqkv2l_weight_name))
    print("concat: {}".format(wqkv2l_weight_scale_name))
    print("concat: {}".format(wqkv2l_weight_bias_name))

    # replace quant op params
    q2l_proj_input_scale = f"model.layers.{layer}.attention.q2l_proj.quant_op.input_scale"
    q2l_proj_input_zp = f"model.layers.{layer}.attention.q2l_proj.quant_op.input_zp"
    q2l_proj_beta = f"model.layers.{layer}.attention.q2l_proj.quant_op.beta"
    kv2l_input_scale = f"model.layers.{layer}.attention.kv2l.quant_op.input_scale"
    kv2l_input_zp = f"model.layers.{layer}.attention.kv2l.quant_op.input_zp"
    kv2l_beta = f"model.layers.{layer}.attention.kv2l.quant_op.beta"
    qkv2l_input_scale = f"model.layers.{layer}.attention.qkv2l.quant_op.input_scale"
    qkv2l_input_zp = f"model.layers.{layer}.attention.qkv2l.quant_op.input_zp"
    qkv2l_beta = f"model.layers.{layer}.attention.qkv2l.quant_op.beta"

    qkv2l_input_scale_value = param_dict[q2l_proj_input_scale]
    qkv2l_input_zp_value = param_dict[q2l_proj_input_zp]
    qkv2l_beta_value = param_dict[q2l_proj_beta]
    param_dict[qkv2l_input_scale] = ms.Parameter(qkv2l_input_scale_value, name=qkv2l_input_scale)
    param_dict[qkv2l_input_zp] = ms.Parameter(qkv2l_input_zp_value, name=qkv2l_input_zp)
    param_dict[qkv2l_beta] = ms.Parameter(qkv2l_beta_value, name=qkv2l_beta)
    ms_meta[qkv2l_input_scale] = f"model_layer_{layer}.safetensors"
    ms_meta[qkv2l_input_zp] = f"model_layer_{layer}.safetensors"
    ms_meta[qkv2l_beta] = f"model_layer_{layer}.safetensors"

    param_dict.pop(q2l_proj_input_scale)
    param_dict.pop(q2l_proj_input_zp)
    param_dict.pop(q2l_proj_beta)
    param_dict.pop(kv2l_input_scale)
    param_dict.pop(kv2l_input_zp)
    param_dict.pop(kv2l_beta)
    ms_meta.pop(q2l_proj_input_scale)
    ms_meta.pop(q2l_proj_input_zp)
    ms_meta.pop(q2l_proj_beta)
    ms_meta.pop(kv2l_input_scale)
    ms_meta.pop(kv2l_input_zp)
    ms_meta.pop(kv2l_beta)
    print("replace {}".format(qkv2l_input_scale))
    print("replace: {}".format(qkv2l_input_zp))
    print("replace: {}".format(qkv2l_beta))

    return param_dict, ms_meta


def infer_quant_param_skip(k, layer, ms_key):
    '''infer_quant_param_skip'''
    return not k.startswith(f"model.layers.{layer}.") or "feed_forward.routed_experts.ffn" in ms_key


def infer_quant_net_param_handler(params_dict, layer, ms_param, ms_meta, kvb_split):
    """infer_quant_net_param_handler"""
    num_head = infer_config['num_head']
    rope_dim = infer_config['rope_dim']
    kv_head_dim = infer_config['kv_head_dim']
    qk_nope_head_dim = infer_config['qk_nope_head_dim']
    v_head_dim = infer_config['v_head_dim']
    for k, value in params_dict.items():
        value = params_dict[k]
        dtype = params_dict[k].dtype
        ms_key = infer_quant_name_replace(k)
        if infer_quant_param_skip(k, layer, ms_key):
            continue
        if "weight_scale" in ms_key or "weight_offset" in ms_key:
            value = value.squeeze(axis=-1)
        if "self_attn" in k and "proj" in k and "kv_b_proj" not in k:
            ms_key = ms_key.replace('.weight', '._layer.weight')
        # l2q_proj
        if "attention.l2q_proj._layer.weight" in ms_key:
            value = value.astype(np.float32).asnumpy()
            value = value.reshape(num_head, rope_dim, -1)
            weight = infer_trans_rope_weight(value)
            weight = weight.reshape(num_head * rope_dim, -1)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        elif "attention.l2q_proj._layer.matmul.dequant_scale" in ms_key or \
                "attention.l2q_proj._layer.matmul.quant_bias" in ms_key:
            value = value.asnumpy()
            value = value.reshape(num_head, rope_dim, -1)
            weight = infer_trans_rope_weight(value)
            weight = weight.reshape(num_head * rope_dim, -1).reshape(-1)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        # kv2l
        elif "attention.kv2l._layer.matmul.dequant_scale" in ms_key or \
                "attention.kv2l._layer.matmul.quant_bias" in ms_key:
            value = value.asnumpy()
            value = value.reshape(kv_head_dim, -1)
            weight = infer_trans_rope_weight(value).reshape(-1)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        elif "attention.kv2l._layer.weight" in ms_key:
            value = value.astype(np.float32).asnumpy()
            value = value.reshape(kv_head_dim, -1)
            weight = infer_trans_rope_weight(value)
            ms_param[ms_key] = ms.Parameter(ms.Tensor(weight, dtype), name=ms_key)
        elif "attention_norm.bias" in ms_key:
            ms_key = ms_key.replace('attention_norm.bias', 'attention.q2l_proj.quant_op.beta')
            ms_param[ms_key] = ms.Parameter(ms.Tensor(value, dtype), name=ms_key)
            ms_key_copy = ms_key.replace('attention.q2l_proj.quant_op.beta', 'attention.kv2l.quant_op.beta')
            ms_param[ms_key_copy] = ms.Parameter(ms.Tensor(value, dtype), name=ms_key_copy)
            ms_meta[ms_key_copy] = f"model_layer_{layer}.safetensors"
        elif "attention.lq_norm.bias" in ms_key:
            ms_key = ms_key.replace('attention.lq_norm.bias', 'attention.l2q_proj.quant_op.beta')
            ms_param[ms_key] = ms.Parameter(ms.Tensor(value, dtype), name=ms_key)
        # .attention.lkv2kv. process
        elif 'input_zp' in ms_key:
            value = value.astype(ms.int8).asnumpy()
            ms_param[ms_key] = ms.Parameter(ms.Tensor(value, ms.int8), name=ms_key)
        elif kvb_split and 'attention.lkv2kv' in ms_key:
            value = value.astype(np.float32).asnumpy()
            lkv2kv_head = qk_nope_head_dim + v_head_dim
            value = value.reshape(num_head, lkv2kv_head, -1)
            value_k_nope, value_v = value[:, :qk_nope_head_dim, :], value[:, qk_nope_head_dim:, :]
            value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
            value_v = value_v.reshape(-1, value_v.shape[-1])
            name_k_nope = ms_key.replace(".attention.lkv2kv.", ".attention.lkv2kv_k_nope.")
            name_v = ms_key.replace(".attention.lkv2kv.", ".attention.lkv2kv_v.")
            ms_param[name_k_nope] = ms.Parameter(ms.Tensor(value_k_nope, dtype), name=name_k_nope)
            ms_param[name_v] = ms.Parameter(ms.Tensor(value_v, dtype), name=name_v)
            ms_meta[name_k_nope] = f"model_layer_{layer}.safetensors"
            ms_key = name_v
        else:
            ms_param[ms_key] = ms.Parameter(ms.Tensor(value, dtype), name=ms_key)
        ms_meta[ms_key] = f"model_layer_{layer}.safetensors"


def infer_quant_net_convert_layer_weight(src_hf_dir, dst_ms_dir, layer, queue, arg):
    '''infer_quant_net_convert_layer_weight'''
    qkv2l_concat = arg.qkv2l_concat
    kvb_split = arg.kvb_split
    print(f"..... start convert layer {layer} .......", flush=True)
    ms_meta = {}
    with open(f"{src_hf_dir}/{arg.param_json}", "r") as fp:
        hf_meta = json.load(fp)['weight_map']

    safetensor_files = set()
    for param_key, param_path in hf_meta.items():
        if f"model.layers.{layer}." in param_key:
            safetensor_files.add(param_path)

    print(f"..... safetensor_files .......", safetensor_files, flush=True)
    params_dict = {}
    for ckpt in safetensor_files:
        src_path = f"{src_hf_dir}/{ckpt}"
        p = ms.load_checkpoint(src_path, format="safetensors")
        params_dict.update(p)

    ms_param = {}
    dst_name = f"model_layer_{layer}.safetensors"

    if layer >= 3:
        infer_quant_process_moe_routed_expert_ffn_weight(params_dict, dst_ms_dir, layer, ms_meta)
        infer_quant_process_moe_shared_expert_ffn_weight((params_dict, ms_param, layer, ms_meta, dst_name))

    if layer < 3:
        infer_quant_process_dense_ffn_weight((params_dict, ms_param, layer, ms_meta, dst_name))

    infer_quant_net_param_handler(params_dict, layer, ms_param, ms_meta, kvb_split)

    # qkv2l concat
    if qkv2l_concat:
        ms_param, ms_meta = infer_convert_qkv2l_concat_weight(ms_param, layer, ms_meta)

    dst_path = f"{dst_ms_dir}/model_layer_{layer}.safetensors"
    ms.save_checkpoint(ms_param, dst_path, format="safetensors")

    queue.put(ms_meta)
    print(f"..... end convert layer {layer} .......", flush=True)


def infer_convert_weight(src_hf_dir, dst_ms_dir, worker_num, ms_meta, arg):
    """convert inference model weight """
    infer_convert_outer_weight(src_hf_dir, dst_ms_dir, ms_meta, arg.param_json)
    layers = infer_config['total_layer_num']
    for index in range(math.ceil(layers / worker_num)):
        process = []
        queue = multiprocessing.Queue()
        for j in range(index * worker_num, (index + 1) * worker_num, 1):
            if j > layers - 1:
                break
            if arg.is_quant:
                p = multiprocessing.Process(target=infer_quant_net_convert_layer_weight,
                                            args=(src_hf_dir, dst_ms_dir, j, queue, arg))
            else:
                p = multiprocessing.Process(target=infer_convert_layer_weight, args=(src_hf_dir, dst_ms_dir, j, queue))
            process.append(p)
            p.start()
        for p in process:
            p.join()

        while not queue.empty():
            meta = queue.get()
            ms_meta.update(meta)


def infer_trans_ckpt_pt_to_ms(src_hf_dir, dst_ms_dir, worker_num, arg):
    """main function of inference weight process"""
    ms_meta = {}
    os.makedirs(dst_ms_dir, exist_ok=True)
    infer_convert_weight(src_hf_dir, dst_ms_dir, worker_num, ms_meta, arg)
    with open(f"{dst_ms_dir}/param_name_map.json", "w") as fp:
        json.dump(ms_meta, fp, indent=4)


def concat_mlp_ffn_weight(w_params, layer_id):
    """concat mlp ffn weight"""
    if layer_id >= 3:
        return
    # mlp
    gate_key = f'model.layers.{layer_id}.feed_forward.w1.weight'
    gate = w_params[gate_key]
    up_proj_key = f'model.layers.{layer_id}.feed_forward.w3.weight'
    up_proj = w_params[up_proj_key]

    gate_np = gate.asnumpy()
    up_proj_np = up_proj.asnumpy()
    weight_concat_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
    concat_weight = ms.Parameter(ms.Tensor(np.concatenate([gate_np, up_proj_np]), gate.dtype),
                                 name=weight_concat_key, requires_grad=False)
    print(f"--------- concat to {weight_concat_key}", flush=True)
    w_params.pop(gate_key)
    w_params.pop(up_proj_key)
    w_params[weight_concat_key] = concat_weight


def quant_concat_mlp_ffn_weight(w_params, layer_id):
    """quant concat mlp ffn weight"""
    if layer_id >= 3:
        return
    # mlp
    gate_key = f'model.layers.{layer_id}.feed_forward.w1._layer.weight'
    gate = w_params[gate_key]
    gate_scale_key = f'model.layers.{layer_id}.feed_forward.w1._layer.matmul.weight_scale'
    gate_scale = w_params[gate_scale_key]
    up_proj_key = f'model.layers.{layer_id}.feed_forward.w3._layer.weight'
    up_proj = w_params[up_proj_key]
    up_proj_scale_key = f'model.layers.{layer_id}.feed_forward.w3._layer.matmul.weight_scale'
    up_proj_scale = w_params[up_proj_scale_key]

    gate_np = gate.asnumpy()
    up_proj_np = up_proj.asnumpy()
    weight_concat_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
    concat_weight = ms.Parameter(ms.Tensor(np.concatenate([gate_np, up_proj_np]), gate.dtype),
                                 name=weight_concat_key, requires_grad=False)
    print(f"--------- concat to {weight_concat_key}", flush=True)

    gate_scale_np = gate_scale.asnumpy()
    up_proj_scale_np = up_proj_scale.asnumpy()
    scale_concat_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.weight_scale"
    concat_scale = ms.Parameter(
        ms.Tensor(np.concatenate([gate_scale_np, up_proj_scale_np], axis=-1), gate_scale.dtype),
        name=scale_concat_key, requires_grad=False)
    print(f"--------- concat to {scale_concat_key}", flush=True)

    gate_offset_key = f'model.layers.{layer_id}.feed_forward.w1._layer.matmul.weight_offset'
    up_proj_offset_key = f'model.layers.{layer_id}.feed_forward.w3._layer.matmul.weight_offset'
    if gate_offset_key in w_params:
        gate_offset = w_params[gate_offset_key]
        up_proj_offset = w_params[up_proj_offset_key]
        gate_zp_np = gate_offset.asnumpy()
        up_proj_zp_np = up_proj_offset.asnumpy()
        zp_concat_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.weight_offset"
        concat_zp = ms.Parameter(
            ms.Tensor(np.concatenate([gate_zp_np, up_proj_zp_np], axis=-1), gate_offset.dtype),
            name=zp_concat_key, requires_grad=False)
        print(f"--------- concat to {zp_concat_key}", flush=True)
        w_params.pop(gate_offset_key)
        w_params.pop(up_proj_offset_key)
        w_params[zp_concat_key] = concat_zp

    w_params.pop(gate_key)
    w_params.pop(gate_scale_key)
    w_params.pop(up_proj_key)
    w_params.pop(up_proj_scale_key)
    w_params[weight_concat_key] = concat_weight
    w_params[scale_concat_key] = concat_scale


def concat_routed_ffn_weight(w_params, layer_id):
    """concat routed ffn weight"""
    if layer_id < 3:
        return
    # route expert
    route_gate_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight'
    route_gate = w_params[route_gate_key]
    route_up_proj_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight'
    route_up_proj = w_params[route_up_proj_key]

    route_gate_np = route_gate.asnumpy()
    route_up_proj_np = route_up_proj.asnumpy()
    route_weight_concat_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden.weight"
    route_concat_weight = ms.Parameter(
        ms.Tensor(np.concatenate([route_gate_np, route_up_proj_np], axis=2), route_gate.dtype),
        name=route_weight_concat_key, requires_grad=False)
    print(f"--------- concat to {route_weight_concat_key}", flush=True)

    w_params.pop(route_gate_key)
    w_params.pop(route_up_proj_key)
    w_params[route_weight_concat_key] = route_concat_weight


def quant_concat_routed_ffn_weight(w_params, layer_id):
    """quant concat routed ffn weight"""
    if layer_id < 3:
        return
    # route expert
    route_gate_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.weight'
    route_gate = w_params[route_gate_key]
    route_gate_scale_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_scale'
    route_gate_scale = w_params[route_gate_scale_key]
    route_up_proj_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.weight'
    route_up_proj = w_params[route_up_proj_key]
    route_up_proj_scale_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_scale'
    route_up_proj_scale = w_params[route_up_proj_scale_key]

    route_gate_np = route_gate.asnumpy()
    route_up_proj_np = route_up_proj.asnumpy()
    route_weight_concat_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.weight"
    route_concat_weight = ms.Parameter(
        ms.Tensor(np.concatenate([route_gate_np, route_up_proj_np], axis=2), route_gate.dtype),
        name=route_weight_concat_key, requires_grad=False)
    print(f"--------- concat to {route_weight_concat_key}", flush=True)

    route_gate_scale_np = route_gate_scale.asnumpy()
    route_up_proj_scale_np = route_up_proj_scale.asnumpy()
    route_scale_concat_key =\
        f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.matmul.weight_scale"
    route_concat_scale = ms.Parameter(
        ms.Tensor(np.concatenate([route_gate_scale_np, route_up_proj_scale_np], axis=-1), route_gate_scale.dtype),
        name=route_scale_concat_key, requires_grad=False)
    print(f"--------- concat to {route_scale_concat_key}", flush=True)

    route_gate_offset_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_offset'
    route_up_proj_offset_key = f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_offset'
    if route_gate_offset_key in w_params:
        route_gate_offset = w_params[route_gate_offset_key]
        route_up_proj_offset = w_params[route_up_proj_offset_key]
        route_gate_zp_np = route_gate_offset.asnumpy()
        route_up_proj_zp_np = route_up_proj_offset.asnumpy()
        route_zp_concat_key =\
            f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.matmul.weight_offset"
        route_concat_zp = ms.Parameter(
            ms.Tensor(np.concatenate([route_gate_zp_np, route_up_proj_zp_np], axis=-1), route_gate_offset.dtype),
            name=route_zp_concat_key, requires_grad=False)
        print(f"--------- concat to {route_zp_concat_key}", flush=True)
        w_params.pop(route_gate_offset_key)
        w_params.pop(route_up_proj_offset_key)
        w_params[route_zp_concat_key] = route_concat_zp

    w_params.pop(route_gate_key)
    w_params.pop(route_gate_scale_key)
    w_params.pop(route_up_proj_key)
    w_params.pop(route_up_proj_scale_key)
    w_params[route_weight_concat_key] = route_concat_weight
    w_params[route_scale_concat_key] = route_concat_scale


def concat_shared_ffn_weight(w_params, layer_id):
    """concat shared ffn weight"""
    if layer_id < 3:
        return
    # shared expert
    shared_gate_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w1.weight'
    shared_gate = w_params[shared_gate_key]
    shared_up_proj_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w3.weight'
    shared_up_proj = w_params[shared_up_proj_key]

    shared_gate_np = shared_gate.asnumpy()
    shared_up_proj_np = shared_up_proj.asnumpy()
    shared_weight_concat_key = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden.weight"
    shared_concat_weight = ms.Parameter(
        ms.Tensor(np.concatenate([shared_gate_np, shared_up_proj_np]), shared_gate.dtype),
        name=shared_weight_concat_key, requires_grad=False)
    print(f"--------- concat to {shared_weight_concat_key}", flush=True)

    w_params.pop(shared_gate_key)
    w_params.pop(shared_up_proj_key)
    w_params[shared_weight_concat_key] = shared_concat_weight


def quant_concat_shared_ffn_weight(w_params, layer_id):
    """quant concat shared ffn weight"""
    if layer_id < 3:
        return
    # shared expert
    shared_gate_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w1._layer.weight'
    shared_gate = w_params[shared_gate_key]
    shared_gate_scale_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w1._layer.matmul.weight_scale'
    shared_gate_scale = w_params[shared_gate_scale_key]
    shared_up_proj_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w3._layer.weight'
    shared_up_proj = w_params[shared_up_proj_key]
    shared_up_proj_scale_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w3._layer.matmul.weight_scale'
    shared_up_proj_scale = w_params[shared_up_proj_scale_key]

    shared_gate_np = shared_gate.asnumpy()
    shared_up_proj_np = shared_up_proj.asnumpy()
    shared_weight_concat_key = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.weight"
    shared_concat_weight = ms.Parameter(
        ms.Tensor(np.concatenate([shared_gate_np, shared_up_proj_np]), shared_gate.dtype),
        name=shared_weight_concat_key, requires_grad=False)
    print(f"--------- concat to {shared_weight_concat_key}", flush=True)

    shared_gate_scale_np = shared_gate_scale.asnumpy()
    shared_up_proj_scale_np = shared_up_proj_scale.asnumpy()
    shared_scale_concat_key =\
        f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.matmul.weight_scale"
    shared_concat_scale = ms.Parameter(
        ms.Tensor(np.concatenate([shared_gate_scale_np, shared_up_proj_scale_np], axis=-1), shared_gate_scale.dtype),
        name=shared_scale_concat_key, requires_grad=False)
    print(f"--------- concat to {shared_scale_concat_key}", flush=True)

    shared_gate_offset_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w1._layer.matmul.weight_offset'
    shared_up_proj_offset_key = f'model.layers.{layer_id}.feed_forward.shared_experts.w3._layer.matmul.weight_offset'
    if shared_gate_offset_key in w_params:
        shared_gate_offset = w_params[shared_gate_offset_key]
        shared_up_proj_offset = w_params[shared_up_proj_offset_key]
        shared_gate_zp_np = shared_gate_offset.asnumpy()
        shared_up_proj_zp_np = shared_up_proj_offset.asnumpy()
        shared_zp_concat_key =\
            f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.matmul.weight_offset"
        shared_concat_zp = ms.Parameter(
            ms.Tensor(np.concatenate([shared_gate_zp_np, shared_up_proj_zp_np], axis=-1), shared_gate_offset.dtype),
            name=shared_zp_concat_key, requires_grad=False)
        print(f"--------- concat to {shared_zp_concat_key}", flush=True)
        w_params.pop(shared_gate_offset_key)
        w_params.pop(shared_up_proj_offset_key)
        w_params[shared_zp_concat_key] = shared_concat_zp

    w_params.pop(shared_gate_key)
    w_params.pop(shared_gate_scale_key)
    w_params.pop(shared_up_proj_key)
    w_params.pop(shared_up_proj_scale_key)
    w_params[shared_weight_concat_key] = shared_concat_weight
    w_params[shared_scale_concat_key] = shared_concat_scale


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape (M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    # Get the original dimensions of weight
    m, n = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    if scale_m != (m + block_size - 1) // block_size:
        raise ValueError("Mismatch in scale rows and weight rows.")
    if scale_n != (n + block_size - 1) // block_size:
        raise ValueError("Mismatch in scale columns and weight columns.")

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:m, :n]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


def dequant_layer_weights(layer_id, pt_layer_weights):
    """Dequanting weights in a layer"""
    dequanted_weights = {}
    for weight_name, weight in pt_layer_weights.items():
        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1 and (f"model.layers.{layer_id}." in weight_name):  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                # Get scale_inv from the correct file
                scale_inv = pt_layer_weights.get(scale_inv_name)
                dequanted_weights[weight_name] = weight_dequant(weight, scale_inv)
            except KeyError:
                print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping dequanting")
                dequanted_weights[weight_name] = weight
        else:
            dequanted_weights[weight_name] = weight
    return dequanted_weights


def plain_name_replace(weight_name: str):
    """Weight name replacing for pre/post-process module"""
    weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
    return weight_name


def mla_name_replace(weight_name: str):
    """Weight name replacing for MLA module weights"""
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.q_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
    weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
    weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
    weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
    weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
    weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


def mlp_name_replace(weight_name: str):
    """Weight name replacing for MLP module, including MoE"""
    weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
    weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
    weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
    weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
    weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                      'feed_forward.routed_experts.router.router.topk_bias')
    return weight_name


def mtp_name_replace(weight_name: str, current_layer_id: int, mtp_layer_id: int):
    """replace weight name for MultiPredictionToken module"""
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.enorm",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.norm_emb")
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.hnorm",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.norm")
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.eh_proj",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.dense")
    return weight_name


def layers_model_file_map(file_path):
    """Get weight-file map"""
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "model.safetensors.index.json")
    with open(weight_map_file) as f:
        weights_map = json.load(f)
    weights_map = weights_map["weight_map"]

    for weight_key, value in weights_map.items():
        if weight_key.startswith("model.layers."):
            layer_name = int(weight_key.split('model.layers.')[1].split('.')[0])
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def load_data_pt(file_name):
    return load_file(file_name, device="cpu")


def read_matched_file_pt(layer_st_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_first:
        st_file_list.extend(list(layer_st_map["model.embed_tokens.weight"]))
    if is_last:
        st_file_list.extend(list(layer_st_map["model.norm.weight"]))
        st_file_list.extend(list(layer_st_map["lm_head.weight"]))
    st_file_list = list(set(st_file_list))
    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_pt(st_file)
        weights.update(current_weight)
    return weights


def _mla_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MLA module"""
    n_head = config['n_head']
    qk_nope_head_dim = config['qk_nope_head_dim']
    qk_rope_head_dim = config['qk_rope_head_dim']
    v_head_dim = config['v_head_dim']

    q_a_proj_key = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
    kv_a_proj_key = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
    o_proj_key = f"model.layers.{layer_id}.self_attn.o_proj.weight"
    q_a_layernorm_key = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
    kv_a_layernorm_key = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
    q_b_proj_key = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
    kv_b_proj_key = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
    input_norm_key = f"model.layers.{layer_id}.input_layernorm.weight"
    post_attn_norm_key = f"model.layers.{layer_id}.post_attention_layernorm.weight"

    q_a_proj = pt_layer_weights.pop(q_a_proj_key)
    kv_a_proj = pt_layer_weights.pop(kv_a_proj_key)
    o_proj = pt_layer_weights.pop(o_proj_key)
    q_a_layernorm = pt_layer_weights.pop(q_a_layernorm_key)
    kv_a_layernorm = pt_layer_weights.pop(kv_a_layernorm_key)
    q_b_proj = pt_layer_weights.pop(q_b_proj_key)
    kv_b_proj = pt_layer_weights.pop(kv_b_proj_key)
    input_norm = pt_layer_weights.pop(input_norm_key)
    post_attn_norm = pt_layer_weights.pop(post_attn_norm_key)

    mla_weight_dict = defaultdict()
    # split q_b_proj
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    qk_nope, qk_rope = q_b_proj.reshape(n_head, qk_head_dim, -1).split([qk_nope_head_dim, qk_rope_head_dim], dim=1)
    qk_rope = qk_rope.reshape(qk_rope.shape[0], qk_rope.shape[1] // 2, 2, -1).permute(0, 2, 1, 3)
    qk_nope = qk_nope.reshape(-1, qk_nope.shape[-1])
    qk_rope = qk_rope.reshape(-1, qk_rope.shape[-1])
    qk_nope_key = mla_name_replace(q_b_proj_key).replace(".l2q_proj.", ".l2q_nope_proj.")
    qk_rope_key = mla_name_replace(q_b_proj_key).replace(".l2q_proj.", ".l2q_pe_proj.")
    mla_weight_dict[qk_nope_key] = qk_nope.clone()
    mla_weight_dict[qk_rope_key] = qk_rope.clone()
    # split kv_a_proj
    kv_lora_rank = kv_a_proj.shape[0] - qk_rope_head_dim
    latent_kv, k_rope = kv_a_proj.split([kv_lora_rank, qk_rope_head_dim], dim=0)
    k_rope = k_rope.reshape(k_rope.shape[0] // 2, 2, -1).permute(1, 0, 2).reshape(-1, k_rope.shape[-1])
    latent_kv_key = mla_name_replace(kv_a_proj_key).replace(".kv2l.", ".kv2l_latent_kv.")
    k_rope_key = mla_name_replace(kv_a_proj_key).replace(".kv2l.", ".kv2l_k_pe.")
    mla_weight_dict[latent_kv_key] = latent_kv.clone()
    mla_weight_dict[k_rope_key] = k_rope.clone()
    # split kv_b_proj
    kv_head_dim = qk_nope_head_dim + v_head_dim
    k_nope, v = kv_b_proj.reshape(n_head, kv_head_dim, -1).split([qk_nope_head_dim, v_head_dim], dim=1)
    k_nope = k_nope.reshape(-1, k_nope.shape[-1])
    v = v.reshape(-1, v.shape[-1])
    k_nope_key = mla_name_replace(kv_b_proj_key).replace(".lkv2kv.", ".lkv2kv_k_nope.")
    v_key = mla_name_replace(kv_b_proj_key).replace(".lkv2kv.", ".lkv2kv_v.")
    mla_weight_dict[k_nope_key] = k_nope
    mla_weight_dict[v_key] = v
    # process q_a_proj, o_proj, and layernorms
    q_a_proj_key = mla_name_replace(q_a_proj_key)
    mla_weight_dict[q_a_proj_key] = q_a_proj.clone()
    o_proj_key = mla_name_replace(o_proj_key)
    mla_weight_dict[o_proj_key] = o_proj.clone()
    q_a_layernorm_key = mla_name_replace(q_a_layernorm_key)
    mla_weight_dict[q_a_layernorm_key] = q_a_layernorm.clone()
    kv_a_layernorm_key = mla_name_replace(kv_a_layernorm_key)
    mla_weight_dict[kv_a_layernorm_key] = kv_a_layernorm.clone()
    input_norm_key = mla_name_replace(input_norm_key)
    mla_weight_dict[input_norm_key] = input_norm.clone()
    post_attn_norm_key = mla_name_replace(post_attn_norm_key)
    mla_weight_dict[post_attn_norm_key] = post_attn_norm.clone()

    return mla_weight_dict


def _mlp_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MLP/MoE module"""
    num_routed_experts = config['num_routed_experts']
    first_k_dense_replace = config['first_k_dense_replace']
    use_gemm = config['use_gemm']

    mlp_weight_dict = defaultdict()
    if layer_id < first_k_dense_replace:
        gate_proj_key = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        up_proj_key = f"model.layers.{layer_id}.mlp.up_proj.weight"
        down_proj_key = f"model.layers.{layer_id}.mlp.down_proj.weight"
        gate_proj = pt_layer_weights.pop(gate_proj_key)
        up_proj = pt_layer_weights.pop(up_proj_key)
        down_proj = pt_layer_weights.pop(down_proj_key)

        gate_proj_key = mlp_name_replace(gate_proj_key)
        up_proj_key = mlp_name_replace(up_proj_key)
        down_proj_key = mlp_name_replace(down_proj_key)
        mlp_weight_dict[gate_proj_key] = gate_proj.clone()
        mlp_weight_dict[up_proj_key] = up_proj.clone()
        mlp_weight_dict[down_proj_key] = down_proj.clone()
    else:
        router_weight_key = f"model.layers.{layer_id}.mlp.gate.weight"
        router_correct_bias_key = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        shared_experts_gate_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        shared_experts_up_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
        shared_experts_down_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        router_weight = pt_layer_weights.pop(router_weight_key)
        router_weight = router_weight[:num_routed_experts, :]
        router_correct_bias = pt_layer_weights.pop(router_correct_bias_key)
        router_correct_bias = router_correct_bias[:num_routed_experts]
        shared_experts_gate_proj = pt_layer_weights.pop(shared_experts_gate_proj_key)
        shared_experts_up_proj = pt_layer_weights.pop(shared_experts_up_proj_key)
        shared_experts_down_proj = pt_layer_weights.pop(shared_experts_down_proj_key)

        gate_proj_list = []
        up_proj_list = []
        down_proj_list = []
        for expert_id in range(num_routed_experts):
            gate_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
            up_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
            down_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
            gate_proj = pt_layer_weights.pop(gate_proj_key)
            up_proj = pt_layer_weights.pop(up_proj_key)
            down_proj = pt_layer_weights.pop(down_proj_key)
            gate_proj_list.append(gate_proj)
            up_proj_list.append(up_proj)
            down_proj_list.append(down_proj)

        expert_gate_proj = torch.stack(gate_proj_list, 0)
        expert_up_proj = torch.stack(up_proj_list, 0)
        expert_down_proj = torch.stack(down_proj_list, 0)

        # replace name and store
        router_weight_key = mlp_name_replace(router_weight_key)
        router_correct_bias_key = mlp_name_replace(router_correct_bias_key)
        shared_experts_gate_proj_key = mlp_name_replace(shared_experts_gate_proj_key)
        shared_experts_up_proj_key = mlp_name_replace(shared_experts_up_proj_key)
        shared_experts_down_proj_key = mlp_name_replace(shared_experts_down_proj_key)
        mlp_weight_dict[router_weight_key] = router_weight.clone()
        mlp_weight_dict[router_correct_bias_key] = router_correct_bias.clone()
        mlp_weight_dict[shared_experts_gate_proj_key] = shared_experts_gate_proj.clone()
        mlp_weight_dict[shared_experts_up_proj_key] = shared_experts_up_proj.clone()
        mlp_weight_dict[shared_experts_down_proj_key] = shared_experts_down_proj.clone()
        # routed experts
        expert_gate_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        expert_up_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"
        expert_down_proj_key = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        if use_gemm:
            expert_gate_proj = expert_gate_proj.transpose(1, 2)
            expert_up_proj = expert_up_proj.transpose(1, 2)
            expert_down_proj = expert_down_proj.transpose(1, 2)
        mlp_weight_dict[expert_gate_proj_key] = expert_gate_proj.clone()
        mlp_weight_dict[expert_up_proj_key] = expert_up_proj.clone()
        mlp_weight_dict[expert_down_proj_key] = expert_down_proj.clone()

    return mlp_weight_dict


def _mtp_pt_to_ms(layer_id, pt_layer_weights, config):
    """Processing weights in MTP module, the shared weights will be ignored"""
    num_layers = config["num_layers"]
    mtp_layer_id = layer_id - num_layers
    # ignore the shared emb_weights and lm head in mtp layers
    pt_layer_weights.pop(f"model.layers.{layer_id}.embed_tokens.weight")
    pt_layer_weights.pop(f"model.layers.{layer_id}.shared_head.norm.weight")
    pt_layer_weights.pop(f"model.layers.{layer_id}.shared_head.head.weight")
    enorm_key = f"model.layers.{layer_id}.enorm.weight"
    hnorm_key = f"model.layers.{layer_id}.hnorm.weight"
    e_proj_key = f"model.layers.{layer_id}.eh_proj.weight"

    enorm = pt_layer_weights.pop(enorm_key)
    hnorm = pt_layer_weights.pop(hnorm_key)
    e_proj = pt_layer_weights.pop(e_proj_key)

    mtp_weight_dict = defaultdict()
    enorm_key = mtp_name_replace(enorm_key, layer_id, mtp_layer_id)
    hnorm_key = mtp_name_replace(hnorm_key, layer_id, mtp_layer_id)
    e_proj_key = mtp_name_replace(e_proj_key, layer_id, mtp_layer_id)
    mtp_weight_dict[enorm_key] = enorm.clone()
    mtp_weight_dict[hnorm_key] = hnorm.clone()
    mtp_weight_dict[e_proj_key] = e_proj.clone()

    return mtp_weight_dict


def _model_preprocess_pt_to_ms(pt_layer_weights):
    """Processing weights in prepross module"""
    emb_weight_key = "model.embed_tokens.weight"
    emb_weight = pt_layer_weights.pop(emb_weight_key)
    emb_weight_key = plain_name_replace(emb_weight_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = emb_weight.clone()

    return plain_weight_dict


def _model_postprocess_pt_to_ms(pt_layer_weights):
    """Processing weights in postpross module"""
    final_norm_key = "model.norm.weight"
    lm_head_key = "lm_head.weight"
    final_norm = pt_layer_weights.pop(final_norm_key)
    lm_head = pt_layer_weights.pop(lm_head_key)

    final_norm_key = plain_name_replace(final_norm_key)
    lm_head_key = plain_name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[final_norm_key] = final_norm.clone()
    plain_weight_dict[lm_head_key] = lm_head.clone()

    return plain_weight_dict


def convert_pt_to_ms(input_path, output_path, config=None):
    """convert hf weight to ms."""
    if config is None:
        config = default_config
    save_format = config['save_format']

    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    os.makedirs(output_path, exist_ok=True)
    layer_st_map = layers_model_file_map(input_path)
    torch.set_default_dtype(torch.bfloat16)

    dtype = config["dtype"]
    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_st_map = defaultdict()
    num_saved_ckpt_files = 0
    for layer_id in range(total_num_layers):
        if layer_id == 0:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif layer_id == total_num_layers - 1:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=True)
        else:
            pt_layer_weights = read_matched_file_pt(layer_st_map, [layer_id], is_first=False, is_last=False)

        # first dequanting weights
        pt_layer_weights = dequant_layer_weights(layer_id, pt_layer_weights)

        ms_layer_weights = defaultdict()
        if layer_id == 0:
            ms_layer_weights.update(_model_preprocess_pt_to_ms(pt_layer_weights))
        ms_layer_weights.update(_mla_pt_to_ms(layer_id, pt_layer_weights, config))
        ms_layer_weights.update(_mlp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id > num_layers - 1:
            ms_layer_weights.update(_mtp_pt_to_ms(layer_id, pt_layer_weights, config))
        if layer_id == total_num_layers - 1:
            ms_layer_weights.update(_model_postprocess_pt_to_ms(pt_layer_weights))

        to_save_ckpt = []
        saving_file = f"ms-model-{num_saved_ckpt_files:05d}.{save_format}"
        for name in list(ms_layer_weights.keys()):
            value = ms_layer_weights.pop(name).to(torch.float32).numpy()
            tmp_dtype = dtype
            if "norm" in name or "router.dense" in name or "topk_bias" in name:
                tmp_dtype = ms.float32
            to_save_ckpt.append({'name': name, 'data': ms.Tensor(value, dtype=tmp_dtype)})
            converted_st_map[name] = saving_file

        ms.save_checkpoint(to_save_ckpt, os.path.join(output_path, saving_file), format=save_format)
        num_saved_ckpt_files += 1
        print(f"saving weights in layer-{layer_id} to file {saving_file}")

    converted_model_index_file = os.path.join(output_path, f"ms-model.{save_format}.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)


def convert_ms_to_gmm(input_path, output_path):
    """convert ms routing ffn weight for gmm."""
    params_dict = ms.load_checkpoint(input_path)
    for k, v in params_dict.items():
        if 'feed_forward.routed_experts.ffn.w1.weight' in k or \
                'feed_forward.routed_experts.ffn.w2.weight' in k or \
                'feed_forward.routed_experts.ffn.w3.weight' in k:
            orig_tensor = ms.Tensor(v)
            gmm_tensor = orig_tensor.transpose((0, 2, 1))
            params_dict[k] = ms.Parameter(gmm_tensor)
            print(f"\rConvertion finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_routed_experts', default=256, type=int)
    parser.add_argument('--torch_ckpt_path', default=None, type=str)
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str)
    parser.add_argument('--use_gmm', action='store_true')
    parser.add_argument('--pre_ckpt_path', default=None, type=str)
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("--num_layers", default=61, type=int)
    parser.add_argument("--num_nextn_predict_layers", default=1, type=int)
    parser.add_argument("--first_k_dense_replace", default=3, type=int)
    parser.add_argument("--n_head", default=128, type=int)
    parser.add_argument("--qk_nope_head_dim", default=128, type=int)
    parser.add_argument("--qk_rope_head_dim", default=64, type=int)
    parser.add_argument("--v_head_dim", default=128, type=int)
    parser.add_argument("--save_format", default="safetensors", choices=["safetensors", "ckpt"])

    parser.add_argument("--infer", default=False, type=bool)
    parser.add_argument('--worker_num', default=10, type=int)
    parser.add_argument('--ffn_concat', default=False, type=bool)
    parser.add_argument('--dense_ffn_hidden_size', default=18432, type=int)
    parser.add_argument('--expert_hidden_size', default=2048, type=int)
    parser.add_argument('--qkv2l_concat', default=False)
    parser.add_argument('--kvb_split', default=True)
    parser.add_argument('--is_quant', default=False, type=bool)
    parser.add_argument('--param_json', default="model.safetensors.index.json", type=str)

    args = parser.parse_args()
    if args.infer:
        ms.set_context(device_target="CPU")
        if args.ffn_concat:
            mindspore_ckpt_path = args.mindspore_ckpt_path
            params = ms.load_checkpoint(mindspore_ckpt_path)
            if args.is_quant:
                for i in range(61):
                    quant_concat_mlp_ffn_weight(params, i)
                    quant_concat_routed_ffn_weight(params, i)
                    quant_concat_shared_ffn_weight(params, i)
            else:
                for i in range(61):
                    concat_mlp_ffn_weight(params, i)
                    concat_routed_ffn_weight(params, i)
                    concat_shared_ffn_weight(params, i)
            if os.path.isfile(mindspore_ckpt_path):
                os.remove(mindspore_ckpt_path)
                ms.save_checkpoint(params, mindspore_ckpt_path)
                print(f"--------- save checkpoint finished, save_path: {mindspore_ckpt_path}.", flush=True)
        else:
            infer_trans_ckpt_pt_to_ms(src_hf_dir=args.torch_ckpt_path,
                                      dst_ms_dir=args.mindspore_ckpt_path,
                                      worker_num=args.worker_num,
                                      arg=args)
    else:
        if args.pre_ckpt_path:
            convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
        else:
            for key in default_config:
                default_config[key] = getattr(args, key, default_config[key])
            default_config['dtype'] = dtype_map.get(default_config['dtype'], default_config['dtype'])

            convert_pt_to_ms(input_path=args.torch_ckpt_path,
                             output_path=args.mindspore_ckpt_path,
                             config=default_config)
