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
from glob import glob

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
    'dtype': ms.bfloat16,
    'save_format': "safetensors"
}


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


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
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
    weight_name = weight_name.replace('mlp.experts.', 'feed_forward.ffn.')
    weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
    weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
    weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
    weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
    weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                      'feed_forward.routed_experts.router.router.topk_bias')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
    weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
    return weight_name


def mtp_name_replace(weight_name: str, current_layer_id: int, mtp_layer_id: int):
    """replace weight name for MultiPredictionToken module"""
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.enorm",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.norm_emb")
    weight_name = weight_name.replace(
        f"model.layers.{current_layer_id}.hnorm", f"model.mtp_hidden_fusers.{mtp_layer_id}.norm")
    weight_name = weight_name.replace(f"model.layers.{current_layer_id}.eh_proj",
                                      f"model.mtp_hidden_fusers.{mtp_layer_id}.dense")
    return weight_name


def convert_pt_to_ms(input_path, output_path, config=None):
    """convert hf weight to ms."""
    if config is None:
        config = default_config

    num_routed_experts = config['num_routed_experts']
    save_format = config['save_format']

    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensors files
    loaded_files = {}

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(input_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
    safetensor_files.sort()

    experts_weight_count_max = 3 * int(num_routed_experts)
    to_convert_weight_cache = {}
    to_convert_layer_id_cache = set()
    num_saved_ckpt_files = 0

    converted_weight_map = {}
    # convert safetensor files one by one
    for safetensor_file in safetensor_files:
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        # first dequanting weights
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    to_convert_weight_cache[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping dequanting")
                    to_convert_weight_cache[weight_name] = weight
            else:
                to_convert_weight_cache[weight_name] = weight
            if "model.layers." in weight_name and ("embed_tokens" in weight_name or "shared_head" in weight_name):
                # ignore mtp shared token embedding and output head
                to_convert_weight_cache.pop(weight_name)
            try:
                current_layer_id = int(weight_name.split('model.layers.')[-1].split('.')[0])
                to_convert_layer_id_cache.add(current_layer_id)
            except (ValueError, IndexError):
                pass

            # Memory management: keep only the 2 most recently used files
            if len(loaded_files) > 2:
                oldest_file = next(iter(loaded_files))
                del loaded_files[oldest_file]

        print("to_convert_layer_id_cache: ", to_convert_layer_id_cache)
        # check whether the decoder layer weights are ready
        ready_layer_id = set()

        for layer_id in to_convert_layer_id_cache:
            # since decoder layer ends with post attention layernorm
            for weight_name in to_convert_weight_cache:
                if f"model.layers.{layer_id}.post_attention_layernorm" in weight_name:
                    ready_layer_id.add(layer_id)

        print("ready_layer_id: ", ready_layer_id)
        # now converting ready layers into mindspore format
        converted_weight_name, finished_weight_name, to_save_weights_list = \
            convert_pt_to_ms_by_layer(experts_weight_count_max, ready_layer_id,
                                      to_convert_weight_cache, config)

        if converted_weight_name:
            num_saved_ckpt_files += 1
            saving_file = f"ms-model-{num_saved_ckpt_files:05d}.{save_format}"
            ms.save_checkpoint(to_save_weights_list, os.path.join(output_path, saving_file), format=save_format)
            for name in converted_weight_name:
                converted_weight_map[name] = saving_file
        # reset cache
        for weight_name in finished_weight_name:
            to_convert_weight_cache.pop(weight_name)
        for layer_id in ready_layer_id:
            to_convert_layer_id_cache.remove(layer_id)

    # finally we rename saved files in format 'ms-model-{id}-of-{total}'
    # generate weight map index file
    converted_mdoel_index_file = os.path.join(output_path, f"ms-model.{save_format}.index.json")
    with open(converted_mdoel_index_file, "w") as f:
        json_string = json.dumps(converted_weight_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)

    return True


def convert_pt_to_ms_by_layer(experts_weight_count_max, ready_layer_id, to_convert_weight_cache, config):
    """convert_pt_to_ms_by_layer"""

    n_head = config['n_head']
    qk_nope_head_dim = config['qk_nope_head_dim']
    qk_rope_head_dim = config['qk_rope_head_dim']
    v_head_dim = config['v_head_dim']
    num_layers = config['num_layers']
    dtype = config['dtype']

    expert_weight_count = 0
    expert_w1_list, expert_w2_list, expert_w3_list = [], [], []
    to_save_weights_list = []
    converted_weight_name = []
    finished_weight_name = []
    for weight_name in to_convert_weight_cache:
        # if there is no ready decoder layer, break
        if not ready_layer_id:
            break
        value = to_convert_weight_cache.get(weight_name)
        try:
            current_layer_id = int(weight_name.split('model.layers.')[-1].split('.')[0])
        except (ValueError, IndexError):
            current_layer_id = -1

        name = name_replace(weight_name)
        if current_layer_id in ready_layer_id:
            # do MLA weight splitting
            if ".attention.l2q_proj" in name:
                q_head = qk_nope_head_dim + qk_rope_head_dim
                value = value.reshape(n_head, q_head, -1)
                value_nope, value_pe = value[:, :qk_nope_head_dim, :], value[:, qk_nope_head_dim:, :]
                value_pe = value_pe.reshape(value_pe.shape[0], value_pe.shape[1] // 2, 2, -1).permute(0, 2, 1, 3)
                value_nope = value_nope.reshape(-1, value_nope.shape[-1]).to(torch.float32).numpy()
                value_pe = value_pe.reshape(-1, value_pe.shape[-1]).to(torch.float32).numpy()
                name_lt = name.split(".attention.l2q_proj.")
                name_nope = name_lt[0] + ".attention.l2q_nope_proj." + name_lt[-1]
                to_save_weights_list.append({'name': name_nope, 'data': ms.Tensor(value_nope, dtype=dtype)})
                name_pe = name_lt[0] + ".attention.l2q_pe_proj." + name_lt[-1]
                to_save_weights_list.append({'name': name_pe, 'data': ms.Tensor(value_pe, dtype=dtype)})
                converted_weight_name.append(name_nope)
                converted_weight_name.append(name_pe)
            elif ".attention.kv2l." in name:
                kv_lora_rank = value.shape[0] - qk_rope_head_dim
                value_latent_kv, value_k_pe = value[:kv_lora_rank, :], value[kv_lora_rank:, :]
                value_k_pe = value_k_pe.reshape(value_k_pe.shape[0] // 2, 2, -1).permute(1, 0, 2)
                value_k_pe = value_k_pe.reshape(-1, value_k_pe.shape[-1]).to(torch.float32).numpy()
                value_latent_kv = value_latent_kv.to(torch.float32).numpy()
                name_lt = name.split(".attention.kv2l.")
                name_k = name_lt[0] + ".attention.kv2l_k_pe." + name_lt[-1]
                to_save_weights_list.append({'name': name_k, 'data': ms.Tensor(value_k_pe, dtype=dtype)})
                name_kv = name_lt[0] + ".attention.kv2l_latent_kv." + name_lt[-1]
                to_save_weights_list.append({'name': name_kv, 'data': ms.Tensor(value_latent_kv, dtype=dtype)})
                converted_weight_name.append(name_k)
                converted_weight_name.append(name_kv)
            elif ".attention.lkv2kv." in name:
                lkv2kv_head = qk_nope_head_dim + v_head_dim
                value = value.reshape(n_head, lkv2kv_head, -1)
                value_k_nope, value_v = value[:, :qk_nope_head_dim, :], value[:, qk_nope_head_dim:, :]
                value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1]).to(torch.float32).numpy()
                value_v = value_v.reshape(-1, value_v.shape[-1]).to(torch.float32).numpy()
                name_lt = name.split(".attention.lkv2kv.")
                name_k = name_lt[0] + ".attention.lkv2kv_k_nope." + name_lt[-1]
                to_save_weights_list.append({'name': name_k, 'data': ms.Tensor(value_k_nope, dtype=dtype)})
                name_v = name_lt[0] + ".attention.lkv2kv_v." + name_lt[-1]
                to_save_weights_list.append({'name': name_v, 'data': ms.Tensor(value_v, dtype=dtype)})
                converted_weight_name.append(name_k)
                converted_weight_name.append(name_v)

            elif 'feed_forward.ffn' in name:
                # concat routed expert mlp weight
                add_to_expert_list(expert_w1_list, expert_w2_list, expert_w3_list, name, value)
                expert_weight_count += 1
                if expert_weight_count == experts_weight_count_max:
                    str_front = name.split('ffn')[0]
                    stack_expert(converted_weight_name, dtype, expert_w1_list,
                                 str_front + 'routed_experts.ffn.w1.weight', to_save_weights_list)
                    stack_expert(converted_weight_name, dtype, expert_w2_list,
                                 str_front + 'routed_experts.ffn.w2.weight', to_save_weights_list)
                    stack_expert(converted_weight_name, dtype, expert_w3_list,
                                 str_front + 'routed_experts.ffn.w3.weight', to_save_weights_list)
                    # reset
                    expert_weight_count = 0
                    expert_w1_list, expert_w2_list, expert_w3_list = [], [], []
            else:
                mtp_layer_id = current_layer_id - num_layers
                # ignore the shared token embedding and head weight
                if "embed_tokens" in name or "shared_head" in name:
                    continue
                if mtp_layer_id > -1:
                    # process mtp_layer
                    name = mtp_name_replace(name, current_layer_id, mtp_layer_id)
                if "norm" in name or "router.dense" in name or "topk_bias" in name:
                    tmp_dtype = ms.float32
                else:
                    tmp_dtype = dtype
                ms_value = value.to(torch.float32).numpy()
                to_save_weights_list.append({'name': name, 'data': ms.Tensor(ms_value, dtype=tmp_dtype)})
                converted_weight_name.append(name)

            finished_weight_name.append(weight_name)
        if current_layer_id == -1:
            ms_value = value.to(torch.float32).numpy()
            to_save_weights_list.append({'name': name, 'data': ms.Tensor(ms_value, dtype=dtype)})
            converted_weight_name.append(name)
            finished_weight_name.append(weight_name)
    return converted_weight_name, finished_weight_name, to_save_weights_list


def stack_expert(converted_weight_name, dtype, expert_w1_list, name,
                 to_save_weights_list):
    """stack_expert"""
    value = torch.stack(expert_w1_list, 0).to(torch.float32).numpy()
    to_save_weights_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})
    converted_weight_name.append(name)


def add_to_expert_list(expert_w1_list, expert_w2_list, expert_w3_list, name, value):
    """add_to_expert_list"""
    if "gate_proj" in name:  # gate_proj
        expert_w1_list.append(value)
    if "down_proj" in name:  # down_proj
        expert_w2_list.append(value)
    if "up_proj" in name:  # up_proj
        expert_w3_list.append(value)


def convert_ms_to_gmm(input_path, output_path):
    """convert ms routing ffn weight for gmm."""
    params = ms.load_checkpoint(input_path)
    for k, v in params.items():
        if 'feed_forward.routed_experts.ffn.w1.weight' in k or \
                'feed_forward.routed_experts.ffn.w2.weight' in k or \
                'feed_forward.routed_experts.ffn.w3.weight' in k:
            orig_tensor = ms.Tensor(v)
            gmm_tensor = orig_tensor.transpose((0, 2, 1))
            params[k] = ms.Parameter(gmm_tensor)
            print(f"\rConvertion finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_routed_experts', default=256)
    parser.add_argument('--torch_ckpt_path', default=None)
    parser.add_argument('--mindspore_ckpt_path', default=None)
    parser.add_argument('--use_gmm', action='store_true')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument("--num_layers", default=61, type=int)
    parser.add_argument("--n_head", default=128, type=int)
    parser.add_argument("--qk_nope_head_dim", default=128, type=int)
    parser.add_argument("--qk_rope_head_dim", default=64, type=int)
    parser.add_argument("--v_head_dim", default=128, type=int)
    parser.add_argument("--save_format", default="safetensors")

    args = parser.parse_args()

    if args.pre_ckpt_path:
        convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
    else:
        for key in default_config:
            default_config[key] = getattr(args, key, default_config[key])
        default_config['dtype'] = dtype_map.get(default_config['dtype'], default_config['dtype'])

        convert_pt_to_ms(input_path=args.torch_ckpt_path, output_path=args.mindspore_ckpt_path, config=default_config)
