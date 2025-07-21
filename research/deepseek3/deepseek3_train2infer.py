#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================

"""
MindSpore Training Checkpoint to Inference Checkpoint Converter

Usage:
    python deepseek3_train2infer.py \\
        --input_path <input_path> \\
        --output_path <output_path> \\
        [--first_k_dense_replace <first_k_dense_replace>] \\
        [--num_heads <num_heads>] \\
        [--qk_nope_head_dim <qk_nope_head_dim>] \\
        [--qk_rope_head_dim <qk_rope_head_dim>] \\
        [--processes <processes>]

Arguments:
    Required:
        --input_path       Path to the directory containing MindSpore .safetensors files
        --output_path      Target directory for the converted inference checkpoint

    Optional:
        --first_k_dense_replace    First layer index where MoE experts appear (default: 3)
        --num_heads           Number of attention heads (default: 128)
        --qk_nope_head_dim     Head dimension for non-PE Q/K (default: 128)
        --qk_rope_head_dim     Head dimension for PE Q/K (default: 64)
        --processes        Number of parallel processes for conversion (default: 32)

Output:
    - Layer-wise safetensors files (layer_*.safetensors)
    - General parameters file (general.safetensors)
    - Parameter mapping file (param_name_map.json)

Example:
    python deepseek3_train2infer.py \\
        --input_path ./mindspore_ckpt \\
        --output_path ./hf_ckpt \\
        --first_k_dense_replace 3 \\
        --processes 16

Note:
    The input checkpoint must follow specific naming conventions for attention and expert weights.
"""

import re
import os
import json
import argparse
from time import time
from multiprocessing import Pool

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from mindformers.tools.logger import logger
from mindformers.tools.utils import set_safe_mode_for_file_or_dir


def process_attention_weights(mode, layer_id, mapping, num_heads, qk_nope_head_dim, qk_rope_head_dim):
    """Processing attention weights."""
    # Processing L2Q parameters
    if mode == 'l2q':
        nope_name = f'model.layers.{layer_id}.attention.l2q_nope_proj.weight'
        pe_name = f'model.layers.{layer_id}.attention.l2q_pe_proj.weight'

        value_nope = get_array(nope_name, mapping)
        value_pe = get_array(pe_name, mapping)

        value_nope = value_nope.reshape(num_heads, qk_nope_head_dim, -1)
        value_pe = value_pe.reshape(num_heads, qk_rope_head_dim, -1)

        value_merged = np.concatenate([value_nope, value_pe], axis=1)
        value_merged = value_merged.reshape(-1, value_merged.shape[-1])

        mapping[nope_name], mapping[pe_name] = None, None

        return f'model.layers.{layer_id}.attention.l2q_proj.weight', value_merged

    # Processing KV2L parameters
    k_pe_name = f'model.layers.{layer_id}.attention.kv2l_k_pe.weight'
    latent_kv_name = f'model.layers.{layer_id}.attention.kv2l_latent_kv.weight'

    value_k_pe = get_array(k_pe_name, mapping)
    value_latent_kv = get_array(latent_kv_name, mapping)

    value_k_pe = value_k_pe.reshape(-1, value_k_pe.shape[-1])

    value_merged = np.concatenate([value_latent_kv, value_k_pe], axis=0)
    value_merged = value_merged.reshape(-1, value_merged.shape[-1])

    mapping[k_pe_name], mapping[latent_kv_name] = None, None

    return f'model.layers.{layer_id}.attention.kv2l.weight', value_merged


def process_expert_weights(layer_id, mapping):
    """Processing expert weights."""
    names, values = [], []

    for i in [f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight',
              f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight',
              f'model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight']:
        orig_tensor = get_array(i, mapping)
        gmm_tensor = orig_tensor.transpose((0, 2, 1))
        names.append(i)
        values.append(gmm_tensor)
        mapping[i] = None
    return names, values


def get_array(name, mapping):
    """Getting specified array."""
    file = os.path.join(INPUT_PATH, mapping[name])
    with safe_open(file, framework='np', device='cpu') as tensor_file:
        return tensor_file.get_tensor(name)


def save_array(array, path):
    """Saving array."""
    save_file(array, path)


def processor(layer_id, mapping, output_path, first_k_dense_replace, num_heads, qk_nope_head_dim, qk_rope_head_dim):
    """Processing the weight of one particular layer."""
    logger.info(f'Processing Layer {layer_id} by pid {os.getpid()}')
    layer_weights = {}
    processor_start = time()

    # Processing attention parameters
    for mode in ['l2q', 'kv2l']:
        name, value = process_attention_weights(mode=mode,
                                                layer_id=layer_id,
                                                mapping=mapping,
                                                num_heads=num_heads,
                                                qk_nope_head_dim=qk_nope_head_dim,
                                                qk_rope_head_dim=qk_rope_head_dim)
        layer_weights[name] = value
    attention_end = time()
    logger.info(f'Layer {layer_id} Attention Transform Finished. Consumed {attention_end - processor_start}s')

    if layer_id >= first_k_dense_replace:
        # Processing expert parameters
        names, values = process_expert_weights(layer_id, mapping)
        for i, name in enumerate(names):
            layer_weights[name] = values[i]
        moe_end = time()
        logger.info(f'Layer {layer_id} MOE Transform Finished. Consumed {moe_end - attention_end}s')

    # Adding parameters that doesn't need to be transferred to layer_weights
    for name in mapping.keys():
        if f'layers.{layer_id}.' in name and name not in layer_weights and mapping[name] is not None:
            layer_weights[name] = get_array(name, mapping)

    unchanged_end = time()
    logger.info(f'Layer {layer_id} Unchanged Weights Loaded. '
                f'Consumed {unchanged_end - (moe_end if layer_id >= first_k_dense_replace else attention_end)}s')

    # Saving
    save_array(layer_weights, f'{output_path}/layer_{layer_id}.safetensors')
    save_end = time()
    logger.info(f'{output_path}/layer_{layer_id}.safetensors is saved. Consumed {save_end - unchanged_end}s')

    # Return a dict with structure {weight name : file name}
    layer_mapping = {}
    for i in layer_weights:
        layer_mapping[i] = f'layer_{layer_id}.safetensors'
    return layer_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MindSpore checkpoint to HuggingFace format")
    parser.add_argument("--input_path", required=True, help="Input directory containing .safetensors files")
    parser.add_argument("--output_path", required=True, help="Output .safetensors file path")
    parser.add_argument("--first_k_dense_replace", type=int, default=3, help="Output .safetensors file path")
    parser.add_argument("--num_heads", type=int, default=128, help="Number of attention heads")
    parser.add_argument("--qk_nope_head_dim", type=int, default=128, help="Q/K head dim without PE")
    parser.add_argument("--qk_rope_head_dim", type=int, default=64, help="Q/K head dim with PE")
    parser.add_argument("--processes", type=int, default=32, help="Number of processed in parallel")
    args = parser.parse_args()

    start = time()
    logger.info(f'Start converting weights with {args.processes} processes')
    print(f'main pid {os.getpid()}')
    INPUT_PATH = args.input_path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info('Output dir is made')

    with open(f"{args.input_path}/param_name_map.json", "r") as f:
        weight_mapping = json.load(f)

    # Getting layer_num
    layer_num = 0
    general = {}
    index_json = {}
    for k in weight_mapping.keys():
        match = re.search(r"layers\.(\d+)", k)
        if match:
            current_layer = int(match.group(1))
            layer_num = max(current_layer, layer_num)
        else:
            general[k] = None
    layer_num += 1
    logger.info(f'Total layer number is {layer_num}')

    # Processing weights with a process pool
    with Pool(processes=args.processes, maxtasksperchild=1) as pool:
        iter_input = [[i,
                       weight_mapping,
                       args.output_path,
                       args.first_k_dense_replace,
                       args.num_heads,
                       args.qk_nope_head_dim,
                       args.qk_rope_head_dim] for i in range(layer_num)]

        process_pool = pool.starmap_async(processor, iter_input)
        pool.close()

        # Processing general parameters
        logger.info('Processing General weights ...')
        for k in general:
            general[k] = get_array(k, weight_mapping)
            index_json[k] = 'general.safetensors'
        save_array(general, f'{args.output_path}/general.safetensors')
        logger.info(f'{args.output_path}/general.safetensors is saved.')

        total_layer_mapping = process_pool.get()
        for m in total_layer_mapping:
            index_json.update(m)

    logger.info('Saving param_name_map.json')
    res_path = f'{args.output_path}/param_name_map.json'
    with open(res_path, 'w') as f:
        json.dump(index_json, f, indent=4)
    set_safe_mode_for_file_or_dir(res_path)
    logger.info('param_name_map.json is saved')

    end = time()
    logger.info(f'Transform finished, consumed {end - start} seconds')
