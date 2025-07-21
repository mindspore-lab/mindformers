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
Convert llama weight.
Support mindspore format and Meta format.
"""

import json
import os
import argparse
from collections import defaultdict
import torch
import mindspore as ms
from mindspore.ops.operations import Cast
from safetensors.torch import save_file
from mindformers import MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import set_safe_mode_for_file_or_dir

ms.set_context(device_target='CPU')
cpu_cast = Cast().set_device('CPU')


def get_torch_storage_size(tensor):
    """Get tensor's storage size, requires torch >= 2.1"""
    return tensor.untyped_storage().nbytes()


def get_dtype(config):
    """Get compute dtype."""
    if config.model.model_config.compute_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif config.model.model_config.compute_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype


def name_replace(weight_name: str):
    """replace ms param weight_name to hf."""
    weight_name = weight_name.replace('tok_embeddings.embedding_weight', 'embed_tokens.weight')
    weight_name = weight_name.replace('model.norm_out.weight', 'model.norm.weight')

    weight_name = weight_name.replace(".attention.wq.", ".self_attn.q_proj.")
    weight_name = weight_name.replace(".attention.wk.", ".self_attn.k_proj.")
    weight_name = weight_name.replace(".attention.wv.", ".self_attn.v_proj.")
    weight_name = weight_name.replace(".attention.wo.", ".self_attn.o_proj.")
    weight_name = weight_name.replace(".attention_norm.", ".input_layernorm.")
    weight_name = weight_name.replace(".ffn_norm.", ".post_attention_layernorm.")

    weight_name = weight_name.replace('feed_forward.w1.', 'mlp.gate_proj.')
    weight_name = weight_name.replace('feed_forward.w2.', 'mlp.down_proj.')
    weight_name = weight_name.replace('feed_forward.w3.', 'mlp.up_proj.')

    return weight_name


def _model_preprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in prepross module"""
    dtype = get_dtype(config)
    emb_weight_key = "model.tok_embeddings.embedding_weight"
    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    emb_weight_key = name_replace(emb_weight_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = torch.from_numpy(emb_weight).to(dtype).clone()

    return plain_weight_dict


def _model_postprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in postpross module"""
    dtype = get_dtype(config)
    final_norm_key = "model.norm_out.weight"
    lm_head_key = "lm_head.weight"
    final_norm = cpu_cast(ms_layer_weights.get(final_norm_key), ms.float32).numpy()

    # 0.5b Model does not have lm_head.weight
    if ms_layer_weights.get(lm_head_key, None) is not None:
        lm_head = cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()

    final_norm_key = name_replace(final_norm_key)
    lm_head_key = name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[final_norm_key] = torch.from_numpy(final_norm).to(dtype).clone()
    if ms_layer_weights.get(lm_head_key, None) is not None:
        plain_weight_dict[lm_head_key] = torch.from_numpy(lm_head).to(dtype).clone()

    return plain_weight_dict


def attention_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in attention module"""
    dtype = get_dtype(config)

    attention_weight_dict = defaultdict()
    attention_weight_list = [
        f"model.layers.{layer_id}.attention.wq.weight",
        f"model.layers.{layer_id}.attention.wq.bias",
        f"model.layers.{layer_id}.attention.wk.weight",
        f"model.layers.{layer_id}.attention.wk.bias",
        f"model.layers.{layer_id}.attention.wv.weight",
        f"model.layers.{layer_id}.attention.wv.bias",
        f"model.layers.{layer_id}.attention.wo.weight",
        f"model.layers.{layer_id}.attention_norm.weight",
        f"model.layers.{layer_id}.ffn_norm.weight"
    ]

    for name in attention_weight_list:
        value = cpu_cast(ms_layer_weights.pop(name), ms.float32).numpy()
        name = name_replace(name)
        attention_weight_dict[name] = torch.from_numpy(value).to(dtype).clone()

    return attention_weight_dict


def feed_forward_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in feed_forward module"""
    dtype = get_dtype(config)
    feed_forward_weight_dict = defaultdict()

    gate_proj_key = f"model.layers.{layer_id}.feed_forward.w1.weight"
    down_proj_key = f"model.layers.{layer_id}.feed_forward.w2.weight"
    up_proj_key = f"model.layers.{layer_id}.feed_forward.w3.weight"

    gate_proj = cpu_cast(ms_layer_weights.pop(gate_proj_key), ms.float32).numpy()
    up_proj = cpu_cast(ms_layer_weights.pop(up_proj_key), ms.float32).numpy()
    down_proj = cpu_cast(ms_layer_weights.pop(down_proj_key), ms.float32).numpy()

    gate_proj_key = name_replace(gate_proj_key)
    up_proj_key = name_replace(up_proj_key)
    down_proj_key = name_replace(down_proj_key)
    feed_forward_weight_dict[gate_proj_key] = torch.from_numpy(gate_proj).to(dtype).clone()
    feed_forward_weight_dict[up_proj_key] = torch.from_numpy(up_proj).to(dtype).clone()
    feed_forward_weight_dict[down_proj_key] = torch.from_numpy(down_proj).to(dtype).clone()

    return feed_forward_weight_dict


def load_data_ms(file_name):
    """Load safetensors"""
    return ms.load_checkpoint(file_name, format="safetensors")


def read_matched_file(layer_st_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_first:
        st_file_list.extend(list(layer_st_map["model.tok_embeddings.embedding_weight"]))
    if is_last:
        st_file_list.extend(list(layer_st_map["model.norm_out.weight"]))
        st_file_list.extend(list(layer_st_map["lm_head.weight"]))
    st_file_list = list(set(st_file_list))
    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_ms(st_file)
        weights.update(current_weight)
    return weights


def has_only_one_safetensors_file(file_path):
    """Check whether there is only one safetensors file."""
    files = os.listdir(file_path)
    safetensors_files = [file for file in files if file.endswith('.safetensors')]
    return len(safetensors_files) == 1


def layers_model_file_map(file_path):
    """Get weight-file map"""
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "param_name_map.json")
    if not os.path.exists(weight_map_file):
        if has_only_one_safetensors_file(file_path):
            weight_map_file = os.path.join(file_path, "param_name_map.json")
            safetensors_file = next((f for f in os.listdir(file_path) if f.endswith('.safetensors')), None)
            weight = load_data_ms(os.path.join(file_path, safetensors_file))
            param_name_map = {key: "model.safetensors" for key in weight.keys()}
            with open(weight_map_file, 'w') as f:
                json.dump(param_name_map, f, indent=4)
            set_safe_mode_for_file_or_dir(weight_map_file)
        else:
            raise ValueError(f"Cannot find weight map file in path {file_path}")

    with open(weight_map_file) as f:
        weights_map = json.load(f)
    try:
        weights_map = weights_map["weight_map"]
    except KeyError:
        pass
    for weight_key, value in weights_map.items():
        if weight_key.startswith("model.layers."):
            layer_name = int(weight_key.split('model.layers.')[1].split('.')[0])
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def ms_ckpt_convertor(input_path, output_path, config):
    """convert ms weight to hf."""
    logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>  Start convert checkpoint from ckpt format.")
    num_layers = config.model.model_config.num_layers
    logger.info(f"Trying to convert mindspore checkpoint in {input_path}.", flush=True)
    ms_weights = ms.load_checkpoint(input_path)

    converted_st_map = defaultdict()
    converted_st_map["weight_map"] = defaultdict()
    converted_st_map["metadata"] = defaultdict()

    total_size = 0
    for layer_id in range(num_layers):
        pt_layer_weights = defaultdict()
        if layer_id == 0:
            pt_layer_weights.update(_model_preprocess_ms_to_pt(ms_weights, config))
        pt_layer_weights.update(attention_ms_to_pt(layer_id, ms_weights, config))
        pt_layer_weights.update(feed_forward_ms_to_pt(layer_id, ms_weights, config))
        if layer_id == num_layers - 1:
            pt_layer_weights.update(_model_postprocess_ms_to_pt(ms_weights, config))

        saving_file_name = f"model-{layer_id + 1:05d}-of-{num_layers:05d}.safetensors"
        for name in list(pt_layer_weights.keys()):
            converted_st_map["weight_map"][name] = saving_file_name
            total_size += get_torch_storage_size(pt_layer_weights.get(name))
        save_file(pt_layer_weights, os.path.join(output_path, saving_file_name))
        logger.info(f"saving weights in layer-{layer_id + 1} to file {saving_file_name}")
    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, f"model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)
    set_safe_mode_for_file_or_dir(converted_model_index_file)


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert safetensors format checkpoint"""
    logger.info(">>>>>>>>>>>>>>>>>>>>>>>  Start convert checkpoint from safetensors format.")
    layer_st_map = layers_model_file_map(input_path)

    num_layers = config.model.model_config.num_layers

    converted_st_map = defaultdict()
    converted_st_map["weight_map"] = defaultdict()
    converted_st_map["metadata"] = defaultdict()

    total_size = 0
    for layer_id in range(num_layers):
        if layer_id == 0:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif 0 < layer_id < num_layers - 1:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=False, is_last=False)
        else:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=False, is_last=True)

        pt_layer_weights = defaultdict()
        if layer_id == 0:
            pt_layer_weights.update(_model_preprocess_ms_to_pt(ms_layer_weights, config))
        pt_layer_weights.update(attention_ms_to_pt(layer_id, ms_layer_weights, config))
        pt_layer_weights.update(feed_forward_ms_to_pt(layer_id, ms_layer_weights, config))
        if layer_id == num_layers - 1:
            pt_layer_weights.update(_model_postprocess_ms_to_pt(ms_layer_weights, config))

        saving_file_name = f"model-{layer_id + 1:05d}-of-{num_layers:05d}.safetensors"
        for name in list(pt_layer_weights.keys()):
            converted_st_map["weight_map"][name] = saving_file_name
            total_size += get_torch_storage_size(pt_layer_weights.get(name))
        save_file(pt_layer_weights, os.path.join(output_path, saving_file_name))
        logger.info(f"saving weights in layer-{layer_id + 1} to file {saving_file_name}")

    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, f"model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)
    set_safe_mode_for_file_or_dir(converted_model_index_file)


def convert_ms_to_pt(input_path, output_path, config_path):
    """convert ms weight to huggingface."""

    # init context
    config = MindFormerConfig(config_path)

    os.makedirs(output_path, exist_ok=True)
    load_format = config.get('load_ckpt_format', "ckpt")
    logger.info(f"load_format---------------->: {load_format}")
    logger.info(f"Loading mindspore checkpoint in {input_path} ...")

    if load_format == "safetensors":
        ms_safetensors_convertor(input_path, output_path, config)
    else:
        ms_ckpt_convertor(input_path, output_path, config)

    logger.info("Finish converting mindspore checkpoints into Huggingface checkpoints!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', required=True, default='transform.ckpt')
    parser.add_argument('--torch_ckpt_path', required=True, default='./qwen2/qwen2-hf/')
    parser.add_argument('--config_path', required=True, type=str, help='config file path.')

    args = parser.parse_args()

    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_ckpt_path,
                     config_path=args.config_path)
