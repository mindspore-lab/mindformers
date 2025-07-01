# Copyright (c) [2023] Deepseek
#
# This code is based on DeepSeek-V3 implementations from the DeepSeek AI team.
#
# Modification points:
# 1. Replace the device from cuda to cpu.
# 2. Change the code structure while keeping the functions unchanged.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""hf fp8 safetensors cast to hf bf16 safetensors"""
import os
import json
from argparse import ArgumentParser
from glob import glob
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import torch


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    """
    row, col = weight.shape
    scale_m, scale_n = scale.shape
    assert scale_m == (row + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (col + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    # Convert to float32 for calculation
    weight = weight.to(torch.float32)

    # Expand scale to match weight shape using broadcasting
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:row, :col]

    # Multiply
    dequantized_weight = weight * scale_expanded

    return dequantized_weight.to(torch.get_default_dtype())


def load_model_index(fp8_path):
    """Load the model index JSON file."""
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    return model_index["weight_map"]


def get_tensor_loader(fp8_path, weight_map):
    """Closure-based loader to fetch tensors from the correct safetensor file."""
    loaded_files = {}

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    return get_tensor, loaded_files


def convert_single_file(
        safetensor_file,
        bf16_path,
        get_tensor_func,
        loaded_files,
        fp8_weight_names,
        progress_bar=None
):
    """Convert a single .safetensors file and save the result."""
    file_name = os.path.basename(safetensor_file)
    current_state_dict = load_file(safetensor_file, device="cpu")
    loaded_files[file_name] = current_state_dict

    new_state_dict = {}
    for weight_name, weight in current_state_dict.items():
        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1:  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                scale_inv = get_tensor_func(scale_inv_name)
                fp8_weight_names.append(weight_name)
                new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
            except KeyError:
                print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                new_state_dict[weight_name] = weight
        else:
            new_state_dict[weight_name] = weight

    new_safetensor_file = os.path.join(bf16_path, file_name)
    save_file(new_state_dict, new_safetensor_file)

    # Memory management: keep only the 2 most recently used files
    if len(loaded_files) > 2:
        oldest_file = next(iter(loaded_files))
        del loaded_files[oldest_file]

    if progress_bar:
        progress_bar.update(1)


def update_model_index(bf16_path, weight_map, fp8_weight_names):
    """Update and save the model index by removing _scale_inv entries."""
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)

    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def main(fp8_path, bf16_path):
    """Main function that orchestrates the entire conversion process."""
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    # Step 1: Load model index
    weight_map = load_model_index(fp8_path)

    # Step 2: Prepare tensor getter and cache
    get_tensor_func, loaded_files = get_tensor_loader(fp8_path, weight_map)

    # Step 3: Collect all safetensor files
    safetensor_files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    fp8_weight_names = []

    # Step 4: Process each file with progress bar
    with tqdm(total=len(safetensor_files), desc="Converting Files") as pbar:
        for safetensor_file in safetensor_files:
            convert_single_file(
                safetensor_file=safetensor_file,
                bf16_path=bf16_path,
                get_tensor_func=get_tensor_func,
                loaded_files=loaded_files,
                fp8_weight_names=fp8_weight_names,
                progress_bar=pbar
            )

    # Step 5: Update model index
    update_model_index(bf16_path, weight_map, fp8_weight_names)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert FP8 weights to BF16.")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True,
                        help="Path to directory containing FP8 weights.")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                        help="Path to output directory for BF16 weights.")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
