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
Generate fake Qwen3 HuggingFace model directory for CLI tests.

**INTERNAL TESTING TOOL - NOT FOR USER USE**

This script creates a minimal fake Qwen3 model directory with all necessary files
(config.json, tokenizer files, etc.) that can be used for testing. This is an
internal testing utility and should not be used directly by users.
"""

import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file


def create_fake_qwen3_model(model_dir: str, vocab_size: int = 1000, hidden_size: int = 512):
    """
    Create a fake Qwen3 HuggingFace model directory.

    Args:
        model_dir: Directory path where the model files will be created
        vocab_size: Vocabulary size (default: 1000, small for testing)
        hidden_size: Hidden size (default: 512, small for testing)
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("Generating fake Qwen3 model directory:")
    print(f"  Model directory: {model_dir}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Hidden size: {hidden_size}")

    # 1. Create config.json
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "max_position_embeddings": 32768,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 8,
        "num_hidden_layers": 4,  # Small number for testing
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.37.0",
        "use_cache": True,
        "vocab_size": vocab_size,
        "pad_token_id": 151643
    }

    config_path = model_path / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Created: {config_path}")

    # 2. Create generation_config.json
    generation_config = {
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "pad_token_id": 151643,
        "transformers_version": "4.37.0"
    }

    gen_config_path = model_path / "generation_config.json"
    with open(gen_config_path, 'w', encoding='utf-8') as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)
    print(f"  Created: {gen_config_path}")

    # 3. Create tokenizer.json (simplified version)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 151643,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": 151644,
                "content": "<|im_start|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": 151645,
                "content": "<|im_end|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "WhitespaceSplit"
        },
        "post_processor": None,
        "decoder": {
            "type": "BPEDecoder",
            "suffix": ""
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {},
            "merges": []
        }
    }

    # Add all tokens to vocab
    # Add all tokens from 0 to vocab_size-1 to ensure they can be encoded
    for i in range(vocab_size):
        tokenizer_json["model"]["vocab"][f"token_{i}"] = i

    # Add special tokens to vocab (matching vocab.json)
    tokenizer_json["model"]["vocab"]["<|endoftext|>"] = 151643
    tokenizer_json["model"]["vocab"]["<|im_start|>"] = 151644
    tokenizer_json["model"]["vocab"]["<|im_end|>"] = 151645

    tokenizer_json_path = model_path / "tokenizer.json"
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    print(f"  Created: {tokenizer_json_path}")

    # 4. Create tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "model_max_length": 32768,
        "clean_up_tokenization_spaces": False,
        "tokenizer_type": "Qwen3Tokenizer"
    }

    tokenizer_config_path = model_path / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"  Created: {tokenizer_config_path}")

    # 5. Create vocab.json (simplified)
    vocab = {}
    for i in range(vocab_size):
        vocab[f"token_{i}"] = i
    # Add special tokens
    vocab["<|endoftext|>"] = 151643
    vocab["<|im_start|>"] = 151644
    vocab["<|im_end|>"] = 151645

    vocab_path = model_path / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"  Created: {vocab_path}")

    # 6. Create merges.txt (empty or minimal for BPE)
    merges_path = model_path / "merges.txt"
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")
        # Add some basic merges
        for i in range(10):
            f.write(f"a{i} b{i}\n")
    print(f"  Created: {merges_path}")

    # 7. Create model.safetensors file with lm_head.weight
    # Create lm_head.weight tensor with shape [vocab_size, hidden_size]
    # This matches the model config dimensions
    lm_head_weight = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    # Create state dict with only lm_head.weight
    state_dict = {
        "lm_head.weight": lm_head_weight
    }

    # Save to model.safetensors
    safetensors_path = model_path / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"  Created: {safetensors_path}")
    print(f"    Contains: lm_head.weight with shape [{vocab_size}, {hidden_size}]")

    print("\nFake Qwen3 model directory generation complete!")
    print(f"  Model directory: {model_dir}")
    print(f"\nUse this path in config: {model_dir}")
