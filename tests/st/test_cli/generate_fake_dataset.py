
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
Generate fake Megatron format dataset for CLI training tests.

**INTERNAL TESTING TOOL - NOT FOR USER USE**

This script creates a minimal fake dataset in Megatron format (.bin and .idx files)
that can be used for testing the CLI training functionality. This is an internal
testing utility and should not be used directly by users.
"""

from pathlib import Path
import numpy as np

from mindformers.dataset.blended_datasets.indexed_dataset import IndexedDatasetBuilder


def generate_fake_dataset(output_prefix: str, num_samples: int = 100, seq_length: int = 1024):
    """
    Generate a fake Megatron format dataset.

    Args:
        output_prefix: Output file prefix (without .bin/.idx extension)
        num_samples: Number of samples to generate
        seq_length: Sequence length for each sample
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate bin and idx file paths
    bin_path = f"{output_prefix}_text_document.bin"
    idx_path = f"{output_prefix}_text_document.idx"

    print("Generating fake dataset:")
    print(f"  Output prefix: {output_prefix}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Bin file: {bin_path}")
    print(f"  Idx file: {idx_path}")

    # Create IndexedDatasetBuilder
    builder = IndexedDatasetBuilder(bin_path, dtype=np.int32)

    # Generate fake token IDs (using a simple vocabulary range)
    # In real scenarios, these would be actual token IDs from a tokenizer
    # For testing, we use a simple range: 0-1000 (excluding special tokens 0 and 1)
    vocab_range = (2, 1000)

    for i in range(num_samples):
        # Generate random token IDs for this sample
        # Using a simple pattern: some random tokens + EOD token (0) at the end
        tokens = np.random.randint(
            vocab_range[0],
            vocab_range[1],
            size=seq_length - 1,
            dtype=np.int32
        )

        # Add EOD token at the end
        tokens = np.append(tokens, np.int32(0))

        # Add document with single sequence
        builder.add_document(tokens, lengths=[len(tokens)])

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    # Finalize the dataset (writes the idx file)
    builder.finalize(idx_path)

    print("\nDataset generation complete!")
    print(f"  Created: {bin_path}")
    print(f"  Created: {idx_path}")
    print(f"\nUse this path in config: {output_prefix}_text_document")
