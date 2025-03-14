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
"""Parallel Decoding for text generation."""
import numpy as np


def la_pre_process(input_ids, slot_mapping, **model_kwargs):
    """ parallel decoding process """
    # adapt warmup stage, `q_seq_lens` is int dtype
    if model_kwargs.get('q_seq_lens') is not None and isinstance(model_kwargs.get('q_seq_lens'), int):
        q_seq_lens = None
    else:
        q_seq_lens = model_kwargs.get('q_seq_lens')

    if q_seq_lens is not None:
        input_ids = input_ids.reshape(-1)
        sum_q_seq_lens = sum(q_seq_lens)
        max_q_seq_lens = max(q_seq_lens)
        if len(input_ids) != sum_q_seq_lens:
            input_ids = input_ids.tolist()
            input_ids_list = list()
            start = 0
            for q_seq_len in q_seq_lens:
                input_ids_list += input_ids[start: start + q_seq_len]
                start += max_q_seq_lens
            input_ids = np.array(input_ids_list, dtype=np.int32)
        if len(slot_mapping) != sum_q_seq_lens:
            slot_mapping = slot_mapping.tolist()
            slot_mapping_list = list()
            start = 0
            for q_seq_len in q_seq_lens:
                slot_mapping_list += slot_mapping[start: start + q_seq_len]
                start += max_q_seq_lens
            slot_mapping = np.array(slot_mapping_list, dtype=np.int32).reshape(-1)

    attention_mask = model_kwargs.get('spec_mask')
    if attention_mask is None:
        attention_mask = np.zeros((1, 1), dtype=np.float16)

    return slot_mapping, attention_mask
