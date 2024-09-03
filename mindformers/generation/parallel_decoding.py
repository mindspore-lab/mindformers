# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore.common.tensor import Tensor


_logits_process = {}


def register_logits_process(name=None):
    def register_fn(fn):
        if name is None:
            _logits_process[fn.__name__] = fn
        else:
            _logits_process[name] = fn
        return fn
    return register_fn


@register_logits_process('la')
def _la_logits_process(logits, q_seq_lens, block_tables, prefill):
    """ lookahead decoding logits process """
    shape = logits.shape
    if len(shape) > 2:
        logits = logits.reshape(-1, shape[-1])
    if q_seq_lens is not None:
        index = []
        batch_size = len(block_tables)
        max_seq_len = logits.shape[0] // batch_size
        if prefill:
            for i in range(batch_size):
                index.append(i * max_seq_len + q_seq_lens[i] - 1)
        else:
            for i in range(batch_size):
                start = i * max_seq_len
                end = start + q_seq_lens[i]
                index += list(range(start, end))
        index = Tensor(index, ms.int32)
        logits = logits[index]
    return logits


def parallel_decoding_logits_process(config, logits, q_seq_lens, block_tables, prefill):
    """ parallel decoding logits process """
    logits_process_fn = None
    if hasattr(config, 'parallel_decoding'):
        logits_process_fn = _logits_process.get(config.parallel_decoding)
    if logits_process_fn is not None:
        return logits_process_fn(logits, q_seq_lens, block_tables, prefill)
    return logits


def _parallel_decoding_pad(inputs, axis, pad_len, value):
    if isinstance(inputs, Tensor):
        inputs = inputs.numpy()
    shape = list(inputs.shape)
    shape[axis] = pad_len - shape[axis]
    if shape[axis] == 0:
        return inputs
    pad = np.full(shape, value, inputs.dtype)
    inputs = np.concatenate((inputs, pad), axis)
    return inputs


def parallel_decoding_process(config, input_ids, block_tables, slot_mapping, prefill, **model_kwargs):
    """ parallel decoding process """
    model_inputs = dict()

    batch_size = len(block_tables)
    params = config.parallel_decoding_params
    inc_seq_len = (params['level'] - 1) * (params['window'] + params['guess_set_size'] + 1)
    max_seq_len = config.seq_length
    pad_seq_len = max_seq_len if prefill else inc_seq_len

    if model_kwargs.get('q_seq_lens') is None:
        input_ids = np.reshape(input_ids, (batch_size, -1))
        input_ids = _parallel_decoding_pad(input_ids, 1, pad_seq_len, 0)
    else:
        input_ids = input_ids.reshape(-1)
        input_ids_pad = np.zeros((batch_size, pad_seq_len), dtype=input_ids.dtype)
        start = 0
        for i, q_seq_len in enumerate(model_kwargs['q_seq_lens']):
            end = start + q_seq_len
            input_ids_pad[i, :q_seq_len] = input_ids[start: end]
            start = end
        input_ids = input_ids_pad

    model_inputs['input_ids'] = Tensor(input_ids, dtype=ms.int32)

    if 'spec_mask' in model_kwargs and model_kwargs['spec_mask'] is not None:
        attention_mask = model_kwargs['spec_mask']
        if model_kwargs.get('q_seq_lens') is None:
            shape = attention_mask.shape
            if len(shape) == 2:
                attention_mask = attention_mask.reshape(batch_size, -1, shape[1])
            attention_mask = _parallel_decoding_pad(attention_mask, 1, pad_seq_len, 1)
            attention_mask = _parallel_decoding_pad(attention_mask, 2, max_seq_len, 1)
        else:
            if isinstance(attention_mask, Tensor):
                attention_mask = attention_mask.numpy().reshape(-1, attention_mask.shape[-1])
            attention_mask_pad = np.ones((batch_size, pad_seq_len, max_seq_len), dtype=attention_mask.dtype)
            start = 0
            for i, q_seq_len in enumerate(model_kwargs['q_seq_lens']):
                end = start + q_seq_len
                attention_mask_pad[i, :q_seq_len, :attention_mask.shape[-1]] = attention_mask[start: end]
                start = end
            attention_mask = attention_mask_pad
    elif 'attention_mask' in model_kwargs and model_kwargs['attention_mask'] is not None:
        attention_mask = model_kwargs['attention_mask']
    else:
        seq_len = input_ids.shape[1]
        attention_mask = np.zeros((batch_size, seq_len, seq_len))
    attention_mask = attention_mask.astype(np.bool_).astype(np.float16)
    model_inputs['attention_mask'] = Tensor(attention_mask, dtype=ms.float16)

    position_ids = model_kwargs.get('position_ids')
    if position_ids is None:
        position_ids = np.zeros((batch_size, input_ids.shape[-1]), dtype=np.int32)
    elif len(position_ids.shape) == 1:
        if model_kwargs.get('q_seq_lens') is None:
            position_ids = position_ids.reshape(batch_size, -1)
        else:
            position_ids = position_ids.reshape(-1)
            position_ids_pad = np.zeros((batch_size, pad_seq_len), dtype=position_ids.dtype)
            start = 0
            for i, q_seq_len in enumerate(model_kwargs['q_seq_lens']):
                end = start + q_seq_len
                position_ids_pad[i, :q_seq_len] = position_ids[start: end]
                start = end
            position_ids = position_ids_pad
    position_ids = _parallel_decoding_pad(position_ids, 1, pad_seq_len, 0)
    model_inputs['position_ids'] = Tensor(position_ids, ms.int32)

    pad_block_len = max_seq_len // config.block_size
    block_tables = _parallel_decoding_pad(block_tables, 1, pad_block_len, -1)
    block_tables = block_tables.astype(np.int32)

    if model_kwargs.get('q_seq_lens') is None:
        slot_mapping = _parallel_decoding_pad(slot_mapping, 0, batch_size * pad_seq_len, -1)
        slot_mapping = slot_mapping.astype(np.int32)
    else:
        slot_mapping_pad = np.full((batch_size * pad_seq_len), -1, dtype=slot_mapping.dtype)
        start = 0
        for i, q_seq_len in enumerate(model_kwargs['q_seq_lens']):
            end = start + q_seq_len
            slot_mapping_pad[i * pad_seq_len: i * pad_seq_len + q_seq_len] = slot_mapping[start: end]
            start = end
        slot_mapping = slot_mapping_pad

    if model_kwargs.get('q_seq_lens') is None:
        q_seq_lens = np.ones((batch_size,))
    else:
        q_seq_lens = model_kwargs['q_seq_lens']
    model_inputs['q_seq_lens'] = Tensor(q_seq_lens, dtype=ms.int32)
    return model_inputs, block_tables, slot_mapping


def parallel_decoding_control(config):
    if hasattr(config, "parallel_decoding"):
        return config.parallel_decoding in ("la", "memory_decoding")
    return False
