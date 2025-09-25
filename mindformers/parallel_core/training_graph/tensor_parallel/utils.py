# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
""" utils """

import hashlib
from typing import Sequence
from mindspore.communication import create_group

_TP_GROUP_NAME = {}


def get_tp_group_name(rank_id, tensor_model_parallel_size):
    """
    Generates a unique group name for a set of ranks involved in tensor parallel
    and creates a communication group with this name.
    """
    rank_start = rank_id // tensor_model_parallel_size * tensor_model_parallel_size
    rand_end = rank_start + tensor_model_parallel_size
    rank_list = [i for i in range(rank_start, rand_end)]

    rank_list_str = "-".join([str(i) for i in rank_list])
    if rank_list_str in _TP_GROUP_NAME:
        return _TP_GROUP_NAME[rank_list_str]

    hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
    tp_group_name = str(hashed)
    create_group(tp_group_name, rank_list)
    _TP_GROUP_NAME[rank_list_str] = tp_group_name
    return tp_group_name


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError(
            f"numerator({numerator}) is not divisible by denominator({denominator})."
        )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """ Split the vocabulary into `world_size` chunks and return the first
        and last index of the vocabulary belonging to the `rank`
        partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
