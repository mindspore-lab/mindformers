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
"""support recomputation"""

from enum import IntEnum

from mindformers.tools.logger import logger

TYPE = IntEnum('RecomputeType', ['NONE', 'SLCT', 'COMM', 'FULL'], start=0)

COEF = {TYPE.NONE: 0, TYPE.COMM: 0.25, TYPE.SLCT: 0.25, TYPE.FULL: 0.5}

YAML_NAME = {
    TYPE.NONE: "", TYPE.COMM: "select_comm_recompute", TYPE.SLCT: "select_recompute",
    TYPE.FULL: "recompute"
}

JSON_MEMORY_NAME = {
    TYPE.NONE: "memory_activation", TYPE.COMM: "memory_select_comm",
    TYPE.SLCT: "memory_select_rec", TYPE.FULL: "memory_recompute"
}

JSON_TIME_NAME = {
    TYPE.NONE: "backward_time", TYPE.COMM: "select_comm_time",
    TYPE.SLCT: "select_rec_time", TYPE.FULL: "recompute_time "
}

JSON_COEF_NAME = {
    TYPE.NONE: "backward_coef", TYPE.SLCT: "select_rec_coef",
    TYPE.COMM: "select_comm_coef", TYPE.FULL: "recompute_coef"
}


def sums(rec_dict):
    x = 0
    for r in TYPE:
        x += rec_dict[r]
    return x


def to_list(rec_dict):
    return list(rec_dict.values())


def right_extend(ll, n):
    all_l = []
    for i in range(n):
        for sublist in ll:
            all_l += [sublist + [i]]
    return all_l


def make_all_indexes_local(used_rec, num_of_interleave, all_indexes: list[list[int]], r: TYPE):
    """make indexes of used recomputation based on the number of interleaves"""
    if r >= len(TYPE) - 1:
        if used_rec[r]:
            all_indexes = right_extend(all_indexes, num_of_interleave)
        return all_indexes
    if used_rec[r]:
        return make_all_indexes_local(used_rec, num_of_interleave,
                                      right_extend(all_indexes, num_of_interleave), TYPE(r + 1))
    return make_all_indexes_local(used_rec, num_of_interleave, all_indexes, TYPE(r + 1))


def make_all_indexes(used_rec, num_of_interleave):
    return make_all_indexes_local(used_rec, num_of_interleave, [[]], TYPE.NONE)


def recomputes_from_indexes(used_rec, indexes):
    recomputes = []
    for idx in indexes:
        recompute = {r: None for r in TYPE}
        for r in TYPE:
            if used_rec[r]:
                recompute[r] = idx[0]
                idx.pop(0)
        recomputes.append(recompute)
    return recomputes


def average(rec_list):
    """return the average number of each type of  recomputation"""
    num = len(rec_list)
    if num == 0:
        return rec_list
    rec_1 = rec_list.pop(0)
    for rec_i in rec_list:
        for r in TYPE:
            if rec_1[r] is not None and rec_i[r] is not None:
                rec_1[r] = rec_1[r] + rec_i[r]
            elif not (rec_1[r] is None and rec_i[r] is None):
                logger.warning("WARNING: Recomputation %s is not taken ",
                               "into consideration by all body layers", r.name)
    for r in TYPE:
        if rec_1[r] is not None:
            rec_1[r] = rec_1[r] / num
    return rec_1


def assign_used(values, unused_rec):
    """find the assignment of each recomputation"""
    assignment = {r: None for r in TYPE}
    value_idx = 0
    for r in TYPE:
        if r not in unused_rec:
            assignment[r] = values[value_idx]
            value_idx += 1
    return assignment


def get_used_list(recompute_considered) -> list[TYPE]:
    """get a list of which recomputation is used"""
    used_rec = []
    for rec in TYPE:
        if recompute_considered[rec]:
            used_rec.append(rec)
    return used_rec


def get_unused_list(recompute_considered) -> list[TYPE]:
    used_rec = []
    for rec in TYPE:
        if not recompute_considered[rec]:
            used_rec.append(rec)
    return used_rec


def least_recomputed(recompute_considered):
    rec = TYPE.NONE
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
            break
    return rec


def most_recomputed(recompute_considered):
    rec = TYPE.FULL
    for r in TYPE:
        if recompute_considered[r]:
            rec = r
    return rec
