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
""" Test ParamAndGradBuffer """
import argparse
import pytest

from mindspore.communication.management import init
from mindspore.communication import get_group_size, get_rank
# pylint: disable=W0401, W0614
from mindformers.experimental.distri_cores.create_comm import *


def run_initialize_and_destroy_model_parallel(order):
    """ run basic test for initialize and destroy comm groups """
    with pytest.raises(AssertionError):
        assert initialize_model_parallel(order=order)
    init()
    world_size = get_group_size()
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(tensor_model_parallel_size=2 * world_size, order=order)
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(pipeline_model_parallel_size=2 * world_size, order=order)
    with pytest.raises(RuntimeError):
        assert initialize_model_parallel(pipeline_model_parallel_size=world_size, \
                                         tensor_model_parallel_size=world_size, order=order)
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, order=order)

    assert get_tp_group() is not None
    assert get_pp_group() is not None
    assert get_model_parallel_group() is not None
    assert get_dp_group() is not None
    destroy_model_parallel()
    with pytest.raises(AssertionError):
        assert get_model_parallel_group()


def run_basic_world_size_rank_parallel_test(order):
    """ test world size and rank """
    init()
    world_size = get_group_size()
    rank = get_rank()

    # test tp
    initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert get_tp_world_size() == world_size
    assert get_tp_rank() == rank
    destroy_model_parallel()

    # test cp
    initialize_model_parallel(context_parallel_size=world_size, order=order)
    assert get_cp_world_size() == world_size
    assert get_cp_rank() == rank
    destroy_model_parallel()

    # test ep
    initialize_model_parallel(expert_model_parallel_size=world_size, order=order)
    assert get_data_modulo_expert_parallel_group() is not None
    assert get_dp_world_size() == world_size
    assert get_dp_rank() == rank
    assert get_ep_world_size() == world_size
    assert get_ep_rank() == rank
    assert get_tensor_and_expert_parallel_world_size() == world_size
    destroy_model_parallel()

    # test dp
    initialize_model_parallel(tensor_model_parallel_size=world_size, order=order)
    assert get_dp_world_size() == 1
    assert get_dp_rank() == 0
    destroy_model_parallel()

    # test pp + vpp
    vpp_size = 4
    initialize_model_parallel(pipeline_model_parallel_size=world_size,
                              virtual_pipeline_model_parallel_size=vpp_size,
                              order=order)
    assert get_pp_world_size() == world_size
    assert get_pp_rank() == rank
    assert get_vpp_world_size() == vpp_size
    set_vpp_rank(0)
    assert is_pipeline_first_stage() == (rank == 0 and get_vpp_rank() == 0)
    assert is_pipeline_first_stage(ignore_virtual=True) == (rank == 0)
    set_vpp_rank(vpp_size - 1)
    assert is_pipeline_last_stage() == (rank == world_size - 1 and get_vpp_rank() == vpp_size - 1)
    assert is_pipeline_last_stage(ignore_virtual=True) == (rank == world_size - 1)
    destroy_model_parallel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', required=True)

    args, rest_args = parser.parse_known_args()
    run_initialize_and_destroy_model_parallel(args.order)
    run_basic_world_size_rank_parallel_test(args.order)
