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
"""some utility functions"""
from typing import List, Dict


import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import DistributedGradReducer

from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_dp_group,
    get_dp_world_size,
)


def train(epoch_num,
          dataset,
          network,
          optimizer,
          save_ckpt_path=None,
          with_attn_input=False,
          reduce_grad=True,
          zero_level=-1,
          ):
    """
    define a train process
    """
    network.set_train()
    grad_func = ops.value_and_grad(
        network, grad_position=None, weights=optimizer.parameters
    )
    if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
            and get_dp_world_size(with_context_parallel=True) > 1):
        grad_reducer = DistributedGradReducer(
            optimizer.parameters,
            group=get_dp_group(with_context_parallel=True),
            mean=True,
            degree=get_dp_world_size(with_context_parallel=True),
        )
    all_loss = []
    for epoch in range(epoch_num):
        step = 0
        for data in dataset:
            if with_attn_input:
                input_ids, labels, attn_mask = data
                loss, grads = grad_func(input_ids, attn_mask, labels)
            else:
                input_ids, labels = data
                loss, grads = grad_func(input_ids, labels)
            if (reduce_grad and ms.get_context("mode") == ms.PYNATIVE_MODE
                    and get_dp_world_size(with_context_parallel=True) > 1):
                if zero_level < 0:
                    print(
                        "reduce gradients on group {}".format(
                            get_dp_group(with_context_parallel=True)
                        )
                    )
                    grads = grad_reducer(grads)
            loss = ops.depend(loss, optimizer(grads))
            print("Epoch {}, step {}, loss {}".format(epoch, step, loss))
            step += 1
            all_loss.append(loss.asnumpy())

    if save_ckpt_path is not None:
        ms.save_checkpoint(network, save_ckpt_path)
    return all_loss


def transform_moe_golden_params_to_pynative_params(
        golden_params: Dict[str, Tensor],
        local_expert_indices: List[int]
        ):
    """
    transform golden moe params to pynative params
    map_dict = {"w1": "gating",
                "w2": "projection",
                "w3": "mapping"}
    """

    print(f"golden to pynative ckpt map:")
    pynative_params = {}
    for name, param in golden_params.items():
        if "feed_forward.ffn" in name:
            for i, local_expert_id in enumerate(local_expert_indices):
                if "w1" in name:  # w1 -> gating
                    pynative_name = f"moe.experts.local_experts.{i}.gating.weight"
                if "w2" in name:  # w2 -> projection
                    pynative_name = f"moe.experts.local_experts.{i}.projection.weight"
                if "w3" in name:  # w3 -> mapping
                    pynative_name = f"moe.experts.local_experts.{i}.mapping.weight"
                if len(param.shape) == 3:
                    pynative_params[pynative_name] = ms.Parameter(param[local_expert_id])
                    print(f"{name} {param[local_expert_id].shape} -> " + \
                          f"{pynative_name} {pynative_params[pynative_name].shape}", flush=True)
                elif len(param.shape) == 2:
                    pynative_params[pynative_name] = ms.Parameter(param)
                    print(f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}", flush=True)
        if "feed_forward.router.dense.weight" in name:
            pynative_name = "moe.router.gating.weight"
            pynative_params[pynative_name] = ms.Parameter(param)
            print(
                f"{name} {param.shape} -> {pynative_name} {pynative_params[pynative_name].shape}",
                flush=True,
            )

    return pynative_params
