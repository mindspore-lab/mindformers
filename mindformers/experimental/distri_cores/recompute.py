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
"""Recompute Cell Overloaded and Graident Checkpointed Wrapper"""

from collections import OrderedDict
from mindspore.common._register_for_recompute import recompute_registry
from mindspore.common.api import _pynative_executor
from mindspore.common.recompute import (
    _RecomputeCell,
    _check_input_args_validate,
    _padding_input_grads,
)
from mindformers.experimental.distri_cores.random import get_rng_tracer, get_rng_state, set_rng_state
from mindformers.experimental.distri_cores.transformer.module import Module

__all__ = ["CheckpointedRecomputeOrientedCell"]


class _RecomputeCellWithRngPersistent(_RecomputeCell):
    """
    Recompute cell considering rng state tracer.
    Note:
     - RecomputeCell now only support pynative mode.
     - When use recompute function, block object should not decorated by @jit.
    """

    def construct(self, *args, **kwargs):
        _check_input_args_validate(self.net, args)
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.save_rng_state = kwargs.pop("save_rng_state", True)
        if self.save_rng_state:
            self.cpu_rng_state = get_rng_state()
            self.rng_tracer_state = get_rng_tracer().get_state()
        return self.net(*args, **kwargs)

    def bprop(self, *args):
        """
        Custom grad method for recompute
        :param args:
        :return: input grad and weight grads
        """
        grad_input = args[-1]
        input_args = self.args[-1]
        kwargs = self.kwargs[-1]
        self.args.pop()
        self.kwargs.pop()
        if kwargs:
            input_args = list(input_args) + list(kwargs.values())
        try:
            pre_rng_state = get_rng_state()
            pre_rng_tracer_state = get_rng_tracer().get_state()
            set_rng_state(self.cpu_rng_state)
            get_rng_tracer().set_state(self.rng_tracer_state)
            _pynative_executor.set_is_run_recompute(True)
            grads = self.grad(self.net, self.internal_params)(*input_args, grad_input)
            _pynative_executor.set_is_run_recompute(False)
            set_rng_state(pre_rng_state)
            get_rng_tracer().set_state(pre_rng_tracer_state)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err
        weights = OrderedDict()
        input_grads = list(grads[0])
        _padding_input_grads(input_args, input_grads)
        for i, param in enumerate(self.internal_params):
            weights[param] = grads[1][i]
        return tuple(input_grads), weights


def recompute_generator(block):
    """
    generator of recompute object.
    :param block:
    :return:
    """
    return _RecomputeCellWithRngPersistent(block)


recompute_registry.register(recompute_generator)


class CheckpointedRecomputeOrientedCell(Module):
    r"""
    Checkpointed recompute layer group for ParallelTransformerLayers.

    Args:
        layers (nn.CellList): CellList of ParallelTransformerLayer(s).

    Inputs:
        - **hidden_states** (Tensor) - Hidden states. Shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Attention mask. Shape :math:`(B, N, S, S)`.
        - **rotary_pos_emb** (Tensor, optional) - Rotary position embedding. Shape math:`(S, H)`. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
    """

    def __init__(self, layers):
        super(CheckpointedRecomputeOrientedCell, self).__init__()
        self.layers = layers

    def construct(self, hidden_states, attention_mask, rotary_pos_emb=None):
        """Construct function of recompute layer group."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)
        return hidden_states
