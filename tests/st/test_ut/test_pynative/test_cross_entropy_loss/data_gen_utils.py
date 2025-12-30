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
"""Data generation utilities for pynative CrossEntropyLoss tests with random data"""
import numpy as np

import mindspore as ms

from mindformers.parallel_core.training_graph.loss_func import CrossEntropyLoss
from mindformers.parallel_core.transformer_config import default_transformer_config

def get_init_params(batch_size, seq_length, vocab_size):
    """
    Generates random initial parameters (inputs) for VocabParallelCrossEntropy.
    """
    np.random.seed(42)

    logits_shape = (batch_size * seq_length, vocab_size)
    logits = 0.01 * np.random.randn(*logits_shape).astype(np.float32)

    target_shape = (batch_size * seq_length,)
    target = np.random.randint(0, vocab_size, size=target_shape).astype(np.int32)
    input_mask = np.random.randint(0, 2, size=target_shape).astype(np.float32)

    if np.sum(input_mask) == 0 and input_mask.size > 0:
        input_mask[0] = 1.0

    return {
        "logits": logits,
        "target": target,
        "input_mask": input_mask,
    }


def get_static_output(logits, target, input_mask):
    """get GRAPH_MODE CELoss result"""
    ms.set_context(mode=0)
    config = default_transformer_config
    config.calculate_per_token_loss = True
    net = CrossEntropyLoss(config)
    grad_fn = ms.value_and_grad(net, grad_position=0)
    result, grad = grad_fn(logits, target, input_mask)
    numerator, denominator = result
    loss = numerator / denominator
    return {
        "numerator": numerator,
        "denominator": denominator,
        "loss": loss,
        "grad": grad
    }


def get_cpu_output(logits, target, input_mask):
    """get cpu (numpy) CELoss result"""
    # forward
    logit_max = np.max(logits, 1, keepdims=True)
    logit_sub = logits - logit_max
    logit_exp = np.exp(logit_sub)
    exp_sum = np.sum(logit_exp, -1, keepdims=True)
    log_exp_sum = np.log(exp_sum)
    logit_neg_logsoftmax = log_exp_sum - logit_sub
    loss_reduce = logit_neg_logsoftmax[np.arange(logits.shape[0]), target]
    numerator = (loss_reduce * input_mask).sum()
    denominator = input_mask.sum() + 1.e-8
    loss = numerator / denominator
    # backward
    dout_reduce = input_mask / input_mask.sum()
    logits_softmax = logit_exp / exp_sum
    logits_softmax[np.arange(logits.shape[0]), target] -= 1
    grad = logits_softmax * dout_reduce.reshape(-1, 1)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "loss": loss,
        "grad": grad
    }
