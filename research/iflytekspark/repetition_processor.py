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
"""Repetition processor."""
import inspect
from mindformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class X1LogitsProcessorList(LogitsProcessorList):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently
    process a `scores` input tensor. This class inherits from list and adds a specific *__call__* method
    to apply each [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def process(self, i, input_ids, scores, **kwargs):
        """apply process"""
        input_ids = input_ids[i:i+1]
        scores_i = scores[i:i+1]
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                penalty_cache = kwargs["repetition_penalty_cache"]
                penalty_cache_i = penalty_cache[i: i + 1]
                scores_i = processor(input_ids, scores_i, penalty_cache_i)
            else:
                scores_i = processor(input_ids, scores_i)
        scores[i] = scores_i


class RepetitionPenaltyIncreaseProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty.
    """

    def __init__(self, repetition_penalty: float):
        repetition_penalty = float(repetition_penalty)
        if repetition_penalty <= 0:
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {repetition_penalty}"
            )

        self.penalty = repetition_penalty

    def __call__(self, input_ids, scores, repetition_penalty_cache):
        if self.penalty != 1.0:
            scores -= repetition_penalty_cache
        return scores
