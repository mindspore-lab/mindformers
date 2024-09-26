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
"""utils"""

import mindspore as ms
from mindspore.communication.management import init

from mindformers.tools.logger import logger
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_world_size,
    initialize_model_parallel,
    destroy_model_parallel
)


# TODO: temporary api
def set_parallel_context(parallel_config):
    """
    Sets the parallel context based on the provided parallel configuration.

    Args:
        parallel_config (MindFormerConfig): The parallel configuration object containing the parallel settings.

    Returns:
        ModelParallelConfig: The updated parallel configuration object.

    """
    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size
    )
    logger.warning(
        f"dp {get_data_parallel_world_size()} | pp {parallel_config.pipeline_model_parallel_size} | "
        + f"tp {parallel_config.tensor_model_parallel_size} | sp {parallel_config.sequence_parallel}"
    )
    return parallel_config


# TODO: temporary api
def destroy_parallel_context():
    """
    Destroys the parallel context.

    Returns:
    None
    """
    destroy_model_parallel()


# TODO: temporary api
def set_seed(seed):
    """
    Set the seed for random number generation.

    Parameters:
    - seed (int): The seed value to set.

    Returns:
    None
    """
    # set global seed, np seed, and dataset seed
    ms.set_seed(seed)
    # set rng seed
    ms.manual_seed(seed)


def mark_only_lora_as_trainable(network):
    """mark only lora parameters as trainable"""
    for param in network.get_parameters():
        if 'lora' in param.name:
            param.requires_grad = True
        else:
            param.requires_grad = False
