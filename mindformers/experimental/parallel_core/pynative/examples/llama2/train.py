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
"""Train llama2 model."""

import argparse

import mindspore as ms
from mindspore.common.api import _no_grad
from mindspore.train import Perplexity
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspore.communication import init

from mindformers.tools import logger
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml
)
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell, train, get_model
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    is_pipeline_last_stage,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from mindformers.experimental.parallel_core.models import Llama2Model
from mindformers.experimental.parallel_core.pynative.parallel_policy_shard.policy_search import search_parallel_policy

from dataset import get_dataset
from utils import set_parallel_context, set_seed


def get_arg_parser():
    """get argument parser"""
    parser = argparse.ArgumentParser(description="Train llama2 model")
    parser.add_argument("--config_path", type=str, default="pretrain_llama2.yaml", help="The path to the config file.")
    parser.add_argument("--run_cmd", type=str, default=None, help="running cmd.")
    parser.add_argument("--model_type", type=str, default="model_config", help="Input model config.")
    return parser

def search_policy(train_config, model_config, optimizer_config):
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()
    search_parallel_policy(train_config, model_config, optimizer_config)

# pylint: disable=W0613
def evaluation(train_one_step_cell, val_dataset_iterator, metrics, **kwargs):
    """
    Evaluates a neural network model using the given dataset.

    Args:
        network_with_loss (nn.Cell): The neural network model with loss function,
            which will output the loss and model final output
        val_dataset_iterator (DatasetIterator): The validation dataset iterator.
        metrics (Dict[str, Metric]): The metrics to evaluate the model.


    Returns:
        Dict[str, Tensor]: The evaluation results.
    """

    with _no_grad():
        results = {}

        train_one_step_cell.set_train(False)
        data_layout = train_one_step_cell.network_with_loss.config.dataset_config.data_layout
        # only calculate metrics on last stage
        if is_pipeline_last_stage():
            if get_data_parallel_world_size() > 1:
                dp_group = get_data_parallel_group()
            if get_tensor_model_parallel_world_size() > 1:
                tp_group = get_tensor_model_parallel_group()

            for metric in metrics.values():
                metric.clear()
            for data in val_dataset_iterator.create_dict_iterator():
                _, logits = train_one_step_cell.forward_backward_func(forward_only=True, **data)
                labels = data["labels"]
                # ensure logits and labels are flattened
                if data_layout == "SBH":
                    logits = logits.swapaxes(0, 1)
                labels = labels.reshape(-1)
                logits = logits.reshape(-1, logits.shape[-1])
                # concat at sequence dimension
                if get_tensor_model_parallel_world_size() > 1:
                    logits = logits.swapaxes(0, 1)
                    logits = all_gather_into_tensor(logits, tp_group)
                    logits = logits.swapaxes(0, 1)
                if get_data_parallel_world_size() > 1:
                    logits = all_gather_into_tensor(logits, dp_group)
                    labels = all_gather_into_tensor(labels, dp_group)
                for metric in metrics.values():
                    metric.update(logits, labels)
            logger.info("Validation Results:")
            for metric_name, metric in metrics.items():
                metric_value = metric.eval()
                results[metric_name] = metric_value
                logger.info(f"{metric_name}: {metric_value}")
        else:
            for data in val_dataset_iterator:
                _ = train_one_step_cell.forward_backward_func(forward_only=True, *data)

        train_one_step_cell.set_train(True)

        return results


def main():
    """main function"""
    parser = get_arg_parser()
    args = parser.parse_args()

    all_config = init_configs_from_yaml(args.config_path)

    training_config = all_config.training_config
    model_config = all_config.llama2_config
    optimizer_config = all_config.optimizer_config
    parallel_config = all_config.parallel_config
    dataset_config = all_config.dataset_config

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    set_parallel_context(parallel_config)
    set_seed(training_config.seed)

    def model_provider_func(pre_process=True, post_process=True):
        """get llama2 model"""
        network_with_loss = Llama2Model(
            model_config, pre_process=pre_process, post_process=post_process
        )
        return network_with_loss

    network_with_loss = get_model(model_provider_func, training_config)

    optimizer = get_optimizer(
        optimizer_config,
        training_config,
        network_with_loss.trainable_params(),
        network_with_loss,
        grad_allreduce_op=training_config.loss_reduction
    )

    train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, None, training_config, model_config)

    train_dataset_iterator, val_dataset_iterator = get_dataset(dataset_config, parallel_config)
    metrics = {
        "perplexity": Perplexity(),
    }
    if training_config.search_parallel:
        search_policy(training_config, model_config, optimizer_config)
    else:
        train(
            train_one_step_cell,
            train_dataset_iterator,
            training_config,
            val_dataset_iterator,
            metrics,
            evaluation,
        )


if __name__ == "__main__":
    main()
