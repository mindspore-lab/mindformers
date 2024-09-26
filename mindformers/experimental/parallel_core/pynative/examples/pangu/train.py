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
"""train and evaluation script for Pangu model."""

import argparse
import os

import mindspore as ms
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspore.communication.management import get_rank
from mindspore.train import Perplexity

from dataset import get_dataset
from mindformers.experimental.parallel_core.models import PanguModel
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    OptimizerConfig,
    DatasetConfig,
    LoraConfig
)
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_tensor_model_parallel_world_size,
    is_pipeline_last_stage
)
from mindformers.experimental.parallel_core.pynative.tensor_parallel import GatherFromModelParallelRegion
from mindformers.experimental.parallel_core.pynative.training import get_loss_func, VocabParallelCrossEntropy, \
    TrainOneStepCell, train, get_model
from mindformers.experimental.parallel_core.pynative.utils import valid_lora_config
from pangu_model import PanguAlphaWithHead, ModelWithLossCell
from pangu_model_config import PanguConfig
from user_defined import set_weight_decay
from utils import set_parallel_context, set_seed, mark_only_lora_as_trainable


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
    results = {}

    train_one_step_cell.set_train(False)

    data_layout = \
        train_one_step_cell.network_with_loss.network.backbone.config.dataset_config.data_layout

    # only calculate metrics on last stage
    if is_pipeline_last_stage():
        for metric in metrics.values():
            metric.clear()
        for data in val_dataset_iterator.create_dict_iterator():
            _, logits = train_one_step_cell.forward_backward_func(forward_only=True, **data)
            input_ids = data["input_ids"]
            labels = input_ids[:, 1:]
            # ensure logits and labels are flattened
            if data_layout == "SBH":
                logits = logits.swapaxes(0, 1)
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])
            if get_tensor_model_parallel_world_size() > 1 and isinstance(
                    train_one_step_cell.network_with_loss.loss_fn.loss_func,
                    VocabParallelCrossEntropy):
                gather_from_mp_region = GatherFromModelParallelRegion()
                logits = gather_from_mp_region(logits)
            if get_data_parallel_world_size() > 1:
                dp_group = get_data_parallel_group()
                logits = all_gather_into_tensor(logits, dp_group)
                labels = all_gather_into_tensor(labels, dp_group)
            for metric in metrics.values():
                metric.update(logits, labels)
        print("Validation Results:")
        for metric_name, metric in metrics.items():
            metric_value = metric.eval()
            results[metric_name] = metric_value
            print(f"{metric_name}: {metric_value}", flush=True)
    else:
        for data in val_dataset_iterator:
            _ = train_one_step_cell.forward_backward_func(forward_only=True, *data)

    train_one_step_cell.set_train(True)

    return results


# pylint: disable=W0613
def main(model_type="ori"):
    # 1. load config
    config_path = "train_pangu.yaml"
    training_config, parallel_config, optimizer_config, dataset_config, lora_config, model_config = (
        init_configs_from_yaml(
            config_path, [TrainingConfig, ModelParallelConfig, OptimizerConfig, DatasetConfig,
                          LoraConfig, PanguConfig]
        )
    )

    # 2. init context
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    set_parallel_context(parallel_config)
    # TODO: wait for the official release of set_rng_seed
    # set_rng_seed(training_config.seed)
    set_seed(training_config.seed)

    if lora_config.use_lora:
        ckpt_dir = lora_config.checkpoint_dir
        ckpt_name = lora_config.checkpoint_prefix + f'{get_rank()}.ckpt'
        pretrain_params = ms.load_checkpoint(os.path.join(ckpt_dir, ckpt_name))
        model_config = valid_lora_config(model_config, pretrain_params)

    # 3. instantiate network and loss
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        if model_type == "ori":
            # include recompute, checkpoint load, parallel initialize
            network = PanguAlphaWithHead(model_config)
            if lora_config.use_lora:
                ms.load_param_into_net(network, pretrain_params)
                mark_only_lora_as_trainable(network)

            # load_checkpoint(model_config.ckpt_path, network)
            loss = get_loss_func(training_config)
            network_with_loss = ModelWithLossCell(network, loss, pad_token=dataset_config.eod_id)
        else:
            network_with_loss = PanguModel(model_config, pre_process=pre_process, post_process=post_process)
        return network_with_loss

    network_with_loss = get_model(model_provider_func, training_config)

    # 4. instantiate optimizer
    model_params = network_with_loss.trainable_params()
    group_params = set_weight_decay(model_params, optimizer_config.weight_decay)
    optimizer = get_optimizer(optimizer_config, group_params, network_with_loss,
                              grad_allreduce_op=training_config.loss_reduction)

    # 5. instantiate train_one_step_cell
    train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, None, training_config, model_config)

    # 6. instantiate dataset iterator
    train_dataset_iterator, val_dataset_iterator = get_dataset(dataset_config)

    # 7. instantiate metrics
    metrics = {
        "perplexity": Perplexity(ignore_label=dataset_config.eod_id),
    }

    # 8. train
    train(train_one_step_cell, train_dataset_iterator, training_config, val_dataset_iterator, metrics, evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="ori",
                        help="use new pangu model or ori model.")
    args = parser.parse_args()
    main(args.model_type)
