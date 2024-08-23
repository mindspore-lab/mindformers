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
"""run train apis"""
import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore.communication import init
from mindspore.train import Perplexity
from mindspore.communication.comm_func import all_gather_into_tensor

from mindformers.experimental.distri_cores.create_comm import (
    initialize_model_parallel,
    get_dp_group,
    get_dp_world_size,
    is_pipeline_last_stage,
    get_tp_world_size,
    get_tp_group,
)
from mindformers.experimental.distri_cores.training import TrainOneStepCell, train
from mindformers.experimental.distri_cores.tensor_parallel.layers import (
    ColumnParallelLinear,
)
from mindformers.experimental.distri_cores.config import (
    init_configs_from_yaml
)
from mindformers.experimental.distri_cores.transformer import ParallelTransformer
from mindformers.experimental.distri_cores.transformer.rotary_pos_embedding import RotaryEmbedding
from mindformers.experimental.distri_cores.loss_func import VocabParallelCrossEntropyLoss
from mindformers.experimental.distri_cores.optimizer import get_optimizer

from dataset import get_random_dataset


class TestNetWithLoss(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, model_config, loss):
        super(TestNetWithLoss, self).__init__()

        self.rope = RotaryEmbedding(kv_channels=model_config.hidden_size//model_config.num_heads,
                                    rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=model_config, post_norm=True)
        self.head = ColumnParallelLinear(
            config=model_config,
            input_size=model_config.hidden_size,
            output_size=model_config.vocab_size,
            skip_weight_param_allocation=False,
            bias=False,
            gather_output=False,
            compute_dtype=model_config.compute_dtype,
            init_method=model_config.init_method,
            bias_init=model_config.bias_init
        )
        self.loss = loss

    def construct(self, hidden_states, attention_mask, labels):
        """ construct. """
        emb = self.rope(max_seq_len=hidden_states.shape[1])
        hidden_states = self.transformer(hidden_states, attention_mask, rotary_pos_emb=emb)
        logits, _ = self.head(hidden_states)

        # flatten the logits and labels
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        # mean reduction
        loss = self.loss(logits, labels).mean()
        return loss, logits


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

    # only calculate metrics on last stage
    if is_pipeline_last_stage():
        for metric in metrics.values():
            metric.clear()
        for data in val_dataset_iterator.create_dict_iterator():
            _, logits = train_one_step_cell.forward_backward_func(forward_only=True, **data)
            labels = data["labels"]
            # ensure logits and labels are flattened
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.shape[-1])
            if get_tp_world_size() > 1:
                logits = logits.swapaxes(0, 1)
                logits = all_gather_into_tensor(logits, get_tp_group())
                logits = logits.swapaxes(0, 1)
            if get_dp_world_size() > 1:
                dp_group = get_dp_group()
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


def run_train(config_path):
    """run train process"""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

    all_config = init_configs_from_yaml(config_path)
    training_config = all_config.training_config
    model_config = all_config.model_config
    optimizer_config = all_config.optimizer_config
    parallel_config = all_config.parallel_config
    dataset_config = all_config.dataset_config

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_parallel,
    )

    ms.set_seed(training_config.seed)
    ms.manual_seed(training_config.seed)

    loss = VocabParallelCrossEntropyLoss()
    network_with_loss = TestNetWithLoss(model_config, loss)

    optimizer = get_optimizer(optimizer_config, network_with_loss.trainable_params(), network_with_loss)

    train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, training_config, model_config)

    train_dataset_iterator, val_dataset_iterator = get_random_dataset(
        dataset_config, model_config, training_config.training_iters, training_config.seed
    )

    # 7. instantiate metrics
    metrics = {
        "perplexity": Perplexity(ignore_label=dataset_config.pad_token),
    }

    train(train_one_step_cell, train_dataset_iterator, training_config, val_dataset_iterator, metrics, evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DistriCore train')
    parser.add_argument('--config_path', type=str, default="config.yaml", help='config file path')
    args = parser.parse_args()
    run_train(args.config_path)
