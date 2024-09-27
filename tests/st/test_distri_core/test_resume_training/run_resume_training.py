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
""" Test Mixtral. """
import argparse
import os

import numpy as np
from mindformers.experimental.parallel_core.pynative.config import init_configs_from_yaml
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_group,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
    get_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_rank,
    initialize_model_parallel,
)

from mindformers.experimental.parallel_core.pynative.training import get_model, TrainOneStepCell, train, get_loss_func
from mindformers.experimental.parallel_core.pynative.optimizer.optimizer import get_optimizer_param_scheduler
from mindformers.experimental.parallel_core.pynative.dist_checkpointing.checkpointing import load_checkpoint
from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits, TransformerLanguageModel
from mindformers.experimental.parallel_core.pynative.transformer.module import Module

from mindformers.tools.resume_ckpt import get_resume_checkpoint

import mindspore as ms
from mindspore import mint
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.communication import get_rank
from mindspore.mint.optim import AdamW
import mindspore.common.dtype as mstype


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, dataset_size=None, input_data=None, label_data=None):
        super().__init__()

        self.dataset_size = dataset_size
        self.input_data = input_data
        self.data_shape = self.input_data.shape
        self.label_data = label_data
        seq_length = self.data_shape[1]
        self.attention_mask = np.tril(np.ones(shape=(1, seq_length-1, seq_length-1))).astype(np.int32)
        self.attention_mask = self.attention_mask < 0.5

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))

    def __len__(self):
        return self.dataset_size

# pylint: disable=W0613
class MixtralModel(Module):
    r"""
    Mixtral Model
    Args:
        config (TransformerConfig): the config of network;
        num_tokentypes (int): if > 0, using tokentypes embedding. Default: 0;
        parallel_output: (bool), Specifies whether return paralleled output on each tensor parallel rank. Default: True;
        pre_process (bool) when using pipeline parallel, indicate whether it's the first stage. Default: True,
        post_process (bool) when using pipeline parallel, indicate whether it's the last stage. Default: True,
        loss_func: loss function

    Returns:
        output (Tensor): mixtral loss or hidden states

    Examples:
    ```python
    def model_provider_func(pre_process=True, post_process=True):
        ''' get mixtral model '''
        loss = get_loss_func(config.training_config)
        network = MixtralModel(
            model_config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
            )
        return network

    network = get_model(model_provider_func, parallel_config)
    ```
    """

    def __init__(
            self,
            config: TransformerConfig,
            num_tokentypes: int = 0,
            parallel_output: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
            loss_func=None,
            **kwargs):
        super(MixtralModel, self).__init__()
        self.config = config
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.pad_token = config.dataset_config.pad_token
        self.compute_dtype = config.compute_dtype
        self.share_embeddings_and_output_weights = not config.untie_embeddings_and_output_weights

        self.language_model = TransformerLanguageModel(
            config,
            encoder_attn_mask_type=None,
            num_tokentypes=num_tokentypes,
            pre_process=self.pre_process,
            post_process=self.post_process
            )
        if self.post_process:
            self.head = ParallelLMLogits(
                config=config,
                bias=False,
                compute_dtype=config.compute_dtype
                )

            self.loss = loss_func

        if self.share_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(
            self, input_ids: ms.Tensor,
            labels: ms.Tensor = None,
            attention_mask: ms.Tensor = None,
            tokentype_ids: ms.Tensor = None,
            inference_params: ms.Tensor = None,
            loss_mask: ms.Tensor = None):
        """
        Forward of mixtral model.

        Args:
            input_ids (Tensor): the tokenized inputs with datatype int32
            attention_mask (Tensor):
        Returns:
            output (Tensor): the output of mixtral decoderlayer
        """
                # ensure `input_ids` and `labels` shape are [bs, seq]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(dim=0)
        elif labels is None:
            labels = input_ids

        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        position_ids = None

        if loss_mask is None:
            if self.pad_token is None:
                raise RuntimeError("If 'pad_token' is not pass into model, the 'loss_mask' must be not None.")
            loss_mask = mint.ne(input_ids, self.pad_token).astype(self.compute_dtype)

        hidden_states = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            )
        # pylint: disable=R1705
        if self.post_process:
            if not self.share_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            # pylint: disable=E1121
            loss = self.post_language_model_processing(
                hidden_states,
                labels,
                logit_weights,
                loss_mask
                )
            return loss
        else:
            return hidden_states

    def post_language_model_processing(
            self,
            lm_output,
            labels,
            logit_weights,
            loss_mask):
        """define post language model process"""
        logits = self.head(lm_output, logit_weights, self.parallel_output)

        logits = logits.reshape(-1, logits.shape[-1]).to(mstype.float32)
        labels = labels.reshape(-1,).to(mstype.int32)

        loss = self.loss(logits, labels, loss_mask)

        return loss


def run_resume_training(config):
    """ Test ParallelTransformer. """
    print(f"config is:\n{config}")
    model_config = config.model_config
    parallel_config = config.parallel_config
    training_config = config.training_config
    optimizer_config = config.optimizer_config
    dataset_config = config.dataset_config

    tp = parallel_config.tensor_model_parallel_size
    ep = parallel_config.expert_model_parallel_size
    pp = parallel_config.pipeline_model_parallel_size
    vpp = parallel_config.virtual_pipeline_model_parallel_size
    vocab_size = model_config.vocab_size
    seq_length = model_config.seq_length
    if training_config.enable_compile_cache:
        print(f"compile_cache will be save to: {training_config.compile_cache_path}")
    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        max_device_memory="58GB",
        deterministic='ON',
        pynative_synchronize=True,
        enable_compile_cache=training_config.enable_compile_cache,
        compile_cache_path=training_config.compile_cache_path
        )

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        )

    dp_group = get_data_parallel_group()
    ep_group = get_expert_model_parallel_group()
    tp_group = get_tensor_model_parallel_group()
    pp_group = get_pipeline_model_parallel_group()
    dp_rank = get_data_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    print(f"dp_group is {dp_group}, ep_group is {ep_group}, tp_group is {tp_group}, pp_group is {pp_group}", flush=True)
    print(f"dp_rank is {dp_rank}, ep_rank is {ep_rank}, tp_rank is {tp_rank}, pp_rank is {pp_rank}", flush=True)

    ms.set_seed(2024)

    # making dataset
    input_data = np.random.randint(low=1, high=vocab_size, size=(10, seq_length+1), dtype=np.int32)
    dataset = TestData(dataset_size=10, input_data=input_data, label_data=input_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
    dataset = dataset.batch(1)

    # build net
    def model_provider_func(pre_process=True, post_process=True):
        """ get mixtral model """
        loss = get_loss_func(training_config)
        network = MixtralModel(
            model_config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
            )
        return network

    network = get_model(model_provider_func, training_config)

    print(f"network construct is:\n{network}")
    print("network parameters are:")
    for param in network.get_parameters():
        print(f"{param.name} {param.dtype} {param.shape}")

    optimizer = AdamW(params=network.trainable_params(), lr=1e-4)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer, optimizer_config, dataset_config, training_config)

    # load golden ckpt
    resume_dict = None
    if training_config.resume_training is True and os.path.exists(training_config.load_checkpoint):
        rank_path = os.path.join(training_config.load_checkpoint, f"rank_{get_rank()}")
        meta_path = os.path.join(rank_path, "meta.json")
        if not os.path.exists(meta_path):
            print(f"Could not find meta.json in directory {rank_path}, using latest ckpt in {rank_path}")
        resume_ckpt_name = get_resume_checkpoint(
            checkpoint_dir=training_config.load_checkpoint,
            resume_training=training_config.resume_training,
            resume_by_meta=True)
        print(f"resume_ckpt_name is {resume_ckpt_name}")
        if resume_ckpt_name is True:
            ckpt_path = training_config.load_checkpoint
        elif isinstance(resume_ckpt_name, str):
            ckpt_path = os.path.join(rank_path, resume_ckpt_name)
        print(f"ckpt_path is {ckpt_path}")
        resume_dict = load_checkpoint(model_config, network, optimizer=optimizer,
                                      opt_param_scheduler=opt_param_scheduler,
                                      ckpt_path=ckpt_path, format=training_config.ckpt_format)

    train_one_step_cell = TrainOneStepCell(network, optimizer, opt_param_scheduler, training_config, model_config)
    # train
    train(train_one_step_cell, dataset, training_config, resume_dict=resume_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config_mixtral_small.yaml", help="test config")
    parser.add_argument('--resume_training', action='store_true', help="resume training")
    parser.add_argument('--enable_compile_cache', action='store_true', help="enable compile cache")
    parser.add_argument('--compile_cache_path', type=str, default=None, help="where to save/load compile_cache")
    parser.add_argument('--crc_check', action='store_true', help="crc check")
    parser.add_argument('--output_dir', type=str, default="./output", help="dir to put log、ckpt and complie cache")
    parser.add_argument('--load_checkpoint', type=str, default="", help="where to load ckpt")
    parser.add_argument('--training_iters', type=int, default=10, help="training_iters")
    cli_args, rest_args = parser.parse_known_args()

    all_config = init_configs_from_yaml(cli_args.config_path)

    all_config.training_config.resume_training = cli_args.resume_training
    all_config.training_config.enable_compile_cache = cli_args.enable_compile_cache
    if cli_args.compile_cache_path is None:
        all_config.training_config.compile_cache_path = os.path.join(cli_args.output_dir, "compile_cache")
    else:
        all_config.training_config.compile_cache_path = cli_args.compile_cache_path
    all_config.training_config.crc_check = cli_args.crc_check
    all_config.training_config.output_dir = cli_args.output_dir
    all_config.training_config.load_checkpoint = cli_args.load_checkpoint
    all_config.training_config.training_iters = cli_args.training_iters

    run_resume_training(all_config)
