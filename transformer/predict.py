# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
Basic model predict/evaluation script
"""
import argparse
import numpy as np

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net

from transformer.data import build_dataset
from transformer.models import build_model
from transformer.build_parallel_config import build_parallel_config
from transformer.tokenization import tokenization
from transformer.tokenization.tokenization import FullTokenizer
from transformer.utils import parse_with_config, _convert_dtype_class
from transformer.logger import get_logger
from transformer.generate import generate


def set_context_env(config):
    """Set the context env"""
    context_args = config.context
    if context_args['device_target'] != "GPU":
        context_args['enable_graph_kernel'] = False
        config.logger.info("Disable graph kernel.")
    context.set_context(**context_args)


def check_args(opt, device_num):
    """Validate the dp and mp"""
    dp = opt.parallel_config['data_parallel']
    mp = opt.parallel_config['model_parallel']
    if mp < 1:
        raise ValueError("The model parallel must be equal or larger than 1. "
                         f"You can fix this by setting --model_parallel=1, for example.")
    if mp > device_num:
        raise ValueError(f"The model parallel must be less or equal to the device_num {device_num}. "
                         f"You can fix this by setting --model_parallel=1, for example")
    if opt.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL) and dp * mp != device_num:
        opt.logger.warn(f"The data_parallel * model_parallel must be equal to the {device_num}. "
                        f"You can remove this warning by setting --data_parallel={device_num // mp}. "
                        f"Now the full_batch will be set False.")
        opt.full_batch = False

    # If the user runs the data_parallel and set full_batch to be true
    if opt.parallel_mode in (ParallelMode.DATA_PARALLEL,) and opt.full_batch:
        raise ValueError("full_batch doesn't support DATA_PARALLEL mode, you can fix it by setting --full_batch=False")


def modify_args(opt):
    # maps fp16 to mstype.float16 and fp32 to mstype.float32
    for k, v in opt.__dict__.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                v[sub_k] = _convert_dtype_class(sub_v)
        else:
            opt.__dict__[k] = _convert_dtype_class(v)


def set_auto_parallel_context_env(config):
    """Set the auto parallel env"""
    if config.parallel_mode != context.ParallelMode.STAND_ALONE:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for multi card training.")
        D.init()
        device_num = D.get_group_size()
        rank_id = D.get_rank()
        context.reset_auto_parallel_context()
        check_args(config, device_num)
        context.set_auto_parallel_context(parallel_mode=config.parallel_mode, gradients_mean=True,
                                          full_batch=config.full_batch,
                                          device_num=device_num, grad_accumulation_step=config.acc_step)

    else:
        config.logger.info(f"Enabling the parallel mode: {config.parallel_mode} for stand alone training.")
        rank_id = 0
        device_num = 1
    if config.full_batch:
        config.logger.info("Enabling the full batch import.")
    return rank_id, device_num


def get_acc(model, dataset):
    """ calculate accuracy for input dataset """
    total_num = 0
    acc_num = 0
    for data in dataset:
        input_ids = data[0].asnumpy().astype(np.int32)
        input_mask = data[1].asnumpy().astype(np.int32)
        label = data[2].asnumpy().astype(np.int32)
        logits = model.predict(Tensor(input_ids, mstype.int32), Tensor(input_mask, mstype.float32)).asnumpy()

        equals = label.reshape(-1) == logits
        total_num += np.prod(label.shape)
        acc_num += sum(equals)

    acc = acc_num / total_num
    return acc


def generate_words(sample, predict_model, opt):
    """
    Generate the word given the input prompt, model and configs

    Args:
        sample(str): The input prompt. For example, it can be "Today is a good day, I want to".
        predict_model(Model): the model that need to run prediction.
        opt(argparse.Namespace): The global configs.

    Inputs:
        input_ids: the tokenized inputs
        attention_mask: the attention mask with [bs, seq_length]. 1 means effective and 0 mean it should be masked.
        labels:
    Returns:
        output: Tensor, the loss of the network
    """
    # Tokenize input sentence to ids
    eval_opts = opt
    tokenizer = FullTokenizer(eval_opts.vocab_path)
    tokens = tokenizer.tokenize(sample)
    input_ids = tokenization.convert_tokens_to_ids(vocab_file=eval_opts.vocab_path,
                                                   tokens=tokens)
    input_ids = np.array(input_ids).reshape(1, -1)
    # eval ops
    output_ids = generate(predict_model,
                          end_token=2,  # For opt model, the end_token is 2
                          origin_inputs=input_ids,
                          model_origin_max_length=eval_opts.model['seq_length'],
                          max_generate_length=eval_opts.model['seq_length'],
                          vocab_size=eval_opts.model["vocab_size"],
                          config=eval_opts)
    # Decode output ids to sentence
    output_samples = tokenization.convert_ids_to_tokens(vocab_file=eval_opts.vocab_path,
                                                        ids=output_ids.tolist())
    output_string = tokenization.convert_tokens_to_string(output_samples)
    print('Output is:', output_string, flush=True)


def run_predict(opt):
    """Main Prediction process"""
    set_context_env(opt)
    rank_id, device_num = set_auto_parallel_context_env(opt)
    parallel_config = build_parallel_config(opt)

    eval_net = build_model(opt, parallel_config)

    opt.logger.info(f"Start to restore from the path {opt.load_checkpoint_path}")
    ckpt = load_checkpoint(opt.load_checkpoint_path)
    load_param_into_net(eval_net, ckpt)

    model = Model(eval_net)

    if opt.generate:
        opt.logger.info("Start to generate the words:")
        generate_words(sample='Hello world!',
                       predict_model=model,
                       opt=opt)
    else:
        opt.logger.info("Start to eval on the datasets.")
        ds = build_dataset(opt, rank_id, device_num, get_eval_dataset=True)
        acc = get_acc(model, ds.create_tuple_iterator())

        opt.logger.info(f"The accuracy is {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/gpt/gpt_base.yaml", help='YAML config files')
    args = parse_with_config(parser)
    args.logger = get_logger()
    modify_args(args)
    set_seed(args.seed)
    args.eval = True
    run_predict(args)
