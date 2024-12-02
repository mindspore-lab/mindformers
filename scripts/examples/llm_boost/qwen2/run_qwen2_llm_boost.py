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

"""Qwen task script"""

import argparse
import os

from mindspore.communication import get_rank
from mindformers import (
    MindFormerConfig,
    logger,
)
from mindformers.core.context import build_context
from mindformers.trainer.utils import load_ckpt
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import str2bool
from research.llm_boost.llm_boost import LlmBoostForCausalLM
from research.llm_boost.llm_boost import LlmBoostConfig
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from research.llm_boost.utils import generate_state_dict, save_strategy_file


# pylint: disable=C0330
def main(
    config="predict_qwen2_72b_instruct.yaml",
    use_parallel=None,
    load_checkpoint=None,
    only_save_strategy=None,
    vocab_file=None,
    merges_file=None,
    seq_length=None,
    max_length=8192,
    device_id=None,
    do_sample=None,
    top_k=None,
    top_p=None,
    batch_size=1,
    repetition_penalty=None,
    device_num=None,
):
    """main function."""
    inputs = [
        "帮我制定一份去上海的旅游攻略",
        "描述一下鲁智深倒拔垂杨柳的场面",
        "描述一下孙悟空大闹天宫的场面",
        "我喜欢看电影，因为",
    ]

    yaml_path = os.path.expanduser(config)
    if not os.path.exists(yaml_path):
        raise ValueError("The yaml_path should exist.")

    config = MindFormerConfig(os.path.realpath(yaml_path))
    if vocab_file:
        if not os.path.exists(vocab_file):
            raise ValueError("The vocab_file should exist.")
        config.processor.tokenizer.vocab_file = vocab_file
    if merges_file:
        if not os.path.exists(merges_file):
            raise ValueError("The merges_file should exist.")
        config.processor.tokenizer.merges_file = merges_file
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id
    if only_save_strategy is not None:
        config.only_save_strategy = only_save_strategy
    if repetition_penalty is not None:
        config.repetition_penalty = repetition_penalty
    if device_num is not None:
        config.parallel_config.model_parallel = device_num
    # init context
    build_context(config)

    if load_checkpoint is not None:
        config.load_checkpoint = load_checkpoint
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if do_sample is not None:
        config.model.model_config.do_sample = do_sample
    if top_k is not None:
        config.model.model_config.top_k = top_k
    if top_p is not None:
        config.model.model_config.top_p = top_p

    config.model.model_config.batch_size = batch_size
    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlmBoostConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)
    # build model
    network = LlmBoostForCausalLM(model_config)

    # get strategy file
    if model_config.llm_backend == "BuildIn" and config.only_save_strategy:
        strategy_ckpt_save_dir = os.path.join(get_output_root_path(), "strategy")
        os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
        strategy_file_path = os.path.join(
            strategy_ckpt_save_dir, "ckpt_strategy_rank_0.ckpt"
        )
        shard_state_dict = generate_state_dict(config)
        if get_rank() == 0:
            save_strategy_file(shard_state_dict, strategy_file_path)
        logger.info(f"Strategy file has been saved in {strategy_file_path}.")
        return

    # load checkpoint
    load_ckpt(config, network)
    # generate
    inputs_ids = tokenizer(inputs, max_length=max_length, padding="max_length")[
        "input_ids"
    ]
    outputs = network.generate(
        input_ids=inputs_ids,
        max_length=max_length,
        do_sample=model_config.do_sample,
        top_k=model_config.top_k,
        top_p=model_config.top_p,
    )
    for output in outputs:
        print(tokenizer.decode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="predict_qwen2_72b_instruct.yaml",
        type=str,
        help="config file path.",
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="checkpoint name or dir to load.",
    )
    parser.add_argument(
        "--only_save_strategy",
        default=None,
        type=str2bool,
        help="only save strategy.",
    )
    parser.add_argument("--vocab_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--merges_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--seq_length", default=None, type=int, help="seq_length")
    parser.add_argument(
        "--max_decode_length",
        default=8192,
        type=int,
        help="max length for predict output.",
    )
    parser.add_argument(
        "--use_parallel", default=None, type=str2bool, help="open parallel for model."
    )
    parser.add_argument(
        "--device_id",
        default=-1,
        type=int,
        help="ID of the target device, the value must be in [0, device_num_per_host-1]",
    )
    parser.add_argument("--do_sample", default=None, type=str2bool, help="do_sample")
    parser.add_argument("--top_k", default=None, type=int, help="top_k")
    parser.add_argument("--top_p", default=None, type=float, help="top_p")
    parser.add_argument(
        "--repetition_penalty", default=1.0, type=float, help="repetition_penalty"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--device_num", default=1, type=int, help="device_num")

    args = parser.parse_args()
    print(args)

    if args.device_id == -1:
        args.device_id = int(os.getenv("RANK_ID", "0"))

    main(
        config=args.config,
        use_parallel=args.use_parallel,
        only_save_strategy=args.only_save_strategy,
        load_checkpoint=args.load_checkpoint,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        seq_length=args.seq_length,
        max_length=args.max_decode_length,
        device_id=args.device_id,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        device_num=args.device_num,
    )
