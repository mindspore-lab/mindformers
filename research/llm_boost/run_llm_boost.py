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

"""LLM Boost task script"""

import argparse
import os
import numpy

from mindspore.communication import get_rank
from mindspore._c_expression import _framework_profiler_step_start, _framework_profiler_step_end


from research.llm_boost.llm_boost import LlmBoostForCausalLM
from research.llm_boost.llm_boost import LlmBoostConfig
from research.llm_boost.utils import generate_state_dict, save_strategy_file

from mindformers import MindFormerConfig, logger
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers.core.context import build_context
from mindformers.trainer.utils import load_ckpt
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import str2bool


# pylint: disable=C0330
# pylint: disable=W0212
def main(
    config,
    max_length=8192,
    batch_size=1,
    measure_throughput=False,
    save_file=""):
    """main function."""
    inputs = [
        "帮我制定一份去上海的旅游攻略",
        "描述一下鲁智深倒拔垂杨柳的场面",
        "描述一下孙悟空大闹天宫的场面",
        "我喜欢看电影，因为",
    ]
    inputs = [
        "hello, I love Beijing",
    ]

    # init context
    build_context(config)

    os.environ['MS_DISABLE_REF_MODE'] = '1'

    config.model.model_config.batch_size = batch_size
    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlmBoostConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    if 'qwen' in config.trainer.model_name:
        # you'll need to set PYHTONPATH to QWEN2 DIR to make this work
        from research.qwen2.qwen2_tokenizer_fast import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer(vocab_file=config.processor.tokenizer.vocab_file,
                                   merges_file=config.processor.tokenizer.merges_file)
    else:
        tokenizer = LlamaTokenizer(vocab_file=config.processor.tokenizer.vocab_file)
    # build model
    network = LlmBoostForCausalLM(model_config)

    # get strategy file
    if config.llm_backend == "BuildIn" and config.only_save_strategy:
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
    if not measure_throughput:
        if len(inputs) == 1 and batch_size != 1:
            inputs = inputs * batch_size
        inputs_ids = tokenizer(inputs, max_length=max_length, padding="max_length")["input_ids"]
        outputs = network.generate(
                input_ids=inputs_ids,
                max_length=max_length,
                do_sample=model_config.do_sample,
                top_k=model_config.top_k,
                top_p=model_config.top_p,
                )

        with open(save_file, 'w') as file:
            for output in outputs:
                print(tokenizer.decode(output))
                file.write(tokenizer.decode(output) + '\n')
        file.close()

    else:
        _framework_profiler_step_start()
        for bs in [1, 4]:
            length = 512
            max_length = length * 2

            print("***************************Warm up for bs {}*************************".format(bs))

            inputs_ids_arr = numpy.random.randint(low=1, high=1000, size=(bs, max_length))
            inputs_ids_arr[:, length:] = 0
            network._exec_add_flags = True
            outputs = network.generate(input_ids=inputs_ids_arr.tolist(), max_length=max_length,
                                       do_sample=model_config.do_sample, top_k=model_config.top_k,
                                       top_p=model_config.top_p)

            for length in [256, 512, 1024, 2048]:
                print("************************ Measure bs={}, length={} *************************".format(bs, length))
                max_length = length * 2
                inputs_ids_arr = numpy.random.randint(low=1, high=1000, size=(bs, length))
                network._exec_add_flags = True
                outputs = network.generate(input_ids=inputs_ids_arr.tolist(), max_length=max_length,
                                           do_sample=model_config.do_sample, top_k=model_config.top_k,
                                           top_p=model_config.top_p)
        _framework_profiler_step_end()


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
        default=False,
        type=bool,
        help="only save strategy.",
    )
    parser.add_argument("--vocab_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--merges_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--seq_length", default=None, type=int, help="seq_length")
    parser.add_argument(
        "--predict_length",
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
    parser.add_argument("--measure_throughput", default=False, type=str2bool, help="measure_throughput")
    parser.add_argument("--save_file", default="results.txt", type=str, help="file to store results")


    args = parser.parse_args()
    print(args)

    if args.device_id == -1:
        args.device_id = int(os.getenv("RANK_ID", "0"))

    yaml_path = os.path.expanduser(args.config)
    if not os.path.exists(yaml_path):
        raise ValueError("The yaml_path should exist.")

    mfconfig = MindFormerConfig(os.path.realpath(yaml_path))
    if args.vocab_file:
        if not os.path.exists(args.vocab_file):
            raise ValueError("The vocab_file should exist.")
        mfconfig.processor.tokenizer.vocab_file = args.vocab_file
    if args.merges_file:
        if not os.path.exists(args.merges_file):
            raise ValueError("The merges_file should exist.")
        mfconfig.processor.tokenizer.merges_file = args.merges_file
    if args.use_parallel is not None:
        mfconfig.use_parallel = args.use_parallel
    if args.device_id is not None:
        mfconfig.context.device_id = args.device_id
    if args.only_save_strategy is not None:
        mfconfig.only_save_strategy = args.only_save_strategy
    if args.repetition_penalty is not None:
        mfconfig.repetition_penalty = args.repetition_penalty
    if args.device_num is not None:
        mfconfig.parallel_config.model_parallel = args.device_num

    if args.load_checkpoint is not None:
        mfconfig.load_checkpoint = args.load_checkpoint
    if args.seq_length is not None:
        mfconfig.model.model_config.seq_length = args.seq_length
    if args.do_sample is not None:
        mfconfig.model.model_config.do_sample = args.do_sample
    if args.top_k is not None:
        mfconfig.model.model_config.top_k = args.top_k
    if args.top_p is not None:
        mfconfig.model.model_config.top_p = args.top_p

    main(config=mfconfig,
         max_length=args.predict_length,
         batch_size=args.batch_size,
         measure_throughput=args.measure_throughput,
         save_file=args.save_file
    )
