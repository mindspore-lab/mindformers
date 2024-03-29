# Copyright 2023 Huawei Technologies Co., Ltd
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
"""model predict script"""
import os
import argparse
import numpy as np

import mindspore as ms
from mindspore import context
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import get_last_checkpoint
from mindformers import TransformerOpParallelConfig, MindFormerConfig, init_context
from mindformers.models import ChatGLM2Config
from mindformers.models import ChatGLM2ForConditionalGeneration
from mindformers.models import ChatGLM3Tokenizer

context.set_context(mode=0, device_target="Ascend", device_id=5)


def main(args, repetition_times=2):
    """main function."""
    # 环境初始化
    config = MindFormerConfig(os.path.realpath(args.config_path))
    config.use_parallel = args.use_parallel

    if not config.use_parallel:
        config.context.device_id = args.device_id
    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    glm32k_config = ChatGLM2Config(**config.model.model_config)
    glm32k_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    glm32k_config.batch_size = args.batch_size
    glm32k_config.use_past = args.use_past
    glm32k_config.seq_length = args.max_length

    if args.checkpoint_path and not config.use_parallel:
        glm32k_config.checkpoint_name_or_path = args.checkpoint_path

    print("starting ......")
    tokenizer = ChatGLM3Tokenizer(config.processor.tokenizer.vocab_file)
    glm32k_model = ChatGLM2ForConditionalGeneration(glm32k_config)
    glm32k_model.set_train(False)

    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard model and load sharded ckpt
        warm_up_model = Model(glm32k_model)
        warm_up_model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, glm32k_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(glm32k_model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    inputs = tokenizer.build_chat_input(args.user_query, history=[])

    for _ in range(repetition_times):
        outputs = glm32k_model.generate(input_ids=inputs["input_ids"],
                                        use_past=args.use_past,
                                        max_length=args.max_length)
        print("output: ")
        print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='run_glm32k.yaml', type=str,
                        help='config')
    parser.add_argument('--max_length', default=512, type=int,
                        help='max length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--device_id', default=5, type=int,
                        help='device_id')
    parser.add_argument('--use_past', default=False, type=str2bool,
                        help="use past")
    parser.add_argument('--use_parallel', default=False, type=bool,
                        help="use parallel")
    parser.add_argument('--checkpoint_path', default='/path/to/glm3_32k.ckpt', type=str,
                        help="checkpoint_path")
    parser.add_argument('--user_query',
                        default='使用python编写快速排序代码',
                        type=str,
                        help='user query input.')
    opt = parser.parse_args()

    main(opt)

    # 启动命令： python infer_generate.py --user_query "晚上睡不着应该怎么办" --device_id 0 --use_past True
