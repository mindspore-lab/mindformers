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
# -*- coding: utf-8 -*-

"""Yi predict example."""
import argparse

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaTokenizer, LlamaForCausalLM, \
    logger
from mindformers import init_context
from mindformers.trainer.utils import transform_and_load_checkpoint


def main(config_path, predict_mode):
    """main function."""
    # 多batch输入
    inputs = ["以雷霆之力", "小明和小红", "生活就像一盒巧克力"]

    # set model config
    config = MindFormerConfig(config_path)

    # 初始化环境
    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)

    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
    model_config.batch_size = len(inputs)

    # build tokenizer
    tokenizer = LlamaTokenizer(**config.processor.tokenizer)
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        seq_length = config.model.model_config.seq_length
        input_ids = Tensor(shape=(1, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    with open('text_generation_result.txt', 'w') as writers:
        for prompt in inputs:
            if predict_mode == 'Chat':
                message = [{"role": "user", "content": prompt}]
                input_id = tokenizer.apply_chat_template(conversation=message,
                                                         tokenize=True,
                                                         add_generation_prompt=True)
            else:
                input_id = tokenizer.encode(prompt)

            outputs = network.generate(input_id,
                                       max_length=model_config.max_decode_length,
                                       do_sample=model_config.do_sample)
            result = tokenizer.decode(outputs[0][len(input_id):], skip_special_tokens=True)
            print(result)
            writers.write(f'text_generation:\n{result}\n')
    writers.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--predict_mode', default="Base", type=str,
                        help='predict_mode is Base or Chat')
    args = parser.parse_args()
    main(args.config_path, args.predict_mode)
