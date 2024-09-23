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
""""Save model strategy"""
import argparse
import mindspore as ms
import numpy as np

from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, LlamaForCausalLM
from mindformers import init_context, AutoModel, build_parallel_config, AutoConfig, build_context
from mindformers.trainer.utils import build_model
from mindformers import logger
from mindformers.tools import set_output_path
from mindformers.model_runner import register_auto_class


def save_strategy_parallel(args):
    """generate strategy and save"""
    # set model config
    config = MindFormerConfig(args.yaml_file)
    register_auto_class(config, args.yaml_file, class_type="AutoModel")
    register_auto_class(config, args.yaml_file, class_type="AutoConfig")
    if config.parallel_config.model_parallel != args.world_size:
        logger.info(
            f"world_size is {args.world_size}, not equal to \
            model_parallel in config:{config.parallel_config.model_parallel}")
        config.parallel_config.model_parallel = args.world_size
        logger.info(f"reset config.parallel_config.model_parallel as :{args.world_size}")

    config.output_dir = args.save_strategy_path
    set_output_path(config.output_dir)
    config.parallel.parallel_mode = 'STAND_ALONE'
    config.parallel.full_batch = False
    model_config = AutoConfig.from_pretrained(args.yaml_file)

    logger.info(f"config.model.model_config.qkv_concat: {config.model.model_config.qkv_concat}")
    logger.info(f"args.qkv_concat: {args.qkv_concat}")
    if args.qkv_concat != '':
        model_config.qkv_concat = args.qkv_concat == 'True'
        logger.info(f"reset qkv_concat as :{config.model.model_config.qkv_concat}")
    build_parallel_config(config)
    model_config.parallel_config = config.parallel_config
    build_context(config)
    model = AutoModel.from_config(model_config)
    from mindformers.experimental.infer.core.utils import generate_state_dict
    from mindformers.experimental.parallel_core.pynative.utils import save_strategy_file

    state_dict = generate_state_dict(network=model)
    save_strategy_file(state_dict=state_dict,
                       strategy_file_name=config.output_dir + '/strategy/ckpt_strategy_rank_0.ckpt')
    logger.info(f"Save strategy finish! Strategies are saved in {config.output_dir}")


def save_strategy_llama(args):
    """generate strategy and save"""
    # set model config
    config = MindFormerConfig(args.yaml_file)
    if config.parallel_config.model_parallel != args.world_size:
        logger.info(
            f"world_size is {args.world_size}, not equal to \
            model_parallel in config:{config.parallel_config.model_parallel}")
        config.parallel_config.model_parallel = args.world_size
        logger.info(f"reset config.parallel_config.model_parallel as :{args.world_size}")

    config.output_dir = args.save_strategy_path
    set_output_path(config.output_dir)
    config.parallel.parallel_mode = 1
    config.parallel.full_batch = True

    init_context(use_parallel=config.use_parallel,
                 context_config=config.context,
                 parallel_config=config.parallel)
    logger.info(f"config.model.model_config.qkv_concat: {config.model.model_config.qkv_concat}")
    logger.info(f"args.qkv_concat: {args.qkv_concat}")
    if args.qkv_concat != '':
        config.model.model_config.qkv_concat = args.qkv_concat == 'True'
        logger.info(f"reset qkv_concat as :{config.model.model_config.qkv_concat}")
    model_config = LlamaConfig(**config.model.model_config)
    model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)

    # build model from config
    model = LlamaForCausalLM(model_config)
    model.set_train(False)
    # if use parallel, load distributed checkpoints
    if config.use_parallel:
        ms_model = ms.Model(model)
        batch_size = model_config.batch_size
        seq_length = model_config.seq_length
        input_ids = np.ones(shape=tuple([batch_size, seq_length]))
        inputs = model.prepare_inputs_for_predict_layout(input_ids)
        build_model(config, ms_model, inputs, do_eval=False, do_predict=True)
    logger.info(f"Save strategy finish! Strategies are saved in {config.output_dir}")


def main(args):
    """save strategy with different arch"""
    if args.precision == 'fp16':
        save_strategy_llama(args)
    else:
        save_strategy_parallel(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default="/home/checkpoint_download/llama57b/predict_llama2_57b_910b.yaml",
                        type=str,
                        help='model yaml path')
    parser.add_argument('--save_strategy_path', default="./output",
                        type=str,
                        help='path to save strategy')
    parser.add_argument('--world_size', default=8, type=int,
                        help='world size')
    parser.add_argument('--qkv_concat', default='', type=str,
                        help='qkv_concat')
    parser.add_argument('--precision', default='', type=str,
                        help='precision')
    uargs = parser.parse_args()
    main(uargs)
