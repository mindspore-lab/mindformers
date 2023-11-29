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
"""export mindir."""

import os
import glob
import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs

from mindformers import AutoModel, LlamaConfig
from mindformers.models import build_model
from mindformers.tools.register import MindFormerConfig
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig

# pylint: disable=W0611
from research.baichuan2.baichuan2_7b import Baichuan7BV2ForCausalLM
from research.baichuan2.baichuan2_13b import Baichuan13BV2ForCausalLM
from research.internlm.internlm_dyn_kvcache_distributed import InternLMForCausalLM


def set_no_pipeline_parallel_context():
    r"""Set parallel context"""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank_id, device_num))
    context.reset_auto_parallel_context()
    # save_dir = '/opt/mnt1/dzx/code/pangu_am_deploy_kvcache/infer_strategy_kvcache_8p_71b.ckpt'
    context.set_auto_parallel_context(
        enable_alltoall=False,  # default False
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
        loss_repeated_mean=True, full_batch=True)
    context.set_auto_parallel_context(communi_parallel_mode='same_server_group_parallel')
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank_id, device_num


def get_glm_prefill_model_input(batch_size, seq_length):
    """get glm model input tuple."""
    x = ms.Tensor(np.ones([batch_size, seq_length]).astype(np.int32))
    position_ids = ms.Tensor(np.ones([batch_size, 2, seq_length]).astype(np.int32))
    attention_mask = ms.Tensor(np.ones([batch_size, 1, seq_length, seq_length]).astype(np.int32))
    return x, position_ids, attention_mask


def get_glm2_prefill_model_input(batch_size, seq_length):
    """get glm2 model input tuple."""
    input_ids = ms.Tensor(np.ones((batch_size, seq_length)), ms.int32)
    input_position = ms.Tensor(np.ones((batch_size, 1)), ms.int32)
    init_reset = ms.Tensor([False], ms.bool_)
    batch_valid_length = ms.Tensor(np.ones([batch_size, 1]), ms.int32)
    return input_ids, None, None, None, None, input_position, init_reset, batch_valid_length


def get_llm_common_prefill_model_input(batch_size, seq_length):
    """get llama model input tuple."""
    x = ms.Tensor(np.ones([batch_size, seq_length]).astype(np.int32))
    return (x,)


def get_bloom_inc_model_input(batch_size, seq_length, prefill):
    """get bloom kv cache model input tuple."""
    if not prefill:
        seq_length = 1
    init_reset = not prefill
    input_ids = ms.Tensor(np.ones((batch_size, seq_length)), mstype.int32)
    input_position = ms.Tensor([127] * batch_size, mstype.int32)
    init_reset = ms.Tensor([init_reset], mstype.bool_)
    batch_valid_length = ms.Tensor([128] * batch_size, mstype.int32)
    return input_ids, input_position, None, None, None, None, init_reset, batch_valid_length


def get_llama_inc_model_input(batch_size, seq_length, prefill):
    """get llama kv cache model input tuple."""
    if not prefill:
        seq_length = 1
    init_reset = not prefill
    input_ids = ms.Tensor(np.ones((batch_size, seq_length)), mstype.int32)
    input_position = ms.Tensor([127] * batch_size, mstype.int32)
    init_reset = ms.Tensor([init_reset], mstype.bool_)
    batch_valid_length = ms.Tensor([128] * batch_size, mstype.int32)
    return input_ids, None, input_position, None, None, None, init_reset, batch_valid_length


def get_glm2_inc_model_input(batch_size, seq_length, prefill):
    """get glm2 kv cache model input tuple."""
    # export first iteration
    if prefill:
        input_ids = ms.Tensor(np.ones((batch_size, seq_length)), ms.int32)
        input_position = ms.Tensor(np.ones((batch_size, 1)), ms.int32)
        init_reset = ms.Tensor([False], ms.bool_)
        batch_valid_length = ms.Tensor(np.ones([batch_size, 1]), ms.int32)
    # export later iteration
    else:
        input_ids = ms.Tensor(np.ones((batch_size, 1)), ms.int32)
        input_position = ms.Tensor(np.ones((batch_size, 1)), ms.int32)
        init_reset = ms.Tensor([True], ms.bool_)
        batch_valid_length = ms.Tensor(np.ones([batch_size, 1]), ms.int32)
    return input_ids, None, None, None, None, input_position, init_reset, batch_valid_length


def get_glm_inc_model_input(batch_size, seq_length, prefill):
    """get glm kv cache model input tuple."""
    if prefill:
        input_ids = ms.Tensor(np.ones((batch_size, seq_length)), mstype.int32)
        position_ids = ms.Tensor(np.ones([batch_size, 2, seq_length]).astype(np.int32))
        attention_mask = ms.Tensor(np.ones([batch_size, 1, seq_length, seq_length]).astype(np.int32))
    else:
        input_ids = ms.Tensor(np.ones((batch_size, 1)), mstype.int32)
        position_ids = ms.Tensor(np.ones([batch_size, 2, 1]).astype(np.int32))
        attention_mask = ms.Tensor(np.ones([batch_size, 1, 1, seq_length]).astype(np.int32))
    input_position = ms.Tensor([127] * batch_size, mstype.int32)
    init_reset = not prefill
    init_reset = ms.Tensor([init_reset], mstype.bool_)
    batch_valid_length = ms.Tensor([128] * batch_size, mstype.int32)
    return input_ids, position_ids, attention_mask, input_position, None, None, init_reset, batch_valid_length


def get_baichuan2_inc_model_input(batch_size, seq_length, prefill):
    """get baichuan2 kv cache model input tuple."""
    if not prefill:
        seq_length = 1
    init_reset = not prefill
    input_ids = ms.Tensor(np.ones((batch_size, seq_length)), mstype.int32)
    input_position = ms.Tensor([127] * batch_size, mstype.int32)
    init_reset = ms.Tensor([init_reset], mstype.bool_)
    batch_valid_length = ms.Tensor([[128] * batch_size], mstype.int32)
    return input_ids, None, input_position, None, None, None, init_reset, batch_valid_length


def get_internlm_dyn_inc_model_input(batch_size, seq_length, act_len, prefill):
    """get internlm kv cache model input tuple."""
    if not prefill:
        seq_length = 1
    input_ids = dummy_tensor(shape=[batch_size, seq_length], dtype=ms.int32)
    input_position = dummy_tensor(shape=[batch_size], dtype=ms.int32)
    batch_valid_length = dummy_tensor(shape=[batch_size], dtype=ms.int64)
    batch_index = dummy_tensor(shape=[batch_size], dtype=ms.int64)
    print(f'input_ids shape: {input_ids.shape}', flush=True)
    print(f'input_position shape: {input_position.shape}', flush=True)
    print(f'batch_valid_length shape: {batch_valid_length.shape}', flush=True)
    if act_len:
        zactivate_len = dummy_tensor(shape=[None], dtype=ms.int64)
        print(f'zactivate_len shape: {zactivate_len.shape}', flush=True)
    else:
        zactivate_len = None
    return input_ids, None, input_position, None, None, None, None, batch_valid_length, batch_index, zactivate_len


def dummy_tensor(shape, dtype):
    if None in shape:
        return ms.Tensor(shape=shape, dtype=dtype)
    return ms.Tensor(np.ones(shape=tuple(shape)), dtype=dtype)


def load_qkv_ckpt(model):
    """Load qkv concat checkpoint."""
    param_dict = model.parameters_dict()
    for name, param in model.parameters_and_names():
        print(f"name == {name}")
        if "wqkv" in name:
            print(f"before load param = {param.value()[0, :50]}")
            print(f"before qkv shape = {param.shape}")
            query = param_dict[name.replace("wqkv", "wq")]
            key = param_dict[name.replace("wqkv", "wk")]
            value = param_dict[name.replace("wqkv", "wv")]
            qkv = ms.ops.cat((query, key, value), 0)
            print(f"qkv shape = ", qkv.shape)
            param.set_data(qkv, param.dtype)
            print(f"end load param = {param.value()[0, :50]}")


PREFILL_MODEL_INPUT_MAP = {
    "bloom": get_llm_common_prefill_model_input,
    "llama": get_llm_common_prefill_model_input,
    "glm": get_glm_prefill_model_input,
    "glm2": get_glm2_prefill_model_input,
    "baichuan": get_llm_common_prefill_model_input,
    "baichuan2": get_llm_common_prefill_model_input,
    "internlm": get_llm_common_prefill_model_input,
}

INCREMENT_MODEL_INPUT_MAP = {
    "bloom": get_bloom_inc_model_input,
    "llama": get_llama_inc_model_input,
    "glm": get_glm_inc_model_input,
    "glm2": get_glm2_inc_model_input,
    "codegeex2": get_glm2_inc_model_input,
    "baichuan": get_baichuan2_inc_model_input,
    "baichuan2": get_baichuan2_inc_model_input,
    "internlm": get_internlm_dyn_inc_model_input
}


def export_single_model(config, batch_size, seq_length, model_type: str = 'MINDIR', model_dir=None):
    """
    export no kvcache model.
    Args:
        config: config from yaml.
        batch_size(int): batch size
        model_type: model type
    """
    model_name = config.trainer.model_name
    if model_dir:
        model = AutoModel.from_pretrained(model_dir)
    else:
        model = build_model(config.model)
    model.set_train(False)
    model_prefix = model_name.split('_')[0]
    if model_prefix in PREFILL_MODEL_INPUT_MAP.keys():
        inputs = PREFILL_MODEL_INPUT_MAP[model_prefix](batch_size, seq_length)
    else:
        raise NotImplementedError(f"model {model_name} not implemented.")

    suffix = model_type.lower()
    filename = config.infer.prefill_model_path.rstrip("." + suffix)
    model.add_flags_recursive(is_first_iteration=True)
    ms.export(model, *inputs,
              file_name=filename,
              file_format=model_type.upper())


def export_inc_model(config, batch_size, seq_length, model_type: str = 'MINDIR', qkv_concat=False, act_len=False,
                     model_dir=None):
    """
        export kvcache model.
        Args:
            config: config from yaml.
            batch_size(int): batch size
            model_type: model type
        """
    model_name = config.trainer.model_name
    if model_dir:
        # model = AutoModel.from_pretrained(model_dir)
        model_config = LlamaConfig(**config.model.model_config)
        parallel_config = TransformerOpParallelConfig(**config.parallel_config)
        model_config.parallel_config = parallel_config
        print(f"==========model_config is {model_config}==========", flush=True)
        model = InternLMForCausalLM(model_config)
    else:
        model = build_model(config.model)
    if qkv_concat:
        load_qkv_ckpt(model)
    model.set_train(False)
    # model.phase = 'predict'
    model_prefix = model_name.split('_')[0]
    if model_prefix in INCREMENT_MODEL_INPUT_MAP.keys():
        func = INCREMENT_MODEL_INPUT_MAP[model_prefix]
    else:
        raise NotImplementedError(f"model {model_name} not implemented.")

    suffix = model_type.lower()
    rank_id = int(os.environ['RANK_ID'])
    # export prefill
    filename = config.infer.prefill_model_path.rstrip("." + suffix) + f"_rank{rank_id}"
    inputs = func(batch_size, seq_length, act_len, True)
    model.add_flags_recursive(is_first_iteration=True)
    ms.export(model, *inputs,
              file_name=filename,
              file_format=model_type.upper())
    print("Prefill model exported.")
    # export inc
    filename = config.infer.increment_model_path.rstrip("." + suffix) + f"_rank{rank_id}"
    inputs = func(batch_size, seq_length, act_len, False)
    model.add_flags_recursive(is_first_iteration=False)
    ms.export(model, *inputs,
              file_name=filename,
              file_format="MINDIR")
    print("Increment model exported.")


def main(args_):
    """export mindir."""
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args_.device_id)

    if args_.config_path is None and args_.model_dir is None:
        raise ValueError(f"config_path or model_dir must exist one.But get none.")

    if args_.config_path:
        config = MindFormerConfig(args_.config_path)
        model_dir = None
    else:
        model_dir = os.path.realpath(args_.model_dir)
        run_config = glob.glob(model_dir + "/*.yaml")
        print(f"==========run_config is {run_config}==========", flush=True)
        if len(run_config) > 1:
            raise ValueError(f"{model_dir} contains multi config files: {run_config}.")
        config = MindFormerConfig(run_config[0])
    print(f"==========MindFormerConfig is {MindFormerConfig}==========", flush=True)

    if config.infer is None:
        raise KeyError(f"infer config not in {args_.config_path}.")
    if not config.infer.prefill_model_path:
        raise ValueError(f"prefill_model_path in {args_.config_path} is empty.")

    batch_size = None if config.model.model_config.is_dynamic else config.model.model_config.batch_size
    seq_length = None if config.model.model_config.is_dynamic else config.model.model_config.seq_length
    qkv_concat = config.model.model_config.qkv_concat
    act_len = config.model.model_config.act_len
    set_no_pipeline_parallel_context()
    if not config.infer.increment_model_path:
        export_single_model(config, batch_size, seq_length, args_.model_type, model_dir)
    else:
        export_inc_model(config, batch_size, seq_length, args_.model_type, qkv_concat, act_len, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path', default=None,
        type=str,
        help='YAML config files')
    parser.add_argument(
        '--model_dir', default=None,
        type=str,
        help='model config and ckpt file dir path')
    parser.add_argument(
        '--batch_size', default=1,
        type=int,
        help='batch size of model.')
    parser.add_argument(
        '--model_type', default="MINDIR",
        type=str,
        help='model type of exported model.')
    parser.add_argument(
        '--qkv_concat', default=False,
        type=str,
        help='concat qkv linear or not.')
    parser.add_argument(
        '--device_id', default=0,
        type=int,
        help='device id to export mindir.')
    args = parser.parse_args()
    main(args)
