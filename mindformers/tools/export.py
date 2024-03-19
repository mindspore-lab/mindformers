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

import argparse
import glob
import os.path

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype

from mindformers import AutoModel
from mindformers.models import build_network
from mindformers.tools.register import MindFormerConfig
# pylint: disable=W0611


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
    return input_ids, None, input_position, None, None, None, init_reset, batch_valid_length


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


def get_gpt2_model_input(batch_size, seq_length, prefill=True):
    """get gpt2 kv cache model input tuple."""
    # export first iteration
    if prefill:
        input_ids = ms.Tensor(np.ones((batch_size, seq_length)), ms.int32)
        input_position = ms.Tensor(np.ones((batch_size,)), ms.int32)
        init_reset = ms.Tensor([False], ms.bool_)
        batch_valid_length = ms.Tensor(np.ones([batch_size, 1]), ms.int32)
    # export later iteration
    else:
        input_ids = ms.Tensor(np.ones((batch_size, 1)), ms.int32)
        input_position = ms.Tensor(np.ones((batch_size,)), ms.int32)
        init_reset = ms.Tensor([True], ms.bool_)
        batch_valid_length = ms.Tensor(np.ones([batch_size, 1]), ms.int32)
    return input_ids, None, None, None, input_position, None, init_reset, batch_valid_length


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
    return input_ids, None, input_position, None, None, None, init_reset, batch_valid_length


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


PREFILL_MODEL_INPUT_MAP = {
    "bloom": get_llm_common_prefill_model_input,
    "llama": get_llm_common_prefill_model_input,
    "llama2": get_llm_common_prefill_model_input,
    "glm": get_glm_prefill_model_input,
    "gpt2": get_gpt2_model_input,
    "glm2": get_glm2_prefill_model_input,
    "glm3": get_glm2_prefill_model_input,
    "baichuan2": get_llm_common_prefill_model_input,
}

INCREMENT_MODEL_INPUT_MAP = {
    "bloom": get_bloom_inc_model_input,
    "llama": get_llama_inc_model_input,
    "llama2": get_llama_inc_model_input,
    "glm": get_glm_inc_model_input,
    "gpt2": get_gpt2_model_input,
    "glm2": get_glm2_inc_model_input,
    "glm3": get_glm2_inc_model_input,
    "codegeex2": get_glm2_inc_model_input,
    "baichuan2": get_baichuan2_inc_model_input,
}


def export_single_model(config, batch_size, model_type: str = 'MINDIR', model_dir=None):
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
        model = build_network(config.model)
    model.set_train(False)
    model_prefix = model_name.split('_')[0]
    if model_prefix in PREFILL_MODEL_INPUT_MAP.keys():
        inputs = PREFILL_MODEL_INPUT_MAP[model_prefix](batch_size, config.infer.infer_seq_length)
    else:
        raise NotImplementedError(f"model {model_name} not implemented.")

    suffix = model_type.lower()
    filename = config.infer.prefill_model_path.rstrip("." + suffix)
    model.add_flags_recursive(is_first_iteration=True)
    ms.export(model, *inputs,
              file_name=filename,
              file_format=model_type.upper())


def export_inc_model(config, batch_size, model_type: str = 'MINDIR', model_dir=None):
    """
        export kvcache model.
        Args:
            config: config from yaml.
            batch_size(int): batch size
            model_type: model type
        """
    model_name = config.trainer.model_name
    if model_dir:
        model = AutoModel.from_pretrained(model_dir)
    else:
        model = build_network(config.model)
    model.set_train(False)
    model_prefix = model_name.split('_')[0]
    if model_prefix in INCREMENT_MODEL_INPUT_MAP.keys():
        func = INCREMENT_MODEL_INPUT_MAP[model_prefix]
    else:
        raise NotImplementedError(f"model {model_name} not implemented.")

    suffix = model_type.lower()
    # export prefill
    filename = config.infer.prefill_model_path.rstrip("." + suffix)
    inputs = func(batch_size, config.infer.infer_seq_length, True)
    model.add_flags_recursive(is_first_iteration=True)
    ms.export(model, *inputs,
              file_name=filename,
              file_format=model_type.upper())
    print("Prefill model exported.")
    # export inc
    filename = config.infer.increment_model_path.rstrip("." + suffix)
    inputs = func(batch_size, config.infer.infer_seq_length, False)
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
        if len(run_config) > 1:
            raise ValueError(f"{model_dir} contains multi config files: {run_config}.")
        config = MindFormerConfig(run_config[0])

    if config.infer is None:
        raise KeyError(f"infer config not in {args_.config_path}.")
    if not config.infer.prefill_model_path:
        raise ValueError(f"prefill_model_path in {args_.config_path} is empty.")

    if not config.infer.increment_model_path:
        export_single_model(config, args_.batch_size, args_.model_type, model_dir)
    else:
        export_inc_model(config, args_.batch_size, args_.model_type, model_dir)


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
        '--device_id', default=0,
        type=int,
        help='device id to export mindir.')
    args = parser.parse_args()
    main(args)
