# Copyright 2025 Huawei Technologies Co., Ltd
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
infer safetensor to ckpt
"""
import os
import argparse
from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import Model, Tensor, save_checkpoint
from mindspore.common import initializer
from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.tools.utils import get_rank_info, set_output_path, set_strategy_save_path
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer


def create_ptq():
    """Create ptq algorithm."""
    from mindspore_gs.ptq import PTQ
    from mindspore_gs.common import BackendTarget
    from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, PrecisionRecovery, QuantGranularity
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                    act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ffn_config = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                           act_quant_dtype=msdtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TOKEN,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ptq = PTQ(config=cfg, layer_policies=OrderedDict({r'.*\.feed_forward\..*': ffn_config}))
    ptq.decoder_layers.append(DeepseekV3DecodeLayer)
    return ptq


def trans_sf_to_ckpt(yaml_file, is_quant, dst_path):
    """trans sf to ckpt"""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    model_config = DeepseekV3Config(**model_config)

    network = DeepseekV3ForCausalLM(model_config)
    if is_quant:
        ptq = create_ptq()
        ptq.apply(network)
        ptq.convert(network)
        ptq.summary(network)

    set_output_path(config.output_dir)
    set_strategy_save_path(config.parallel)
    ms_model = Model(network)
    if config.load_checkpoint:
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    rank_id, _ = get_rank_info()
    save_path = f"{dst_path}/rank_{rank_id}"
    print(f"--------- start to save checkpoint, save_path: {save_path}.", flush=True)
    os.makedirs(save_path, exist_ok=True)
    save_checkpoint(network, os.path.join(f"{save_path}/dsv3_{rank_id}.ckpt"),
                    choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    print(f"--------- save checkpoint finished, save_path: {save_path}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default=None, type=str)
    parser.add_argument('--is_quant', default=False, type=bool)
    parser.add_argument('--dst_path', default=None, type=str)
    args = parser.parse_args()
    trans_sf_to_ckpt(args.yaml_file, args.is_quant, args.dst_path)
