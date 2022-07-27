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
Model operations
"""
import mindspore.common.dtype as mstype

from .bert.bert import BertConfig, get_bert_network
from .gpt.gpt import GPTConfig, get_gpt_network
from .t5.t5 import TransformerConfig, get_t5_network
from .vit.vit import VitConfig


def get_model_config(opt):
    """Get the model config"""
    opt.logger.info(f"Start to build model config")
    micro_batch_interleaved = opt.speed_up['micro_batch_num']
    global_batch_size = opt.model.pop('global_batch_size')
    micro_batch_size = opt.model.pop('micro_batch_size')
    compute_dtype = opt.model.pop('compute_dtype')
    compute_dtype = mstype.float16 if compute_dtype == "fp16" else mstype.float32
    if global_batch_size % micro_batch_interleaved != 0:
        raise ValueError(f"global_batch_size:{global_batch_size} must be a multiple of micro_batch_interleaved:"
                         f"{micro_batch_interleaved}.")

    config_mapper = {"gpt": GPTConfig, "bert": BertConfig, "t5": TransformerConfig, "vit": VitConfig}

    opt.logger.info(f"Model Name: {opt.arch}")
    config_func = config_mapper[opt.arch]
    config = config_func(**opt.model,
                         compute_dtype=compute_dtype,
                         batch_size=global_batch_size//micro_batch_interleaved)
    opt.model['global_batch_size'] = global_batch_size
    opt.model['micro_batch_size'] = micro_batch_size
    return config


def build_model(opt, parallel_config):
    """Return the backbone and the net with loss wrapper"""
    opt.logger.info(f"Start to build model")
    model_config = get_model_config(opt)
    model_config.parallel_config = parallel_config

    model_name = opt.arch

    net = None
    if model_name == 'gpt':
        net = get_gpt_network(opt, model_config)
    elif model_name == 'bert':
        net = get_bert_network(opt, model_config)
    elif model_name == 't5':
        net = get_t5_network(opt, model_config)
    else:
        raise RuntimeError(f"Model {model_name} is not supported yet.")
    opt.logger.info(f"Build model finished")
    return net
