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
from mindspore import context
from mindspore.context import ParallelMode

from .bert.bert import BertConfig, get_bert_network
from .gpt.gpt import GPTConfig, get_gpt_network
from .t5.t5 import TransformerConfig, get_t5_network
from .vit.vit import VitConfig
from .opt.opt import OPTConfig, get_opt_network


def get_downstream_config(opt):
    """Get the model config"""
    opt.logger.info("Start to build model config")
    train_batch_size = opt.model.pop('train_batch_size')
    eval_batch_size = opt.model.pop('eval_batch_size')
    compute_dtype = opt.model.pop('compute_dtype')

    config_mapper = {"gpt": GPTConfig, "bert": BertConfig, "t5": TransformerConfig, "vit": VitConfig,
                     "opt": OPTConfig}

    data_dp = 1
    if opt.parallel_mode in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL) and \
            not context.get_auto_parallel_context('full_batch'):
        data_dp = context.get_auto_parallel_context("device_num")
    opt.logger.info(f"Model Name: {opt.arch}")
    config_func = config_mapper[opt.arch]
    config = config_func(**opt.model,
                         compute_dtype=compute_dtype,
                         batch_size=data_dp * train_batch_size)
    opt.model['train_batch_size'] = train_batch_size
    opt.model['eval_batch_size'] = eval_batch_size
    return config

def get_model_config(opt):
    """Get the model config"""
    opt.logger.info("Start to build model config")
    micro_batch_interleaved = opt.speed_up['micro_batch_num']
    global_batch_size = opt.model.pop('global_batch_size')
    micro_batch_size = opt.model.pop('micro_batch_size')
    compute_dtype = opt.model.pop('compute_dtype')
    if global_batch_size % micro_batch_interleaved != 0:
        raise ValueError(f"global_batch_size:{global_batch_size} must be a multiple of micro_batch_interleaved:"
                         f"{micro_batch_interleaved}.")

    config_mapper = {"gpt": GPTConfig, "bert": BertConfig, "t5": TransformerConfig, "vit": VitConfig,
                     "opt": OPTConfig}

    data_dp = 1
    if opt.parallel_mode in (ParallelMode.AUTO_PARALLEL, ParallelMode.SEMI_AUTO_PARALLEL) and \
            not context.get_auto_parallel_context('full_batch'):
        data_dp = context.get_auto_parallel_context("device_num")
    opt.logger.info(f"Model Name: {opt.arch}")
    config_func = config_mapper[opt.arch]
    config = config_func(**opt.model,
                         compute_dtype=compute_dtype,
                         batch_size=data_dp * global_batch_size // micro_batch_interleaved)
    opt.model['global_batch_size'] = global_batch_size
    opt.model['micro_batch_size'] = micro_batch_size
    return config


def build_model(opt, parallel_config):
    """Return the backbone and the net with loss wrapper"""
    opt.logger.info(f"Start to build model")
    model_config = get_model_config(opt)
    model_config.parallel_config = parallel_config

    model_name = opt.arch

    config_mapper = {"gpt": get_gpt_network, "bert": get_bert_network, "t5": get_t5_network,
                     "opt": get_opt_network}
    net = None
    model_func = config_mapper.get(model_name, None)
    if model_func:
        net = model_func(opt, model_config)
    else:
        raise RuntimeError(f"Model {model_name} is not supported yet.")
    opt.logger.info(f"Build model finished")
    return net
