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
"""Functions to check rules"""
import os
import json
import mindspore as ms
from .utils import get_real_group_size
from .logger import logger


def get_parallel_strategy(config):
    dp = config.parallel_config.data_parallel
    mp = config.parallel_config.model_parallel
    pp = config.parallel_config.pipeline_stage
    return dp, mp, pp


def get_device_num():
    return get_real_group_size()


def get_server_num():
    path = os.getenv('RANK_TABLE_FILE', None)
    if path is None:
        return 1
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return int(data['server_count'])


def _check_mode(config, mode):
    """rules with different mode"""
    if mode == 'train':
        if config.model.model_config.use_past:
            config.model.model_config.use_past = False
            logger.warning("use_past could not be used in train mode, "
                           "it has been forced to False")
    elif mode == 'predict':
        if config.model.model_config.compute_type == 'bfloat16':
            config.model.model_config.compute_type = 'float16'
            logger.warning("cast compute_type because predict param need float16 but get bfloat16")
        if config.model.model_config.param_init_type == 'bfloat16':
            config.model.model_config.param_init_type = 'float16'
            logger.warning("cast param_init_type because predict param need float16 but get bfloat16")
        _rule_fa_only_for_train(config, mode)
        _rule_pp_only_for_train(config, mode)
    elif mode == 'eval':
        _rule_fa_only_for_train(config, mode)
        _rule_pp_only_for_train(config, mode)
    elif mode == 'export':
        _rule_fa_only_for_train(config, mode)
        _rule_pp_only_for_train(config, mode)
    else:
        raise ValueError(f"mode should be in ['train', 'predict', 'eval', 'export'], but get {mode}")


def _rule_fa_only_for_train(config, mode):
    """flash attention only support training for now."""
    if config.model.model_config.use_flash_attention:
        config.model.model_config.use_flash_attention = False
        logger.warning("Flash attention only support training process for now, "
                       f"disable use_flash_attention in {mode} mode.")


def _rule_pp_only_for_train(config, mode):
    """pp only support training for now"""
    _, _, pp = get_parallel_strategy(config)
    if pp > 1:
        raise ValueError(f"pp only support training process for now, set pp=1 to {mode} ")


def _check_full_batch():
    """check full_batch"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    full_batch = ms.get_auto_parallel_context("full_batch")
    if parallel_mode not in ["semi_auto_parallel", "auto_parallel"] and full_batch:
        ms.set_auto_parallel_context(full_batch=False)
        logger.warning(f"full_batch could only be used under semi_auto_parallel or auto_parallel, "
                       f"but get {parallel_mode}, full_batch has been forced to False")


def _check_parallel(config):
    """check parallel config"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    dp, mp, pp = get_parallel_strategy(config)
    device_num = get_device_num()
    server_num = get_server_num()
    if parallel_mode in ["semi_auto_parallel"]:
        if dp * mp * pp != device_num:
            raise ValueError(f"The parallel config data_parallel * model_parallel * pipeline_stage should "
                             f"be equal to device_num, but get dp*mp*pp = {dp}*{mp}*{pp} = {dp * mp * pp} "
                             f"!= device_num({device_num})")

        if config.model.model_config.num_layers and config.model.model_config.num_layers < pp:
            raise ValueError(f"num_layers of model should be greater than or equal to pipeline_stage, but get "
                             f"num_layers ({config.model.model_config.num_layers}) < pp({pp})")

        if server_num > 1:
            if server_num % pp != 0:
                logger.warning(f"server_num % pipeline_stage = {server_num} % {pp} = {server_num % pp} != 0, "
                               f"which may cause parallel error when using multiple servers")

        if config.parallel.enable_parallel_optimizer:
            if config.model.model_config.vocab_size and config.model.model_config.vocab_size % device_num != 0:
                logger.warning(f"vocab_size({config.model.model_config.vocab_size}) % device_num({device_num})"
                               f" = {config.model.model_config.vocab_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")
            if config.model.model_config.hidden_size and config.model.model_config.hidden_size % device_num != 0:
                logger.warning(f"hidden_size({config.model.model_config.hidden_size}) % device_num({device_num})"
                               f" = {config.model.model_config.hidden_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")


def _check_keyword_gen_dataset(config, mode, **kwargs):
    """
    check seq_len keyword_gen_dataset satisfy
    `seq_length = max_source_length + max_target_length + 1` in train_dataset or
    `seq_length = max_source_length` in eval_dataset
    """
    dataset = kwargs.get('dataset', None)
    model = kwargs.get('network', None)
    # if model or dataset was not generate from config,
    # skip this verification
    if model or dataset:
        return

    train_dataset = config.train_dataset
    eval_dataset = config.eval_dataset
    seq_length = config.model.model_config.seq_length

    def raise_error_msg(phase, max_source_length, max_target_length, dataset_phase):
        """generate error message"""
        if phase == "train" and max_source_length + max_target_length + 1 != seq_length:
            raise ValueError(f"make sure `seq_length = max_source_length + max_target_length + 1`, "
                             f"but got seq_length={seq_length}, "
                             f"max_source_length={max_source_length}, "
                             f"max_target_length={max_target_length} in {dataset_phase}.")
        if phase == "eval" and max_source_length != seq_length:
            raise ValueError(f"make sure `seq_length = max_source_length`, "
                             f"but got seq_length={seq_length}, "
                             f"max_source_length={max_source_length} in {dataset_phase}.")

    if train_dataset and train_dataset.data_loader.type == "ADGenDataLoader" and mode == 'train':
        # verify train_dataset
        raise_error_msg("train", train_dataset.max_source_length, train_dataset.max_target_length, "train_dataset")

        # when do_eval == True, using ADGENMetric, verify eval_dataset
        if config.do_eval and config.metric.type == "ADGENMetric":
            raise_error_msg("eval", eval_dataset.max_source_length, eval_dataset.max_target_length, "eval_dataset")

        # when do_eval == True, using PerplexityMetric, verify eval_dataset
        if config.do_eval and config.metric.type == "PerplexityMetric":
            raise_error_msg("train", eval_dataset.max_source_length, eval_dataset.max_target_length, "eval_dataset")

    if eval_dataset and eval_dataset.data_loader.type == "ADGenDataLoader" and mode == 'eval':
        # verify eval_dataset
        if config.metric.type == "ADGENMetric":
            raise_error_msg("eval", eval_dataset.max_source_length, eval_dataset.max_target_length, "eval_dataset")

        if config.metric.type == "PerplexityMetric":
            raise_error_msg("train", eval_dataset.max_source_length, eval_dataset.max_target_length, "eval_dataset")

    # when do_eval == True, eval_dataset should be in train mode
    if config.metric and config.metric.type == "PerplexityMetric" and \
       eval_dataset and eval_dataset.data_loader.phase != 'train':
        logger.warning("when using 'PerplexityMetric', eval_dataset.data_loader.phase would be set to 'train'.")
        eval_dataset.data_loader.phase = 'train'
        config.eval_dataset_task.dataset_config.data_loader.phase = eval_dataset.data_loader.phase


def check_rules(config, mode='train', **kwargs):
    """check rules"""
    _check_mode(config, mode)
    _check_full_batch()
    _check_parallel(config)
    _check_keyword_gen_dataset(config, mode, **kwargs)
