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
"""Get resume ckpt."""
import os

from mindspore import load_checkpoint

from mindformers.tools.logger import logger
from mindformers.tools.resume_ckpt import get_resume_checkpoint_by_meta
from mindformers.tools.utils import get_real_rank, is_publicly_accessible_path, replace_rank_id_in_ckpt_name
from mindformers.trainer.utils import get_last_checkpoint, is_hyper_param_existed_in_sf_dir


def get_resume_checkpoint(checkpoint_dir, resume_training, ckpt_format='ckpt'):
    """get resume checkpoint."""
    rank_id = get_real_rank()
    checkpoint_dir_path = os.path.join(checkpoint_dir, f"rank_{rank_id}")

    if not resume_training:
        return None

    if isinstance(resume_training, str):
        if not resume_training.endswith(f".{ckpt_format}"):
            resume_training = f"{resume_training}.{ckpt_format}"
        resume_training = replace_rank_id_in_ckpt_name(resume_training, rank_id)
        logger.info("Specify resume checkpoint: %s",
                    os.path.join(checkpoint_dir, f"rank_{rank_id}", resume_training))
        return os.path.join(checkpoint_dir, f"rank_{rank_id}", resume_training)

    if not is_publicly_accessible_path(checkpoint_dir):
        return get_last_checkpoint(checkpoint_dir_path, ckpt_format)

    # if load checkpoint is complete safetensors, return checkpoint_dir
    if is_hyper_param_existed_in_sf_dir(checkpoint_dir, ckpt_format):
        return checkpoint_dir

    resume_ckpt = get_resume_checkpoint_by_meta(checkpoint_dir, ckpt_format)
    if not isinstance(resume_ckpt, str):
        return get_last_checkpoint(checkpoint_dir_path, ckpt_format)
    return os.path.join(checkpoint_dir, f"rank_{rank_id}", resume_ckpt)


def load_resume_checkpoint(load_checkpoint_path, remove_redundancy, load_ckpt_format):
    """resume training, load training info from checkpoint to config"""
    if not os.path.realpath(load_checkpoint_path) or \
            not os.path.exists(load_checkpoint_path):
        raise FileNotFoundError(f"The load_checkpoint_path must be correct, but get {load_checkpoint_path}")

    if os.path.isdir(load_checkpoint_path):
        hyper_param_file = os.path.join(load_checkpoint_path, 'hyper_param.safetensors')
        resume_param = load_checkpoint(hyper_param_file, format='safetensors')
        resume_dict = {'loss_scale': resume_param['loss_scale'],
                       'epoch_num': resume_param['epoch_num'],
                       'step_num': resume_param['step_num'],
                       'global_batch_size': resume_param['global_batch_size']}
    else:
        resume_dict = load_checkpoint(
            load_checkpoint_path,
            format=load_ckpt_format, remove_redundancy=remove_redundancy)

    return resume_dict
