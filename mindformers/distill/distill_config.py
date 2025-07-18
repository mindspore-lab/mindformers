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
Note: The config module of Parameter Efficient Tuning module.
"""
import os

from mindspore._checkparam import args_type_check

from mindformers.tools import DictConfig, MindFormerConfig
from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    is_publicly_accessible_path,
    clear_auto_trans_output,
    get_output_root_path,
    get_device_num_per_node
)


__all__ = ['DistillationConfig']


class DistillationConfig(DictConfig):
    """
    The configuration base class for Distillation algorithms.

    Args:
        teacher_config (MindFormerConfig):
            The config instance of teacher model.
        skip_lm_loss (bool, optional):
            Whether to skip lm loss of student model. Default: True.
        temperature (float, optional):
            The soften coefficient of Knowledge-Distillation. Default: 1.0.
        kd_loss_scale (float, optional):
            The coefficient of kd loss. Only take effects when skip_lm_loss = False. Default: 1.0.
        student_need_map (bool, optional):
            Whether the student checkpoint files need to add a "student_model" prefix,
            e.g. when loading HuggingFace checkpoint. Default: True.
        teacher_need_map (bool, optional):
            Whether the teacher checkpoint files need to add a "teacher_model" prefix,
            e.g. when loading HuggingFace checkpoint. Default: True.

    Returns:
        An instance of DistillConfig.
    """

    @args_type_check(teacher_config=MindFormerConfig, skip_lm_loss=bool, temperature=(float, int),
                     kd_loss_scale=(float, int), student_need_map=bool, teacher_need_map=bool)
    def __init__(self,
                 teacher_config,
                 skip_lm_loss: bool = True,
                 temperature: float = 1.0,
                 kd_loss_scale: float = 1.0,
                 student_need_map: bool = True,
                 teacher_need_map: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_config = teacher_config
        self.skip_lm_loss = skip_lm_loss
        self.temperature = temperature
        self.kd_loss_scale = kd_loss_scale
        self.student_need_map = student_need_map
        self.teacher_need_map = teacher_need_map
        if self.temperature <= 0:
            raise ValueError(f"The `temperature` should be a positive number, but get {self.temperature}.")
        if self.kd_loss_scale <= 0:
            raise ValueError(f"The `kd_loss_scale` should be a positive number, but get {self.kd_loss_scale}.")
        self.check_config()

    def check_config(self):
        """Check config rules."""
        logger.info("Checking teacher_config...")
        self._check_config_type()
        if not self.teacher_config.model or not self.teacher_config.model.model_config:
            raise ValueError("The model.model_config is not found in teacher_config, Please check the config setting.")
        self._try_use_pretrained_model_dir_as_ckpt()
        self._validate_auto_trans_ckpt_requirements()
        self._using_checkpoint_name_or_path_if_needed()
        self._clear_redundant_model_checkpoint_name()
        if not self.teacher_config.load_checkpoint:
            logger.warning("The load_checkpoint in teacher_config is not found, "
                           "so teacher model will not load any checkpoint!")

    def _check_config_type(self):
        """Check config rules."""
        config = self.teacher_config
        if config.auto_trans_ckpt is not None and not isinstance(config.auto_trans_ckpt, bool):
            raise TypeError(f"auto_trans_ckpt must be bool, but get {config.auto_trans_ckpt}")
        if isinstance(config.metric, dict):
            config.metric = [config.metric]

    def _try_use_pretrained_model_dir_as_ckpt(self):
        """Use `pretrained_model_dir` as fallback for checkpoint if applicable."""
        config = self.teacher_config
        if not config.load_checkpoint and config.pretrained_model_dir:
            from mindformers.utils import contains_safetensors_files
            if contains_safetensors_files(config.pretrained_model_dir):
                config.load_checkpoint = config.pretrained_model_dir
                logger.info(f'Parameter load_checkpoint does not set the weight path. Defaulting to '
                            f'pretrained_model_dir: {config.pretrained_model_dir}')
            else:
                logger.info(f'Pretrained_model_dir: {config.pretrained_model_dir} '
                            f'does not contain any safetensors file and load_checkpoint is empty. '
                            f'No weights will be loaded.')

    def _validate_auto_trans_ckpt_requirements(self):
        """Ensure auto_trans_ckpt has shared directory requirements met."""
        config = self.teacher_config
        if config.auto_trans_ckpt and config.load_ckpt_format == 'ckpt':
            if not is_publicly_accessible_path(get_output_root_path()):
                raise ValueError(f"When device num > {get_device_num_per_node()} and auto_trans_ckpt is set to True, "
                                 f"the output_dir should be a shared directory that can be accessed by all nodes. "
                                 f"But {os.path.abspath(config.output_dir)} is not a shared directory.")
            clear_auto_trans_output(config.load_checkpoint, config.src_strategy_path_or_dir)

    def _using_checkpoint_name_or_path_if_needed(self):
        """Using model_config.checkpoint_name_or_path if possible."""
        config = self.teacher_config
        if config.auto_trans_ckpt and not config.load_checkpoint:
            if config.model and config.model.model_config.checkpoint_name_or_path:
                config.load_checkpoint = config.model.model_config.checkpoint_name_or_path
                config.model.model_config.checkpoint_name_or_path = None
            else:
                raise ValueError("When `auto_trans_ckpt` is True, `load_checkpoint` should not be empty or None.")

    def _clear_redundant_model_checkpoint_name(self):
        """Clear model_config.checkpoint_name_or_path if load_checkpoint is set."""
        config = self.teacher_config
        if config.load_checkpoint and config.model and config.model.model_config.checkpoint_name_or_path:
            config.model.model_config.checkpoint_name_or_path = None
            logger.info("The `load_checkpoint` is set; `checkpoint_name_or_path` will be cleared.")
