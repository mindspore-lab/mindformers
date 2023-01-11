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
功能: model tasks
"""

from execution import execute_by_subprocess
from lk_utils import PathPair, is_valid_path, change_file_mod
from common_consts import MODEL_TASK_TYPES

from ma import FileTransporter
from ma.constants import MA_LOGGER, MA_DAVINCI_TOOL, CACHE_MODEL_CONFIG_PATH


def finetune(code_path, boot_file_path, model_config_path, pretrained_model_path, params):
    try:
        cmd = ["python", MA_DAVINCI_TOOL, "tk", "finetune", "--quiet", "--boot_file_path", boot_file_path]
        cmd.extend(params)
        if pretrained_model_path:
            if not is_valid_path(pretrained_model_path, is_folder=True):
                raise ValueError(f"Value of parameter pretrained_model_path is not valid.")
            cmd.extend(["--pretrained_model_path", pretrained_model_path])
        if model_config_path:
            model_config_path = FileTransporter(path_pair=PathPair(src_path=model_config_path,
                                                                   dst_path=CACHE_MODEL_CONFIG_PATH),
                                                is_folder=False,
                                                existence_detection=True,
                                                info="model_config").transport()
            change_file_mod(model_config_path)
            cmd.extend(["--model_config_path", model_config_path])
        execute_by_subprocess(cmd, cwd=code_path)
    except ValueError as ve:
        MA_LOGGER.error(ve)
        raise RuntimeError("Failed to launch model task due to invalid path value.")
    except RuntimeError as re:
        MA_LOGGER.error(re)
        raise RuntimeError("Failed to launch model task due to runtime exception.")


def evaluate(code_path, boot_file_path, model_config_path, ckpt_path, params):
    try:
        cmd = ["python", MA_DAVINCI_TOOL, "tk", "evaluate", "--quiet", "--boot_file_path", boot_file_path]
        cmd.extend(params)
        if not is_valid_path(ckpt_path, is_folder=True):
            raise ValueError(f"Value of parameter pretrained_model_path is not valid.")
        cmd.extend(["--ckpt_path", ckpt_path])
        if model_config_path:
            model_config_path = FileTransporter(path_pair=PathPair(src_path=model_config_path,
                                                                   dst_path=CACHE_MODEL_CONFIG_PATH),
                                                is_folder=False,
                                                existence_detection=True,
                                                info="model_config").transport()
            change_file_mod(model_config_path)
            cmd.extend(["--model_config_path", model_config_path])
        execute_by_subprocess(cmd, cwd=code_path)
    except ValueError as ve:
        MA_LOGGER.error(ve)
        raise RuntimeError("Failed to launch model task due to invalid path value.")
    except RuntimeError as re:
        MA_LOGGER.error(re)
        raise RuntimeError("Failed to launch model task due to runtime exception.")


def infer(code_path, boot_file_path, model_config_path, ckpt_path, params):
    try:
        cmd = ["python", MA_DAVINCI_TOOL, "tk", "infer", "--quiet", "--boot_file_path", boot_file_path]
        cmd.extend(params)
        if not is_valid_path(ckpt_path, is_folder=True):
            raise ValueError(f"Value of parameter pretrained_model_path is not valid.")
        cmd.extend(["--ckpt_path", ckpt_path])
        if model_config_path:
            model_config_path = FileTransporter(path_pair=PathPair(src_path=model_config_path,
                                                                   dst_path=CACHE_MODEL_CONFIG_PATH),
                                                is_folder=False,
                                                existence_detection=True,
                                                info="model_config").transport()
            change_file_mod(model_config_path)
            cmd.extend(["--model_config_path", model_config_path])
        execute_by_subprocess(cmd, cwd=code_path)
    except ValueError as ve:
        MA_LOGGER.error(ve)
        raise RuntimeError("Failed to launch model task due to invalid path value.")
    except RuntimeError as re:
        MA_LOGGER.error(re)
        raise RuntimeError("Failed to launch model task due to runtime exception.")


def call_task_func(args, params):
    if args.task_type not in MODEL_TASK_TYPES:
        MA_LOGGER.error("Failed to launch model task due to invalid model task type.")
        return False

    # call corresponding interface
    try:
        MA_LOGGER.info(f"Model task (type: {args.task_type}) starts.")
        if args.task_type == "finetune":
            finetune(code_path=args.code_path,
                     boot_file_path=args.boot_file_path,
                     model_config_path=args.model_config_path,
                     pretrained_model_path=args.pretrained_model_path,
                     params=params)
        elif args.task_type == "evaluate":
            evaluate(code_path=args.code_path,
                     boot_file_path=args.boot_file_path,
                     model_config_path=args.model_config_path,
                     ckpt_path=args.ckpt_path,
                     params=params)
        elif args.task_type == "infer":
            infer(code_path=args.code_path,
                  boot_file_path=args.boot_file_path,
                  model_config_path=args.model_config_path,
                  ckpt_path=args.ckpt_path,
                  params=params)
    except RuntimeError as re:
        MA_LOGGER.error(re)
        return False
    MA_LOGGER.info(f"Model task (type: {args.task_type}) is completed.")
    return True
