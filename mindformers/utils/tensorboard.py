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
"""TENSORBOARD Module"""
from tensorboardX import SummaryWriter

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig

_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TENSORBOARD_WRITER_ARGS = {'log_timers_to_tensorboard': False,
                                   'log_loss_scale_to_tensorboard': False
                                   }


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def _set_tensorboard_writer(config):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER

    if hasattr(config, 'tensorboard_dir') and config.tensorboard_dir:
        if _GLOBAL_TENSORBOARD_WRITER is not None:
            logger.warning("tensorboard writer is already initialized.")
        else:
            logger.info('......setting tensorboard......')
            queue_size = getattr(config, 'tensorboard_queue_size', 10)
            if not queue_size:
                queue_size = 10
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=config.tensorboard_dir,
                max_queue=queue_size)


def _unset_tensorboard_writer():
    """Unset tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    if _GLOBAL_TENSORBOARD_WRITER is not None:
        if isinstance(_GLOBAL_TENSORBOARD_WRITER, SummaryWriter):
            _GLOBAL_TENSORBOARD_WRITER.close()
        else:
            logger.warning("The global tensorboard writer is not an instance of SummaryWriter.")
        _GLOBAL_TENSORBOARD_WRITER = None


def write_args_to_tensorboard(config):
    """Write arguments to tensorboard."""
    global _GLOBAL_TENSORBOARD_WRITER
    if _GLOBAL_TENSORBOARD_WRITER is None:
        raise ValueError("tensorboard writer is not set.")
    writer = get_tensorboard_writer()
    if isinstance(config, MindFormerConfig):
        for key, value in config.items():
            writer.add_text(key, str(value), global_step=1)
    else:
        logger.warning("Only MindFormerConfig will be written.")


def update_tensorboard_args(config):
    """Update args for tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER_ARGS
    for key in _GLOBAL_TENSORBOARD_WRITER_ARGS:
        if isinstance(config, MindFormerConfig):
            if config.get_value(key) is not None:
                _GLOBAL_TENSORBOARD_WRITER_ARGS[key] = config.get_value(key)
        else:
            if hasattr(config, key) and getattr(config, key) is not None:
                _GLOBAL_TENSORBOARD_WRITER_ARGS[key] = getattr(config, key)


def get_tensorboard_args():
    """Return args for tensorboard writer."""
    return _GLOBAL_TENSORBOARD_WRITER_ARGS
