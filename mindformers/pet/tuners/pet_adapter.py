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
"""
Note: PET Adapter is the base adapter class for Parameter Efficient Tuning of MindFormers.
"""
import os

from mindspore import nn

from tk.graph.freeze_utils import freeze_delta

from mindformers.auto_class import AutoModel
from mindformers.mindformer_book import MindFormerBook
from ..pet_config import PetConfig
from ..constants import BaseModelInitType, PetType


class PetAdapter:
    r"""
    PetAdapter is the base class of adapter to modify the pretrained model.
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        """Add efficient tuning parameters to ptm."""
        raise NotImplementedError("should implemented by the certain tuning algorithm.")

    @classmethod
    def get_pretrained_model(cls, config):
        """
        Get pretrained model from config.
        """
        init_type = config.base_model.init_type
        if init_type == BaseModelInitType.MODEL_CONFIG:
            config_path = os.path.join(MindFormerBook.get_project_path(),
                                       'configs', 'clip', 'model_config', config.base_model.model_config)
            base_model = AutoModel.from_config(config_path)
            if config.base_model.model_ckpt is None:
                raise ValueError("init base from model config, the model ckpt cannot be None")
        elif init_type == BaseModelInitType.MODEL_NAME:
            base_model = AutoModel.from_pretrained(config.base_model_name)
        elif init_type == BaseModelInitType.MODEL_DIR:
            checkpoint_dir = os.path.join(MindFormerBook.get_default_checkpoint_download_folder(),
                                          config.base_model_dir_name)
            base_model = AutoModel.from_pretrained(checkpoint_dir)
        else:
            raise ValueError("type of init base model must in BaseModelInitType")
        return base_model

    @classmethod
    def freeze_pretrained_model(cls, model, pet_type: PetType):
        """
        Freeze the parameters of ptm which no update in the tuning process.

        Notes:
            Refer to mindpet api.
        """
        freeze_delta(model, pet_type)
