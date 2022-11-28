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
"""Base Dataset."""
import os

import mindspore.dataset as ds


class BaseDataset:
    """Base Dataset."""
    def __init__(self, config: dict = None):
        self.dataset_config = config
        self.init_dataset_config()

    def init_dataset_config(self):
        """Init dataset config."""
        ds.config.set_seed(self.dataset_config.seed)
        ds.config.set_prefetch_size(self.dataset_config.prefetch_size)
        ds.config.set_numa_enable(self.dataset_config.numa_enable)

        if self.dataset_config.auto_tune:
            if self.dataset_config.profile:
                raise EnvironmentError(
                    "MindSpore's AutoTune is enabled, so Profile cannot be enabled,"
                    "now Profile's flag is True, please set to False!")
            os.makedirs(self.dataset_config.filepath_prefix, exist_ok=True)
            self.dataset_config.filepath_prefix = os.path.join(self.dataset_config.filepath_prefix, "autotune")
            ds.config.set_enable_autotune(True, filepath_prefix=self.dataset_config.filepath_prefix)
            ds.config.set_autotune_interval(self.dataset_config.autotune_per_step)
