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
"""Check Model Input Config."""


def check_mim_model_config(config):
    """Check masked image modeling config."""
    if config.model.arch is not None and config.model.encoder is None:
        config.model.arch.image_size = config.runner_config.image_size
        config.model.arch.batch_size = config.runner_config.batch_size
    if config.model.encoder is not None:
        config.model.encoder.image_size = config.runner_config.image_size
        config.model.encoder.batch_size = config.runner_config.batch_size
