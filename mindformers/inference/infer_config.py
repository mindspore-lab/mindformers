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
"""lite config."""

from mindspore_lite import ModelType

from mindformers.tools import DictConfig


def convert_lite_model_type(model_type: str):
    """
    convert str to ModelType
    :param model_type: str
    :return: ModelType
    """
    if model_type == "mindir":
        return ModelType.MINDIR
    if model_type == "mindir_lite":
        return ModelType.MINDIR_LITE

    raise KeyError(f"Supported data type keywords include: "
                   f"[mindir, mindir_lite], but get {model_type}")


class InferConfig(DictConfig):
    """
    Inference config class.
    """
    def __init__(self,
                 prefill_model_path: str = "",
                 increment_model_path: str = "",
                 model_type: str = "mindir",
                 model_name: str = "common",
                 infer_seq_length: int = 1024,
                 target: str = "Ascend",
                 device_id: int = 0,
                 rank_id: int = 0,
                 ge_config_path: str = "",
                 dynamic: bool = False,
                 paged_attention: bool = False,
                 pa_block_size: int = 16,
                 pa_num_blocks: int = 512,
                 **kwargs):
        super(InferConfig, self).__init__(**kwargs)
        self.prefill_model_path = prefill_model_path
        self.increment_model_path = increment_model_path
        self.model_type = convert_lite_model_type(model_type)
        self.model_name = model_name
        self.seq_length = infer_seq_length
        self.target = target
        self.device_id = device_id
        self.rank_id = rank_id
        self.config_path = ge_config_path
        self.dynamic = dynamic
        self.paged_attention = paged_attention
        self.block_size = pa_block_size
        self.num_blocks = pa_num_blocks
