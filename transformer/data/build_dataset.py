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
Data operations.
"""
from transformer.utils import download_data

from .gpt_dataset import create_dataset
from .bert_dataset import create_bert_dataset
from .t5_dataset import create_t5_dataset


def build_dataset(opt, rank_id, device_num):
    """get dataset from local or obs"""
    model_name = opt.arch
    if opt.data_url.startswith == "s3://":
        # copy data from the cloud to the /cache/Data
        cache_url = '/cache/Data/'
        opt.logger.info(f"Find the data url { opt.data_url} startswith s3. Start to cache the data_url "
                        f"to the local path {cache_url}.")
        download_data(src_data_url=opt.data_url, tgt_data_path=cache_url, rank=rank_id)
        opt.logger.info(f"Data cache the finished.")
    else:
        cache_url = opt.data_url

    opt.logger.info("Start to build the dataset.")
    ds = None
    if model_name == 'gpt':
        ds = create_dataset(opt.model['global_batch_size'], data_path=cache_url, device_num=device_num, rank=rank_id)
    elif model_name == 'bert':
        ds = create_bert_dataset(device_num, rank_id, data_dir=cache_url, batch_size=opt.model['global_batch_size'])
    elif model_name == 't5':
        ds = create_t5_dataset(opt.model['global_batch_size'], data_path=cache_url,
                               device_num=device_num, rank=rank_id)
    else:
        raise RuntimeError(f"Model name {opt.arch} is not supported yet.")
    opt.logger.info("End to build the dataset.")
    return ds
