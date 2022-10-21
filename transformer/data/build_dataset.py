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

from mindspore import context
from .downstream_dataset import create_classification_dataset, create_squad_dataset, create_language_model_dataset
from .gpt_dataset import create_gpt_dataset
from .bert_dataset import create_bert_dataset
from .t5_dataset import create_t5_dataset
from .image_dataset import create_image_dataset
from .wiki_dataset import create_wiki_dataset


def build_dataset(opt, rank_id, device_num, get_eval_dataset=False):
    """get dataset from local or obs"""
    model_name = opt.arch
    url = opt.train_data_path if not get_eval_dataset else opt.eval_data_path
    if url.startswith == "s3://":
        # copy data from the cloud to the /cache/Data
        cache_url = '/cache/Data/'
        opt.logger.info(f"Find the data url {url} startswith s3. Start to cache the data_path "
                        f"to the local path {cache_url}.")
        download_data(src_data_path=url, tgt_data_path=cache_url, rank=rank_id)
        opt.logger.info(f"Data cache the finished.")
    else:
        cache_url = url

    opt.logger.info("Start to build the dataset.")
    ds = None

    if context.get_auto_parallel_context('full_batch'):
        opt.logger.info("Detect the full_batch import is true, modify the shard_num and shard_id to be 1 and 0."
                        "So each card will receive the same input data with "
                        f"batch size: {opt.model['global_batch_size']}")
        device_num = 1
        rank_id = 0

    opt.dataset_device_num = device_num
    opt.dataset_rank = rank_id
    opt.dataset_batch_size = opt.model['global_batch_size']
    opt.dataset_path = cache_url
    opt.dataset_drop_remainder = True
    opt.dataset_do_shuffle = True
    opt.dataset_schema_dir = None
    opt.dataset_bucket_list = None

    if model_name == 'gpt':
        ds = create_gpt_dataset(opt)
    elif model_name  in ('bert', 'nezha'):
        ds = create_bert_dataset(opt)
    elif model_name == 't5':
        ds = create_t5_dataset(opt)
    elif model_name == 'opt':
        ds = create_wiki_dataset(opt)
    elif model_name == 'vit':
        ds = create_image_dataset(opt)
    else:
        raise RuntimeError(f"Model name {opt.arch} is not supported yet.")
    opt.logger.info("End to build the dataset.")
    return ds



def build_downstream_dataset(opt, rank_id, device_num, get_eval_dataset=False, dataset_format='tfrecord',
                             schema_file_path=None, batch_size=1, data_file_path='', do_shuffle="true",
                             task_name="classifier", is_training=True):
    """get dataset from local or obs"""
    url = data_file_path if not get_eval_dataset else opt.eval_data_path
    if is_training and url.startswith == "s3://":
        # copy data from the cloud to the /cache/Data
        cache_url = '/cache/Data/'
        opt.logger.info(f"Find the data url { url} startswith s3. Start to cache the data_path "
                        f"to the local path {cache_url}.")
        download_data(src_data_path=url, tgt_data_path=cache_url, rank=rank_id)
        opt.logger.info(f"Data cache the finished.")
    else:
        cache_url = url

    opt.logger.info("Start to build the dataset.")
    ds = None

    if context.get_auto_parallel_context('full_batch'):
        opt.logger.info("Detect the full_batch import is true, modify the shard_num and shard_id to be 1 and 0."
                        "So each card will receive the same input data with "
                        f"batch size: {opt.model['global_batch_size']}")
        device_num = 1
        rank_id = 0

    opt.dataset_format = dataset_format
    opt.dataset_device_num = device_num
    opt.dataset_rank = rank_id
    opt.dataset_batch_size = batch_size
    opt.dataset_path = cache_url
    opt.dataset_drop_remainder = True
    opt.dataset_do_shuffle = do_shuffle
    opt.dataset_schema_dir = schema_file_path
    opt.dataset_bucket_list = None
    opt.repeat_count = 1

    if task_name == "classifier":
        ds = create_classification_dataset(opt)
    elif task_name == "squad":
        ds = create_squad_dataset(opt)
    elif task_name == "language_model":
        ds = create_language_model_dataset(opt)
    opt.logger.info("End to build the dataset.")
    return ds
