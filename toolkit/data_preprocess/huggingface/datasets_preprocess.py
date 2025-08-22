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
"""packing huggingface datasets"""

import sys
import argparse
import os
import time

from mindformers.tools import MindFormerConfig, logger
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader, process_legacy_args
from mindformers.core.context.build_context import MFContextOperator


class ProcessHFDataLoader(HFDataLoader):
    """HF dataset process module"""

    def __new__(cls, **kwargs):
        kwargs = process_legacy_args(**kwargs)
        config = cls._build_config(**kwargs)
        dataset = cls.load_dataset(config)
        cls.process_dataset(config, dataset)


def prepare_args():
    """prepare arguments for packing dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='A path of config file containing CommonDataLoader.')
    parser.add_argument('--save_path',
                        default="./packed_data",
                        type=str,
                        help='A directory to save packed dataset.')
    parser.add_argument('--register_path',
                        type=str,
                        default=None,
                        help='The register path of outer API. '
                             'This is usually the parent path of the Python file where the outer API is located.'
                             'This configuration can be ignored if it does not involve registering the outer API.')
    args = parser.parse_args()

    if args.register_path is not None:
        work_path = os.path.dirname(os.path.abspath(__file__))
        work_path = '/'.join(work_path.split('/')[:-3])  # set work path as project root
        if not os.path.isabs(args.register_path):
            args.register_path = os.path.join(work_path, args.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = args.register_path
        if args.register_path not in sys.path:
            sys.path.append(args.register_path)

    return args


def main():
    """main processing function"""
    start_time = time.time()
    # prepare arguments and load config
    args = prepare_args()
    config = MindFormerConfig(args.config)
    MFContextOperator(config)

    # build dataset
    dataloader_config = config.train_dataset.data_loader
    dataloader_config.handler.append(dict(type='save_to_disk', dataset_path=args.save_path))
    ProcessHFDataLoader(**dataloader_config)

    end_time = time.time()
    logger.info(f"Processed datasets saved in {args.save_path}, spend {end_time - start_time:.4f}s.")


if __name__ == '__main__':
    main()
