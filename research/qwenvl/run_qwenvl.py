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
"""QwenVL task script"""
import argparse
import os
import shutil
import sys
from typing import Optional, List

import mindspore as ms
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

from mindformers import Trainer, MindFormerConfig, CLIPVisionConfig, MindFormerRegister, MindFormerModuleType, \
    BatchNormalize, BatchToTensor, build_profile_cb
from mindformers.models import build_processor
from mindformers.core.context import build_context
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import check_in_modelarts, str2bool

from qwen.qwen_config import QwenConfig
from qwenvl import QwenVL
from qwenvl_config import QwenVLConfig
from qwenvl_dataloader import QwenVLDataLoader
from qwenvl_dataset import QwenVLDataset
from qwenvl_processor import QwenVLImageProcessor
from qwenvl_processor import QwenVLProcessor
from qwenvl_tokenizer import QwenVLTokenizer
from qwenvl_transform import QwenVLTransform


def register_modules():
    MindFormerRegister.register_cls(QwenVLConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLDataset, MindFormerModuleType.DATASET)
    MindFormerRegister.register_cls(QwenVLDataLoader, MindFormerModuleType.DATASET_LOADER)
    MindFormerRegister.register_cls(QwenVLTokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(QwenVLTransform, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(QwenVLProcessor, MindFormerModuleType.PROCESSOR)
    MindFormerRegister.register_cls(QwenVLImageProcessor, MindFormerModuleType.PROCESSOR)

    MindFormerRegister.register_cls(BatchNormalize, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(BatchToTensor, MindFormerModuleType.TRANSFORMS)


if check_in_modelarts():
    import moxing as mox


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        obs_strategy_dir = os.path.join(config.remote_save_url, "strategy")
        if mox.file.exists(obs_strategy_dir) and config.local_rank == 0:
            mox.file.remove(obs_strategy_dir, recursive=True)
        obs_transformed_ckpt_dir = os.path.join(config.remote_save_url, "transformed_checkpoint")
        if mox.file.exists(obs_transformed_ckpt_dir) and config.local_rank == 0:
            mox.file.remove(obs_transformed_ckpt_dir, recursive=True)
        mox.file.make_dirs(obs_strategy_dir)
        mox.file.make_dirs(obs_transformed_ckpt_dir)
    else:
        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(strategy_dir)
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(transformed_ckpt_dir)
        os.makedirs(strategy_dir, exist_ok=True)
        os.makedirs(transformed_ckpt_dir, exist_ok=True)


def main(config='run_qwenvl_910b.yaml',
         run_mode='predict',
         use_parallel=False,
         use_past=None,
         ckpt=None,
         auto_trans_ckpt=None,
         vocab_file=None,
         image_path=Optional[List[str]],
         image_size=None,
         prompt="",
         seq_length=None,
         max_length=512,
         device_id=0,
         do_sample=None,
         top_k=None,
         top_p=None):
    """main function."""

    yaml_path = os.path.expanduser(config)
    assert os.path.exists(yaml_path)

    config = MindFormerConfig(os.path.realpath(yaml_path))
    if vocab_file:
        assert os.path.exists(vocab_file)
        config.processor.tokenizer.vocab_file = vocab_file
        config.train_dataset.tokenizer.vocab_file = vocab_file
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if device_id is not None:
        config.context.device_id = device_id

    # init context
    build_context(config)

    if config.profile:
        config.profile_cb = build_profile_cb(config)

    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
        if config.auto_trans_ckpt:
            clear_auto_trans_output(config)

    if use_past is not None:
        config.model.model_config.use_past = use_past
        config.model.model_config.text_config.use_past = use_past

    if seq_length is not None:
        config.model.model_config.text_config.seq_length = seq_length
        config.processor.tokenizer.max_length = seq_length
        if run_mode != "predict":
            config.train_dataset.text_transforms.max_length = seq_length + 1

    if do_sample is not None:
        config.model.model_config.do_sample = do_sample
    if top_k is not None:
        config.model.model_config.top_k = top_k
    if top_p is not None:
        config.model.model_config.top_p = top_p

    if ckpt is not None and run_mode == "predict":
        config.model.model_config.checkpoint_name_or_path = ckpt

    if image_size is None:
        image_size = config.model.model_config.vision_config.image_size

    config.model.model_config.vision_config.image_size = image_size
    if run_mode in ['train', 'finetune']:
        config.model.model_config.use_past = False
        config.model.model_config.text_config.use_past = False
        config.model.model_config.vision_config.image_size = image_size
    else:
        config.processor.image_processor.image_size = image_size

    model_config = config.model.model_config

    if run_mode == 'predict':
        query = []
        if image_path:
            query = [{'image': path} for path in image_path]
        if prompt:
            query.append({'text': prompt})

        model_config.text_config.batch_size = len(query)
        model_config.text_config = QwenConfig(**model_config.text_config)
        model_config.vision_config = CLIPVisionConfig(**model_config.vision_config)
        model_config = QwenVLConfig(**model_config)
        model = QwenVL(model_config)
        model.load_checkpoint(model_config)

        processor = build_processor(config.processor)
        tokenizer = processor.tokenizer

        text_input = tokenizer.from_list_format(query)
        process_res = processor(text_input=text_input)
        input_id = np.expand_dims(process_res.get("text")[0], axis=0)
        input_image = process_res.get("image")[0].unsqueeze(0)
        img_pos = ms.Tensor(process_res.get("img_pos"), ms.int32)
        for _ in range(5):
            result = model.generate(input_id, images=input_image, img_pos=img_pos)
            result = processor.tokenizer.decode(result, skip_special_tokens=False)
            result = processor.tokenizer.post_process(result[0], query)
            print(result)
    elif run_mode == 'train':
        trainer = Trainer(args=config, task="contrastive_language_image_pretrain")
        trainer.train()
    elif run_mode == 'finetune':
        trainer = Trainer(args=config, task="contrastive_language_image_pretrain")
        trainer.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=auto_trans_ckpt)
    else:
        raise NotImplementedError(f"run_mode '${run_mode}' not supported yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='qwenvl/run_qwenvl_stage1_910b.yaml',
                        type=str,
                        help='config file path.')
    parser.add_argument('--run_mode', default='predict', type=str,
                        help='set run mode for model.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--vocab_file',
                        default="qwenvl/qwen.tiktoken",
                        type=str,
                        help='tokenizer model')
    parser.add_argument('--image_path',
                        default='', type=str, nargs='*',
                        help='input predict data.')
    parser.add_argument('--image_size', default=None, type=int, help='image size')
    parser.add_argument('--prompt', default='', type=str, help='input predict data.')
    parser.add_argument('--seq_length', default=None, type=int, help='seq_length')
    parser.add_argument('--predict_length', default=512, type=int, help='max length for predict output.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--optimizer_parallel', default=False, type=str2bool,
                        help='whether use optimizer parallel. Default: False')
    parser.add_argument('--device_id', default=-1, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    parser.add_argument('--use_past', default=None, type=str2bool,
                        help='use_past')
    parser.add_argument('--do_sample', default=None, type=str2bool,
                        help='do_sample')
    parser.add_argument('--top_k', default=None, type=int,
                        help='top_k')
    parser.add_argument('--top_p', default=None, type=float,
                        help='top_p')

    args = parser.parse_args()

    if args.device_id == -1:
        args.device_id = int(os.getenv("RANK_ID", "0"))

    register_modules()

    main(config=args.config,
         run_mode=args.run_mode,
         use_parallel=args.use_parallel,
         use_past=args.use_past,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         vocab_file=args.vocab_file,
         image_path=args.image_path,
         image_size=args.image_size,
         prompt=args.prompt,
         seq_length=args.seq_length,
         max_length=args.predict_length,
         device_id=args.device_id,
         do_sample=args.do_sample,
         top_k=args.top_k,
         top_p=args.top_p)
