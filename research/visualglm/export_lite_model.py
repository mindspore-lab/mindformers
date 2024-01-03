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

""" export mindir model for mindspore lite."""

import argparse
import os

import mindspore as ms

from mindformers.tools import logger
from mindformers.tools.utils import str2bool
from mindformers.tools.utils import get_output_subpath
from visualglm import VisualGLMImageToTextGeneration
from visualglm_config import VisualGLMConfig


def init_context(device_id):
    """
    init context
    :param device_id: npu device id
    """
    ms.set_context(mode=0, device_target="Ascend", device_id=device_id, max_device_memory="30GB", save_graphs=False)


def main(args):
    init_context(device_id=args.device_id)

    model_config = VisualGLMConfig.from_pretrained(args.config_path)

    model_config.max_txt_len = args.seq_length

    if args.checkpoint is not None:
        print(f"checkpoint: {args.checkpoint}")
        model_config.checkpoint_name_or_path = args.checkpoint

    if args.batch_size > 1:
        model_config.batch_size = args.batch_size
    else:
        model_config.batch_size = 1

    model_config.text_config.batch_size = model_config.batch_size
    model_config.text_config.seq_length = args.seq_length + model_config.qformer_config.query_length
    model_config.text_config.do_sample = args.do_sample
    model_config.text_config.top_p = args.top_p
    model_config.text_config.top_k = args.top_k
    model_config.text_config.use_past = args.use_past

    model = VisualGLMImageToTextGeneration(model_config)

    if args.mode == "export":
        model.set_train(False)
        rank_id = int(os.getenv("RANK_ID", "0"))
        model.add_flags_recursive(is_first_iteration=True)
        full_inputs = model.llm_model.prepare_inputs_for_export(full_model=True)
        save_path = get_output_subpath(
            f"mindir_full_checkpoint_bs_{model_config.batch_size}_{model_config.text_config.seq_length}",
            rank_id)
        ms.export(model.llm_model, *full_inputs, file_name=save_path, file_format='MINDIR')

        if model_config.text_config.use_past:
            model.add_flags_recursive(is_first_iteration=False)
            inc_inputs = model.llm_model.prepare_inputs_for_export(full_model=False)
            save_path = get_output_subpath(
                f"mindir_inc_checkpoint_bs_{model_config.batch_size}_{model_config.text_config.seq_length}",
                rank_id)
            ms.export(model.llm_model, *inc_inputs, file_name=save_path, file_format='MINDIR')

        logger.info(".........Export Over!.............")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="visualglm_6b", type=str, required=False,
                        help='model type')

    parser.add_argument('--config_path', default="run_visualglm_6b_image_to_text_generation.yaml", type=str,
                        required=False, help='config path')

    parser.add_argument('--device_id', type=int, default=1, required=False, help='device id')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch_size')

    parser.add_argument('--checkpoint', type=str, default=None, required=False, help='checkpoint path')

    parser.add_argument('--generate_repeat_time', type=int, default=5, required=False,
                        help='generate repeat time')

    parser.add_argument('--use_past', type=str2bool, default=False, required=False,
                        help='whether use past')

    parser.add_argument('--do_sample', type=str2bool, default=False, required=False,
                        help='whether do sample')

    parser.add_argument('--top_p', type=float, default=1, required=False, help='top p')

    parser.add_argument('--top_k', type=int, default=0, required=False, help='top k')

    parser.add_argument('--seq_length', type=int, default=32, required=False, help='seq length')

    parser.add_argument('--mode', type=str, default="export", required=False, help='')

    args_ = parser.parse_args()
    print(args_)
    main(args_)
