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
"""export wizardcoder inc"""
import argparse
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch_size')
parser.add_argument('--seq_length', default=2048, type=int,
                    help='batch_size')
parser.add_argument('--model_path', default='', type=str,
                    help='model path')
parser.add_argument('--device_id', default=0, type=int,
                    help='set device id.')
args = parser.parse_args()

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
batch_size = args.batch_size
seq_length = args.seq_length


config = WizardCoderConfig(
    batch_size=batch_size,
    seq_length=seq_length,
    n_position=8192,
    vocab_size=49153,
    hidden_size=6144,
    num_layers=40,
    num_heads=48,
    eos_token_id=0,
    pad_token_id=49152,
    checkpoint_name_or_path=args.model_path,
    use_past=True
)

model = WizardCoderLMHeadModel(config)
model.set_train(False)

# 全量推理 prefill
model.add_flags_recursive(is_first_iteration=True)
input_ids = ms.Tensor(np.ones((batch_size, seq_length)), mstype.int32)
input_position = ms.Tensor([127]*batch_size, mstype.int32)
init_reset = ms.Tensor([False], mstype.bool_)
batch_valid_length = ms.Tensor([[128]*batch_size], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length,
          file_name=f"wizardcoder-15b_mslite_inc/prefill_seq{seq_length}_bs{batch_size}", file_format="MINDIR")

# 增量推理 decode
model.add_flags_recursive(is_first_iteration=False)
input_ids = ms.Tensor(np.ones((batch_size, 1)), mstype.int32)
input_position = ms.Tensor([128]*batch_size, mstype.int32)
init_reset = ms.Tensor([True], mstype.bool_)
batch_valid_length = ms.Tensor([[129]*batch_size], mstype.int32)
ms.export(model, input_ids, None, None, input_position, init_reset, batch_valid_length,
          file_name=f"wizardcoder-15b_mslite_inc/decode_seq{seq_length}_bs{batch_size}", file_format="MINDIR")
