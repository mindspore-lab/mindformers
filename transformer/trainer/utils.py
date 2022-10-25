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
Use for generating words.
"""
import numpy as np

from mindspore import Tensor
from mindspore.common import dtype as mstype
from transformer.tokenization import tokenization
from transformer.tokenization.tokenization import FullTokenizer
from transformer.generate import generate


def get_acc(model, dataset, opt=None):
    """ calculate accuracy for input dataset """
    if opt.dataset_name == 'imagenet':
        # get accuracy for ViT on the imagenet dataset
        acc_num = 0
        for data in dataset:
            input_image = data[0].asnumpy().astype(np.float32)
            label = data[1].asnumpy().astype(np.int32)
            logits = model.predict(Tensor(input_image, mstype.float32)).asnumpy()
            y_pred = np.argmax(logits, axis=1)

            equals = label.reshape(-1) == y_pred
            acc_num += sum(equals)

        acc = acc_num / 50000
        return acc

    total_num = 0
    acc_num = 0
    for data in dataset:
        input_ids = data[0].asnumpy().astype(np.int32)
        input_mask = data[1].asnumpy().astype(np.int32)
        label = data[2].asnumpy().astype(np.int32)
        logits = model.predict(Tensor(input_ids, mstype.int32), Tensor(input_mask, mstype.float32)).asnumpy()

        equals = label.reshape(-1) == logits
        total_num += np.prod(label.shape)
        acc_num += sum(equals)

    acc = acc_num / total_num
    return acc


def generate_words(sample, predict_model, opt):
    """
    Generate the word given the input prompt, model and configs

    Args:
        sample(str): The input prompt. For example, it can be "Today is a good day, I want to".
        predict_model(Model): the model that need to run prediction.
        opt(argparse.Namespace): The global configs.

    Inputs:
        input_ids: the tokenized inputs
        attention_mask: the attention mask with [bs, seq_length]. 1 means effective and 0 mean it should be masked.
        labels:
    Returns:
        output: Tensor, the loss of the network
    """
    # Tokenize input sentence to ids
    eval_opts = opt
    tokenizer = FullTokenizer(eval_opts.vocab_path)
    tokens = tokenizer.tokenize(sample)
    input_ids = tokenization.convert_tokens_to_ids(vocab_file=eval_opts.vocab_path,
                                                   tokens=tokens)
    input_ids = np.array(input_ids).reshape(1, -1)
    # eval ops
    output_ids = generate(predict_model,
                          end_token=2,  # For opt model, the end_token is 2
                          origin_inputs=input_ids,
                          model_origin_max_length=eval_opts.model['seq_length'],
                          max_generate_length=eval_opts.model['seq_length'],
                          vocab_size=eval_opts.model["vocab_size"],
                          cache_encoder=None,
                          config=eval_opts)
    # Decode output ids to sentence
    output_samples = tokenization.convert_ids_to_tokens(vocab_file=eval_opts.vocab_path,
                                                        ids=output_ids.tolist())
    output_string = tokenization.convert_tokens_to_string(output_samples)
    print('Output is:', output_string, flush=True)
