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
"""DPO process"""
import argparse
import os
import json
import sys
import numpy as np
from tqdm import tqdm

from mindspore.mindrecord import FileWriter, FileReader
import mindspore as ms
from mindspore import ops as P

from mindformers import AutoModel
from mindformers.tools.utils import str2bool
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.core.context import build_context


ROLE_MAPPING = {
    "human": "<|user|>",
    "gpt": "<|assistant|>",
    "system": "<|system|>"
}


def build_message(tokenizer, messages, metadata=""):
    """Build message"""
    encoded_messages = []
    for i, msg in enumerate(messages):
        role = ROLE_MAPPING.get(msg['from'], "")
        if not role:
            raise ValueError(f"Unsupported role {msg['from']}")
        message = f"{role}{metadata}\n{msg['value']}"
        tokens = tokenizer.encode(message)
        if i != 0:
            tokens = tokens[2:]  # remove prefix
        encoded_messages.append(tokens)
    prompt_ids = []
    for encoded_ids in encoded_messages[:-1]:
        prompt_ids += encoded_ids
    answer_ids = encoded_messages[-1]
    return prompt_ids, answer_ids


def build_message_cvalues(tokenizer, prompt, ans):
    """Build message cvalues"""
    msg = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
    msg += "<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(msg)
    msg = f"{ans}<|im_end|>"
    answer_ids = tokenizer.encode(msg)
    return prompt_ids, answer_ids


def divide_data_equal_first(data_nums, interval_nums):
    """Divide data equal first"""
    nums_per_interval, nums_of_data_remaining = divmod(data_nums, interval_nums)
    if nums_of_data_remaining == 0:
        return {i: nums_per_interval for i in range(interval_nums)}

    result = {}
    for i in range(interval_nums):
        if i < interval_nums - 1:
            result[i] = nums_per_interval
        else:
            result[i] = nums_per_interval + nums_of_data_remaining
    return result


def get_logps(model, input_ids, labels, attention_mask, loss_mask, input_sliced_sig, is_dynamic):
    """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities.
                    Label tokens with value of label_pad_token_id are ignored. Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log
            probabilities of the given labels under the given logits.
    """
    valid_length = np.array(attention_mask).sum(axis=-1)
    batch_length = int(max(valid_length))
    input_ids = ms.Tensor(input_ids, dtype=ms.int32)
    if len(input_ids.shape) == 1:
        input_ids = ms.ops.unsqueeze(input_ids, 0)

    labels = ms.Tensor(labels, dtype=ms.int32)
    if len(labels.shape) == 1:
        labels = ms.ops.unsqueeze(labels, 0)

    loss_mask = ms.Tensor(loss_mask, dtype=ms.int32)

    if len(loss_mask.shape) == 1:
        loss_mask = ms.ops.unsqueeze(loss_mask, 0)

    if not input_sliced_sig:
        input_ids = P.StridedSlice()(input_ids, (0, 0), (input_ids.shape[0], input_ids.shape[1] - 1), (1, 1))
        labels = P.StridedSlice()(labels, (0, 1), (labels.shape[0], labels.shape[1]), (1, 1))
        loss_mask = P.StridedSlice()(loss_mask, (0, 1), (loss_mask.shape[0], loss_mask.shape[1]), (1, 1))
    if is_dynamic:
        input_ids = P.StridedSlice()(input_ids, (0, 0), (input_ids.shape[0], min(batch_length, input_ids.shape[1])),
                                     (1, 1))
        labels = P.StridedSlice()(labels, (0, 0), (labels.shape[0], min(batch_length, labels.shape[1])), (1, 1))
        loss_mask = P.StridedSlice()(loss_mask, (0, 0), (loss_mask.shape[0], min(batch_length, loss_mask.shape[1])),
                                     (1, 1))

    outputs = model(input_ids)
    logits = outputs[0]
    # [bs, seq_len] -> [bs, seq_len]
    labels = labels * loss_mask
    logits = logits.to(ms.float32)
    # [bs, seq_len, vocab_size]
    log_probs = ms.ops.log_softmax(logits, -1)
    # [bs, seq_len] -> [bs, seq_len, 1]
    index = ms.ops.unsqueeze(labels, -1)
    index = index.to(ms.int32)
    # [bs, seq_len, 1]
    per_token_logps = ms.ops.gather_elements(log_probs, -1, index)
    # [bs, seq_len, 1] -> [bs, seq_len]
    per_token_logps = ms.ops.squeeze(per_token_logps, -1)
    logps = ms.ops.sum(per_token_logps * loss_mask, -1)
    return logps.asnumpy()


def preprocess(data_path: str, dst_file: str, config_path: str, load_checkpoint_path: str, src_strategy_path: str,
               auto_trans_ckpt: bool, seq_len: int, dataset_type: str, save_interval: int):
    """DPO Process"""
    dst_file_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_path):
        os.makedirs(dst_file_path)
    dst_file_name = os.path.basename(dst_file)
    config = MindFormerConfig(config_path)
    input_sliced_sig = config.model.model_config.input_sliced_sig
    is_dynamic = config.model.model_config.is_dynamic
    if load_checkpoint_path is not None:
        config.load_checkpoint = load_checkpoint_path
    if src_strategy_path is not None and os.path.exists(src_strategy_path):
        config.src_strategy_path_or_dir = src_strategy_path
    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
    logger.info("..........Build Context Config..........")
    print('config', config)
    build_context(config)
    logger.info("..........Build Parallel Config..........")
    build_parallel_config(config)
    logger.info("parallel config is: %s", config.parallel_config)
    rank_id = config.local_rank

    tokenizer = build_tokenizer(config.processor.tokenizer)
    model = AutoModel.from_config(config)
    model.set_train(False)
    if is_dynamic:
        dynamic_input_ids = ms.Tensor(shape=[None, None], dtype=ms.int32)
        model.set_inputs(dynamic_input_ids)
    if dataset_type == 'dpo':
        with open(data_path, "r", encoding='utf-8') as file:
            pairs = json.load(file)
    elif dataset_type == 'cvalues':
        pairs = []
        with open(data_path, "r", encoding='utf-8') as file:
            for line in file:
                pairs.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if save_interval > len(pairs) or save_interval <= 0:
        raise ValueError(f"Save interval must be greater than 0 and less than {len(pairs)}, but get {save_interval}")

    schema = {
        "chosen_input_ids": {"type": "int32", "shape": [-1]},
        "chosen_labels": {"type": "int32", "shape": [-1]},
        "chosen_attention_mask": {"type": "int32", "shape": [-1]},
        "chosen_loss_mask": {"type": "int32", "shape": [-1]},
        "chosen_ref_logps": {"type": "float32", "shape": [-1]},
        "rejected_input_ids": {"type": "int32", "shape": [-1]},
        "rejected_labels": {"type": "int32", "shape": [-1]},
        "rejected_attention_mask": {"type": "int32", "shape": [-1]},
        "rejected_loss_mask": {"type": "int32", "shape": [-1]},
        "rejected_ref_logps": {"type": "float32", "shape": [-1]},
    }
    #  data import configuration full batch: True
    if rank_id == 0:
        file_dict = divide_data_equal_first(len(pairs), save_interval)
        data_nums = 0
        file_nums = 0
        file_name = os.path.join(dst_file_path, f'{dst_file_name.split(".")[0]}_{file_nums}.mindrecord')
        writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
        writer.add_schema(schema)

    nums = 0
    batch_chosen_input_ids = []
    batch_chosen_labels = []
    batch_chosen_attention_mask = []
    batch_chosen_loss_mask = []
    batch_rejected_input_ids = []
    batch_rejected_labels = []
    batch_rejected_attention_mask = []
    batch_rejected_loss_mask = []
    num_records = len(pairs)
    for step in tqdm(range(len(pairs))):
        pair = pairs[step]
        if dataset_type == 'dpo':
            chosen_messages = pair['conversations'] + [pair['chosen']]
            rejected_messages = pair['conversations'] + [pair['rejected']]
            prompt_ids, chosen_ids = build_message(tokenizer, chosen_messages)
            _, rejected_ids = build_message(tokenizer, rejected_messages)
        elif dataset_type == 'cvalues':
            prompt_ids, chosen_ids = build_message_cvalues(tokenizer, pair['prompt'], pair['pos_resp'])
            _, rejected_ids = build_message_cvalues(tokenizer, pair['prompt'], pair['neg_resp'])
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        def _build(prompt_ids, resp_ids):
            # check input_ids > seq_length
            input_ids = prompt_ids + resp_ids
            attention_mask = [1] * len(input_ids)
            if input_sliced_sig:
                labels = input_ids[1:] + [tokenizer.pad_token_id]
                loss_mask = [0] * len(prompt_ids) + [1] * (len(resp_ids) - 1) + [0]
            else:
                labels = input_ids[:]
                loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

            input_len = len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * (seq_len - input_len)
            labels = labels + [tokenizer.pad_token_id] * (seq_len - input_len)
            attention_mask = attention_mask + [0] * (seq_len - input_len)
            loss_mask = loss_mask + [0] * (seq_len - input_len)

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]
                attention_mask = attention_mask[:seq_len]
                loss_mask = loss_mask[:seq_len]

            input_ids = np.array(input_ids, dtype=np.int32)
            labels = np.array(labels, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.int32)
            loss_mask = np.array(loss_mask, dtype=np.int32)
            return input_ids, labels, attention_mask, loss_mask

        chosen_input_ids, chosen_labels, chosen_attention_mask, chosen_loss_mask = \
            _build(prompt_ids, chosen_ids)
        rejected_input_ids, rejected_labels, rejected_attention_mask, rejected_loss_mask = \
            _build(prompt_ids, rejected_ids)

        batch_chosen_input_ids.append(chosen_input_ids)
        batch_chosen_labels.append(chosen_labels)
        batch_chosen_attention_mask.append(chosen_attention_mask)
        batch_chosen_loss_mask.append(chosen_loss_mask)
        batch_rejected_input_ids.append(rejected_input_ids)
        batch_rejected_labels.append(rejected_labels)
        batch_rejected_attention_mask.append(rejected_attention_mask)
        batch_rejected_loss_mask.append(rejected_loss_mask)
        nums += 1

        bs = config.model.model_config.batch_size
        num_batch = bs if step != num_records // bs * bs else num_records % bs
        if len(batch_chosen_input_ids) == bs or step == num_records // bs * bs:
            batch_chosen_ref_logps = get_logps(model, batch_chosen_input_ids, batch_chosen_labels,
                                               batch_chosen_attention_mask, batch_chosen_loss_mask,
                                               input_sliced_sig, is_dynamic)
            batch_rejected_ref_logps = get_logps(model, batch_rejected_input_ids, batch_rejected_labels,
                                                 batch_rejected_attention_mask, batch_rejected_loss_mask,
                                                 input_sliced_sig, is_dynamic)
            for i in range(num_batch):
                sample = {
                    "chosen_input_ids": batch_chosen_input_ids[i],
                    "chosen_labels": batch_chosen_labels[i],
                    "chosen_attention_mask": batch_chosen_attention_mask[i],
                    "chosen_loss_mask": batch_chosen_loss_mask[i],
                    "chosen_ref_logps": np.array([batch_chosen_ref_logps[i]]),
                    "rejected_input_ids": batch_rejected_input_ids[i],
                    "rejected_labels": batch_rejected_labels[i],
                    "rejected_attention_mask": batch_rejected_attention_mask[i],
                    "rejected_loss_mask": batch_rejected_loss_mask[i],
                    "rejected_ref_logps": np.array([batch_rejected_ref_logps[i]]),
                }
                if rank_id == 0:
                    writer.write_raw_data([sample])
                    data_nums += 1
                    if data_nums == file_dict[file_nums]:
                        writer.commit()
                        file_nums += 1
                        data_nums = 0
                        if file_nums != len(file_dict):
                            file_name = os.path.join(dst_file_path,
                                                     f'{dst_file_name.split(".")[0]}_{file_nums}.mindrecord')
                            writer = FileWriter(file_name=file_name, shard_num=1, overwrite=True)
                            writer.add_schema(schema)

            nums = 0
            batch_chosen_input_ids = []
            batch_chosen_labels = []
            batch_chosen_attention_mask = []
            batch_chosen_loss_mask = []
            batch_rejected_input_ids = []
            batch_rejected_labels = []
            batch_rejected_attention_mask = []
            batch_rejected_loss_mask = []


def merge(src_dir, dst_file):
    """Merge dataset file"""
    schema = {
        "chosen_input_ids": {"type": "int32", "shape": [-1]},
        "chosen_labels": {"type": "int32", "shape": [-1]},
        "chosen_attention_mask": {"type": "int32", "shape": [-1]},
        "chosen_loss_mask": {"type": "int32", "shape": [-1]},
        "chosen_ref_logps": {"type": "float32", "shape": [-1]},
        "rejected_input_ids": {"type": "int32", "shape": [-1]},
        "rejected_labels": {"type": "int32", "shape": [-1]},
        "rejected_attention_mask": {"type": "int32", "shape": [-1]},
        "rejected_loss_mask": {"type": "int32", "shape": [-1]},
        "rejected_ref_logps": {"type": "float32", "shape": [-1]},
    }
    writer = FileWriter(file_name=dst_file, shard_num=1, overwrite=True)
    writer.add_schema(schema)

    files = os.listdir(src_dir)
    files = sorted(files)
    for file in tqdm(files):
        if not file.endswith(".mindrecord"):
            continue
        path = os.path.join(src_dir, file)
        reader = FileReader(path)
        for _, item in enumerate(reader.get_next()):
            writer.write_raw_data([item])
        reader.close()
    writer.commit()
    print("merge success")


def main():
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Path to source json file.")
    parser.add_argument("--dst", type=str, help="Path to target mindrecrod file.")
    parser.add_argument("--config", type=str, help="Path to model config file.")
    parser.add_argument('--load_checkpoint', default='./merges.txt', type=str, help='checkpoint name or dir to load')
    parser.add_argument('--src_strategy_path', default=None, type=int, help="strategy of load_checkpoint.")
    parser.add_argument('--auto_trans_ckpt', default=None, type=str2bool,
                        help="whether to transform checkpoint to the checkpoint matching current distribute strategy.")
    parser.add_argument('--seq_len', default=1024, type=int, help="Sequence length.")
    parser.add_argument('--dataset_type', type=str, default='dpo', help="Dataset type to process.")
    parser.add_argument('--save_interval', type=int, default=2, help='Save the data interval.')
    parser.add_argument('--merge', default=False, type=bool, help='Whether to merge dataset file.')
    parser.add_argument('--register_path', default=None, type=str, help='the register path of outer API.')
    args = parser.parse_args()
    if args.config is not None and not os.path.isabs(args.config):
        args.config = os.path.join(work_path, args.config)

    if args.register_path is not None:
        if not os.path.isabs(args.register_path):
            args.register_path = os.path.join(work_path, args.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = args.register_path
        if args.register_path not in sys.path:
            sys.path.append(args.register_path)

    if args.merge:
        merge(args.src, args.dst)
    else:
        preprocess(args.src, args.dst, args.config, args.load_checkpoint, args.src_strategy_path,
                   args.auto_trans_ckpt, args.seq_len, args.dataset_type, args.save_interval)


if __name__ == "__main__":
    main()
