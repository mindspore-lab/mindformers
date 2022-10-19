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
"""QATask"""
import collections

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from transformer.tasks import Task, TaskConfig
from transformer.trainer import parse_config
from transformer.data import create_squad_dataset
from transformer.processor.create_squad_data import read_squad_examples, convert_examples_to_features
from transformer.processor.squad_get_predictions import write_predictions
from transformer.processor.squad_postprocess import squad_postprocess

from transformer.tokenization import tokenization


class QATask(Task):
    """
    QATask
    """

    def preprocess(self):
        """
        process input dataset
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.config.vocab_file_path, do_lower_case=True)
        self.config.eval_examples = read_squad_examples(self.config.eval_json_path, False)
        self.config.eval_features = convert_examples_to_features(
            examples=self.config.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=self.config.seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            vocab_file=self.config.vocab_file_path)

        self.config.dataset_batch_size = self.config.eval_batch_size
        self.config.dataset_path = self.config.eval_features
        self.config.dataset_do_shuffle = self.config.eval_data_shuffle
        self.config.is_training = False
        return create_squad_dataset(self.config)

    def process(self, preprocess_output, model):
        """
        process inference result
        """
        output = []
        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
        columns_list = ["input_ids", "input_mask", "segment_ids", "unique_ids"]
        for data in preprocess_output.create_dict_iterator(num_epochs=1):
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, segment_ids, unique_ids = input_data
            start_positions = Tensor([1], mstype.float32)
            end_positions = Tensor([1], mstype.float32)
            is_impossible = Tensor([1], mstype.float32)
            logits = model.predict(input_ids, input_mask, segment_ids, start_positions,
                                   end_positions, unique_ids, is_impossible)
            ids = logits[0].asnumpy()
            start = logits[1].asnumpy()
            end = logits[2].asnumpy()

            for i in range(self.config.eval_batch_size):
                unique_id = int(ids[i])
                start_logits = [float(x) for x in start[i].flat]
                end_logits = [float(x) for x in end[i].flat]
                output.append(RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))
        return output

    def postprocess(self, process_output):
        """
        process final result
        """
        all_predictions = write_predictions(self.config.eval_examples, self.config.eval_features, process_output, 20,
                                            30, True)
        squad_postprocess(self.config.eval_json_path, all_predictions, output_metrics="output.json")


if __name__ == "__main__":
    config = TaskConfig()
    config.device_target = "GPU"
    config.device_id = 0
    config.epoch_num = 3
    config.num_class = 2
    config.eval_data_shuffle = False
    config.eval_batch_size = 1
    config.vocab_file_path = "./vocab.txt"
    config.load_pretrain_checkpoint_path = "./checkpoint/bert_base1.ckpt"
    config.load_finetune_checkpoint_path = "./squad_ckpt"
    config.checkpoint_prefix = 'squad'
    config.eval_json_path = "./squad_data/dev-v1.1.json"

    config.vocab_size = 30522
    config.embedding_size = 768
    config.num_layers = 12
    config.num_heads = 12
    config.seq_length = 384
    config.max_position_embeddings = 512

    parse_config(config)
    trainer = QATask(config)
    trainer.run()
