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

from mindtransformer.tasks import Task, TaskConfig
from mindtransformer.trainer import parse_config
from mindtransformer.data import create_squad_dataset
from mindtransformer.processor.create_squad_data import read_squad_examples, convert_examples_to_features
from mindtransformer.processor.squad_get_predictions import write_predictions
from mindtransformer.processor.squad_postprocess import squad_postprocess

from mindtransformer.tokenization import tokenization


class QATask(Task):
    """
    QATask
    """

    def preprocess(self):
        """
        process input dataset
        """
        if self.input_kwargs is not None:
            if "vocab_file_path" in self.input_kwargs.keys():
                self.config.vocab_file_path = self.input_kwargs["vocab_file_path"]
            if "eval_data_path" in self.input_kwargs.keys():
                self.config.eval_data_path = self.input_kwargs["eval_data_path"]

        tokenizer = tokenization.FullTokenizer(vocab_file=self.config.vocab_file_path, do_lower_case=True)
        self.config.eval_examples = read_squad_examples(self.config.eval_data_path, False)
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
        squad_postprocess(self.config.eval_data_path, all_predictions, output_metrics="output.json")


class QATaskConfig(TaskConfig):
    """
    QATaskConfig
    """
    def __init__(self, *args, **kwargs):
        super(QATaskConfig, self).__init__(*args, **kwargs)
        self.auto_model = "bert_squad"
        self.device_target = "GPU"
        self.device_id = 0
        self.epoch_num = 3
        self.num_class = 2
        self.eval_data_shuffle = False
        self.eval_batch_size = 12
        self.checkpoint_prefix = 'tmp'

        self.vocab_file_path = "./vocab.txt"
        self.eval_data_path = "./squad_data/dev-v1.1.json"

        self.vocab_size = 30522
        self.embedding_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.seq_length = 384
        self.max_position_embeddings = 512


if __name__ == "__main__":
    config = QATaskConfig()
    parse_config(config)
    task = QATask(config)
    task.run()
