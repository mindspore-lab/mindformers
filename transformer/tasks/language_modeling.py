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
"""LMTask"""
import math
from transformer.tasks import Task, TaskConfig
from transformer.trainer import parse_config
from transformer.data import create_language_model_dataset

class LMTask(Task):
    """
    LMTask
    """

    def preprocess(self):
        """
        process input dataset
        """
        self.config.dataset_batch_size = self.config.eval_batch_size
        self.config.dataset_path = self.config.eval_data_path
        self.config.dataset_do_shuffle = self.config.eval_data_shuffle
        self.config.is_training = False
        self.config.dataset_device_num = 1
        self.config.dataset_rank = 0
        self.config.repeat_count = 1
        return create_language_model_dataset(self.config)

    def process(self, preprocess_output, model):
        """
        process inference result
        """

        if self.config.metric.lower() == "ppl":
            print("Prepare to calculate the ppl score ...")

            columns_list = ["input_ids", "input_mask", "label_ids"]
            print("==================== [PPL] Testing ====================")
            num_data = 1
            total_loss = 0.0
            avg_loss = 0.0
            for data in preprocess_output.create_dict_iterator():
                input_data = []
                for i in columns_list:
                    input_data.append(data[i])
                input_ids, input_mask, label_ids = input_data
                loss = model.predict(input_ids, input_mask, label_ids)
                print("loss is", loss)
                loss = float(loss.asnumpy())
                total_loss += loss
                avg_loss = float(total_loss / num_data)
                print(" | Current Loss: {:.6f}".format(avg_loss))
                print(" | Current PPL: {}\n\n".format(math.exp(avg_loss)))
                num_data += 1

            print("\n\n")
            print("**************************************************************")
            print("Average Loss: {:.6f}".format(avg_loss))
            print("Average PPL: {:.6f}".format(math.exp(avg_loss)))
            print("********************** Testing Finished **********************")
        else:
            raise ValueError("metric method not supported, support: [ppl]")



if __name__ == "__main__":
    config = TaskConfig()
    config.device_target = "GPU"
    config.device_id = 0
    config.epoch_num = 3
    config.eval_data_shuffle = False
    config.eval_batch_size = 1
    config.eval_data_path = './test-mindrecord'
    config.checkpoint_prefix = 'gpt2_language_model'

    config.vocab_size = 50257
    config.embedding_size = 768
    config.num_layers = 12
    config.num_heads = 12
    config.seq_length = 1024
    config.max_position_embeddings = 1024
    config.metric = 'ppl'
    parse_config(config)
    trainer = LMTask(config)
    trainer.run()
