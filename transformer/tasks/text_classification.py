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
"""TextClassificationTask"""
from transformer.tasks import Task, TaskConfig
from transformer.trainer import parse_config
from transformer.data.downstream_dataset import create_classification_dataset
from transformer.processor.assessment_method import Accuracy, F1, MCC, SpearmanCorrelation


class TextClassificationTask(Task):
    """
    TextClassificationTask
    """

    def preprocess(self):
        """
        process input dataset
        """

        self.config.get_eval_dataset = True
        self.config.dataset_path = self.download_dataset()

        self.config.dataset_batch_size = self.config.eval_batch_size
        self.config.dataset_do_shuffle = self.config.eval_data_shuffle
        self.config.is_training = False
        self.config.dataset_format = "mindrecord"
        ds = create_classification_dataset(self.config)
        return ds

    def process(self, preprocess_output, model):
        """
        process inference result
        """
        if self.config.assessment_method == "accuracy":
            callback = Accuracy()
        elif self.config.assessment_method == "f1":
            callback = F1(False, self.config.num_class)
        elif self.config.assessment_method == "mcc":
            callback = MCC()
        elif self.config.assessment_method == "spearman_correlation":
            callback = SpearmanCorrelation()
        else:
            raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
        for data in preprocess_output.create_dict_iterator(num_epochs=1):
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, token_type_id, label_ids = input_data
            logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
            callback.update(logits, label_ids)

        if self.config.assessment_method == "accuracy":
            print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                      callback.acc_num / callback.total_num))
        elif self.config.assessment_method == "f1":
            print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
            print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
            print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
        elif self.config.assessment_method == "mcc":
            print("MCC {:.6f} ".format(callback.cal()))
        elif self.config.assessment_method == "spearman_correlation":
            print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
        else:
            raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")


if __name__ == "__main__":
    config = TaskConfig()
    parse_config(config)
    trainer = TextClassificationTask(config)
    trainer.run()
