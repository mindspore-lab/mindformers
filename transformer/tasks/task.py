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

"""Task"""
from mindspore.train.model import Model

from transformer.auto_class import AutoClass
from transformer.trainer import Trainer, TrainingConfig, parse_config


class TaskConfig(TrainingConfig):
    """
    TaskConfig
    """

    def __init__(self, *args, **kwargs):
        super(TaskConfig, self).__init__(*args, **kwargs)
        self.epoch_size = 1
        self.eval_data_shuffle = False
        self.is_training = False
        self.parallel_mode = "stand_alone"
        self.full_batch = False
        self.global_batch_size = 1
        self.checkpoint_prefix = ""

class Task(Trainer):
    """
    Task
    """

    def build_model(self, model_config):
        """build model"""
        network = AutoClass.get_network_with_loss_class(self.config.auto_model)
        if network is not None:
            return network(model_config)
        raise ValueError("invalid auto_model %s." % self.config.auto_model)

    def preprocess(self):
        """task preprocess"""
        print("task preprocess")
        return {}

    def process(self, preprocess_output, model):
        """task process"""
        print("task process", preprocess_output, model)
        return {}

    def postprocess(self, process_output):
        """task postprocess"""
        print("task postprocess", process_output)
        return {}

    def run(self):
        """task run"""
        self.set_context_env()
        self.set_auto_parallel_context_env()

        # This should be called before any cell construction
        self.set_fused_kernel()

        # Build model
        self.logger.info("Start to build model")
        model_config = self.check_and_build_model_config()
        parallel_config = self.build_parallel_config()
        model_config.parallel_config = parallel_config
        model_config.batch_size = self.config.eval_batch_size
        model_config.is_training = False
        net = self.build_model(model_config)
        net.set_train(False)

        # load checkpoint
        self.load_checkpoint(net)

        model = Model(net)

        preprocess_output = self.preprocess()
        process_output = self.process(preprocess_output, model)
        postprocess_output = self.postprocess(process_output)
        return postprocess_output


if __name__ == "__main__":
    config = TaskConfig()
    parse_config(config)
    task = Task(config)
    task.run()
