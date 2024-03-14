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
"""iFlytekSpark model infer APIs."""
import os
import glob
from threading import Thread

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindspore.train import Model
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools import logger
from mindformers import AutoConfig
from mindformers.pet import get_pet_model, LoraConfig
from iflytekspark_sampler import Sampler
from iflytekspark_model import IFlytekSparkModelForCasualLM


class IFlytekSparkInfer:
    """iFlytekSpark Inference class."""
    def __init__(self, config, tokenizer=None, ckpt_path=None, max_length=128, max_batch=1,
                 streamer=None, use_parallel=False, rank_id=0, parallel_config=None):
        super(IFlytekSparkInfer, self).__init__()
        self.max_length = max_length
        self.max_batch = max_batch
        self.tokenizer = tokenizer
        self.streamer = streamer
        self.use_parallel = use_parallel
        self.rank_id = rank_id
        self.parallel_config = parallel_config
        self.build_model(config)
        self.load_ckpt(ckpt_path)
        self.show_infer_state()

    def predict(self, question_list):
        """predict"""
        if self.max_batch == len(question_list) and self.max_batch > 1:
            out = self.batch_infer(question_list)
        else:
            out = self.infer(question_list)
        return out

    def build_model(self, config):
        """iFlytekSpark initialize"""
        self.config = AutoConfig.from_pretrained(config)
        self.config.seq_length = self.max_length
        self.config.batch_size = self.max_batch
        self.is_dynamic = self.config.is_dynamic and \
                            self.config.is_lite_infer and not self.use_parallel
        self.use_sample_model = self.config.top_p == 1.0 and self.config.do_sample
        if self.use_parallel:
            self.config.parallel_config = self.parallel_config
        # main model
        self.model = IFlytekSparkModelForCasualLM(self.config)
        self.model.set_train(False)
        # postprocess model
        self.sample_model = Sampler(self.config.top_k)
        self.sample_model.set_train(False)

        if self.config.pet_config and self.config.pet_config.pet_type == 'lora':
            logger.info("----------------Init lora params----------------")
            pet_config = LoraConfig(
                param_init_type=self.config.pet_config.param_init_type,
                compute_dtype=self.config.pet_config.compute_dtype,
                lora_rank=self.config.pet_config.lora_rank,
                lora_alpha=self.config.pet_config.lora_alpha,
                lora_dropout=self.config.pet_config.lora_dropout,
                target_modules=self.config.pet_config.target_modules
            )
            self.model = get_pet_model(self.model, pet_config)

    def load_ckpt(self, ckpt_path):
        """load checkpoint"""
        if ckpt_path is not None and ckpt_path != "":
            ckpt_file = glob.glob(os.path.join(ckpt_path, "rank*"))
            distribute_load = (len(ckpt_file) > 1) and (os.path.join(ckpt_path, "rank_1") in ckpt_file)
            rank_id = 0
            if self.use_parallel and distribute_load:
                logger.info("Distributed ckpt is found, it will be loaded with distribute way.")
                rank_id = self.rank_id
                model = Model(self.model)
                self.model.add_flags_recursive(is_first_iteration=True)
                input_np = Tensor(np.ones(shape=(self.max_batch, self.max_length)), ms.int32)
                current_index_np = Tensor(np.ones(shape=(self.max_batch,)), ms.int32)
                valid_length_each_example_np = Tensor(np.ones(shape=(self.max_batch, 1)), ms.int32)
                init_reset_np = Tensor([True], ms.bool_)
                model.infer_predict_layout(input_np, None, current_index_np, None, None, None,
                                           init_reset_np, valid_length_each_example_np, None)
            logger.info("Network parameters are loading.")
            ckpt_dir = os.path.join(ckpt_path, "rank_{}".format(rank_id))
            ckpt_dict = load_checkpoint(get_last_checkpoint(ckpt_dir))
            not_load_network_params = load_param_into_net(self.model, ckpt_dict)
            logger.info("Network parameters are not loaded：%s", str(not_load_network_params))

    def infer(self, question_list):
        """inference"""
        result = []
        sample_model = self.sample_model if self.use_sample_model else None
        for idx, question in enumerate(question_list):
            if self.streamer is not None:
                output = self.stream_print(question, sample_model)
            else:
                cur = self.model.generate([question],
                                          streamer=self.streamer,
                                          max_length=self.max_length,
                                          sampler=sample_model)
                output = self.tokenizer.decode(cur, skip_special_tokens=True)
            result.extend(output)
            print(f"\nquestion {idx}, output: \n", output)
        return result

    def export_mindir(self, mindir_save_dir):
        """export mindir model file."""
        if not os.path.exists(mindir_save_dir):
            os.makedirs(mindir_save_dir)
        full_mindir_file = os.path.join(mindir_save_dir, "full_bs{}_seq{}_rank{}".format(self.max_batch,
                                                                                         self.max_length,
                                                                                         self.rank_id))
        inc_mindir_file = os.path.join(mindir_save_dir, "inc_bs{}_seq{}_rank{}".format(self.max_batch,
                                                                                       self.max_length,
                                                                                       self.rank_id))
        sampler_mindir_file = os.path.join(mindir_save_dir, "sample_bs{}_rank{}".format(self.max_batch,
                                                                                        self.rank_id))
        logger.info(f"Prefill mindir save dir: {full_mindir_file}")
        logger.info(f"Decode mindir save dir: {inc_mindir_file}")
        logger.info(f"Sampler mindir save dir: {sampler_mindir_file}")

        logger.info(f"Export batch size : {self.max_batch}")
        logger.info("Start export full mindir")
        input_ids = Tensor(np.ones(shape=(self.max_batch, self.config.seq_length)), dtype=mstype.int32)
        current_index = Tensor(np.ones(shape=(self.max_batch,)), mstype.int32)
        batch_valid_length = Tensor(np.ones(shape=(self.max_batch,)), mstype.int32)
        init_true = Tensor([True], mstype.bool_)

        self.model.add_flags_recursive(is_first_iteration=True)
        if self.is_dynamic:
            seq = None
            bs = None
            input_ids = ms.Tensor(shape=[self.max_batch, seq], dtype=ms.int32)
            batch_valid_length = Tensor(shape=[bs,], dtype=ms.int32)
            self.model.set_inputs(input_ids, None, current_index, None, None, None,
                                  init_true, batch_valid_length, None)
        ms.export(self.model,
                  input_ids, None, current_index, None, None, None, init_true, batch_valid_length, None,
                  file_name=full_mindir_file, file_format='MINDIR')

        logger.info("Start export inc mindir")
        inputs_np_1 = Tensor(np.ones(shape=(self.max_batch, 1)), mstype.int32)
        batch_valid_lengthp_np_1 = Tensor(np.ones(shape=(self.max_batch,)), mstype.int32)
        init_flase = Tensor([False], mstype.bool_)
        self.model.add_flags_recursive(is_first_iteration=False)
        ms.export(self.model,
                  inputs_np_1, None, current_index, None, None, None, init_flase, batch_valid_lengthp_np_1, None,
                  file_name=inc_mindir_file, file_format='MINDIR')

        logger.info("Start export sampler mindir")
        logits = Tensor(np.abs(np.random.rand(self.max_batch, self.config.vocab_size)), dtype=mstype.float16)
        temperature = Tensor([self.config.temperature], dtype=mstype.float16)
        ms.export(self.sample_model, logits, temperature, 1.0, None,
                  file_name=sampler_mindir_file, file_format='MINDIR')

        logger.info("Finish export mindir")

    def batch_infer(self, question_list):
        """batch infer."""
        sample_model = self.sample_model if self.use_sample_model else None
        if self.streamer is not None:
            output = self.stream_print(question_list, sample_model)
        else:
            result = self.model.generate(question_list,
                                         streamer=self.streamer,
                                         max_length=self.max_length,
                                         sampler=sample_model)
            output = self.tokenizer.batch_decode(result, skip_special_tokens=True)
        return output

    def stream_print(self, question, sample_model=None):
        """stream inference print"""
        question_list = [question] if self.max_batch == 1 else question
        generation_kwargs = dict(input_ids=question_list,
                                 streamer=self.streamer,
                                 max_length=self.max_length,
                                 sampler=sample_model)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        output = [""] * self.max_batch
        for new_text in self.streamer:
            print("stream output:", new_text)
            for i in range(len(new_text)):
                output[i] += new_text[i]
        thread.join()
        return output

    def show_infer_state(self):
        logger.info(f"use_past -> {self.config.use_past}")
        logger.info(f"max_length -> {self.config.seq_length}")
        logger.info(f"batch_size -> {self.config.batch_size}")
        logger.info(f"is_dynamic -> {self.config.is_dynamic}")
        logger.info(f"use_parallel -> {self.use_parallel}")
