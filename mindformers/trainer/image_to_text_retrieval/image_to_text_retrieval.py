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
"""Image-to-text Retrieval Trainer."""
import os
from typing import List, Optional, Union
from pprint import pprint

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore.train.model import Model
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn import TrainOneStepCell, Optimizer
from mindspore.train import Callback

from mindformers.dataset import build_dataset, check_dataset_config, BaseDataset
from mindformers.models import build_model, BaseModel
from mindformers.core.callback import build_callback
from mindformers.core.lr import build_lr
from mindformers.core.optim import build_optim
from mindformers.wrapper import build_wrapper
from mindformers.trainer.base_trainer import BaseTrainer
from mindformers.trainer.utils import check_runner_config
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .eval_utils import compute_itm_scores, extract_image_text_mapping
from ..config_args import ConfigArguments
from ..base_trainer import BaseTrainer
from ..utils import check_runner_config



@MindFormerRegister.register(MindFormerModuleType.TRAINER, alias="image_to_text_retrieval")
class ImageToTextRetrievalTrainer(BaseTrainer):
    """Image-to-text Retrieval Trainer."""
    def __init__(self, model_name: str = None):
        super(ImageToTextRetrievalTrainer, self).__init__("image_to_text_retrieval", model_name)
        self.model_name = model_name
        self.kwargs = None

    def train(self,
              config: Optional[Union[dict, ConfigArguments]] = None,
              network: Optional[Union[str, BaseModel]] = None,
              dataset: Optional[Union[str, BaseDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        """
        Training task for ImageToTextRetrievalTrainer Trainer.
        """
        # check mindspore version
        # currently, filip only support training under mindspore2.0
        if not ms.__version__.startswith('2.0'):
            raise NotImplementedError(f"Currently, filip only support training under mindspore2.0, "
                                      f"but with mindspore {ms.__version__}")

        self.kwargs = kwargs

        # no_weight_decay_params filter
        def decay_filter(param_name):
            no_decay_params = config.no_decay_params
            for keyword in no_decay_params:
                if keyword in param_name.name:
                    return False
            return True

        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.train_dataset_task)
        check_runner_config(config, dataset)

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config})

        network = network.to_float(mstype.float16)

        logger.info("网络参数量：%s M.", str(count_params(network)))

        # build optimizer
        logger.info(".........Build Optimizer..........")
        if optimizer is None:
            # build learning rate schedule
            logger.info(".........Build LR Schedule..........")
            lr_schedule = build_lr(config.lr_schedule)

            params = network.trainable_params()
            decay_params = list(filter(decay_filter, params))
            other_params = list(filter(lambda x: not decay_filter(x), params))
            group_params = [{'params': decay_params, 'weight_decay': config.optimizer.weight_decay},
                            {'params': other_params},
                            {'order_params': params}]
            if lr_schedule is not None:
                optimizer = build_optim(
                    config.optimizer,
                    default_args={"params": group_params,
                                  "learning_rate": lr_schedule})
            else:
                assert config.optimizer.learning_rate, "learning_rate must be input"
                optimizer = build_optim(
                    config.optimizer,
                    default_args={"params": group_params})

        # build callback
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(
                config.callbacks, default_args={"learning_rate": optimizer.learning_rate}))

        # build runner wrapper
        logger.info(".........Build Running Wrapper..........")
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2. ** 20,
                                                 scale_factor=2,
                                                 scale_window=1000)
        if wrapper is None:
            model = build_wrapper(config.runner_wrapper,
                                  default_args={"network": network,
                                                "optimizer": optimizer,
                                                "scale_update_cell": update_cell})
        elif isinstance(wrapper, TrainOneStepCell):
            model = wrapper
        else:
            raise NotImplementedError(f"Now not support this wrapper,"
                                      f"it should be TrainOneStepCell type, but get {wrapper}")

        # define Model and begin training
        logger.info(".........Starting Init Train Model..........")
        model = Model(model)

        if int(os.getenv("RANK_ID", '0')) % 8 == 0:
            pprint(config)
        model.train(
            config.runner_config.epochs, dataset, callbacks=callbacks,
            dataset_sink_mode=config.runner_config.sink_mode,
            sink_size=config.runner_config.sink_size)


    def evaluate(self,
                 config: Optional[Union[dict, ConfigArguments]] = None,
                 network: Optional[Union[str, BaseModel]] = None,
                 dataset: Optional[Union[str, BaseDataset]] = None,
                 callbacks: Optional[Union[Callback, List[Callback]]] = None,
                 **kwargs):
        """
        Evaluation task for ImageToTextRetrievalTrainer Trainer.
        """
        self.kwargs = kwargs
        is_full_config = kwargs.get("is_full_config", False)
        config = self.set_config(config, is_full_config)
        # build dataset
        logger.info(".........Build Dataset..........")
        check_dataset_config(config)
        if dataset is None:
            dataset = build_dataset(config.eval_dataset_task)
        logger.info("Create eval dataset finish, dataset size:%d", dataset.get_dataset_size())

        # build network
        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model, default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config,
                "is_training": False})

        network = network.to_float(mstype.float16)

        # checkpoint
        logger.info(".........Loading Checkpoint..........")
        if config.model.model_config.checkpoint_name_or_path is not None:
            network.load_checkpoint(config.model.model_config)
        logger.info("Network Parameters: %s M.", str(count_params(network)))

        # build callback
        logger.info(".........Build Callbacks for Evaluate..........")
        if callbacks is None:
            callbacks = []
            if config.profile:
                callbacks.append(config.profile_cb)
            callbacks.extend(build_callback(config.eval_callbacks))

        logger.info(".........Starting Evaling Model..........")
        if int(os.getenv("RANK_ID", '0')) % 8 == 0:
            pprint(config)

        # prepare inputs for computing simliarity matrix
        eval_inputs = network.prepare_inputs_for_itm_eval(dataset)

        # k_test value, for topk
        k_test = config.eval_dataset.k_test if \
            config.eval_dataset.k_test is not None else 128
        # trainer arguments overrides top_k
        k_test = kwargs.pop('k_test', k_test)

        # whether adding additional itm score
        add_extra_itm_score = config.eval_dataset.add_extra_itm_score
        # trainer arguments overrides add_extra_itm_score
        add_extra_itm_score = kwargs.pop('add_extra_itm_score', add_extra_itm_score)

        # compute image-to-text/text-to-image similarity scores
        score_i2t, score_t2i = compute_itm_scores(network,
                                                  eval_inputs,
                                                  k_test=k_test,
                                                  add_extra_itm_score=add_extra_itm_score)

        # ground-truth image-text mapping
        img2txt, txt2img = extract_image_text_mapping(config.eval_dataset, score_i2t, score_t2i)

        # report evaluation results
        eval_result = self._report_metrics(
            score_i2t,
            score_t2i,
            img2txt,
            txt2img,
        )

        logger.info(eval_result)
        logger.info(".........Evaluate Over!.............")
        return eval_result

    def _report_metrics(self, scores_i2t, scores_t2i, img2txt, txt2img):
        """
        report metrics for image-text matching

        Args:
            scores_i2t: image-to-text similarity score matrix
            scores_t2i: text-to-image similarity score matrix
            img2txt: image-to-text ground truth mapping
            txt2img: text-to-image ground truth mapping

        Returns:
            eval_result: A dictionary containing r1, r5, r10 scores
        """
        def get_lowest_from_ranks(ranks, ground_truth):
            if isinstance(ground_truth, int):
                return np.where(ranks == ground_truth)[0][0]
            assert isinstance(ground_truth, list), "img2txt or txt2img should be list[int] or list[list[int]]!"
            rank = 1e20
            for i in ground_truth:
                tmp = np.where(ranks == i)[0][0]
                if tmp < rank:
                    rank = tmp
            return rank
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            print("img2txt[{}]: {}".format(index, img2txt[index]))
            inds = np.argsort(score)[::-1]
            print("most similarity:", inds)
            # Score
            ranks[index] = get_lowest_from_ranks(inds, img2txt[index])

        # 计算度量, 100 是百分比基数,
        # ranks为每张图ground-truth对应的
        # text在similarity score中排位的最小值。
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            print("txt2img[{}]: {}".format(index, txt2img[index]))
            inds = np.argsort(score)[::-1]
            print("most similarity:", inds)
            ranks[index] = get_lowest_from_ranks(inds, txt2img[index])

        # 此段逻辑同上注释，以text为基准。
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result
