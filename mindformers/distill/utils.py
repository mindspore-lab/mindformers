# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Distill utilities."""
import json
import os
from types import MethodType

import mindspore as ms
from mindspore.communication.comm_func import barrier

from mindformers.core.context.build_context import is_legacy_model
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_main_rank, clear_auto_trans_output, FILE_PERMISSION
from mindformers.trainer.utils import load_ckpt
from mindformers.utils.load_checkpoint_utils import (
    _check_checkpoint_path,
    _get_src_strategy,
    get_merged_dst_strategy_path,
    process_for_stand_alone_mode,
    load_safetensors_checkpoint,
    get_load_checkpoint_files,
    build_model,
    _get_checkpoint_mode
)


def load_safetensors_to_distill_network(config, network, optimizer=None):
    """load safetensors checkpoint to distill model"""
    if config.load_ckpt_async:
        logger.warning("The configuration 'load_ckpt_async=True' is not supported for safetensors files currently.")
    config.load_checkpoint = _check_checkpoint_path(config.load_checkpoint)
    logger.info(f"Load checkpoint from {config.load_checkpoint}.")
    pet_config = config.model.model_config.get("pet_config")
    if pet_config and pet_config.pet_type == "slora" and network.lora_list:
        raise ValueError(f"slora only support .ckpt file, {config.load_ckpt_format} file will be compatible soon.")
    strategy_path = ms.get_auto_parallel_context('strategy_ckpt_save_file')
    load_checkpoint_files, load_checkpoint = get_load_checkpoint_files(config)
    process_for_stand_alone_mode(config, network, strategy_path)
    # merge dst strategy
    strategy_path = get_merged_dst_strategy_path(config, strategy_path)
    load_safetensors_checkpoint(
        config, load_checkpoint_files, network,
        strategy_path, load_checkpoint, optimizer
    )


def load_ckpt_to_distill_network(config, network, optimizer=None):
    """load ckpt checkpoint to distill model. The full ckpt checkpoint will be converted to safetensors."""
    load_checkpoint = config.load_checkpoint
    if os.path.isfile(load_checkpoint):
        safetensors_file = os.path.abspath(load_checkpoint.replace(".ckpt", ".safetensors"))
        if not os.path.exists(safetensors_file):
            logger.warning(f"The ckpt format checkpoint will not be supported soon. "
                           f"Please update the checkpoint to safetensors.")
            logger.warning(f"Convert checkpoint file {load_checkpoint} from ckpt format to safetensors format. "
                           f"Please make sure there is no other safetensors checkpoint file in the directory.")
            if is_main_rank():
                ms.ckpt_to_safetensors(load_checkpoint, os.path.dirname(load_checkpoint))
            if config.use_parallel:
                barrier()
        else:
            logger.info(f"Existing converted safetensors checkpoint file {safetensors_file} found.")
        config.load_checkpoint = os.path.dirname(load_checkpoint)
        config.load_ckpt_format = "safetensors"
        config.auto_trans_ckpt = True
        load_safetensors_to_distill_network(config, network, optimizer)
    else:
        load_ckpt(config, network, optimizer)


def _check_distill_src_strategy(config):
    """check whether the location of the source strategy of load_checkpoint is valid."""
    if not config.load_checkpoint or config.load_ckpt_format == "ckpt":
        return
    ckpt_file_mode = _get_checkpoint_mode(config)
    if not (ckpt_file_mode == "multi_checkpoint_file_with_rank_id" and config.auto_trans_ckpt):
        return
    src_strategy_path = os.path.realpath(_get_src_strategy(config))
    dst_strategy_path = os.path.realpath(os.path.dirname(ms.get_auto_parallel_context('strategy_ckpt_save_file')))
    if os.path.samefile(src_strategy_path, dst_strategy_path):
        if not os.listdir(src_strategy_path):
            raise ValueError(f"The source strategy under `{src_strategy_path}` is not allowed to be the same as "
                             f"the output strategy path `{dst_strategy_path}` in distillation training. "
                             f"Please move the checkpoint and strategy files to a different location, "
                             f"or specify a different output folder by setting `config.output_dir`.")
    if is_main_rank():
        clear_auto_trans_output()
    if config.use_parallel:
        barrier()


def generate_name_map(config, network, optimizer=None):
    """generate name map"""
    config.name_map = {}
    maxsplit = 1 if is_legacy_model() else 2
    for _, param in network.parameters_and_names():
        name = param.name
        prefix, *_, suffix = name.split(".", maxsplit=maxsplit)
        config.name_map[suffix] = name
    if optimizer:
        for _, param in optimizer.parameters_and_names():
            name = param.name
            if prefix not in name:
                continue
            opt_prefix, *_, suffix = name.split(".", maxsplit=maxsplit + 1)
            config.name_map[opt_prefix + "." + suffix] = name
    if config.load_ckpt_format == "safetensors" and _get_checkpoint_mode(config) == "multi_checkpoint_file":
        param_json = os.path.join(config.load_checkpoint, "param_name_map.json")
        if os.path.exists(param_json) and is_main_rank():
            with open(param_json, 'r') as f:
                data = json.load(f)
            logger.warning(f"Rename the original param_name_map.json to param_name_map.json.back.")
            os.rename(param_json, os.path.join(config.load_checkpoint, "param_name_map.json.back"))
            weight_map = data.get("weight_map", data)
            new_weight_map = {}
            for key, value in weight_map.items():
                new_weight_map[config.name_map.get(key, key)] = value
            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            with os.fdopen(os.open(param_json, flags_, FILE_PERMISSION), 'w') as f:
                json.dump(new_weight_map, f, indent=2)
                logger.warning(f"Converted new param_name_map.json")
        if config.use_parallel:
            barrier()


def roll_back_json(config):
    """recover the original `param_name_map.json` from `param_name_map.json.back`."""
    if config.load_ckpt_format == "safetensors" and _get_checkpoint_mode(config) == "multi_checkpoint_file":
        param_json_back = os.path.join(config.load_checkpoint, "param_name_map.json.back")
        if os.path.exists(param_json_back) and is_main_rank():
            logger.warning(f"roll back the original param_name_map.json")
            os.remove(os.path.join(config.load_checkpoint, "param_name_map.json"))
            os.rename(param_json_back, os.path.join(config.load_checkpoint, "param_name_map.json"))
        if config.use_parallel:
            barrier()

def load_distill_network(config, model, network, dataset, optimizer=None):
    """compile and load checkpoint to distill model"""
    distill_config = config.distill_config
    teacher_config = config.distill_config.teacher_config
    _check_distill_src_strategy(config)
    _check_distill_src_strategy(teacher_config)
    # compile distill model
    build_model(config, model, dataset)
    # pylint: disable=W0212
    # get inner distill model
    network_with_optim = None
    if config.resume_training or config.get("remove_redundancy", False):
        network = model._train_network
        network_with_optim = network
    network = getattr(network, "_backbone", network)
    while hasattr(network, 'network'):
        network = network.network
        network = getattr(network, "_backbone", network)
    # load student model
    if config.load_checkpoint:
        if distill_config.student_need_map:
            generate_name_map(config, network.student_model, optimizer)
        logger.info(f"......Start load checkpoint to student from {config.load_ckpt_format}......")
        if network_with_optim is not None:
            logger.warning(f"When load to student network with optimizer, there will be warnings shows that "
                           f"some parameters are not loaded. No need to take care about the warnings.")
        student_network = network_with_optim or network.student_model
        if config.load_ckpt_format == "ckpt":
            load_ckpt_to_distill_network(config, student_network, optimizer)
        else:
            load_safetensors_to_distill_network(config, student_network, optimizer)
        config.pop("name_map", None)
        roll_back_json(config)
    # load teacher model
    if teacher_config.load_checkpoint:
        if distill_config.teacher_need_map:
            generate_name_map(teacher_config, network.teacher_model)
        logger.info(f"......Start load checkpoint to teacher from {config.load_ckpt_format}......")
        if teacher_config.load_ckpt_format == "ckpt":
            load_ckpt_to_distill_network(teacher_config, network.teacher_model)
        else:
            load_safetensors_to_distill_network(teacher_config, network.teacher_model)
        teacher_config.pop("name_map", None)
        roll_back_json(teacher_config)


def adjust_distillation_model_for_legacy(model, distill_config):
    """modify the output of legacy model"""
    logger.warning("modify the original construct function...")
    student = model.student_model
    teacher = model.teacher_model
    def teacher_loss(self, logits, labels, input_mask):
        return logits
    teacher.loss.construct = MethodType(teacher_loss, teacher.loss)

    if distill_config.skip_lm_loss:
        def student_loss(self, logits, labels, input_mask):
            return ms.Tensor([0], ms.float32), logits, input_mask
    else:
        def student_loss(self, logits, labels, input_mask):
            # need to call the original `construct` function of class to get the loss
            loss = type(self).construct(self, logits, labels, input_mask)
            return loss, logits, input_mask
    student.loss.construct = MethodType(student_loss, student.loss)


def adjust_distillation_model_for_mcore(model, distill_config):
    """modify the output of mcore model"""
    # pylint: disable=W0212
    logger.warning("modify the original construct function...")

    def expert_load_balancing(self, scores, top_indices, alpha):
        return ms.Tensor([0], ms.float32)

    def add_loss(self, extra_loss, router_aux_loss):
        return extra_loss

    student = model.student_model.get_gpt_model()
    teacher = model.teacher_model.get_gpt_model()

    def teacher_loss(self, labels, logits, input_mask):
        return logits

    def teacher_construct(self, *args, **kwargs):
        logits, _, _ = type(self).construct(self, *args, **kwargs)
        return logits

    teacher.compute_language_model_loss = MethodType(teacher_loss, teacher)
    teacher.construct = MethodType(teacher_construct, teacher)
    teacher.mtp_process = False
    for _, cell in teacher.cells_and_names():
        if "TopKRouter" in str(type(cell)):
            cell._expert_load_balancing = MethodType(expert_load_balancing, cell)
        elif "MoELayer" in str(type(cell)):
            cell.add_loss = MethodType(add_loss, cell)

    def student_loss(self, labels, logits, input_mask):
        return logits

    def student_construct(
            self,
            input_ids,
            position_ids=None,
            attention_mask=None,
            decoder_input=None,
            labels=None,
            extra_block_kwargs=None,
            prefix_keys_values=None,
            loss_mask=None,
            actual_seq_len=None
    ):
        # need to call the original `construct` function of class to get the results
        logits, mtp_loss, extra_loss = type(self).construct(
            self, input_ids, position_ids, attention_mask, decoder_input, labels,
            extra_block_kwargs, prefix_keys_values, loss_mask, actual_seq_len
        )
        if loss_mask is None:
            loss_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), ms.float32)
        label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), ms.float32)
        loss_mask = self.mul(loss_mask, label_mask)
        loss_mask = self.reshape(loss_mask, (-1,))
        if not self.config.skip_lm_loss:
            labels = self.reshape(labels, (-1,))
            lm_loss = self.loss(logits, labels, loss_mask)
        else:
            lm_loss = ms.Tensor([0], ms.float32)

        return logits, loss_mask, lm_loss, mtp_loss, extra_loss

    student.compute_language_model_loss = MethodType(student_loss, student)
    student.construct = MethodType(student_construct, student)
    if distill_config.skip_lm_loss:
        student.mtp_process = False
        for _, cell in student.cells_and_names():
            if "TopKRouter" in str(type(cell)):
                cell._expert_load_balancing = MethodType(expert_load_balancing, cell)
            elif "MoELayer" in str(type(cell)):
                cell.add_loss = MethodType(add_loss, cell)
