#  Copyright 2024 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""utils of load checkpoint file"""
import os
import re
import shutil
import time
from enum import Enum
from glob import glob
from multiprocessing import Process
from safetensors import safe_open

import mindspore as ms
from mindspore import Parameter
from mindspore.common.api import _pynative_executor
from mindspore.communication.comm_func import barrier

from mindformers.tools.logger import logger
from mindformers.tools.utils import (
    is_main_rank,
    get_real_rank,
    clear_auto_trans_output,
    barrier_world
)
from mindformers.utils import convert_hf_safetensors_multiprocess, check_safetensors_key, is_hf_safetensors_dir
from mindformers.utils.safetensors.convert_safetensors import _convert_index_json
from mindformers.checkpoint.utils import compile_model
from mindformers.version_control import check_safetensors_addition_param_support
from ..version_control import check_tft_valid


class CkptFormat(Enum):
    """
    Enum class for MindFormers support checkpoints formats.
    """

    CKPT = 'ckpt'
    SAFETENSORS = 'safetensors'

    @classmethod
    def support_type(cls):
        return [member.value for member in cls]


class CheckpointFileMode(Enum):
    """
    Enum class for MindFormers load checkpoint file cases.
    """
    SINGLE_CHECKPOINT_FILE = 'single_checkpoint_file'
    MULTI_CHECKPOINT_FILE = 'multi_checkpoint_file'
    MULTI_CHECKPOINT_FILE_WITH_RANK_ID = 'multi_checkpoint_file_with_rank_id'


def _get_origin_network(network):
    """recursive find if cells which have function <convert_name>"""
    if 'convert_name' in dir(network):
        return network, True
    #DFS for network
    for cell in list(network.cells()):
        network, find_cell = _get_origin_network(cell)
        if find_cell:
            return network, True
    return network, False


def get_load_path_after_hf_convert(config, network):
    """check if it is hf safetensors and convert"""
    if (config.load_checkpoint and config.get('load_ckpt_format', 'ckpt') == 'safetensors' and
            is_hf_safetensors_dir(config.load_checkpoint, network)):
        # 'qkv_concat is True' save ms safetensors
        if (config.model.model_config.get("qkv_concat", False) or not check_safetensors_addition_param_support()):
            logger.info(".......Load Checkpoint format is hf safetensors, start convert to ms safetensors!.......")
            converted_sf_path = process_hf_checkpoint(network, config.output_dir, config.load_checkpoint)
            # Wait for main rank to convert HF safetensors
            if config.use_parallel:
                barrier()
            return converted_sf_path
    return config.load_checkpoint


def _check_checkpoint_path(path):
    """check checkpoint path."""
    if not isinstance(path, str) or isinstance(path, os.PathLike):
        raise ValueError(f"config.load_checkpoint must be a `str`, but got `{path}` as type `{type(path)}`.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.load_checkpoint `{path}` does not exist.")

    if path[-1] == '/':  # remove last '/' in path
        return path[:-1]
    return path


def _get_checkpoint_mode(config):
    """get checkpoint place mode."""
    checkpoint_path = config.load_checkpoint

    if os.path.isfile(checkpoint_path):
        # check if checkpoint_path upper folder is rank_x/
        upper_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        if upper_dir_name.startswith('rank_'):
            return CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value
        return CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value

    # check path is dir
    if not os.path.isdir(checkpoint_path):
        raise ValueError("Provided path is neither a file nor a directory.")

    dir_files = os.listdir(checkpoint_path)
    if any(folder_name.startswith('rank_') for folder_name in dir_files):
        return CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value

    if any(file_name.endswith(config.load_ckpt_format) for file_name in dir_files):
        return CheckpointFileMode.MULTI_CHECKPOINT_FILE.value

    raise ValueError("not support mode: no valid checkpoint files found")


def _get_src_strategy(config):
    """search and get strategy file path from load_checkpoint directory."""
    if os.path.isfile(config.load_checkpoint):
        upper_dir = '/'.join(config.load_checkpoint.split('/')[:-3])
    else:
        upper_dir = os.path.dirname(config.load_checkpoint)

    input_src_strategy = config.get('src_strategy_path_or_dir')
    if input_src_strategy and os.path.isdir(input_src_strategy):
        src_strategy_path = input_src_strategy
    elif os.path.exists(os.path.join(upper_dir, 'strategy')):
        src_strategy_path = os.path.join(upper_dir, 'strategy')
        logger.info(f"src_strategy_path_or_dir is empty, load source strategy from {src_strategy_path}.")
    else:
        raise ValueError("when use checkpoint after train/finetune, src_strategy_path_or_dir should be set "
                         "as a folder contained strategy ckpt files.")
    logger.info(f"load source strategy from {src_strategy_path}.")
    return src_strategy_path


def extract_suffix(file_path):
    """
    Extracts the suffix from safetensor filenames based on specific patterns.

    The function handles two filename patterns:
    1. {prefix}_rank_{rank_id}-{epoch}_{step}.safetensors
    2. {prefix}_rank_{rank_id}_{task_id}-{epoch}_{step}.safetensors

    Args:
        file_path: Path to the file or just the filename

    Returns:
        str: Extracted suffix in the format:
             - "{epoch}_{step}" for pattern 1
             - "{task_id}-{epoch}_{step}" for pattern 2
             Returns None if the filename doesn't match either pattern or if epoch/step are missing.
    """
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]

    # Regular expression pattern to match both filename formats
    # Pattern groups: (task_id), (epoch), (step)
    pattern = r'^.+_rank_\d+(?:_(\d+))?-(\d+)_(\d+)$'
    match = re.match(pattern, base_name)

    if not match:
        logger.info(f"Filename '{filename}' does not match expected pattern. "
                    "Skipping suffix extraction.")
        return None

    # Extract matched groups
    task_id = match.group(1)  # Will be None if no task_id in filename
    epoch = match.group(2)
    step = match.group(3)

    if not epoch or not step:
        logger.info(f"Filename '{filename}' is missing epoch or step information. "
                    "Skipping suffix extraction.")
        return None

    # Construct the appropriate suffix based on presence of task_id
    if task_id:
        return f"_{task_id}-{epoch}_{step}"
    return f"-{epoch}_{step}"


def _get_src_file_suffix(config):
    """get file_suffix from config.load_checkpoint."""
    if isinstance(config.resume_training, str):
        file_suffix = extract_suffix(config.resume_training)
        return config.load_checkpoint, file_suffix

    if os.path.isfile(config.load_checkpoint):
        # only support path format: path/rank_x/prefix-{epoch}_{step}.{config.load_ckpt_format}
        file_suffix = extract_suffix(config.load_checkpoint)
        checkpoint_dir = '/'.join(config.load_checkpoint.split('/')[:-2])
        return checkpoint_dir, file_suffix

    # config.load_checkpoint is folder
    rank_id = get_real_rank()
    rank_path = f"{config.load_checkpoint}/rank_{rank_id}"
    if not os.path.exists(rank_path):
        raise FileNotFoundError(f"{rank_path} not found.")

    last_checkpoint = get_last_checkpoint(rank_path, config.load_ckpt_format)
    file_suffix = extract_suffix(last_checkpoint)
    return config.load_checkpoint, file_suffix


def _get_src_file(checkpoint_dir, checkpoint_name=None, ckpt_format='ckpt'):
    """
    Retrieves the path to a checkpoint source file.

    This function constructs the path to a checkpoint file by first determining the
    rank-specific checkpoint directory (appending "rank_{real_rank}" to the base
    checkpoint directory). If a specific checkpoint name is provided, it directly
    forms the path using that name within the rank directory. If no name is given,
    it fetches the most recent checkpoint file in the specified format from the
    rank directory. It validates the existence of the constructed path and raises
    a FileNotFoundError if the file does not exist.

    Args:
        checkpoint_dir (str): Base directory where checkpoints are stored.
        checkpoint_name (str, optional): Specific name of the checkpoint file to retrieve.
            If None, the latest checkpoint in the specified format will be used. Defaults to None.
        ckpt_format (str, optional): Format/extension of the checkpoint files (e.g., 'ckpt').
            Used when fetching the latest checkpoint. Defaults to 'ckpt'.

    Returns:
        str: Full path to the valid checkpoint source file.

    Raises:
        FileNotFoundError: If the constructed checkpoint path does not exist.
    """
    checkpoint_rank_dir = os.path.join(checkpoint_dir, f"rank_{get_real_rank()}")
    if isinstance(checkpoint_name, str):
        ckpt_path = os.path.join(checkpoint_rank_dir, checkpoint_name)
    else:
        ckpt_path = get_last_checkpoint(checkpoint_rank_dir, ckpt_format)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} is not found.")
    return ckpt_path


def load_checkpoint_with_safetensors(config, model, network, input_data, do_eval=False,
                                     do_predict=False, optimizer=None):
    """load different format checkpoint interface."""
    logger.info(f"......Start load checkpoint from {config.load_ckpt_format}......")
    if config.load_ckpt_async:
        logger.warning("The configuration 'load_ckpt_async=True' is not supported for safetensor files currently.")
    config.load_checkpoint = _check_checkpoint_path(config.load_checkpoint)
    load_checkpoint = config.load_checkpoint
    logger.info(f"Load checkpoint from {config.load_checkpoint}.")

    pet_config = config.model.model_config.get("pet_config")
    if pet_config and pet_config.pet_type == "slora" and network.lora_list:
        raise ValueError(f"slora only support .ckpt file, {config.load_ckpt_format} file will be compatible soon.")
    ckpt_file_mode = _get_checkpoint_mode(config)
    validate_config_with_file_mode(ckpt_file_mode, config.use_parallel, config.auto_trans_ckpt)
    # reduce compile time in prediction
    if do_eval or do_predict:
        logger.info("Set network.set_train=False, reduce compile time in prediction.")
        network.set_train(False)

    load_checkpoint_files = []
    strategy_path = ms.get_auto_parallel_context('strategy_ckpt_save_file')
    if ckpt_file_mode == CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value:
        logger.info(f"......Use single checkpoint file mode......")
        load_checkpoint_files = [config.load_checkpoint]
    if ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE.value:
        logger.info(f"......Use multi checkpoint file mode......")
        load_checkpoint_files = glob(
            os.path.join(load_checkpoint, f"*.{config.load_ckpt_format}"))
        load_checkpoint_files.sort()
        config.remove_redundancy = False
    elif ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value:
        logger.info(f"......Use multi checkpoint file with rank id mode......")
        # change strategy
        if config.auto_trans_ckpt:
            logger.info(f"......auto_trans is True, will unify all rank files and slice to dst parallel strategy......")
            src_strategy_path = get_merged_src_strategy_path(config)
            unified_safetensors_path = os.path.join(config.output_dir, 'unified_checkpoint/')
            load_checkpoint, file_suffix = _get_src_file_suffix(config)
            unify_safetensors(load_checkpoint,
                              src_strategy_path,
                              unified_safetensors_path,
                              use_parallel=config.use_parallel,
                              file_suffix=file_suffix,
                              remove_redundancy=config.get('remove_redundancy', False))
            load_checkpoint = unified_safetensors_path
            load_checkpoint_files = glob(
                os.path.join(load_checkpoint, f"*.{config.load_ckpt_format}"))
            load_checkpoint_files.sort()
        else:
            logger.info(f"......auto_trans is False, will not unify or slice rank files......")
            if check_tft_valid() and not config.remove_redundancy:
                sf_file_name = load_checkpoint
                logger.info(f"......tft is enabled and not enable remove_redundancy, sf_file_name={sf_file_name}......")
            else:
                sf_file_name = _get_src_file(load_checkpoint, config.resume_training, config.load_ckpt_format)
                logger.info(f"......sf_file_name={sf_file_name}......")

            load_checkpoint_files = glob(sf_file_name, recursive=False)

    # use resume_training in train/finetune mode
    if config.resume_training or (config.get('remove_redundancy', False) and not do_predict):
        # pylint: disable=W0212
        network = model._train_network
    #build model
    if config.use_parallel:
        compile_model(
            model=model,
            dataset=input_data,
            mode=config.context.mode,
            sink_mode=config.runner_config.sink_mode,
            epoch=config.runner_config.epochs,
            sink_size=config.runner_config.sink_size,
            do_eval=do_eval, do_predict=do_predict
        )
        #wait generate all rank strategy files
        barrier()

    # only execute qkv concat check on the main rank in predict mode
    if do_predict and is_main_rank(ignore_check_modelarts=True):
        qkv_concat_config = config.model.model_config.get("qkv_concat", False)
        validate_qkv_concat(network, qkv_concat_config, load_checkpoint)
    # wait for the main rank to complete qkv check
    if config.use_parallel:
        barrier()

    process_for_stand_alone_mode(config, network, strategy_path)
    #merge dst strategy
    strategy_path = get_merged_dst_strategy_path(config, strategy_path)
    load_safetensors_checkpoint(config, load_checkpoint_files, network, strategy_path, load_checkpoint, optimizer)


def process_for_stand_alone_mode(config, network, strategy_path):
    """process for stand alone mode"""
    enable_stand_alone = (config.parallel.parallel_mode == 'STAND_ALONE')
    if config.use_parallel and enable_stand_alone:
        from mindformers.parallel_core.inference.utils import generate_state_dict, save_strategy_file
        strategy_ckpt_save_dir = os.path.dirname(strategy_path)
        if is_main_rank():
            if os.path.exists(strategy_ckpt_save_dir):
                shutil.rmtree(strategy_ckpt_save_dir)
                logger.info(f"Existed strategy directory {strategy_ckpt_save_dir} has been deleted.")
            os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
        barrier()
        _pynative_executor.sync()

        shard_state_dict = generate_state_dict(network)
        save_strategy_file(shard_state_dict, strategy_path)
        logger.info(f"Strategy file for stand alone mode has been saved in {strategy_path}.")
        barrier()
        _pynative_executor.sync()


def validate_config_with_file_mode(ckpt_file_mode, use_parallel, auto_trans_ckpt):
    """validate use_parallel and auto_trans_ckpt config with different file mode"""
    if ckpt_file_mode == CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value:
        if use_parallel:
            raise ValueError("When load checkpoint is a single file and use_parallel is True, please change "
                             "load_checkpoint in yaml from file name to the directory where only this file is located.")
    elif ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE.value:
        if use_parallel and not auto_trans_ckpt:
            raise ValueError("When load checkpoint is complete and use_parallel is True, please set auto_trans_ckpt: "
                             "True to enable automatic slicing function.")
    elif ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value:
        if not use_parallel:
            raise ValueError("when input checkpoint file is rank dir, Please set use_parallel: True to enable "
                             "distributed ckpt load.")
    else:
        raise ValueError("not support mode: no valid checkpoint files found")


def unify_safetensors(src_checkpoint, src_strategy_path, unified_path, use_parallel=False,
                      file_suffix=None, remove_redundancy=False):
    """merge strategy and unified safetensors."""
    logger.info("Start unify safetensors.")
    if is_main_rank():
        # unify checkpoints
        logger.info(f"unified safetensors with file_suffix:{file_suffix}, remove_redundancy: {remove_redundancy}")
        logger.info(f"unified safetensors with save path:{unified_path}")
        unify_time_start = time.time()
        ms.unified_safetensors(
            src_dir=src_checkpoint,
            src_strategy_file=src_strategy_path,
            dst_dir=unified_path,
            file_suffix=file_suffix,
            merge_with_redundancy=not remove_redundancy
        )
        unify_time_end = time.time()
        logger.info("Time spent unifying safetensors: %.2fs", unify_time_end - unify_time_start)
        clear_auto_trans_output()
    if use_parallel:
        barrier()
    logger.info("Unified safetensors finished.")


def load_safetensors_checkpoint(config, load_checkpoint_files, network, strategy_path, load_ckpt_path, optimizer):
    """load checkpoint into net."""
    origin_network, _ = _get_origin_network(network)
    if config.use_parallel and config.auto_trans_ckpt:
        logger.info("......Start load distributed checkpoint to model......")
        addition_args = {}
        # convert HF name map directly with ms 2.6.0 version
        if check_safetensors_addition_param_support():
            name_map = None
            if not config.model.model_config.get("qkv_concat", False) \
                    and is_hf_safetensors_dir(load_ckpt_path, origin_network):
                try:
                    logger.info("......obtain name map for HF safetensors.....")
                    name_map = origin_network.obtain_name_map(load_checkpoint_files)
                except Exception as e:
                    raise TypeError(f"Please complete abstract function obtain_name_map. Details: {e}") from e
                if is_main_rank():
                    _convert_index_json(load_ckpt_path, load_ckpt_path, origin_network.convert_map_dict, False)
                barrier()
            addition_args["name_map"] = name_map
        ms.load_distributed_checkpoint(
            network=network,
            predict_strategy=strategy_path,
            unified_safetensors_dir=load_ckpt_path,
            format=config.load_ckpt_format,
            **addition_args
        )
        # Load optimizer param in resume_training
        logger.info(f"load checkpoint unified: {load_ckpt_path}")
        hyper_param_file = os.path.join(load_ckpt_path, 'hyper_param.safetensors')
        if optimizer and config.resume_training:
            if not os.path.exists(hyper_param_file):
                raise FileNotFoundError(rf"No hyper_param.safetensors in given dir: {load_ckpt_path}")
            logger.info("......Start load hyper param into optimizer......")
            hyper_param_dict = ms.load_checkpoint(ckpt_file_name=hyper_param_file, format='safetensors')
            update_global_step(config, hyper_param_dict)
            ms.load_param_into_net(optimizer, hyper_param_dict)
    else:
        logger.info("......Start load checkpoint to model......")
        params_dict = dict()
        remove_redundancy = config.get('remove_redundancy', False)
        for checkpoint_file in load_checkpoint_files:
            logger.info(f"load checkpoint file: {checkpoint_file}")
            with safe_open(checkpoint_file, framework='np') as f:
                remove_redundancy = _revise_remove_redundancy_with_file(remove_redundancy, f)
            params_dict.update(ms.load_checkpoint(
                ckpt_file_name=checkpoint_file,
                format=config.load_ckpt_format
            ))
        if not config.model.model_config.get("qkv_concat", False) \
           and is_hf_safetensors_dir(load_ckpt_path, origin_network):
            logger.info("......HuggingFace weights convert name......")
            params_dict = origin_network.convert_weight_dict(params_dict, model_config=config.model.model_config)
        if optimizer and config.resume_training:
            logger.info("......Start load hyper param into optimizer......")
            update_global_step(config, params_dict)
        ms.load_param_into_net(network, params_dict, remove_redundancy=remove_redundancy)


def update_global_step(config, hyper_param_dict):
    if "global_step" in hyper_param_dict and config.runner_config.step_scale is not None:
        resume_global_step = int(hyper_param_dict["global_step"].data * config.runner_config.step_scale)
        logger.info("Set global_step from %d to: %d", \
                    int(hyper_param_dict["global_step"]), resume_global_step)
        hyper_param_dict["global_step"] = Parameter([resume_global_step])


def process_hf_checkpoint(model, output_dir=None, load_checkpoint=None):
    """process huggingface checkpoint."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = './output'
        logger.warning(f'Output directory is set to ./output, '
                       f'due to the output_dir {output_dir} does not exist.')

    # Get `model_type` from model.config as the folder prefix. If not, `model` is used by default.
    model_type = getattr(model.config, 'model_type', 'model')

    # Generate the base directory name.
    base_dir_name = f"{model_type}_ms_converted_weight"
    converted_dir = os.path.join(output_dir, base_dir_name)

    # If the directory exists, it will be overwritten.
    if os.path.exists(converted_dir):
        logger.warning(f"Notice: The converted path {converted_dir} has already exist! "
                       f"The old converted weight will be overwritten!")

    logger.info(f"The Hugging Face checkpoint will be converted to MindSpore checkpoint at: '{converted_dir}'.")

    if is_main_rank():
        p = Process(target=convert_hf_safetensors_multiprocess,
                    args=[load_checkpoint, converted_dir, model, model.config])
        p.start()
        p.join()

        # If the child process exits abnormally, the main process should also throw an exception.
        if p.exitcode != 0:
            raise RuntimeError("convert HuggingFace weight failed.")

    # If used parallel mode, other cards are waiting for the main card to complete the weight conversion.
    barrier_world("for the main rank to convert HuggingFace weight...")

    logger.info(f"Convert Hugging Face checkpoint successfully! The MindSpore checkpoint path is: '{converted_dir}'.")

    return converted_dir


def get_last_checkpoint(checkpoint_dir, ckpt_format='ckpt'):
    """get last checkpoint for resuming or finetune."""
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"{checkpoint_dir} is not a real directory,"
            f"When distributed loads are sliced weights,"
            f"load_checkpoint should be a checkpoint directory containing the directory of rank_{{0-*}},"
            f"The directory structure is as follows: **checkpoint_root_dir/rank_{{0-*}}/**.{ckpt_format}")
    output_checkpoint_path = [
        checkpoint
        for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith(f'.{ckpt_format}')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])


def validate_qkv_concat(model_cls_or_instance, qkv_concat_config, load_checkpoint):
    """
    Check whether qkv_concat configuration and qkv concat weight convert are the same.
    Currently only safetensors format is supported.
    """
    # check the type of model_cls_or_instance
    from mindformers.models.modeling_utils import PreTrainedModel
    if not (
            isinstance(model_cls_or_instance, PreTrainedModel) or
            (isinstance(model_cls_or_instance, type) and issubclass(model_cls_or_instance, PreTrainedModel))
    ):
        logger.warning(f"Cur model_cls_or_instance: {model_cls_or_instance} is not "
                       f"a subclass or an instance of PreTrainedModel, "
                       f"will not execute qkv concat check.")
        return

    concat_key_list = model_cls_or_instance.obtain_qkv_ffn_concat_keys()
    if concat_key_list is None:
        return

    logger.info(".........Starting qkv concat check.........")
    is_qkv_concat = True
    for concat_key in concat_key_list:
        is_qkv_concat = check_safetensors_key(load_checkpoint, concat_key) and is_qkv_concat
        if not is_qkv_concat:
            break

    if is_qkv_concat and not qkv_concat_config:
        raise ValueError("The qkv concat check failed! The qkv in the model weights has been concatenated,"
                         " but qkv_concat is set to false.")
    if not is_qkv_concat and qkv_concat_config:
        raise ValueError("The qkv concat check failed! The qkv in the model weights has been not concatenated,"
                         " but qkv_concat is set to true.")
    if is_qkv_concat and qkv_concat_config:
        logger.info("The qkv concat check succeed! The qkv in the model weights has been concatenated and "
                    "qkv_concat is set to true.")
    if not is_qkv_concat and not qkv_concat_config:
        logger.info("The qkv concat check succeed! The qkv in the model weights has been not concatenated and "
                    "qkv_concat is set to false.")


def get_merged_src_strategy_path(config):
    """prepare for src strategy."""
    # prepare merged strategy directory
    merged_strategy = os.path.join(config.output_dir, 'merged_strategy')
    os.makedirs(merged_strategy, exist_ok=True)

    # set src_strategy_path
    src_strategy = _get_src_strategy(config)
    dst_strategy = os.path.join(merged_strategy, 'src_strategy.ckpt')
    src_strategy_path = (src_strategy, dst_strategy)
    if is_main_rank():
        # merge src strategy
        logger.info("merge src strategy in parallel mode.")
        ms.merge_pipeline_strategys(
            src_strategy_dirs=src_strategy_path[0],
            dst_strategy_file=src_strategy_path[1])
    barrier()
    logger.info("merge src strategy finished.")
    return src_strategy_path[1]


def get_merged_dst_strategy_path(config, strategy_path):
    """prepare for dst strategy."""
    enable_stand_alone = (config.parallel.parallel_mode == 'STAND_ALONE')
    if config.use_parallel and config.auto_trans_ckpt and not enable_stand_alone:
        # prepare merged strategy directory
        merged_strategy = os.path.join(config.output_dir, 'merged_strategy')
        os.makedirs(merged_strategy, exist_ok=True)
        # set dst_strategy_path
        dst_strategy_path = (
            os.path.dirname(strategy_path),
            os.path.join(merged_strategy, 'dst_strategy.ckpt')
        )
        if is_main_rank():
            # merge dst strategy
            logger.info("merge dst strategy in parallel mode.")
            ms.merge_pipeline_strategys(
                src_strategy_dirs=dst_strategy_path[0],
                dst_strategy_file=dst_strategy_path[1])
        barrier()
        logger.info("merge dst strategy finished.")
        strategy_path = dst_strategy_path[1]
    return strategy_path


def _revise_remove_redundancy_with_file(remove_redundancy, f):
    """Check whether remove_redundancy is consistent with the safetensors file."""
    if f.metadata() is not None and "remove_redundancy" in f.metadata().keys():
        if f.metadata()["remove_redundancy"] == "True" and not remove_redundancy:
            logger.warning("For 'load_checkpoint', the safetensors file is without redundancy "
                           "but remove_redundancy in yaml is False and it will revise to True.")
            return True
        if f.metadata()["remove_redundancy"] == "False" and remove_redundancy:
            logger.warning("For 'load_checkpoint', the safetensors file is with redundancy, "
                           "but remove_redundancy is set to True and it will revise to False.")
            return False
    logger.warning("no metadata info in file.Please make sure remove_redundancy is consistent in file and config. ")
    return remove_redundancy
