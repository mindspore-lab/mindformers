import os
import argparse
import importlib
import shutil
import yaml
import numpy as np
from glob import glob

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

import mindformers
from mindformers import MindFormerConfig
from mindformers.tools import logger
from mindformers.tools.download_tools import download_with_progress_bar
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models import build_model, build_tokenizer
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.pet import get_pet_model

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(mindformers.__file__))


def convert_path(src_path, save_path):
    src_path = os.path.realpath(src_path)
    file_name = os.path.basename(src_path)
    file_path = os.path.join(save_path, file_name)
    return src_path, file_path


def get_parallel_status():
    device_num = os.getenv('MS_WORKER_NUM')
    if device_num:
        return True, int(device_num)
    return False, 1


class ModelPredict:

    def __init__(self,
                 model_name: str = 'llama2_7b',
                 model_path: str = 'predict_model',
                 config_path: str = '',
                 config=None,
                 predict_data=None,
                 input_args=None):

        self.model_name = model_name
        self.model_prefix = self.model_name.split('_')[0]
        self.model_path = model_path
        self.config_path = config_path
        self.config = config
        self.input_args = input_args

        # load supported models from file
        with open(f"{cur_dir}/supported_models.yaml", 'r') as file:
            self.supported_models = yaml.safe_load(file)
        self.model_list = list(self.supported_models['configs'].keys())

        # prepare predict data
        data_seq_len = self.input_args.get('data_seq_len')
        data_batch_size = self.input_args.get('data_batch_size')
        if data_seq_len and data_batch_size:
            self.custom_data = True
            self.batch_size = data_batch_size
            self.inputs = self.prepare_fake_data(data_seq_len)
        else:
            self.custom_data = False
            self.inputs, self.batch_size = self.prepare_data(predict_data)

        # prepare model path
        self.init_model_path()

        # prepare config
        self.custom_model = False
        self.use_pet_model = False
        self.init_config()
        self.model_config = self.config.model.model_config

        # register model for research directory
        if self.model_name in list(self.supported_models['register'].keys()):
            self.init_register()

        # build context
        build_context(self.config)
        build_parallel_config(self.config)

        # build model
        logger.info("build base model.")
        self.network = build_model(self.config.model)
        if self.use_pet_model:
            self.build_pet_model()
        self.model = Model(self.network)

    def load_checkpoint(self, load_checkpoint=None):
        if load_checkpoint and os.path.exists(load_checkpoint):
            load_checkpoint = os.path.realpath(load_checkpoint)
            self.config.load_checkpoint = load_checkpoint
        elif not self.custom_model:
            # process checkpoint
            self.process_checkpoint()
        else:
            self.config.load_checkpoint = ""
            logger.warning(f"use custom model, will not load checkpoint.")

        if self.config.load_checkpoint:
            seq_length = self.model_config.seq_length
            input_ids = Tensor(shape=(self.batch_size, seq_length), dtype=ms.int32, init=init.One())
            infer_data = self.network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(self.config, self.model, self.network, infer_data, do_predict=True)

    def predict(self, generate_config=None):
        if not self.custom_data:
            # process tokenizer
            self.process_tokenizer()
            tokenizer = build_tokenizer(self.config.processor.tokenizer)
        else:
            tokenizer = None
            logger.warning(f"use custom data, will not build tokenizer.")

        if generate_config is None:
            generate_config = self.network.generation_config
        for k, v in self.input_args.items():
            if hasattr(generate_config, k) and not callable(getattr(generate_config, k)):
                setattr(generate_config, k, v)

        if tokenizer:
            inputs_ids = tokenizer(self.inputs,
                                   max_length=self.model_config.seq_length,
                                   padding="max_length")["input_ids"]
        else:
            inputs_ids = self.inputs
        outputs = self.network.generate(inputs_ids, generation_config=generate_config)

        # 时间统计设置环境变量
        if tokenizer:
            for output in outputs:
                print(tokenizer.decode(output))
        else:
            print(outputs)

    def build_pet_model(self):
        logger.info("build pet model.")
        pet_config = self.config.model.model_config.pet_config
        self.network = get_pet_model(self.network, pet_config)

    def init_model_path(self):
        self.model_path = f"{self.model_path}/{self.model_name}"
        os.makedirs(self.model_path, exist_ok=True)

        if self.model_name not in self.model_list:
            logger.warning(f"{self.model_name} is not in standard supported model list, "
                           f"may cause some error. supported: {self.model_list}")

    def prepare_data(self, predict_data):
        if not isinstance(predict_data, list):
            raise ValueError("input predict data should be a file path or multiple str data.")

        if len(predict_data) == 1 and os.path.isfile(predict_data[0]):
            with open(predict_data[0], 'r') as file:
                input_data = file.readlines()
                input_data = [_.strip() for _ in input_data]
        else:
            input_data = predict_data

        batch_size = len(predict_data)
        if self.model_prefix not in list(self.supported_models['template'].keys()):
            return input_data, batch_size

        inputs = []
        before, after = self.supported_models['template'][self.model_prefix]
        for data in input_data:
            inputs.append(f"{before}{data}{after}")
        return inputs, batch_size

    def prepare_fake_data(self, seq_length):
        inputs = np.random.randint((self.batch_size, seq_length), dtype=np.int32)
        return inputs.tolist()

    def init_register(self):
        for register in self.supported_models['register'][self.model_name]:
            try:
                logger.info(f"import {register}")
                importlib.import_module(register)
            except ModuleNotFoundError:
                logger.error(f"{register} not found.")
            except ImportError:
                logger.error(f'failed to import {register}.')

    def process_model_config(self):
        self.config.model.model_config.checkpoint_name_or_path = None
        self.config.model.model_config.batch_size = self.batch_size
        self.config.model.model_config.parallel_config = self.config.parallel_config

        if getattr(self.config.model.model_config, 'pet_config'):
            self.use_pet_model = True

        for k, v in self.input_args.items():
            if self.config.model.model_config.get(k):
                self.custom_model = True
                self.config.model.model_config[k] = v

    def init_config(self):
        # process config file
        self.process_config()

        # process parallel config
        self.process_parallel_config()

        # process model config
        self.process_model_config()

        # set auto transform ckpt
        if self.config.use_parallel and os.path.isfile(self.config.load_checkpoint):
            auto_trans_ckpt = True
        else:
            auto_trans_ckpt = False
        self.config.auto_trans_ckpt = self.input_args.get('auto_trans_ckpt', auto_trans_ckpt)

    def process_config(self):
        config_files = glob(f"{self.model_path}/*.yaml")
        if isinstance(self.config, MindFormerConfig):  # copy config file to model_path
            src_path, config_path = convert_path(self.config_path, self.model_path)
            shutil.copy(src_path, config_path)
        elif config_files:  # load first config file in model_path
            if len(config_files) > 1:
                logger.warning(f"find {len(config_files)} config files in {self.model_path}.")
            config_path = config_files[0]
            self.config = MindFormerConfig(config_path)
        else:
            src_path = os.path.join(project_root, self.supported_models['configs'][self.model_name])
            src_path, config_path = convert_path(src_path, self.model_path)
            shutil.copy(src_path, config_path)
            self.config = MindFormerConfig(config_path)
        logger.info(f"use config file: {config_path}.")

    def process_checkpoint(self):
        ckpt_files = glob(f"{self.model_path}/*.ckpt")
        load_checkpoint = self.input_args.get('load_checkpoint', None)
        if load_checkpoint == "" or load_checkpoint is False:
            # not load checkpoint later
            logger.warning(f"set load checkpoint: False.")
            self.config.load_checkpoint = ""
            return

        if not load_checkpoint or not os.path.exists(load_checkpoint):
            load_checkpoint = self.config.load_checkpoint

        if load_checkpoint and os.path.exists(load_checkpoint):
            # make soft link of checkpoint file in model_path
            src_path, load_checkpoint = convert_path(load_checkpoint, self.model_path)
            try:
                os.symlink(src_path, load_checkpoint)
            except OSError:
                logger.warning(f"link of {src_path} -> {load_checkpoint} is existed.")
        elif ckpt_files:  # use first checkpoint file in model_path
            if not self.use_pet_model or len(ckpt_files) != 2:
                logger.info(f"use pet model and find 2 ckpt files in {self.model_path}.")
                load_checkpoint = self.model_path
            else:
                logger.info(f"find {len(ckpt_files)} ckpt files in {self.model_path}.")
                load_checkpoint = ckpt_files[0]
        else:
            ckpt_url = self.supported_models['checkpoints'][self.model_name]['checkpoint']
            load_checkpoint = os.path.join(self.model_path, f"{self.model_name}.ckpt")
            download_with_progress_bar(ckpt_url, load_checkpoint)
        logger.info(f"use checkpoint file: {load_checkpoint}.")

        self.config.load_checkpoint = load_checkpoint

    def process_tokenizer(self):
        tokenizer_files = glob(f"{self.model_path}/tokenizer.model")
        vocab_file = self.input_args.get('vocab_file', None)
        if not vocab_file or not os.path.exists(vocab_file):
            vocab_file = self.config.processor.tokenizer.vocab_file

        if vocab_file and os.path.exists(vocab_file):
            # make soft link of tokenizer file in model_path
            src_path, vocab_file = convert_path(vocab_file, self.model_path)
            try:
                os.symlink(src_path, vocab_file)
            except OSError:
                logger.warning(f"link of {vocab_file} is existed.")
        elif tokenizer_files:
            vocab_file = tokenizer_files[0]
        else:
            tokenizer_url = self.supported_models['tokenizers'][self.model_prefix]
            vocab_file = os.path.join(self.model_path, "tokenizer.model")
            download_with_progress_bar(tokenizer_url, vocab_file)
        logger.info(f"use tokenizer file: {vocab_file}.")

        self.config.processor.tokenizer.vocab_file = vocab_file

    def process_parallel_config(self):
        parallel, device_num = get_parallel_status()
        self.config.use_parallel = parallel

        if not parallel:
            return

        dp = self.input_args.get('model_parallel', None)
        mp = self.input_args.get('data_parallel', None)
        if dp and mp:
            self.config.parallel_config.model_parallel = dp
            self.config.parallel_config.data_parallel = mp
        else:
            self.config.parallel_config.model_parallel = device_num
            self.config.parallel_config.data_parallel = 1
        self.config.parallel_config.pipeline_stage = 1


def convert_type(data):
    if data.lower() == 'true':
        return True
    if data.lower() == 'false':
        return False

    try:
        if '.' in data:
            data = float(data)
        else:
            data = int(data)
    except ValueError:
        logger.debug(f"try to convert '{data}' to float or int failed.")
    return data


def build_args(_args):
    error_info = "script input args should be aligned '--key value',"
    if divmod(len(_args), 2)[1] != 0:
        raise ValueError(f"{error_info} length of args is not correct.")

    input_args = dict()
    for arg_key, arg_value in zip(_args[0::2], _args[1::2]):
        if len(arg_key) <= 2 or arg_key[:2] != '--':
            raise ValueError(f"{error_info} got '{arg_key}' in keys.")
        if arg_value[:2] == '--':
            raise ValueError(f"{error_info} got '{arg_value}' in values.")

        value = convert_type(arg_value)
        input_args[arg_key[2:]] = value
    return input_args


def init_model_name(model_name, config_path):
    config = None
    if config_path and config_path.endswith('.yaml') and os.path.exists(config_path):
        config = MindFormerConfig(config_path)
        config_model_name = config.trainer.model_name
        if not model_name:
            logger.warning(f"input model name is None, use model name '{config_model_name}' in config.")
            model_name = config_model_name
        elif model_name != config_model_name:
            logger.warning(f"model name '{config_model_name}' in config file different from input "
                           f"model name '{model_name}', will use '{model_name}'.")

    return model_name, config


def main(
        model_name: str = 'llama2_7b',
        model_path: str = 'predict_model',
        config_path: str = None,
        predict_data=None,
        input_args=None,
):
    # init model name from config or input
    model_name, config = init_model_name(model_name, config_path)

    model_predict = ModelPredict(model_name,
                                 model_path,
                                 config_path,
                                 config,
                                 predict_data,
                                 input_args)
    model_predict.load_checkpoint()
    model_predict.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str, default="llama2_7b",
        help='input model name with size, e.g., llama2_7b.')
    parser.add_argument(
        '--model_path', type=str, default="predict_model",
        help='input model local path or remote repo.')
    parser.add_argument(
        '--config_path', type=str, default=None,
        help='input model config file path.')
    parser.add_argument(
        '--predict_data', metavar='N', type=str, nargs='+',
        help='multiple input predict data.')

    args, unknown_args = parser.parse_known_args()
    unknown_args = build_args(unknown_args)

    # convert input path to real path
    if args.config_path:
        args.config_path = os.path.realpath(args.config_path)
    if args.model_path:
        args.model_path = os.path.realpath(args.model_path)

    main(
        args.model_name,
        args.model_path,
        args.config_path,
        args.predict_data,
        unknown_args
    )
