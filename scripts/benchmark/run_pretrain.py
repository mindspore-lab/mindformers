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
"""Run pretrain process"""
import os
import zipfile
import subprocess
import tempfile
from enum import Enum
from glob import glob
from pathlib import Path
import argparse

import mindformers
from mindformers import MindFormerConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools import logger
from mindformers.trainer import Trainer
from scripts.benchmark.base_init_model import BaseInitModel

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(mindformers.__file__))


def convert_path(src_path, save_path):
    src_path = os.path.realpath(src_path)
    file_name = os.path.basename(src_path)
    file_path = os.path.join(save_path, file_name)
    return src_path, file_path


class EncodingFormat(Enum):
    """
    Encoding Format Enumeration Class
    """
    UTF_8 = "utf-8"


DATASET_TYPES = ['wiki', 'alpaca']


class DatasetType(Enum):
    ZIP = "zip"
    JSON = "json"
    TXT = "txt"
    OTHER = "other"


class ModelPretrain(BaseInitModel):
    """Inherited from BaseInitModel"""

    def __init__(self, model_name_or_dir: str = 'llama2_7b', input_args=None):
        super().__init__(model_name_or_dir=model_name_or_dir, input_args=input_args)

    def train(self, model_name_or_dir, train_data):
        """Start training process"""
        if train_data is None or train_data == '':
            raise ValueError("The dataset should not be None. Please provide a valid dataset.")

        # Set the training dataset directory
        self.config.train_dataset.data_loader.dataset_dir = train_data
        load_checkpoint = self.input_args.get('load_checkpoint', None)

        # If the model directory exists, look for checkpoint files
        if Path(model_name_or_dir).is_dir():
            ckpt_files = glob(str(Path(model_name_or_dir) / '*.ckpt'))
            if ckpt_files:
                load_checkpoint = ckpt_files[0]

        # Set the checkpoint in the config
        self.config.load_checkpoint = load_checkpoint

        # Determine if it's a pretraining task or fine-tuning task
        if load_checkpoint is None or load_checkpoint == '':
            logger.info("Starting a pretraining task as no checkpoint is provided.")
        else:
            logger.info(f"Starting a fine-tuning task using checkpoint: {load_checkpoint}.")

        # Initialize the trainer and start training
        train = Trainer(args=self.config)
        train.train()

    def generate_pretrain_data(self, pretrain_data, temp_dir) -> str:
        """
        Download or check the dataset based on the provided pretraining data parameter,
        and directly call the pretrain function.

        Parameters:
        pretrain_data (str): This can be a dataset name, a download URL, or a local directory path.
        - If it's a dataset name, the function will check if the dataset is supported and download it if necessary.
        - If it's a download URL, the function will download the file directly.
        - If it's a local directory path, the function will return the path and call the pretrain function directly.

        Returns:
        str: The path to the dataset that will be used for pretraining.
        """
        model_name = self.config.trainer.model_name.lower()

        if pretrain_data.lower() == 'wikitext2':
            return self._process_wikitext2(pretrain_data, temp_dir)

        if os.path.isfile(pretrain_data):
            filename = os.path.basename(pretrain_data)
            filetype = os.path.splitext(filename)[-1].lower()

            if filetype == '.mindrecord':
                return pretrain_data
            if 'wiki' in filename.lower():
                return self._process_wiki_file(pretrain_data, model_name, temp_dir)
            if 'alpaca' in filename.lower():
                return self._process_alpaca_file(pretrain_data, model_name, temp_dir)

            return pretrain_data

        raise ValueError(f"File {pretrain_data} does not exist or is not a file.")

    @staticmethod
    def _run_preprocess_script(
            script_path,
            dataset_type,
            input_glob,
            tokenizer_path,
            seq_length,
            output_file
    ):
        """run script"""

        logger.info(f"Running preprocessing script: {script_path}")
        try:
            subprocess.run([
                'python', script_path,
                '--dataset_type', dataset_type,
                '--input_glob', input_glob,
                '--model_file', tokenizer_path,
                '--seq_length', str(seq_length),
                '--output_file', output_file
            ], check=True)
            logger.info(f"Executed {script_path} successfully with output file {output_file}.")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to execute {script_path}: {e}") from e

    def _process_wikitext2(self, pretrain_data, temp_dir) -> str:
        """process_wikitext2"""
        tokenizer_path = self.config.processor.tokenizer.vocab_file
        if tokenizer_path is not None:
            unzip_dir = os.path.join(temp_dir, os.path.splitext(pretrain_data)[0])
            self.unzip_file(pretrain_data, unzip_dir)
            input_glob = os.path.join(unzip_dir, 'wikitext-2', 'wiki.train.tokens')

            script_path = os.path.join(
                project_root,
                'mindformers',
                'tools',
                'dataset_preprocess',
                'llama',
                'llama_preprocess.py'
            )
            seq_length = 4096
            dataset_name = self.generate_dataset_name("wiki", seq_length)
            output_file = os.path.join(temp_dir, dataset_name)

            self._run_preprocess_script(
                script_path,
                'wiki',
                input_glob,
                tokenizer_path,
                seq_length,
                output_file
            )
            return output_file

        raise ValueError(f"Please specify the vocabulary [tokenizer.model].")

    def _process_wiki_file(self, pretrain_data, model_name, temp_dir) -> str:
        """process_wiki_file"""
        tokenizer_path = self.config.processor.tokenizer.vocab_file
        if tokenizer_path is not None:
            seq_length = self.config.model.model_config.seq_length
            dataset_name = self.generate_dataset_name("wiki", seq_length)
            if 'llama3' in model_name:
                script_path = os.path.join(
                    project_root,
                    'research',
                    'llama3',
                    'llama3_preprocess.py'
                )
                output_file = os.path.join(temp_dir, dataset_name)
            else:
                script_path = os.path.join(
                    project_root,
                    'mindformers',
                    'tools',
                    'dataset_preprocess',
                    'llama',
                    'llama_preprocess.py'
                )
                output_file = os.path.join(temp_dir, dataset_name)

            self._run_preprocess_script(
                script_path,
                'wiki',
                pretrain_data,
                tokenizer_path,
                seq_length,
                output_file
            )
            return output_file

        raise ValueError(f"Please specify the vocabulary [tokenizer.model].")

    def _process_alpaca_file(self, pretrain_data, model_name, temp_dir) -> str:
        """process_alpaca_file"""
        tokenizer_path = self.config.processor.tokenizer.vocab_file
        if tokenizer_path is not None:
            alpaca_converter_script = os.path.join(
                project_root,
                'mindformers',
                'tools',
                'dataset_preprocess',
                'llama',
                'alpaca_converter.py'
            )
            converted_file = os.path.join(temp_dir, 'alpaca-data-conversation.json')

            try:
                subprocess.run([
                    'python', alpaca_converter_script,
                    '--data_path', pretrain_data,
                    '--output_path', converted_file
                ], check=True)
                logger.info(f"Executed {alpaca_converter_script} successfully with output file {converted_file}.")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to execute {alpaca_converter_script}: {e}") from e


            seq_length = self.config.model.model_config.seq_length
            dataset_name = self.generate_dataset_name("alpaca", seq_length)
            if 'llama3' in model_name:
                script_path = os.path.join(
                    project_root,
                    'research',
                    'llama3',
                    'llama3_preprocess.py'
                )
                output_file = os.path.join(temp_dir, dataset_name)
            else:
                script_path = os.path.join(
                    project_root,
                    'mindformers',
                    'tools',
                    'dataset_preprocess',
                    'llama',
                    'llama_preprocess.py'
                )
                output_file = os.path.join(temp_dir, dataset_name)

            self._run_preprocess_script(
                script_path,
                'qa',
                converted_file,
                tokenizer_path,
                seq_length,
                output_file
            )
            return output_file

        raise ValueError(f"Please specify the vocabulary [tokenizer.model].")

    @staticmethod
    def generate_dataset_name(dataset_type, seq_length):
        """generate_dataset_name"""
        if dataset_type == 'wiki':
            return f"{dataset_type}{seq_length}.mindrecord"

        if dataset_type == 'alpaca':
            return f"{dataset_type}-fastchat{seq_length}.mindrecord"

        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of {DATASET_TYPES}.")

    @staticmethod
    def unzip_file(zip_path, extract_to):
        """
        Extract the ZIP file to the specified directory.

        Parameters:
        zip_path (str): The path to the ZIP file.
        extract_to (str): The directory where the extracted files will be saved.

        Raises:
        Exception: If extracting the file fails or the file is not a valid ZIP format.
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract the file: {e}") from e

    def process_tokenizer(self):
        """Process tokenizer"""
        try:
            # Check if config, processor, tokenizer, or vocab_file is None
            if not self.config or not self.config.processor or not self.config.processor.tokenizer:
                raise AttributeError("Tokenizer configuration is missing in self.config.processor.tokenizer.")

            # Check if vocab file is already set in the config
            if self.config.processor.tokenizer.vocab_file:
                return

            # Look for tokenizer files
            tokenizer_files = glob(f"{self.model_path}/tokenizer.model")
            vocab_file = self.input_args.get('vocab_file', None)

            # If no vocab file is provided or exists, use the one in config
            if not vocab_file or not os.path.exists(vocab_file):
                vocab_file = self.config.processor.tokenizer.vocab_file

            # If a valid vocab file exists
            if vocab_file and os.path.exists(vocab_file):
                # make a soft link of the tokenizer file in model_path
                src_path, vocab_file = convert_path(vocab_file, self.model_path)
                try:
                    os.symlink(src_path, vocab_file)
                except OSError:
                    logger.warning(f"Link of {vocab_file} already exists.")
            elif tokenizer_files:
                vocab_file = tokenizer_files[0]
            else:
                logger.error(f"Currently does not support downloading the tokenizer "
                             f"and vocab file of {self.model_name} from online.")
                return

            logger.info(f"Using tokenizer file: {vocab_file}.")
            # Set the vocab file in the config
            self.config.processor.tokenizer.vocab_file = vocab_file

        except AttributeError as e:
            logger.error(f"Tokenizer configuration error: {e}")
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"An error occurred while processing the tokenizer: {e}")


def convert_type(data):
    """convert_type"""
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


def build_args(args_):
    """Build input_args"""
    error_info = "script input args should be aligned '--key value',"
    if divmod(len(args_), 2)[1] != 0:
        raise ValueError(f"{error_info} length of args is not correct.")

    input_args = dict()
    for arg_key, arg_value in zip(args_[0::2], args_[1::2]):
        if len(arg_key) <= 2 or arg_key[:2] != '--':
            raise ValueError(f"{error_info} got '{arg_key}' in keys.")
        if arg_value[:2] == '--':
            raise ValueError(f"{error_info} got '{arg_value}' in values.")

        value = convert_type(arg_value)
        input_args[arg_key[2:]] = value
    return input_args


def main(model_name_or_dir: str = 'llama2_7b', pretrain_data=None, input_args=None):
    """mian function"""
    model_realpath = os.path.realpath(model_name_or_dir)
    if not os.path.exists(model_realpath):
        raise ValueError(f"The provided model path does not exist: {model_realpath}")

    if Path(model_name_or_dir).is_dir():
        yaml_files = glob(str(Path(model_name_or_dir) / '*.yaml'))
        if yaml_files:
            config = MindFormerConfig(yaml_files[0])
            build_context(config)
            build_parallel_config(config)
    else:
        raise RuntimeError('Currently only support offline infer')
    model_train = ModelPretrain(model_name_or_dir, input_args)
    model_train.process_tokenizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        train_data = model_train.generate_pretrain_data(pretrain_data, temp_dir)
        model_train.train(model_name_or_dir, train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_dir', type=str, default="llama2_7b",
        help='input model name with size, e.g., llama2_7b.')
    parser.add_argument(
        '--pretrain_data', type=str, default="wikitext2",
        help='dataset directory of data loader to train/finetune. '
             'Default: None')

    args, unknown_args = parser.parse_known_args()
    unknown_args = build_args(unknown_args)

    main(args.model_name_or_dir, args.pretrain_data, unknown_args)
