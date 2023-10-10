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
"""SFT DataLoader"""
import os
import json
from json.decoder import JSONDecodeError
from typing import Callable
from pyarrow.json import read_json
from pyarrow.csv import read_csv, ParseOptions
from pyarrow.parquet import ParquetDataset
from pyarrow.lib import Table

from mindspore.dataset import GeneratorDataset
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.dataset.dataloader.datareaders import _DATA_READER_MAP
from mindformers.dataset.dataloader.sft_dataparsers import _DATA_PARSER_MAP


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class SFTDataLoader:
    """SFT DataLoader"""
    def __new__(cls,
                dataset_dir: str,
                dataset_name: str = "",
                file_format: str = None,
                customized_reader: Callable = None,
                customized_parser: Callable = None,
                shuffle: bool = False,
                **kwargs):
        r"""
        SFT DataLoader API.

        Args:
            dataset_dir (str): The directory path to parquet text with hdfs.
            dataset_name (str): Dataset name. Currently, ["wikitext"] is supported.
            file_format (str): Retrieves the end character of the desired file name.
            customized_reader (Callable): User-defined functions for reading data.
                The input parameter is the path of the dataset file.
                The return value is a list of many sentences.
            customized_parser (Callable): User-defined function for parsing data.
                The input parameter is a dictionary that contains a single line of data.
                There are three return values: prompt, answerh and label. If a value is not required,
                an empty string is returned.
            shuffle (Optional[bool]): Whether or not to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.

        Return:
            A GeneratorDataset object.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import SFTDataLoader
            >>> data_loader = SFTDataLoader(dataset_dir="The required task dataset path",
            ...                                    dataset_name="alpaca",
            ...                                    file_format="json",
            ...                                    shuffle=True)
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
        """
        dataset = SFTDataSet(dataset_dir, dataset_name, file_format, customized_reader, customized_parser)
        column_names = ["prompt", "answer", "label"]
        return GeneratorDataset(dataset, column_names=column_names, shuffle=shuffle, **kwargs)


class SFTDataSet:
    r"""
    SFT DataSet API.

    Args:
        dataset_dir (str): The directory path to parquet text with hdfs.
        dataset_name (str): Dataset name. Currently, ["wikitext"] is supported.
        file_format (str): Retrieves the end character of the desired file name.
        customized_reader (Callable): User-defined functions for reading data.
            The input parameter is the path of the dataset file.
            The return value is a list of many sentences.
        customized_parser (Callable): User-defined function for parsing data.
            The input parameter is a dictionary that contains a single line of data.
            There are three return values: prompt, answerh and label. If a value is not required,
            an empty string is returned.

    Return:
        A GeneratorDataset object.

    Raises:
        ValueError: Error input for dataset_dir.
        TypeError: Type error for column_names.
    """
    def __init__(self, dataset_dir, dataset_name=None, file_format=None,
                 customized_reader=None, customized_parser=None):
        self._general_reader_map = {
            "json": self._read_json,
            "jsonl": read_json,
            "csv": read_csv,
            "tsv": self._read_tsv,
            "parquet": self._read_parquet,
        }
        file_format = self._check_format(dataset_dir, file_format)
        dataset_name = dataset_name.lower() if dataset_name else "default"
        if customized_reader:
            self.table = customized_reader(dataset_dir)
        elif dataset_name in _DATA_READER_MAP:
            self.table = _DATA_READER_MAP[dataset_name](dataset_dir)
        else:
            self.table = self._general_reader_map[file_format](dataset_dir)
        self.data_parser_fun = customized_parser \
            if customized_parser else _DATA_PARSER_MAP.get(dataset_name, _DATA_PARSER_MAP["default"])

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, i):
        row_dict = self.table.take([i]).to_pylist()[0]
        prompt, answer, label = self.data_parser_fun(row_dict)
        return prompt, answer, label

    def _check_format(self, dataset_dir, file_format):
        """Check and correct the `format`."""
        if not file_format:
            if os.path.isfile(dataset_dir):
                file_format = os.path.splitext(dataset_dir)[1].strip('.')
            elif os.path.isdir(dataset_dir):
                # For a directory, the suffix of the first file is used. Only the parquet format is supported.
                file_format = os.path.splitext(next(os.scandir(dataset_dir)))[1].strip('.')
                if file_format != "parquet":
                    raise ValueError("The dataset directory supports only the parquet format.")
            else:
                raise FileNotFoundError(rf"No such file or directory: {dataset_dir}")

        file_format = file_format.strip(".").lower()
        if file_format in ("json", "jsonl"):
            try:
                with open(dataset_dir, 'r') as f:
                    json.load(f)
                file_format = "json"
            except JSONDecodeError:
                file_format = "jsonl"

        if file_format in self._general_reader_map:
            return file_format
        raise ValueError("The dataset file format can only be json, jsonl, csv, tsv, and parquet.")

    @staticmethod
    def _read_parquet(path):
        """Reads data in parquet format."""
        ds = ParquetDataset(path, memory_map=True)
        return ds.read()

    @staticmethod
    def _read_tsv(path):
        """Reads data in TSV format."""
        return read_csv(path, parse_options=ParseOptions(delimiter="\t"))

    @staticmethod
    def _read_json(path):
        """Reads data in JSON format."""
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return Table.from_pydict(data)
        if isinstance(data, list):
            # For pyarrow 12.0.1, pyarrow.lib.Table does not support from_pylist.
            pydict = {k: [i[k] for i in data] for k in data[0]}
            return Table.from_pydict(pydict)
        raise NotImplementedError
