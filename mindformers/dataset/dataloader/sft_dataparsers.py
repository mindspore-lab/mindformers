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
"""SFT DataParsers."""


def default_parser(row_dict):
    """Default data parsing function.Returns the first three values of `row_dict`."""
    values = list(row_dict.values())
    if len(values) == 1:
        return values[0], "", ""
    if len(values) == 2:
        return values[0], values[1], ""
    return values[0], values[1], values[2]


def alpaca_parser(row_dict):
    """Parsing the Alpaca dataset."""
    if row_dict.get("input"):
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        ).format_map(row_dict)
    else:
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}"
        ).format_map(row_dict)
    return text, "", ""


def advertisegen_parser(row_dict):
    """Parsing the AdvertiseGen dataset."""
    return row_dict.get("content"), row_dict.get("summary"), ""


def cola_parser(row_dict):
    """Parsing the COLA dataset."""
    values = list(row_dict.values())
    return values[3], "", values[1]


def imdb_parser(row_dict):
    """Parsing the IMDB dataset."""
    label = 1 if row_dict.get("sentiment") == 'positive' else 0
    return row_dict.get("review"), "", label


def sst2_parser(row_dict):
    """Parsing the SST-2 dataset."""
    return row_dict.get("sentence"), "", row_dict.get("label")


def agnwes_parser(row_dict):
    """Parsing the AG-News dataset."""
    return row_dict.get("sentence"), "", row_dict.get("label")


def tnews_parser(row_dict):
    """Parsing the TNEWS dataset."""
    label = int(row_dict.get("label")) - 100
    return row_dict.get("sentence"), "", label


_DATA_PARSER_MAP = {
    "default": default_parser,
    "alpaca": alpaca_parser,
    "advertisegen": advertisegen_parser,
    "cola": cola_parser,
    "imdb": imdb_parser,
    "sst-2": sst2_parser,
    "ag-news": agnwes_parser,
    "tnews": tnews_parser,
}
