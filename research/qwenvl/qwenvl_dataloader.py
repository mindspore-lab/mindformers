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
"""QwenVL DataLoader."""

import json
import os
from typing import Optional, List, Tuple
from typing import Union, Callable

import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision
from mindformers.dataset.dataloader.sft_dataloader import SFTDataSet
from mindformers.dataset.transforms.vision_transforms import BatchResize
from mindformers.tools import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from PIL import Image
from pyarrow import RecordBatch


class BatchResizeV2(BatchResize):
    def __init__(self, img_resolution, interpolation='cubic'):
        super().__init__(img_resolution, interpolation)
        self.resize = vision.Resize(img_resolution, self.interpolation)


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class QwenVLDataLoader:
    """dataloader for QwenVL format dataset"""
    _default_column_names = ["image", "text"]

    def __new__(cls,
                dataset_dir: str,
                annotation_file: str,
                column_names: Optional[Union[List[str], Tuple[str]]] = None,
                shuffle: Optional[bool] = True,
                extra_kwargs: Optional[dict] = None,
                **kwargs):
        if column_names is None:
            column_names = cls._default_column_names

        if not os.path.exists(dataset_dir):
            raise FileExistsError(f"The dataset_dir {dataset_dir} is not exist.")

        if not os.path.exists(annotation_file):
            raise FileExistsError(f"The annotation_file {annotation_file} is not exist.")

        qwenvl_dataset = QwenVLSFTDataset(dataset_dir, annotation_file, **extra_kwargs)
        return GeneratorDataset(qwenvl_dataset,
                                column_names,
                                shuffle=shuffle,
                                **kwargs)


class QwenVLSFTDataset(SFTDataSet):
    """dataset for QwenVL format dataset"""

    def __init__(self,
                 image_dir: str,
                 annotation_file: str,
                 max_length: int = 1025,
                 read_function: Callable = None,
                 map_function_kwargs: dict = None,
                 out_img_shape: int = 448,
                 max_img_len: int = 5,
                 max_chunk_size: int = 1024):
        self.out_img_shape = (out_img_shape, out_img_shape)
        self.resize = BatchResizeV2(self.out_img_shape, interpolation='cubic')

        self.max_chunk_size = max_chunk_size

        super().__init__(annotation_file, None, None, file_format="json",
                         max_length=max_length,
                         read_function=read_function, map_function=self._qwenvl_map,
                         map_function_kwargs=map_function_kwargs)
        self.image_dir = image_dir
        self.max_img_len = max_img_len
        self.num_samples = sum(len(batch) for batch in self.table)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        chunk_idx, item_idx = divmod(i, self.max_chunk_size)
        example = self.table[chunk_idx].take([item_idx]).to_pylist()[0]
        result = self.map_function(example, **self.map_function_kwargs)
        img_list, img_idx = self._img_padding(result["img_path_list"], result["img_idx"])
        new_dict = {key: value for key, value in result.items() if key not in ("img_path_list", "img_idx")}
        new_dict["img_idx"] = img_idx
        return img_list, new_dict

    def _img_padding(self, img_list, img_idx):
        """padding image if size of img_list is small then max_img_len"""
        if not img_list:
            return [np.ones(self.out_img_shape + (3,), dtype=np.uint8)] * self.max_img_len, [-1] * self.max_img_len
        new_img_list = []
        new_img_idx = []
        for i, img_path in enumerate(img_list):
            new_img_list.append(self.resize(np.array(Image.open(img_path).convert("RGB"))))
            new_img_idx.append(img_idx[i])
        if len(new_img_list) <= self.max_img_len:
            for _ in range(self.max_img_len - len(new_img_list)):
                new_img_list.append(new_img_list[-1])
                new_img_idx.append(-1)
        else:
            raise ValueError("The number of images is greater than the maximum number of images.")
        return new_img_list, new_img_idx

    @staticmethod
    def _find_img_tags(s):
        """find img tags in the caption and return the index"""
        start_tag = '<img>'
        end_tag = '</img>'
        start_positions = []
        end_positions = []
        start = 0
        while start < len(s):
            start = s.find(start_tag, start)
            if start == -1:
                break
            start_positions.append(start + 5)
            start += len(start_tag)
        start = 0
        while start < len(s):
            start = s.find(end_tag, start)
            if start == -1:
                break
            end_positions.append(start)
            start += len(end_tag)
        return start_positions, end_positions

    def _qwenvl_map(self, example, **kwargs):
        """map function to convert sample"""
        data_field = kwargs.get("data_field", "conversations")
        from_keyword, value_keyword = kwargs.get("from_keyword", "from"), kwargs.get("value_keyword", "value")
        user_role_name = kwargs.get("user_role_name", "human")
        assistant_role_name = kwargs.get("assistant_role_name", "gpt")
        user_prompt, assistant_prompt = kwargs.get("user_prompt", ""), kwargs.get("assistant_prompt", "")
        system_message = kwargs.get("system_message", "You are a helpful assistant.")

        raw_data = []
        raw_data_role = []
        img_idx = []
        img_path_list = []
        img_pos = 0

        # add system info
        system = "<|im_start|>system\n" + system_message + "<|im_end|>\n"
        raw_data.append(system)
        raw_data_role.append('system')

        for message in example[data_field]:
            from_ = message[from_keyword]
            value = message[value_keyword]
            sub_img_path_list = []
            if '<img>' in value:
                img_start_pos, img_end_pos = self._find_img_tags(value)
                sub_img_path_list, img_string = self._get_img_path(value, img_start_pos, img_end_pos)
                for _ in img_string:
                    img_idx.append(img_pos)
                    img_pos += 1
            raw_data_role.append(from_)
            if from_ == user_role_name:
                raw_data.append("<|im_start|>" + from_ + '\n' + user_prompt + value + '<|im_end|>\n')
            elif from_ == assistant_role_name:
                raw_data.append("<|im_start|>" + from_ + '\n' + assistant_prompt + value + '<|im_end|>\n')
            else:
                raise ValueError(f"Incorrect role name: {from_}. Check the values of `user_role_name` "
                                 f"and `assistant_role_name` in `map_function_kwargs`.")
            img_path_list.extend(sub_img_path_list)

        if len(img_path_list) > self.max_img_len:
            logger.warning("The number of images in some samples exceeds the max_img_len. "
                           "The excess images will be discarded. max_img_len is %s and the actual image size is %s",
                           self.max_img_len, len(img_path_list))
            img_idx = img_idx[:self.max_img_len]
            img_path_list = img_path_list[:self.max_img_len]

        return {
            "img_path_list": img_path_list,
            "raw_data": raw_data,
            "raw_data_role": raw_data_role,
            "img_idx": img_idx,
            "user_role_name": user_role_name,
            "assistant_role_name": assistant_role_name,
            "task": 'sft'
        }

    def _get_img_path(self, value, img_start_pos, img_end_pos):
        img_img_path = []
        img_string = []
        for i, img_start_pos_item in enumerate(img_start_pos):
            img_path_in_value = value[img_start_pos_item:img_end_pos[i]]
            img_img_path.append(os.path.join(self.image_dir, img_path_in_value))
            img_string.append(img_path_in_value)
        return img_img_path, img_string

    def _read_json(self, path):
        """Reads JSON format with pyarrow and return list of RecordBatch"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [RecordBatch.from_pylist(data[i:i + self.max_chunk_size])
                for i in range(0, len(data), self.max_chunk_size)]
