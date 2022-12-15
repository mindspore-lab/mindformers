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
"""Flickr8k DataLoader."""
import os
from collections import defaultdict

from mindspore.dataset import GeneratorDataset

from ...tools.image_tools import load_image
from ...tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class Flickr8kDataLoader:
    """Flicker8k Dataloader"""
    _default_column_names = ["image", "text"]
    def __new__(cls, dataset_dir, annotation_dir, column_names=None, stage="train"):
        """
        Flicker8k Dataloader API

        Args:
            dataset_dir: the directory to images
            annotation_dir: the directory to Flickr_8k.trainImages.txt, Flickr_8k.testImages.txt,
                            Flickr_8k.devImages.txt, and Flickr8k.token.txt
            stege: the supported key words are in ["train"、"test"、"del"、"all"]
            column_names: the output column names, a tuple or a list of string with length 2

        Return:
            a GeneratorDataset for Flickr8k dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if not os.path.isdir(annotation_dir):
            raise TypeError(f"{annotation_dir} is not existed.")

        if not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 2, but got {type(column_names)}")

        if len(column_names) != 2:
            raise ValueError(f"the length of column_names should be 2,"
                             f" but got {len(column_names)}")

        if not isinstance(column_names[0], str) or not isinstance(column_names[1], str):
            raise ValueError(f"the item type of column_names should be string,"
                             f" but got {type(column_names[0])} and {type(column_names[1])}")

        if column_names is None:
            column_names = cls._default_column_names
        flick8k_dataset = Flickr8kDataSet(dataset_dir, annotation_dir, stage)
        return GeneratorDataset(flick8k_dataset, column_names)

class Flickr8kDataSet:
    """Flickr8k DataSet"""
    def __init__(self, dataset_dir, annotation_dir, stage="train"):
        """
        Flicker8k Dataset

        Args:
            dataset_dir: the directory to images
            annotation_dir: the directory to Flickr_8k.trainImages.txt, Flickr_8k.testImages.txt,
                            Flickr_8k.devImages.txt, and Flickr8k.token.txt
            stege: the supported key words are in ["train"、"test"、"del"、"all"]

        Return:
            a iterable dataset for Flickr8k dataset
        """
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if not os.path.isdir(annotation_dir):
            raise TypeError(f"{annotation_dir} is not existed.")

        self.dataset_dir = dataset_dir

        if stage == "train":
            train_file = os.path.join(annotation_dir, "Flickr_8k.trainImages.txt")
            with open(train_file, 'r', encoding='utf-8') as file:
                image_names = file.read().splitlines()
        elif stage == "test":
            test_file = os.path.join(annotation_dir, "Flickr_8k.testImages.txt")
            with open(test_file, 'r', encoding='utf-8') as file:
                image_names = file.read().splitlines()
        elif stage == "dev":
            dev_file = os.path.join(annotation_dir, "Flickr_8k.devImages.txt")
            with open(dev_file, 'r', encoding='utf-8') as file:
                image_names = file.read().splitlines()
        elif stage == "all":
            image_names = [file for file in os.listdir(dataset_dir) if file.endswith(".jpg")]
        else:
            raise KeyError("unsupported stage.")

        annotation_file = os.path.join(annotation_dir, "Flickr8k.token.txt")
        with open(annotation_file, 'r', encoding='utf-8') as file:
            annotation_list = file.read().splitlines()

        dataset_dict = defaultdict(list)
        for line in annotation_list:
            image_name = line.split("#")[0]
            if image_name in image_names:
                image_anno = line.split("\t")[-1]
                dataset_dict[image_name].append(image_anno)

        self.image_names = image_names
        self.dataset_dict = dataset_dict

    def __getitem__(self, item):
        image_name = self.image_names[item]
        image_path = os.path.join(self.dataset_dir, image_name)
        image = load_image(image_path)

        image_anno = self.dataset_dict[image_name]
        return image, image_anno

    def __len__(self):
        return len(self.image_names)
