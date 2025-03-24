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
"""get mock data for dataset"""
import os
import json
import pickle
import numpy as np
import cv2
from mindspore.mindrecord import FileWriter

np.random.seed(0)


# pylint: disable=W0703
def get_cifar100_data(data_path, data_num: int = 1):
    """get cifar100 data"""
    np.random.seed(0)
    meta_dict = {
        b"fine_label_names": [b"fine_1", b"fine_2"],
        b"coarse_label_names": [b"coarse_1", b"coarse_2"]
    }

    train_dict = {
        b"fine_labels": list(np.random.randint(0, 2, size=data_num)),
        b"coarse_labels": list(np.random.randint(0, 2, size=data_num)),
        b"data": np.random.randint(0, 256, size=(data_num, 3*32*32))
    }

    test_dict = {
        b"fine_labels": list(np.random.randint(0, 2, size=data_num)),
        b"coarse_labels": list(np.random.randint(0, 2, size=data_num)),
        b"data": np.random.randint(0, 256, size=(data_num, 3*32*32))
    }

    meta_path = os.path.join(data_path, "meta")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(meta_path, "wb") as w_meta, open(train_path, "wb") as w_train, \
                    open(test_path, "wb") as w_test:
                pickle.dump(meta_dict, w_meta)
                pickle.dump(train_dict, w_train)
                pickle.dump(test_dict, w_test)
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(test_path):
                os.remove(test_path)
            print(f"cifar100 data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"cifar100 data initialize failed for {count} times.")


# pylint: disable=W0703
def get_llava_data(data_path, data_num: int = 1):
    """get llava data"""
    test_image_name = "test.jpg"
    os.makedirs(os.path.join(data_path, "train2014"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "text"), exist_ok=True)
    test_jpg_path = os.path.join(data_path, "train2014", f"COCO_train2014_{test_image_name}")
    data = [
        {
            "id": "000000442786",
            "image": test_image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "What do you see happening in this image?\n<image>"
                },
                {
                    "from": "gpt",
                    "value": "The scene depicts a lively plaza area with several people walking and enjoying their "
                             "time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. "
                             "The kite has multiple sections attached to it, spread out in various directions as "
                             "if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and "
                             "interacting with others. Some of these individuals are carrying handbags, and others have"
                             " backpacks. The image captures the casual, social atmosphere of a bustling plaza on a "
                             "nice day."
                }
            ]
        }
    ] * data_num

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            txt_path = os.path.join(data_path, "text", "detail_23k.json")
            with open(txt_path, "w", encoding="utf-8") as w_data:
                w_data.write(json.dumps(data))

            image = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(test_jpg_path, image)
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(test_jpg_path):
                os.remove(test_jpg_path)
            print(f"llava data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

        if not success_sig:
            raise RuntimeError(f"llava data initialize failed for {count} times.")


# pylint: disable=W0703
def get_mindrecord_data(data_path, data_num: int = 1, seq_len: int = 16):
    """get mindrecord data"""
    np.random.seed(0)
    output_file = os.path.join(data_path, "test.mindrecord")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            writer = FileWriter(output_file)
            data_schema = {
                "input_ids": {"type": "int32", "shape": [-1]},
                "attention_mask": {"type": "int32", "shape": [-1]},
                "labels": {"type": "int32", "shape": [-1]}
            }
            writer.add_schema(data_schema, "test-schema")
            for _ in range(data_num):
                features = {}
                features["input_ids"] = np.random.randint(0, 64, size=seq_len).astype(np.int32)
                features["attention_mask"] = features["input_ids"]
                features["labels"] = features["input_ids"]
                writer.write_raw_data([features])
            writer.commit()
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(output_file + ".db"):
                os.remove(output_file + ".db")
            print(f"mindrecord data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"mindrecord data initialize failed for {count} times.")


# pylint: disable=W0703
def get_adgen_data(data_path, is_json_error: bool = False):
    """get adgen data"""
    np.random.seed(0)
    if not is_json_error:
        data1 = {
            "content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
            "summary": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"
        }
        data2 = {
            "mock_key_1": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
            "mock_key_2": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"
        }
        data3 = {
            "content": "",
            "summary": ""
        }
        data = [data1, data2, data3]
        train_path = os.path.join(data_path, "train.json")
    else:
        data = ["类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤", "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"]
        train_path = os.path.join(data_path, "json_error_train.json")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w:
                if not is_json_error:
                    write_data = [json.dumps(item, ensure_ascii=False) for item in data] + ["   "]
                    w.write("\n".join(write_data))
                else:
                    w.write("\n".join(data))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"adgen data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"adgen data initialize failed for {count} times.")


# pylint: disable=W0703
def get_wikitext_data(data_path, data_num: int = 1):
    """get wikitext data"""
    train_path = os.path.join(data_path, "wiki.train.tokens")

    data = ["= 华为 =", "华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。", "== 华为 ==",
            "An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13."]
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write("\n".join(data * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"flickr8k data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"flickr8k data initialize failed for {count} times.")


# pylint: disable=W0703
def get_json_data(data_path, data_num: int = 1):
    """get json data"""
    train_path = os.path.join(data_path, "train.json")

    data = {"input": ["华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。"]}
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write("\n".join([json.dumps(data, ensure_ascii=False)] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"json data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"json data initialize failed for {count} times.")


# pylint: disable=W0703
def get_agnews_data(data_path, data_num: int = 1):
    """get agnews data"""
    data = ["\"1\"", "\"华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。\"",
            "\"An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.\""]

    data_path = os.path.join(data_path, "agnews")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(data_path, "w", encoding="utf-8") as w:
                w.write("\n".join([",".join(data)] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(data_path):
                os.remove(data_path)
            print(f"agnews data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"agnews data initialize failed for {count} times.")


# pylint: disable=W0703
def get_alpaca_data(data_path, data_num: int = 1):
    """get alpaca data"""
    train_path = os.path.join(data_path, "train.json")

    data = {
        "instruction": "华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。",
        "input": "",
        "output": "An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13."
    }
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write(json.dumps([data] * data_num, ensure_ascii=False))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"json data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"json data initialize failed for {count} times.")
