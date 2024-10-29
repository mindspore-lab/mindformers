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

"""Computation cost analyzer for pipeline balancing."""

import json
import os
import re
from tqdm import tqdm
from mindformers.tools.logger import logger


class ComputationAnalyzer:
    """Parser & Analyzer for profiling timelines"""

    is_msprof_file = False

    def __init__(self, timeline_folder_path, model_name, num_of_stage, layer_list=None):
        self.timeline_folder_path = timeline_folder_path
        self.model_name = model_name
        self.num_of_stage = num_of_stage
        self.select_step_number = 0
        self.step_time = 0.0
        if layer_list:
            self.layer_list = layer_list
        else:
            self.layer_list = self._get_layer_list()
        self.timeline_data = self._get_timeline_data()
        logger.info("parsing layer objs")
        self.auto_partition_layer_objects, self.pre_defined_layer_objects = (
            self._parse_layer_objects())
        logger.info("parsing auto partition layer name")
        self.auto_partition_layer_name_list = (self.parse_auto_partition_layer_name_list())
        logger.info("parse layer with computation time list")
        self.layer_with_computation_time_list = (self.parse_layer_with_computation_time_list())
        logger.info("transform layer with cost list")
        self.layer_with_cost_list = self.transform_layer_with_cost_list()

    def _get_layer_list(self):
        """Return cfgs from model config file"""

        model_config_file = os.path.join(os.getcwd(), "cfgs", "model_layers.json")
        with open(model_config_file) as json_file:
            model_layers_data = json.load(json_file)
            for layer_list in model_layers_data:
                if self.model_name in layer_list["name"]:
                    return layer_list
        logger.info("ERROR: Not found model in model config file")
        return False

    def _get_timeline_data(self):
        """Return timeline objects from json file."""
        timeline_data = []
        for file_name in [file for file in os.listdir(self.timeline_folder_path) if
                          file.endswith(".json")]:
            if file_name.startswith("ascend_timeline_display"):
                self.is_msprof_file = False
            elif file_name.startswith("msprof"):
                self.is_msprof_file = True
            else:
                logger.error("ERROR: Not support timeline file type")
            with open(self.timeline_folder_path + file_name) as json_file:
                timeline_data += json.load(json_file)
        return timeline_data

    def _parse_step_duration(self, timeline_data):
        """Return timeline objects during a training step."""

        op_name = ""
        step_start = 0.0
        step_end = 0.0
        cpt = 0
        for obj in timeline_data:
            if "MatMul-op" in obj["name"]:
                op_name = obj["name"]
                break
        for obj in timeline_data:
            if obj["name"] == op_name:
                cpt += 1
                if cpt == self.select_step_number * self.num_of_stage:
                    step_start = float(obj["ts"])
                if cpt == (self.select_step_number + 1) * self.num_of_stage:
                    step_end = float(obj["ts"])
        step_time = step_end - step_start
        self.step_time = step_time
        return (step_start, step_end)

    def _load_json_data(self, file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    def _initialize_step_duration(self, timeline_data, file_name, step_start, step_end):
        if step_start == 0 or step_end == 0:
            step_start, step_end = self._parse_step_duration(timeline_data)
            if file_name.startswith("ascend_timeline_display"):
                self.is_msprof_file = False
            elif file_name.startswith("msprof"):
                self.is_msprof_file = True
        return step_start, step_end

    def _add_layer_object(self, objects_list, condition, obj):
        if condition and obj not in objects_list:
            objects_list.append(obj)

    def _process_file(self, file_name, step_start, step_end, pre_defined_layer_objects,
                      auto_partition_layer_objects):
        """_process_file"""
        timeline_data = self._load_json_data(self.timeline_folder_path + file_name)
        logger.info("processing file: %s. %d objects in it.", file_name, len(timeline_data))
        step_start, step_end = self._initialize_step_duration(timeline_data, file_name, step_start,
                                                              step_end)
        for obj in timeline_data:
            if "dur" not in obj:
                continue
            for layer_name in self.layer_list["pre_defined_layer"]:
                if (not obj["name"].startswith("lm_head") and "Linear" in obj[
                        "name"]) or "network-PanguAlphaHeadModel" in obj["name"]:
                    continue
                if layer_name in obj["name"]:
                    condition = self.is_msprof_file or (
                        not self.is_msprof_file and float(obj["ts"]) >= step_start and float(
                            obj["ts"]) + float(obj["dur"]) <= step_end)
                    self._add_layer_object(pre_defined_layer_objects, condition, obj)
            for layer_name in self.layer_list["auto_partition_layer"]:
                if layer_name in obj["name"]:
                    condition = self.is_msprof_file or (
                        not self.is_msprof_file and float(obj["ts"]) >= step_start and float(
                            obj["ts"]) + float(obj["dur"]) <= step_end)
                    self._add_layer_object(auto_partition_layer_objects, condition, obj)
        return step_start, step_end

    def _parse_layer_objects(self):
        auto_partition_layer_objects = []
        pre_defined_layer_objects = []
        step_start, step_end = 0, 0
        for file_name in tqdm(
                [file for file in os.listdir(self.timeline_folder_path) if file.endswith(".json")]):
            step_start, step_end = self._process_file(file_name, step_start, step_end,
                                                      pre_defined_layer_objects,
                                                      auto_partition_layer_objects)
        return auto_partition_layer_objects, pre_defined_layer_objects

    def parse_auto_partition_layer_name_list(self):
        """example: [42-TransformerEncoderLayer,43-TransformerEncoderLayer]"""
        auto_partition_layer_name_list = []
        for auto_partition_name in self.layer_list["auto_partition_layer"]:
            for obj in self.timeline_data:
                object_name = obj["name"]
                if auto_partition_name in object_name:
                    if self.is_msprof_file:
                        find_layer_name = re.findall(r"[0-9]*-" + auto_partition_name,
                                                     object_name)
                        layer_name = find_layer_name[0]
                    else:
                        layer_name = object_name
                    if layer_name not in auto_partition_layer_name_list:
                        auto_partition_layer_name_list.append(layer_name)

        return auto_partition_layer_name_list

    def parse_layer_with_computation_time_list(self):
        """
        Map each layer_name with its duration time. 例
        如[46-TransformerEncoderLayer':
        37.24729124999999, '47-TransformerEncoderLayer': 37.36572429687501]
        """
        layer_with_computation_time_list = {}
        for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
            layer_with_computation_time_list[pre_defined_layer_name] = 0
        for auto_partition_layer_name in self.auto_partition_layer_name_list:
            layer_with_computation_time_list[auto_partition_layer_name] = 0

        for obj in self.pre_defined_layer_objects:
            for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
                if pre_defined_layer_name in obj["name"]:
                    layer_with_computation_time_list[pre_defined_layer_name] += (obj["dur"] / 1000)
        for obj in self.auto_partition_layer_objects:
            for auto_partition_layer_name in self.auto_partition_layer_name_list:
                if auto_partition_layer_name in obj["name"]:
                    layer_with_computation_time_list[auto_partition_layer_name] += (
                        obj["dur"] / 1000)
        return layer_with_computation_time_list

    def transform_layer_with_cost_list(self):
        """calculating the average value of layer cost"""
        total_cost_auto_partition_layer = {}
        number_of_auto_partition_layer = {}
        transform_layer_with_cost_list = {}
        for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
            transform_layer_with_cost_list[pre_defined_layer_name] = 0

        for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
            total_cost_auto_partition_layer[auto_partition_layer_name] = 0
            number_of_auto_partition_layer[auto_partition_layer_name] = 0
            transform_layer_with_cost_list[auto_partition_layer_name] = 0

        # test
        for layer_name in self.layer_with_computation_time_list:
            for pre_defined_layer_name in self.layer_list["pre_defined_layer"]:
                if pre_defined_layer_name in layer_name:
                    var_tmp = self.layer_with_computation_time_list[layer_name]
                    transform_layer_with_cost_list[pre_defined_layer_name] += var_tmp
            for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
                if auto_partition_layer_name in layer_name:
                    # assuming that the duration time of a layer can not exceed 10% of step time
                    # in order to
                    # avoid some specific long time layers that caused from the error caused by
                    # time_line.json
                    if (not self.is_msprof_file and self.layer_with_computation_time_list[
                            layer_name] > self.step_time / 1000 / 10):
                        continue
                    total_cost_auto_partition_layer[auto_partition_layer_name] += \
                        self.layer_with_computation_time_list[layer_name]
                    number_of_auto_partition_layer[auto_partition_layer_name] += 1

        for auto_partition_layer_name in self.layer_list["auto_partition_layer"]:
            transform_layer_with_cost_list[auto_partition_layer_name] = (
                total_cost_auto_partition_layer[auto_partition_layer_name] /
                number_of_auto_partition_layer[auto_partition_layer_name])
        return transform_layer_with_cost_list


if __name__ == "__main__":
    path = "/home/czrz/pp_interleave/pp-balancing-master/timeline/"
    example_model_name = "LLaMA_prof"
    comp1 = ComputationAnalyzer(path, example_model_name, 8)
    logger.info(comp1.layer_with_computation_time_list)
    logger.info(comp1.layer_with_cost_list)
