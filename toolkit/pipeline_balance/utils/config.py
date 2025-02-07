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
"""config json generator"""
import json
import os
from dataclasses import dataclass, asdict

import yaml

from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.layer import Layer
from toolkit.pipeline_balance.utils.compute_memory import Stage, ComputeMemory
import toolkit.pipeline_balance.utils.recompute as Recompute


@dataclass
class LayersDescription:
    """layers description"""

    name: str
    type: Layer.type_enum
    model_name: str
    time: int
    nb_layer: int
    memory_parameter: int

    def __init__(
            self, layer_type: Layer.type_enum, time: int, nb_layer: int, model_name: str
    ):
        self.type = layer_type.name
        self.time = time
        self.name = layer_type.name
        self.nb_layer = nb_layer
        self.model_name = model_name


@dataclass
class ModelInfo:
    """basic info of a model"""

    name: str
    pre_defined_layer: dict
    auto_partition_layer: dict
    layers_description: list[LayersDescription]

    def __init__(self, model_name, head_time, body_time, tail_time, nb_layer):
        self.name = model_name
        self.pre_defined_layer = {"HEAD": 0, "TAIL": -1}
        self.auto_partition_layer = {"NumberOfLayers": nb_layer}
        self.layers_description = []
        self.layers_description.append(
            LayersDescription(Layer.type_enum.HEAD, head_time, 1, model_name)
        )
        self.layers_description.append(
            LayersDescription(Layer.type_enum.BODY, body_time, nb_layer, model_name)
        )
        self.layers_description.append(
            LayersDescription(Layer.type_enum.TAIL, tail_time, 1, model_name)
        )

    def get_layer_by_type(self, layer_type: Layer.type_enum):
        for layer in self.layers_description:
            if layer.type == layer_type.name:
                return layer
        return None

    def memory_update(self, mem_act, mem_par, mem_head, mem_tail):
        """update input memories to layer description"""
        self.get_layer_by_type(Layer.type_enum.HEAD).memory_parameter = mem_head
        self.get_layer_by_type(Layer.type_enum.TAIL).memory_parameter = mem_tail
        self.get_layer_by_type(Layer.type_enum.BODY).memory_parameter = mem_par

        json_data = asdict(self)

        for rec in Recompute.TYPE:
            rec_mem = mem_act.get(rec)
            if rec_mem is None:
                continue
            for layer in json_data["layers_description"]:
                if layer["type"] == Layer.type_enum.BODY.name:
                    layer[Recompute.JSON_MEMORY_NAME[rec]] = rec_mem
        self.to_json_ = json_data

    def dump_json(self, file_name):
        with open(file_name, "w") as json_file:
            json.dump(self.to_json_, json_file, indent=4)


def time_parser(file_name: str):
    """parse time given by yaml"""
    if file_name is None:
        logger.error("input file cannot be none")
        raise ValueError("input file cannot be none")

    if not file_name.endswith("yaml") and not file_name.endswith("yml"):
        logger.error("Only accept yaml as input format")
        raise ValueError(f"Only accept yaml as input format. not {file_name}")

    filepath = os.path.realpath(file_name)
    with open(filepath, encoding="utf-8") as fp:
        cfg_dict = yaml.safe_load(fp)
    head_time = cfg_dict["time_config"]["head"]
    body_time = cfg_dict["time_config"]["body"]
    tail_time = cfg_dict["time_config"]["tail"]

    return head_time, body_time, tail_time


def process_offset(offset, pipeline_num):
    """process input offset"""
    rounds = 1
    if isinstance(offset, int) and offset == 0:
        offset = [0] * pipeline_num
    # if offset is list of lists (usually when pp=4)
    elif isinstance(offset, list) and any(isinstance(item, list) for item in offset):
        tmp_offset = []
        for item in offset:
            if isinstance(item, int) and item == 0:
                tmp_offset.append([0] * pipeline_num)
            elif not (isinstance(item, list) and len(item) == pipeline_num):
                raise ValueError(f"Unsupported input format offset: {item},",
                                 f"please check the length of your offset list and the pipeline number")
            else:
                tmp_offset.append(item)
        offset = tmp_offset
        rounds = len(offset)
    elif not (isinstance(offset, list) and len(offset) == pipeline_num):
        raise ValueError(f"Unsupported input format offset: {offset},",
                         "please check the length of your offset list and the pipeline number")

    return offset, rounds


def process_rec_config(
        layer_per_stage: int, pipeline_num: int, offset: list[int], rec_config
):
    """process recomputation config into a dict"""
    if rec_config is None or offset is None:
        return None
    if isinstance(rec_config, bool):
        if rec_config:
            rec_config = [layer_per_stage] * pipeline_num
            rec_config = [recom + bias for recom, bias in (rec_config, offset)]
        else:
            rec_config = [0] * pipeline_num
        rec_config = [rec_config]
    elif isinstance(rec_config, list) and len(rec_config) == pipeline_num:
        # in order to be compatible with internal_from_yaml, change list into double list
        rec_config = [rec_config]
    else:
        raise ValueError(f"Unsupported input format recompute: {rec_config}, please check the length of list")

    return rec_config


def instantiate_stage(stage_id, pipeline_num, nb_layer, layer_per_recompute, memory):
    """instantiate a stage"""
    stage = Stage(
        sid=stage_id,
        nb_stage=pipeline_num,
        nb_layer=nb_layer,
        nb_layer_rec={
            Recompute.TYPE.COMM: layer_per_recompute[Recompute.TYPE.COMM][0][stage_id],
            Recompute.TYPE.FULL: layer_per_recompute[Recompute.TYPE.FULL][0][stage_id],
            Recompute.TYPE.SLCT: layer_per_recompute[Recompute.TYPE.SLCT][0][stage_id],
            Recompute.TYPE.BOTH: layer_per_recompute[Recompute.TYPE.BOTH][0][stage_id],
        },
        memory_usage=memory,
    )
    return stage


def memory_parser(file_name: str):
    """parse input given by yaml"""
    if file_name is None:
        logger.error("input file cannot be none")
        raise ValueError("input file cannot be none")
    if not file_name.endswith("yaml") and not file_name.endswith("yml"):
        logger.error("Only accept yaml as input format")
        raise ValueError(f"Only accept yaml as input format. not {file_name}")

    filepath = os.path.realpath(file_name)
    with open(filepath, encoding="utf-8") as fp:
        cfg_dict = yaml.safe_load(fp)

    # get pipeline config
    pipeline_num = cfg_dict["pipeline_config"]["pipeline_num"]
    num_layer = cfg_dict["pipeline_config"]["num_layer"]
    offset = cfg_dict["pipeline_config"]["offset"]

    offset, rounds = process_offset(offset, pipeline_num)

    layer_per_stage = int(num_layer / pipeline_num)

    # get recompute config
    if rounds > 1:
        layer_per_recompute = []
        for i in range(rounds):
            rec_config = {}
            for rec in Recompute.YAML_NAME.values():
                rec_list = cfg_dict["recompute_config"].get(rec)
                if rec_list is None:
                    continue
                rec_config[rec] = process_rec_config(layer_per_stage, pipeline_num, offset[i], rec_list[i])
            rec_config[Recompute.OFFSET] = [offset[i]]
            layer_per_recompute.append(
                Recompute.internal_from_yaml(1, pipeline_num, rec_config, [[layer_per_stage] * pipeline_num])
            )
    else:
        rec_config = {}
        for rec in Recompute.YAML_NAME.values():
            rec_list = cfg_dict["recompute_config"].get(rec)
            rec_config[rec] = process_rec_config(layer_per_stage, pipeline_num, offset, rec_list)
        rec_config[Recompute.OFFSET] = [offset]
        layer_per_recompute = Recompute.internal_from_yaml(1, pipeline_num, rec_config,
                                                           [[layer_per_stage] * pipeline_num])
    # get memory usage
    stage_id = cfg_dict["memory_usage"]["body_memories"]["stage_id"]
    mem_head_stage = cfg_dict["memory_usage"]["head_memory"]
    mem_tail_stage = cfg_dict["memory_usage"]["tail_memory"]
    body_memories = cfg_dict["memory_usage"]["body_memories"]["memories"]
    stages_a = []
    if rounds > 1:
        for i in range(rounds):
            for idx, sg_id in enumerate(stage_id[i]):
                stages_a.append(
                    instantiate_stage(
                        sg_id, pipeline_num,
                        layer_per_stage + offset[i][sg_id],
                        layer_per_recompute[i], body_memories[i][idx],
                    )
                )
        stages_a.append(
            instantiate_stage(
                0, pipeline_num, layer_per_stage + offset[i][0],
                layer_per_recompute[i], mem_head_stage,
            )
        )
        stages_a.append(
            instantiate_stage(
                pipeline_num - 1, pipeline_num,
                layer_per_stage + offset[i][pipeline_num - 1],
                layer_per_recompute[i], mem_tail_stage,
            )
        )
    else:
        for idx, sg_id in enumerate(stage_id):
            stages_a.append(
                instantiate_stage(
                    sg_id, pipeline_num, layer_per_stage + offset[sg_id],
                    layer_per_recompute, body_memories[idx],
                )
            )
        stages_a.append(
            instantiate_stage(
                0, pipeline_num, layer_per_stage + offset[0],
                layer_per_recompute, mem_head_stage,
            )
        )
        stages_a.append(
            instantiate_stage(
                pipeline_num - 1, pipeline_num, layer_per_stage + offset[pipeline_num - 1],
                layer_per_recompute, mem_tail_stage,
            )
        )

    return pipeline_num, stages_a, num_layer


def initialize_layer_json(model_name: str, file_name: str):
    """initialize layer description json file"""
    num_stage, stages_a, num_layer = memory_parser(file_name)
    head_time, body_time, tail_time = time_parser(file_name)
    comp_mem = ComputeMemory(number_of_stage=num_stage, stages_A=stages_a)

    mem_act = {}
    print("[INFO] memory_parameter  =", int(comp_mem.get_memory_parameter()))
    if comp_mem.get_memory_const() is not None:
        print("[INFO] memory_const       =", int(comp_mem.get_memory_const()))
    for r in Recompute.TYPE:
        if comp_mem.recompute_considered_[r]:
            mem_act[r] = int(comp_mem.get_memory_activation(r))
            print(
                "[INFO]" + Recompute.JSON_MEMORY_NAME[r],
                "=",
                int(comp_mem.get_memory_activation(r)),
            )
    print("[INFO] memory_tail       =", int(comp_mem.get_memory_tail()))
    print("[INFO] memory_head       =", int(comp_mem.get_memory_head()))

    mi = ModelInfo(model_name, head_time, body_time, tail_time, num_layer)
    mi.memory_update(
        mem_act,
        int(comp_mem.get_memory_parameter()),
        int(comp_mem.get_memory_head()),
        int(comp_mem.get_memory_tail()),
    )
    mi.dump_json(os.path.join("./layers", model_name + ".json"))
