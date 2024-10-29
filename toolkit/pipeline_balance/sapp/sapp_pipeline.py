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
"""sapp pipeline"""

import sys
from mindformers.tools.logger import logger
import toolkit.pipeline_balance.simulator.simulator as sim
import toolkit.pipeline_balance.simulator.pp_simulator as sim_lm
from toolkit.pipeline_balance.sapp.sapp_solver import SappSolver
from toolkit.pipeline_balance.utils.layer import Layer, filter_layer_type
import toolkit.pipeline_balance.utils.recompute as Recompute


class SappPipeline:
    """pipeline balancer"""

    def __init__(self, model_name: str, num_of_stage: int, num_of_micro_batch: int, max_memory: int,
                 layers: list[Layer], vpp_less_memory: bool = False, num_of_interleave: int = 1):
        self.model_name_ = model_name
        self.num_of_stage_ = num_of_stage
        self.num_of_micro_batch_ = num_of_micro_batch
        self.num_of_interleave_ = num_of_interleave
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        self.problem_ = None
        self.layers_ = layers
        self.layers_sorted_ = {
            Layer.type_enum.HEAD: filter_layer_type(layers, Layer.type_enum.HEAD),
            Layer.type_enum.BODY: filter_layer_type(layers, Layer.type_enum.BODY),
            Layer.type_enum.TAIL: filter_layer_type(layers, Layer.type_enum.TAIL),
        }

    def has_some_memory_info(self) -> bool:
        """Check if there is all information for memory constraint."""
        return self.problem_.has_some_memory_info()

    def construct_problem(self, solver: str = "pulp"):
        """Construct the problem to solve, chose the solver"""
        if solver == "pulp":
            self.problem_ = self._construct_problem_pulp_()
        elif solver == "other":
            logger.warning("No other solver available..., automatically switch to pulp!!!")
            self.problem_ = self._construct_problem_pulp_()
        else:
            logger.warning("No other solver available..., automatically switch to pulp!!!")
            self.problem_ = self._construct_problem_pulp_()

    def solve_problem(self, time_limit=90, dump_folder=None):
        """ "Solve the problem to have the schedule pipeline"""
        self.problem_.solve(time_limit, dump_folder)

    def get_result(self) -> dict[str, list[list[str]]]:
        """Get result distribution of the solution (compact form)"""
        return self.problem_.result()

    def get_memory_activation(self) -> list[float]:
        """Give the activation memory per stage for simulator."""
        return self.problem_.get_simulator_memory_activation()

    def get_memory_parameter(self) -> list[float]:
        """Give the parameter memory per stage for simulator."""
        return self.problem_.get_simulator_memory_parameter()

    def get_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        return self.problem_.get_simulator_time()

    def naive_layer_per_stage(self, layer_num: int, num_of_interleave=1) -> list[list[int]]:
        """Give the even layer to stage assignment of LLM"""
        flat_lyr_per_stg = [0] * self.num_of_stage_ * num_of_interleave
        pp_dis = max(int((layer_num + 1) / (self.num_of_stage_ * num_of_interleave)), 1)
        for a in range(layer_num):
            pp_id = min(a // pp_dis, (self.num_of_stage_ * num_of_interleave) - 1)
            flat_lyr_per_stg[pp_id] += 1
        lyr_per_stg = []
        for i in range(num_of_interleave):
            lyr_per_stg.append([])
            for s in range(self.num_of_stage_):
                lyr_per_stg[i].append(flat_lyr_per_stg[i * self.num_of_stage_ + s])
        return lyr_per_stg

    def print_yaml_results(self):
        """Print the results"""
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            semi_layer_per_stage = self.naive_layer_per_stage(layer.nb_layer_,
                                                              self.num_of_interleave_)
            offset = []
            recomputes = {r: [] for r in Recompute.TYPE}
            layer_name = layer.name_
            for inter in range(self.num_of_interleave_):
                offset.append([])
                for r in Recompute.TYPE:
                    recomputes[r].append([])
                for stage in range(self.num_of_stage_):
                    sum_layer = 0
                    for r in Recompute.TYPE:
                        if self._recompute_considered()[r]:
                            sum_layer += self.problem_.variables_.get(layer.name_)[r][inter][
                                stage].varValue
                    offset[inter].append(int(sum_layer - semi_layer_per_stage[inter][stage]))

                    if self._recompute_considered()[Recompute.TYPE.FULL]:
                        recomputes[Recompute.TYPE.FULL][inter].append(int(
                            self.problem_.variables_.get(layer_name)[Recompute.TYPE.FULL][inter][
                                stage].varValue))
                    for r in Recompute.TYPE:
                        if (r is not Recompute.TYPE.FULL and r is not Recompute.TYPE.NONE and
                                self._recompute_considered()[r]):
                            recomputes[r][inter].append(
                                int(self.problem_.variables_.get(layer_name)[r][inter][stage].varValue))
                            stage_value = (int(
                                self.problem_.variables_.get(layer_name)[Recompute.TYPE.FULL][inter][
                                    stage].varValue) if self._recompute_considered()[
                                        Recompute.TYPE.FULL] else 0)
                            recomputes[r][inter][stage] += stage_value

        logger.info("layer-to-stage assignment baseline is %s", semi_layer_per_stage)

        yaml_results = "\nTo put in yaml configuration:"
        if self.num_of_interleave_ == 1:
            offset = flatten(offset)
        yaml_results += f"\n\toffset: {offset}"
        for r in Recompute.TYPE:
            if self._recompute_considered()[r] and r is not Recompute.TYPE.NONE:
                recompute_layers = recomputes[r]
                if self.num_of_interleave_ == 1:
                    recompute_layers = flatten(recompute_layers)
                yaml_results += f"\n\t{Recompute.YAML_NAME[r]}: {recompute_layers}"
        yaml_results += f"\n\tpp_interleave_num: {self.num_of_interleave_}"
        logger.info(yaml_results)

    def get_naive_memory_activation(self, all_recompute=False, interleave_num=1) -> list[float]:
        """
        Give the activation memory per stage for an even layer assignment without
        interleave for simulator.
        """
        memory_active = []
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if all_recompute:
                rec = Recompute.most_recomputed(layer.recompute_considered_)
            else:
                rec = Recompute.least_recomputed(layer.recompute_considered_)
            lyr_per_stg = self.naive_layer_per_stage(layer.nb_layer_, interleave_num)
            if self.has_some_memory_info():
                for inter in range(interleave_num):
                    memory_active.append([])
                    for stage in range(self.num_of_stage_):
                        memory_active[inter].append(
                            lyr_per_stg[inter][stage] * layer.memory_activation_rec_[rec])
        return memory_active

    def get_naive_memory_parameter(self, interleave_num=1) -> list[float]:
        """
        Give the parameter memory per stage for a naive layer assignment
        without interleave for simulator.
        """
        memory_param = []
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            lyr_per_stg = self.naive_layer_per_stage(layer.nb_layer_, interleave_num)
            if layer.memory_parameter_ is not None:
                for inter in range(interleave_num):
                    memory_param.append([])
                    for stage in range(self.num_of_stage_):
                        memory_param[inter].append(
                            lyr_per_stg[inter][stage] * layer.memory_parameter_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param[0][0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param[interleave_num - 1][self.num_of_stage_ - 1] += tail.memory_parameter_
        return memory_param

    def get_naive_time(self, all_recompute=False, interleave_num=1) -> list[float]:
        """
        Give the time per stage for a naive layer assignment
        without interleave for simulator.
        """
        time = []
        rec = Recompute.TYPE.NONE
        for i in range(interleave_num):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    if all_recompute:
                        rec = Recompute.most_recomputed(layer.recompute_considered_)
                    else:
                        rec = Recompute.least_recomputed(layer.recompute_considered_)
                    lyr_per_stg = self.naive_layer_per_stage(layer.nb_layer_, interleave_num)
                    if not all_recompute:
                        time[i][s] += lyr_per_stg[i][s] * layer.time_
                    else:
                        time[i][s] += lyr_per_stg[i][s] * (
                            layer.forward_time_ + layer.backward_time_rec_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[interleave_num - 1][self.num_of_stage_ - 1] += tail.time_
        if all_recompute:
            logger.info("even partitioning computed with %s", Recompute.YAML_NAME[rec])
        return time

    def simulate(self, show=True, file_name=None):
        """Use simulator to visualize output."""
        if self.has_some_memory_info():
            stage_mem_par = self.get_memory_parameter()
            stage_mem_act = self.get_memory_activation()
            if self.vpp_less_memory_:
                simulator = sim_lm.PipelineSimulator(self.get_time(), self.num_of_micro_batch_,
                                                     block_mem_act=stage_mem_act,
                                                     block_mem_par=stage_mem_par)
            else:
                simulator = sim.PipelineSimulator(self.get_time(), self.num_of_micro_batch_,
                                                  stage_mem_act=stage_mem_act,
                                                  stage_mem_par=stage_mem_par)
        else:
            if self.vpp_less_memory_:
                simulator = sim_lm.PipelineSimulator(self.get_time(), self.num_of_micro_batch_)
            else:
                simulator = sim.PipelineSimulator(self.get_time(), self.num_of_micro_batch_)

        if self.vpp_less_memory_:
            simulator.run(comm=False)
        else:
            simulator.run()
        if show:
            simulator.show(file_name)
        return simulator.end_time

    def simulate_naive(self, show=True, all_recompute=False, interleave_num=1, file_name=None):
        """Use simulator to visualize output."""
        if self.has_some_memory_info():
            stage_mem_par = self.get_memory_parameter()
            stage_mem_act = self.get_memory_activation()
            if self.vpp_less_memory_:
                simulator = sim_lm.PipelineSimulator(
                    block_time=self.get_naive_time(all_recompute, interleave_num),
                    micro_num=self.num_of_micro_batch_, block_mem_act=stage_mem_act,
                    block_mem_par=stage_mem_par)
            else:
                simulator = sim.PipelineSimulator(
                    self.get_naive_time(all_recompute, interleave_num), self.num_of_micro_batch_,
                    stage_mem_act=stage_mem_act, stage_mem_par=stage_mem_par)
        else:
            if self.vpp_less_memory_:
                simulator = sim_lm.PipelineSimulator(self.get_naive_time(False),
                                                     self.num_of_micro_batch_)
            else:
                simulator = sim.PipelineSimulator(self.get_naive_time(False),
                                                  self.num_of_micro_batch_)
        if self.vpp_less_memory_:
            simulator.run(comm=False)
        else:
            simulator.run()
        if show:
            simulator.show(file_name)
        return simulator.end_time

    def _construct_problem_pulp_(self) -> SappSolver:
        """construct the problem using pulp"""
        prob = SappSolver(num_of_stage=self.num_of_stage_,
                          num_of_micro_batch=self.num_of_micro_batch_,
                          num_of_interleave=self.num_of_interleave_, max_memory=self.max_memory_,
                          vpp_less_memory=self.vpp_less_memory_, layers=self.layers_,
                          layers_sorted=self.layers_sorted_)
        return prob

    def _recompute_considered(self):
        return self.problem_.recompute_considered_


def choose_interleave(model_name: str, number_of_stage: int, number_of_micro_batch: int,
                      max_memory: int, layers: list[Layer]) -> tuple[
                          int, int, dict[str, list[list[str]]]]:
    """Simulates different interleaves and returns the best."""
    max_inter = 4
    best_time = int(sys.maxsize)
    best_inter = 1
    best_distribution = {}

    for inter in range(1, max_inter + 1):
        pipe = SappPipeline(model_name=model_name, num_of_stage=number_of_stage,
                            num_of_micro_batch=number_of_micro_batch, max_memory=max_memory,
                            layers=layers, num_of_interleave=inter)

        pipe.construct_problem(solver="pulp")
        pipe.solve_problem()
        time = pipe.simulate(show=False)
        logger.info("[INFO] for interleave %d, time = %d", inter, time)
        if time < best_time:
            best_time = time
            best_inter = inter
            best_distribution = pipe.get_result()

    return (best_inter, best_time, best_distribution)


def flatten(inter_stage_list):
    """Flatten an interleave x stage list to a stage list"""
    stage_list = [0] * len(inter_stage_list[0])
    for inter, _ in enumerate(inter_stage_list):
        for stage, _ in enumerate(inter_stage_list[inter]):
            stage_list[stage] += inter_stage_list[inter][stage]
    return stage_list
