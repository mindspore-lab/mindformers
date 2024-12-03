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
"""Solver Class"""

import os
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

import pulp as lpSolver
from pulp import PULP_CBC_CMD

import toolkit.pipeline_balance.utils.recompute as Recompute
from toolkit.pipeline_balance.utils.layer import Layer
from toolkit.pipeline_balance.utils.logger import logger


@dataclass
class PipelineMemoryConstraint:
    prob: Any
    variables: Any
    layers_sorted: dict[Any]
    num_of_stage: int
    num_of_interleave: int
    micro_batch: int
    memory_limit: int


class SappSolver:
    """solver for pipeline balance"""

    MEM_OVERHEAD_NAME = "memory_overhead"
    TOTAL_SUM = "var_sum_FPi_BPi"
    WINGS_SUM = "wings_sum"
    PREV_DIFF = "prev_diff"
    NEXT_DIFF = "next_diff"
    MAX_STAGE_TIME = "max_stage_time"
    MAX_LAST_CHUNK = "max_last_chunk"

    PROP_PHASE = IntEnum("Propagation", ["FW", "BW"], start=0)

    def __init__(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            num_of_micro_batch: int,
            max_memory: int,
            layers: list[Layer],
            layers_sorted: dict[Layer.type_enum, list[Layer]],
            vpp_less_memory: bool = False,
            constant_memory: int = 0,
            optimization_level: int = 1,
            description: str = "Pipeline_execution_time_minimize",
    ):

        self.num_of_stage_ = num_of_stage
        self.num_of_interleave_ = num_of_interleave
        self.num_of_micro_batch_ = num_of_micro_batch
        self.max_memory_ = max_memory
        self.vpp_less_memory_ = vpp_less_memory
        self.constant_memory_ = constant_memory
        self.optimization_level_ = optimization_level
        self.layers_ = layers
        self.layers_sorted_ = layers_sorted

        self.recompute_considered_ = self.find_recompute_considered(
            layers_sorted)
        self.variables_ = self._create_variables_to_solve_(
            num_of_stage, num_of_interleave, layers_sorted)
        self.problem_ = self._create_problem_(description)

    @staticmethod
    def compute_forward_in_backward(num_of_stage: int,
                                    micro_batch: int) -> list[int]:
        """Computes the number of forward propagation happening after a backward"""
        n = num_of_stage - 1
        factors = []
        for _ in range(num_of_stage):
            factors.append(abs(n))
            n -= 2
        if micro_batch < 2 * num_of_stage:
            for i in range(num_of_stage // 2):
                factors[i] = 0
        return factors

    @staticmethod
    def compute_lm_forward_in_backward(num_of_stage: int) -> list[int]:
        """Function compute_forward_in_backward in less_memory schedule"""
        return [s for s in range(num_of_stage)]

    @staticmethod
    def compute_activation_nums(num_of_stage: int, num_of_interleave: int,
                                micro_batch: int) -> list[list[int]]:
        """compute the number of activation"""
        activation_nums = []
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[0][s] += max(0, num_of_stage - 2 * s - 1)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] += min(
                    0, num_of_stage - 2 * s - 1)
            for i in range(num_of_interleave):
                for s in range(num_of_stage):
                    activation_nums[i][s] = min(activation_nums[i][s],
                                                micro_batch)
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)

        return activation_nums

    @staticmethod
    def compute_less_activation_nums(
            num_of_stage: int, num_of_interleave: int) -> list[list[int]]:
        """compute number of less_mem activation"""
        activation_nums = []
        if num_of_interleave > 1:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for _ in range(num_of_stage):
                    activation_nums[i].append(num_of_stage)
            for s in range(num_of_stage):
                activation_nums[num_of_interleave - 1][s] -= s
        else:
            for i in range(num_of_interleave):
                activation_nums.append([])
                for s in range(num_of_stage):
                    activation_nums[i].append(num_of_stage - s)
        return activation_nums

    def add_total_nb_layer_constraint(self, prob, variables, sorted_layers):
        """Constraints to respect the total number of layer"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            prob += (lpSolver.lpSum(
                variables[layer.name_][rec] for rec in Recompute.TYPE
                if self.recompute_considered_[rec]) == layer.nb_layer_)
        return prob

    def add_stage_nb_layer_constraint(self, prob, variables, sorted_layers):
        """Constraints to respect the total number of layer"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            for i in range(self.num_of_interleave_):
                for s in range(self.num_of_stage_):
                    prob += (lpSolver.lpSum(variables[layer.name_][rec][i][s]
                                            for rec in Recompute.TYPE
                                            if self.recompute_considered_[rec])
                             >= 1)
        return prob

    @staticmethod
    def find_recompute_considered(layers_sorted):
        """Find the recomputation types considered"""
        return layers_sorted[Layer.type_enum.BODY][0].recompute_considered_

    def max_stage_micro_eq_stage(self, prob, variables, layers_sorted,
                                 pipeline_total_time):
        """apply optimizations on vpp when pp=#mb"""
        last_chunk = self.num_of_interleave_ - 1

        for i_stage in range(self.num_of_stage_):
            for inter in range(last_chunk):
                prob += self.variables_[self.MAX_STAGE_TIME] >= (
                    self._max_stage_bound_i_bp(variables, layers_sorted,
                                               i_stage, inter) +
                    self._max_stage_bound_head_tail(layers_sorted, i_stage,
                                                    inter, inter))

        if self.vpp_less_memory_:
            factors = self.compute_lm_forward_in_backward(self.num_of_stage_)
        else:
            factors = self.compute_forward_in_backward(
                self.num_of_stage_, self.num_of_micro_batch_)

        for i_stage in range(self.num_of_stage_):
            logger.debug(
                f"v={last_chunk}, s={i_stage}: (BP + HT) + "
                f"({factors[i_stage]} / {self.num_of_micro_batch_} * FP")
            prob += self.variables_[self.MAX_LAST_CHUNK] >= (
                self._max_stage_bound_i_bp(variables, layers_sorted, i_stage,
                                           last_chunk) +
                self._max_stage_bound_head_tail(layers_sorted, i_stage, last_chunk, last_chunk) +
                (factors[i_stage] / self.num_of_micro_batch_) *
                self._max_stage_bound_i_fp(variables, layers_sorted, i_stage,
                                           last_chunk))

        if self.optimization_level_ >= 2:
            prob += (self.variables_[self.MAX_STAGE_TIME]
                     >= self.variables_[self.MAX_LAST_CHUNK])

            for v in range(self.num_of_interleave_ - 1):
                prob += self.variables_[self.PREV_DIFF][v] >= (
                    self._prev_diff_sum(variables, layers_sorted, prob, v))
                prob += self.variables_[self.WINGS_SUM][v] >= (
                    (self.num_of_interleave_ - v) / self.num_of_interleave_ *
                    self._wings_sum(variables, layers_sorted, v))

            prob += self.variables_[self.TOTAL_SUM] >= (self._total_sum(
                variables, layers_sorted))

            prob += self.variables_[self.NEXT_DIFF] >= (self._next_diff_sum(
                variables, layers_sorted, prob))

            prob += pipeline_total_time >= (
                (self.variables_[self.TOTAL_SUM] +
                 lpSolver.lpSum(self.variables_[self.WINGS_SUM]) +
                 lpSolver.lpSum(self.variables_[self.PREV_DIFF]) +
                 self.variables_[self.NEXT_DIFF]) /
                max(1, (self.num_of_interleave_ - 2)) +
                self.variables_[self.MAX_STAGE_TIME] *
                (self.num_of_micro_batch_ - 2))
        else:
            logger.info("Approach 2b")
            prob += (self.variables_[self.MAX_LAST_CHUNK]
                     >= self.variables_[self.MAX_STAGE_TIME])

            prob += pipeline_total_time >= (
                self.variables_[self.MAX_STAGE_TIME] +
                self.variables_[self.MAX_LAST_CHUNK])

        return prob

    def add_max_stage_constraint(self, prob, variables, layers_sorted,
                                 num_of_stage, pipeline_total_time):
        """add constrains based on max stage time"""
        if (self.num_of_interleave_ > 1 and self.optimization_level_ >= 1
                and self.num_of_micro_batch_ == self.num_of_stage_):
            prob = self.max_stage_micro_eq_stage(prob, self.variables_,
                                                 layers_sorted,
                                                 pipeline_total_time)
        else:
            # Constraints on sub-main-part of a stage that it may take (for all stage)
            max_stage_time = lpSolver.LpVariable(self.MAX_STAGE_TIME, 0, None,
                                                 lpSolver.LpContinuous)

            for i_stage in range(num_of_stage):
                for inter_f in range(self.num_of_interleave_):
                    for inter_b in range(self.num_of_interleave_):
                        prob += max_stage_time >= self._max_stage_bound_i_fp(
                            variables, layers_sorted, i_stage,
                            inter_f) + self._max_stage_bound_i_bp(
                                variables, layers_sorted, i_stage,
                                inter_b) + self._max_stage_bound_head_tail(
                                    layers_sorted, i_stage, inter_f, inter_b)

            var_sum_f_pi_b_pi = lpSolver.LpVariable(self.TOTAL_SUM, 0, None,
                                                    lpSolver.LpContinuous)

            prob += var_sum_f_pi_b_pi >= self._total_sum(
                variables, layers_sorted)

            prob += pipeline_total_time >= var_sum_f_pi_b_pi + max_stage_time * (
                self.num_of_micro_batch_ - 2)

        return prob

    ############################################
    #            Memory Constraint             #
    ############################################

    def stage_param_memory(self, variables, layers_sorted, stage_id,
                           num_of_stage, num_of_interleave):
        """memory constraint for parameters"""
        bound = lpSolver.LpAffineExpression()
        for inter_id in range(num_of_interleave):
            for layer in layers_sorted[Layer.type_enum.BODY]:
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        bound += (
                            variables[layer.name_][rec][inter_id][stage_id] *
                            layer.memory_parameter_)
        if stage_id == 0:
            for head in layers_sorted[Layer.type_enum.HEAD]:
                bound += head.memory_parameter_
        if stage_id == num_of_stage - 1:
            for tail in layers_sorted[Layer.type_enum.TAIL]:
                bound += tail.memory_parameter_
        return bound

    def stage_active_memory_per_micro(self, variables, layers_sorted, stage_id,
                                      inter_id):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (variables[layer.name_][rec][inter_id][stage_id] *
                              layer.memory_activation_rec_[rec])
        return bound

    def stage_active_memory(self, variables, layers_sorted, stage_id,
                            num_of_interleave, activation_nums):
        """calculate active memory of a stage"""
        bound = lpSolver.LpAffineExpression()
        for inter_id in range(num_of_interleave):
            for layer in layers_sorted[Layer.type_enum.BODY]:
                for rec in Recompute.TYPE:
                    if self.recompute_considered_[rec]:
                        bound += (
                            variables[layer.name_][rec][inter_id][stage_id] *
                            layer.memory_activation_rec_[rec] *
                            activation_nums[inter_id][stage_id])
        return bound

    def init_overhead_variables(self, prob, variables):
        for s in range(self.num_of_stage_):
            for v in range(self.num_of_interleave_ - 1):
                prob += variables[self.MEM_OVERHEAD_NAME][s][v] >= (
                    self.stage_active_memory_per_micro(
                        variables, self.layers_sorted_, s, v + 1) -
                    self.stage_active_memory_per_micro(
                        variables, self.layers_sorted_, s, v))
        return prob

    def stage_overhead_memory(self, variables, stage_id):
        bound = lpSolver.LpAffineExpression()
        for v in range(self.num_of_interleave_ - 1):
            bound += variables[self.MEM_OVERHEAD_NAME][stage_id][v]
        return bound

    def add_pipeline_memory_constraint(self,
                                       constraint: PipelineMemoryConstraint):
        """add constrain on the memory of pipeline"""
        prob = constraint.prob
        variables = constraint.variables
        layers_sorted = constraint.layers_sorted
        num_of_stage = constraint.num_of_stage
        num_of_interleave = constraint.num_of_interleave
        micro_batch = constraint.micro_batch
        memory_limit = constraint.memory_limit

        if self.vpp_less_memory_:
            activation_nums = self.compute_less_activation_nums(
                num_of_stage, num_of_interleave)
            overhead_factors = self.compute_lm_forward_in_backward(
                num_of_stage)
        else:
            activation_nums = self.compute_activation_nums(
                num_of_stage, num_of_interleave, micro_batch)
            overhead_factors = self.compute_forward_in_backward(
                num_of_stage, micro_batch)

        # prob = self.init_overhead_variables(prob, variables)

        logger.output("activation nums = %s", activation_nums)
        logger.output("overhead factors = %s", overhead_factors)

        for s in range(num_of_stage):
            prob += memory_limit >= (
                self.stage_param_memory(variables, layers_sorted, s,
                                        num_of_stage, num_of_interleave) +
                self.stage_active_memory(variables, layers_sorted, s,
                                         num_of_interleave, activation_nums) +
                # self.stage_overhead_memory(variables, s) * overhead_factors[s] +
                self.constant_memory_)

    def get_simulator_memory_activation(self) -> list[float]:
        """Give the activation memory per stage for simulator."""
        memory_active = []
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if self.has_some_memory_info():
                for inter in range(self.num_of_interleave_):
                    memory_active.append([])
                    for stage in range(self.num_of_stage_):
                        memory_active[inter].append(0)
                        memory_active[inter][stage] = sum(
                            self.variables_.get(
                                layer.name_)[rec][inter][stage].varValue *
                            layer.memory_activation_rec_[rec]
                            for rec in Recompute.TYPE
                            if self.recompute_considered_[rec])
        return memory_active

    def get_simulator_memory_parameter(self) -> list[float]:
        """Give the parameter memory per stage for simulator."""
        memory_param_stage = [0] * self.num_of_stage_
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            if self.has_some_memory_info():
                for inter in range(self.num_of_interleave_):
                    for stage in range(self.num_of_stage_):
                        memory_param_stage[stage] += sum(
                            self.variables_.get(layer.name_)[rec][inter]
                            [stage].varValue * layer.memory_parameter_
                            for rec in Recompute.TYPE
                            if self.recompute_considered_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            if head.memory_parameter_ is not None:
                memory_param_stage[0] += head.memory_parameter_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            if tail.memory_parameter_ is not None:
                memory_param_stage[self.num_of_stage_ -
                                   1] += tail.memory_parameter_
        memory_param = [memory_param_stage] * self.num_of_interleave_
        return memory_param

    def get_simulator_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_.get(
                                layer.name_)[rec][i][s].varValue * (
                                    layer.forward_time_ +
                                    layer.backward_time_rec_[rec])

        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_
        return time

    def get_simulator_forward_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time = []
        for i in range(self.num_of_interleave_):
            time.append([])
            for s in range(self.num_of_stage_):
                time[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time[i][s] += self.variables_[layer.name_][rec][i][
                                s].varValue * (layer.forward_time_)
        for head in self.layers_sorted_[Layer.type_enum.HEAD]:
            time[0][0] += head.time_ / 3
        for tail in self.layers_sorted_[Layer.type_enum.TAIL]:
            time[self.num_of_interleave_ - 1][self.num_of_stage_ -
                                              1] += tail.time_ / 3
        return time

    def get_simulator_recompute_time(self) -> list[float]:
        """Give the time per stage for simulator."""
        time_all_rec = []
        time_no_rec = []
        for i in range(self.num_of_interleave_):
            time_all_rec.append([])
            time_no_rec.append([])
            for s in range(self.num_of_stage_):
                time_all_rec[i].append(0)
                time_no_rec[i].append(0)
                for layer in self.layers_sorted_[Layer.type_enum.BODY]:
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            time_all_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[rec])
                            time_no_rec[i][s] += self.variables_[
                                layer.name_][rec][i][s].varValue * (
                                    layer.backward_time_rec_[
                                        Recompute.TYPE.NONE])
        return [[r - n for r, n in zip(ar, nr)]
                for ar, nr in zip(time_all_rec, time_no_rec)]

    def has_some_memory_info(self) -> bool:
        """Check if there is some information for memory constraint."""
        some_info = False
        for rec in Recompute.TYPE:
            if self.recompute_considered_[rec]:
                some_info = True
        return some_info

    ############################################
    #            General Constraint            #
    ############################################
    def add_optional_recompute_constraint(self, prob, variables,
                                          sorted_layers):
        """Constraints to put unused recompute variables at 0"""
        for layer in sorted_layers[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if not self.recompute_considered_[rec]:
                    prob += lpSolver.lpSum(variables[layer.name_][rec]) == 0

    def dump_problem(self, folder=None):
        """
        dump pulp problem inside folder
        """
        dump_name = "problem_" + str(self.layers_[0].model_name_)
        dump_name += "_" + str(self.max_memory_)
        dump_name += "_" + str(self.num_of_interleave_)
        dump_name += "_" + str(self.num_of_stage_)

        logger.info("dump_problem:out folder = %s", folder)
        if folder is not None:
            dump_name = os.path.join(folder, dump_name)
        dump_name += ".lp"
        logger.info("dump problem file: %s", dump_name)
        self.problem_.writeLP(dump_name)

    def print_results(self):
        """Print the detailed results"""
        if self.has_some_memory_info():
            print("For max memory ", self.max_memory_)
            print("==============")
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = layer.name_
            print("For layer:", layer_name)
            print("=========")
            print("  Forward Prop time: ", layer.forward_time_)
            for rec in Recompute.TYPE:
                if layer.recompute_considered_[rec]:
                    print(
                        "  Backward Prop",
                        Recompute.YAML_NAME[rec],
                        "time:",
                        layer.backward_time_rec_[rec],
                    )
            for inter in range(self.num_of_interleave_):
                for stage in range(self.num_of_stage_):
                    print("    Assign", layer_name, end=": ")
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            value = str(int(self.variables_[layer_name][rec][inter][stage].varValue))
                            print(value if rec is Recompute.TYPE.NONE else f"+ {value} {rec.name}", end=" ")
                    print(
                        (" of chunk " +
                         str(inter) if self.num_of_interleave_ != 1 else ""),
                        " to stage " + str(stage),
                    )
        for s in range(self.num_of_stage_):
            for i in range(self.num_of_interleave_ - 1):
                logger.debug(
                    f"{self.MEM_OVERHEAD_NAME}[{s}][{i}] ="
                    f"{self.variables_[self.MEM_OVERHEAD_NAME][s][i].varValue}"
                )

        # Memory overhead currently disabled
        # if self.vpp_less_memory_:
        #     factors = self.compute_lm_forward_in_backward(
        #         self.num_of_stage_)
        # else:
        #     factors = self.compute_forward_in_backward(
        #         self.num_of_stage_, self.num_of_micro_batch_)

        # for s in range(self.num_of_stage_):
        #     overhead = 0
        #     for i in range(self.num_of_interleave_ - 1):
        #          overhead += self.variables_[self.MEM_OVERHEAD_NAME][s][i].varValue
        #     logger.debug(f"overhead for stage {s} is estimated at {factors[s] * overhead}")

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(
                f"{self.WINGS_SUM}[{v}] = {self.variables_[self.WINGS_SUM][v].varValue}"
            )

        for v in range(self.num_of_interleave_ - 1):
            logger.debug(
                f"{self.PREV_DIFF}[{v}] = {self.variables_[self.PREV_DIFF][v].varValue}"
            )

        logger.debug(
            f"{self.NEXT_DIFF} = {self.variables_[self.NEXT_DIFF].varValue}")
        logger.debug(
            f"{self.TOTAL_SUM} = {self.variables_[self.TOTAL_SUM].varValue}")
        logger.debug(
            f"{self.MAX_STAGE_TIME} = {self.variables_[self.MAX_STAGE_TIME].varValue}"
        )
        logger.debug(
            f"{self.MAX_LAST_CHUNK} = {self.variables_[self.MAX_LAST_CHUNK].varValue}"
        )

    def solve(self, time_limit=90, dump_folder=None):
        """Solve the problem and print the results"""
        logger.info("solve:out folder = %s", dump_folder)
        self.dump_problem(dump_folder)
        solver = lpSolver.getSolver(PULP_CBC_CMD, timeLimit=time_limit)
        self.problem_.solve(solver)

        self.print_results()
        for name, result in self.result().items():
            logger.output("%s %s %s", name, result, "\n")

    def result(self) -> dict[str, list[list[str]]]:
        """return schedule distribution for each layer (in the form of a dict)"""
        r = {}
        for layer in self.layers_sorted_[Layer.type_enum.BODY]:
            layer_name = layer.name_
            inter = []
            for i in range(self.num_of_interleave_):
                stage = []
                for s in range(self.num_of_stage_):
                    for rec in Recompute.TYPE:
                        if self.recompute_considered_[rec]:
                            stage.append(
                                str(
                                    self.variables_.get(layer_name)[rec][i]
                                    [s].varValue) + " + ")
                inter.append(stage)
            r[layer_name] = inter
        return r

    def _create_problem_(self, description: str) -> lpSolver.LpProblem:
        """create the problem"""
        prob = lpSolver.LpProblem(description, lpSolver.LpMinimize)
        layers_sorted = self.layers_sorted_
        num_of_stage = self.num_of_stage_
        num_of_interleave = self.num_of_interleave_
        num_of_micro_batch = self.num_of_micro_batch_
        max_memory = self.max_memory_
        # Local variable declaration
        # max time that a "main" stage have to take (var to minimize)
        pipeline_total_time = lpSolver.LpVariable("pipeline_total_time", 0,
                                                  None, lpSolver.LpContinuous)

        # Var to Minimize
        prob += pipeline_total_time

        self.add_total_nb_layer_constraint(prob, self.variables_,
                                           layers_sorted)
        # self.add_stage_nb_layer_constraint(prob, self.variables_, layers_sorted)
        self.add_max_stage_constraint(prob, self.variables_, layers_sorted,
                                      num_of_stage, pipeline_total_time)

        constraint = PipelineMemoryConstraint(
            prob=prob,
            variables=self.variables_,
            layers_sorted=layers_sorted,
            num_of_stage=num_of_stage,
            num_of_interleave=num_of_interleave,
            micro_batch=num_of_micro_batch,
            memory_limit=max_memory,
        )
        if self.has_some_memory_info():
            self.add_pipeline_memory_constraint(constraint)
        return prob

    def _create_variables_to_solve_(
            self,
            num_of_stage: int,
            num_of_interleave: int,
            layers: dict[Layer.type_enum, list[Layer]],
    ):
        """create variables to solve"""
        variables = {}

        variables[self.TOTAL_SUM] = lpSolver.LpVariable(
            self.TOTAL_SUM, 0, None, lpSolver.LpContinuous)

        wings_sum_dict = lpSolver.LpVariable.dicts(
            name=self.WINGS_SUM,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )
        wings_sum_list = list(wings_sum_dict.values())
        variables[self.WINGS_SUM] = wings_sum_list

        prev_diff_dict = lpSolver.LpVariable.dicts(
            name=self.PREV_DIFF,
            indices=(range(0, self.num_of_interleave_ - 1)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )
        prev_diff_list = list(prev_diff_dict.values())
        variables[self.PREV_DIFF] = prev_diff_list

        variables[self.NEXT_DIFF] = lpSolver.LpVariable(
            self.NEXT_DIFF, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_STAGE_TIME] = lpSolver.LpVariable(
            self.MAX_STAGE_TIME, 0, None, lpSolver.LpContinuous)

        variables[self.MAX_LAST_CHUNK] = lpSolver.LpVariable(
            self.MAX_LAST_CHUNK, 0, None, lpSolver.LpContinuous)

        lp_variable_dict = lpSolver.LpVariable.dicts(
            name=self.MEM_OVERHEAD_NAME,
            indices=(
                range(0, self.num_of_stage_),
                range(0, self.num_of_interleave_ - 1),
            ),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpInteger,
        )
        variables_list = [x for x in lp_variable_dict.values()]
        variables[self.MEM_OVERHEAD_NAME] = variables_list

        for layer in layers[Layer.type_enum.BODY]:
            variable_dict = lpSolver.LpVariable.dicts(
                name=layer.name_,
                indices=(
                    range(0, len(Recompute.TYPE)),
                    range(0, num_of_interleave),
                    range(0, num_of_stage),
                ),
                lowBound=0,
                upBound=None,
                cat=lpSolver.LpInteger,
            )
            variable_values = list(variable_dict.values())
            interleave_values = []
            for interleave in variable_values:
                interleave_value = [y for y in interleave.values()]
                interleave_values.append(interleave_value)
            variables[layer.name_] = interleave_values

        return variables

    ############################################
    #             Time Constraint              #
    ############################################
    def _max_stage_bound_i_fp(self, variables, layers_sorted, stage_id,
                              inter_f):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (variables[layer.name_][rec][inter_f][stage_id] *
                              layer.forward_time_)
        return bound

    def _max_stage_bound_i_bp(self, variables, layers_sorted, stage_id,
                              inter_b):
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    bound += (variables[layer.name_][rec][inter_b][stage_id] *
                              layer.backward_time_rec_[rec])
        return bound

    def _max_stage_bound_head_tail(self, layers_sorted, stage_id, inter_f,
                                   inter_b):
        """maximize the stage bound of head and tail"""
        bound = lpSolver.LpAffineExpression()
        if stage_id == 0:
            if inter_f == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ / 3
            if inter_b == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2 / 3
        if stage_id == self.num_of_stage_ - 1:
            if inter_f == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ / 3
            if inter_b == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2 / 3
        return bound

    def _total_sum(self, variables, layers_sorted):
        """sum up the layer time"""
        bound = lpSolver.LpAffineExpression()
        for layer in layers_sorted[Layer.type_enum.BODY]:
            for rec in Recompute.TYPE:
                if self.recompute_considered_[rec]:
                    for inter in range(self.num_of_interleave_):
                        for stage in range(self.num_of_stage_):
                            bound += variables[layer.name_][rec][inter][
                                stage] * (layer.forward_time_ +
                                          layer.backward_time_rec_[rec])
        return bound

    def body_layer_time(self, prop, layer, variables, inter, stage):
        """calculate body layer time"""
        if prop == self.PROP_PHASE.FW:
            bound = lpSolver.lpSum(variables[layer.name_][rec][inter][stage] * layer.forward_time_
                                   for rec in Recompute.TYPE if self.recompute_considered_[rec])
        else:
            bound = lpSolver.lpSum(variables[layer.name_][rec][inter][stage] * layer.backward_time_rec_
                                   for rec in Recompute.TYPE if self.recompute_considered_[rec])

        return bound

    def micro_batch_time(self, prop, variables, layers_sorted, inter, stage):
        """computes the time taken by a given micro batch propagation"""
        bound = lpSolver.LpAffineExpression()
        if prop == self.PROP_PHASE.FW:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, variables, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ / 3
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ / 3
        else:
            for layer in layers_sorted[Layer.type_enum.BODY]:
                bound = self.body_layer_time(prop, layer, variables, inter, stage)
            if stage == 0 and inter == 0:
                for head in layers_sorted[Layer.type_enum.HEAD]:
                    bound += head.time_ * 2 / 3
            if stage == self.num_of_stage_ - 1 and inter == self.num_of_interleave_ - 1:
                for tail in layers_sorted[Layer.type_enum.TAIL]:
                    bound += tail.time_ * 2 / 3
        return bound

    def _wings_sum(self, variables, layers_sorted, v):
        """sum up the warm-up and cool-down time of a given chunk"""
        bound = lpSolver.LpAffineExpression()
        for stage in range(self.num_of_stage_):
            bound += self.micro_batch_time(self.PROP_PHASE.FW, variables,
                                           layers_sorted, v, stage)
            bound += self.micro_batch_time(self.PROP_PHASE.BW, variables,
                                           layers_sorted, v, stage)
        # normalize
        bound = bound / self.num_of_stage_
        return bound

    def _prev_diff_sum(self, variables, layers_sorted, prob, v):
        """models bubble time for the first diagonal (forward, interleave 0)"""
        max_prev_stages = lpSolver.LpVariable.dicts(
            name="max_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_prev_stages = lpSolver.LpVariable.dicts(
            name="diff_with_prev_stages_" + str(v),
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        prob += max_prev_stages[0] >= (self.micro_batch_time(
            self.PROP_PHASE.FW, variables, layers_sorted, v, 0))

        for stage in range(1, self.num_of_stage_):
            prob += max_prev_stages[stage] >= max_prev_stages[stage - 1]
            prob += max_prev_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, variables, layers_sorted, v, stage))

            prob += diff_with_prev_stages[stage] >= (
                max_prev_stages[stage - 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, variables, layers_sorted, v, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_prev_stages[s] for s in range(1, self.num_of_stage_))
        return bound

    def _next_diff_sum(self, variables, layers_sorted, prob):
        """models bubble time for the last diagonal (forward, last chunk)"""
        last_chunk = self.num_of_interleave_ - 1
        max_next_stages = lpSolver.LpVariable.dicts(
            name="max_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        diff_with_next_stages = lpSolver.LpVariable.dicts(
            name="diff_with_next_stages",
            indices=(range(self.num_of_stage_)),
            lowBound=0,
            upBound=None,
            cat=lpSolver.LpContinuous,
        )

        bound = lpSolver.LpAffineExpression()

        prob += max_next_stages[self.num_of_stage_ -
                                1] >= (self.micro_batch_time(
                                    self.PROP_PHASE.FW, variables, layers_sorted, last_chunk,
                                    self.num_of_stage_ - 1))

        for stage in reversed(range(0, self.num_of_stage_ - 1)):
            prob += max_next_stages[stage] >= max_next_stages[stage + 1]
            prob += max_next_stages[stage] >= (self.micro_batch_time(
                self.PROP_PHASE.FW, variables, layers_sorted, last_chunk, stage))

            prob += diff_with_next_stages[stage] >= (
                max_next_stages[stage + 1] - self.micro_batch_time(
                    self.PROP_PHASE.FW, variables, layers_sorted, last_chunk, stage))

        bound += self.num_of_micro_batch_ * lpSolver.lpSum(
            diff_with_next_stages[s] for s in range(self.num_of_stage_ - 1))
        return bound
