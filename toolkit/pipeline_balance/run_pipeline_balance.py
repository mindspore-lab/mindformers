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
"""run pipeline balance"""
import os
import sys
import argparse

from toolkit.pipeline_balance.utils.logger import logger
from toolkit.pipeline_balance.utils.config import initialize_layer_json
from toolkit.pipeline_balance.utils.layer import generate_layers_list
from toolkit.pipeline_balance.utils.compute_memory import compute_memories
from toolkit.pipeline_balance.sapp.sapp_pipeline import SappPipeline
import toolkit.pipeline_balance.utils.interactive as Interactive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SAPP AutoBalancing', description=(
                'Balance layers onto pipeline stages, '
                + 'considering recomputation and interleaving'),
                                     epilog='')

    # Pipeline info
    parser.add_argument('-s', '--stage', type=int, default=4, help="Number of stages")
    parser.add_argument('-mb', '--micro_batch', type=int, default=4, help="Number of micro batch")
    parser.add_argument('-i', '--interleave_degree', type=int, default=1, help="Interleave level")

    # Memory size
    parser.add_argument('-mem', '--max_memory', type=int, default=56000,
                        help="Maximum memory available (MB)")

    parser.add_argument('-lm', '--less_memory',
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Compute Memory with 'Less Memory interleave' option")

    parser.add_argument('-mc', '--constant_memory', type=int, default=0,
                        help="Constant memory per stages")

    parser.add_argument('-o', '--output_folder', type=str, default="./output",
                        help="output files location")

    # Model info
    parser.add_argument('-m', '--model_name', type=str, default="model_name", help="")

    # Search time
    parser.add_argument('-t', '--time_limit', type=int, default=90,
                        help="Limitation on searching time")

    # Optimization level
    parser.add_argument('-O', '--optimization_level', type=int, default=1,
                        help="Defines optimization level when Stage (S) = Micro Batch number (M)."
                        + " 0 for same approach as M > S. "
                        + " 1 (default) generally better. "
                        + " 2 better for memory constrained cases. ")

    # Simulate naive or manual config
    parser.add_argument('-naive', '--simulate_naive', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Simualte naive configs")
    parser.add_argument('-manual', '--manual_config', type=str, default=None,
                        help="Path of manual config")

    # Layer info
    parser.add_argument('-lf', '--layer_folder', type=str, default="./layers/",
                        help="Path to the layer folder")
    parser.add_argument('-dump', '--dump_layer',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Dump the layers")

    # For Computation of memory
    parser.add_argument('-mf', '--memory_folder', type=str, default="./memory/",
                        help="Path to the profiler memory folder")
    # For Initialization
    parser.add_argument('-init', '--init', type=str, default=None,
                        help="Path to the init file")

    # Computation argument
    parser.add_argument('-cm', '--compute_memory',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Parse Mindspore log to generate MEMORY of the layer (unavailable)",)
    parser.add_argument('-exec', '--exec',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Compute solver")



    args = parser.parse_args()

    if len(sys.argv) == 1:
        Interactive.main()
        sys.exit(0)

    layer_folder = args.layer_folder
    memory_folder = args.memory_folder
    model_name = args.model_name
    number_of_stage = args.stage
    number_of_micro_batch = args.micro_batch
    time_limit = args.time_limit
    less_memory = args.less_memory

    if args.init:
        init_file = os.path.join(os.path.dirname(__file__), args.init)
        initialize_layer_json(model_name, init_file)

    output_folder = os.path.join(os.path.dirname(__file__), args.output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    manual_config = None
    if args.manual_config:
        manual_config = os.path.join(os.path.dirname(__file__), args.manual_config)
        manual_config = None if (not manual_config.endswith('yaml') and
                                 not manual_config.endswith('yml')) else manual_config

    max_memory = args.max_memory
    constant_memory = args.constant_memory
    interleave_degree = args.interleave_degree
    optimization_level = args.optimization_level

    layers = generate_layers_list(layer_folder, model_name)

    if args.compute_memory:
        layers = compute_memories(layers=layers, memory_folder=memory_folder,
                                  number_of_stage=number_of_stage)
    for layer in layers:
        logger.output("%s", layer)

    if args.dump_layer:
        for layer in layers:
            layer.dump()

    if args.exec:
        pipe = SappPipeline(model_name=model_name, num_of_stage=number_of_stage,
                            num_of_micro_batch=number_of_micro_batch, max_memory=max_memory,
                            layers=layers, num_of_interleave=interleave_degree,
                            vpp_less_memory=less_memory, constant_memory=constant_memory,
                            optimization_level=optimization_level)

        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=time_limit, dump_folder=output_folder)
        pipe.print_yaml_results()
        total_time = pipe.simulate(show=True, file_name=os.path.join(output_folder, "result.svg"))

        if args.simulate_naive:
            logger.output("Simulating naive configs")
            pipe.simulate_naive(layers, output_folder)
        if manual_config:
            logger.output("Simulating manual configs")
            pipe.simulate_file(manual_config, output_folder)

        logger.output("total_time: %d", total_time)
        logger.output("time: %s", pipe.get_time())
        logger.output("mem_par: %s", pipe.get_memory_parameter())
        logger.output("mem_act: %s", pipe.get_memory_activation())
