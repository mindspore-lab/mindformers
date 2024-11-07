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

import sys
import argparse
import json

from mindformers.tools.logger import logger
import toolkit.pipeline_balance.utils.interactive as Interactive
from toolkit.pipeline_balance.utils.layer import generate_layers_list
from toolkit.pipeline_balance.utils.compute_memory import compute_memories
from toolkit.pipeline_balance.sapp.sapp_pipeline import SappPipeline

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

    parser.add_argument('-o', '--overlap', type=list, default=[1],
                        help="List of overlap coefficient")

    # Model info
    parser.add_argument('-m', '--model_name', type=str, default="Llama_special", help="")

    # Model info
    parser.add_argument('-t', '--time_limit', type=int, default=90,
                        help="Limitation on searching time")

    # Layer info
    parser.add_argument('-lf', '--layer_folder', type=str, default="./layers/",
                        help="Path to the layer folder")
    parser.add_argument('-dump', '--dump_layer',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Dump the layers")
    # For Computation of time
    parser.add_argument('-tf', '--timeline_folder', type=str, default="./timeline/",
                        help="Path to the profiler timeline folder")
    # For Computation of memory
    parser.add_argument('-mf', '--memory_folder', type=str, default="./memory/",
                        help="Path to the profiler memory folder")

    # Computation argument
    parser.add_argument('-ct', '--compute_timer',  # type=bool,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False,
                        help="Parse timeline_folder to generate TIME of the layer")
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
    timeline_folder = args.timeline_folder
    memory_folder = args.memory_folder
    model_name = args.model_name
    number_of_stage = args.stage
    number_of_micro_batch = args.micro_batch
    time_limit = args.time_limit
    less_memory = args.less_memory

    overlap_coeff = args.overlap

    max_memory = args.max_memory
    interleave_degree = args.interleave_degree

    layers = generate_layers_list(layer_folder, model_name)

    if args.compute_timer:
        # With this way of computing the layers object parse many time the timeline profiling...
        json_layer = layer_folder + '/' + model_name + '.json'
        logger.info("json_layer=%s", json_layer)
        with open(json_layer) as json_file:
            layer_datas_json = json.load(json_file)
            for layer in layers:
                layer.compute_timer(timeline_folder, layer_datas_json)

    if args.compute_memory:
        layers = compute_memories(layers=layers, memory_folder=memory_folder,
                                  number_of_stage=number_of_stage)
    for layer in layers:
        logger.info("%s", layer)

    if args.dump_layer:
        for layer in layers:
            layer.dump()

    if args.exec:
        pipe = SappPipeline(model_name=model_name, num_of_stage=number_of_stage,
                            num_of_micro_batch=number_of_micro_batch, max_memory=max_memory,
                            layers=layers, num_of_interleave=interleave_degree,
                            vpp_less_memory=less_memory,)


        pipe.construct_problem(solver="pulp")
        pipe.solve_problem(time_limit=time_limit)
        pipe.print_yaml_results()
        total_time = pipe.simulate(show=True, file_name="./output/result.svg")
        pipe.simulate_naive(all_recompute=True, show=True, interleave_num=interleave_degree,
                            file_name='./output/result_all_rec.svg',)
        pipe.simulate_naive(all_recompute=False, show=True, interleave_num=interleave_degree,
                            file_name='./output/result_no_rec.svg',)

        logger.info("total_time: %d", total_time)
        logger.info("time: %d", pipe.get_time())
        logger.info("mem_par: %d", pipe.get_memory_parameter())
        logger.info("mem_act: %d", pipe.get_memory_activation())
