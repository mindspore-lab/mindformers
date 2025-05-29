"""search parallel shard policy for heterogeneous pipeline"""

from mindformers.experimental.parallel_core.pynative.parallel_policy_shard.memory_cost import MemoryCostModel
from mindformers.experimental.parallel_core.pynative.parallel_policy_shard.time_cost import TimeCostModel
from mindformers.experimental.parallel_core.pynative.parallel_policy_shard.system_config import ClusterConfig
from mindformers.experimental.parallel_core.pynative.parallel_policy_shard.utils import get_all_combincations, calculate_layer_allocations
from mindformers.tools import logger

def check_oom(memory_cost, single_npu_max_available_memory):
    if memory_cost * 1.2 > single_npu_max_available_memory:
        return False
    return True

def get_policy(pipeline_stage_device, memory_cost, single_npu_max_available_memory):
    """get policy of cur pipeline stage device"""
    tensor_model_parallel_size = []
    for stage_device in pipeline_stage_device:
        for tp in range(1, stage_device):
            if tp & (tp - 1) == 0 and check_oom(memory_cost/tp, \
                                                single_npu_max_available_memory):
                tensor_model_parallel_size.append(tp)
                break
    return tensor_model_parallel_size

def search_parallel_policy(train_config, model_config, optimizer_config):
    """search parallel policy"""
    num_layers = model_config.num_layers
    nnodes = train_config.nnodes
    nproc_per_node = train_config.nproc_per_node
    cluster_config = ClusterConfig(nnodes=nnodes, nproc_per_node=nproc_per_node)
    pipeline_stage_devices = get_all_combincations(cluster_config.nproc_per_node) # whole search space
    best_parallel_policy = []

    for pipeline_stage_device in pipeline_stage_devices:
        pipeline_stage_layer_allocations = calculate_layer_allocations(num_layers, pipeline_stage_device)[:5]
        time_min_cost = float('inf')
        cur_best_parallel_policy = []
        parallel_policy = {}
        for stage_allocations in pipeline_stage_layer_allocations:
            pipeline_stage_layer = stage_allocations['allocation']
            memory_model = MemoryCostModel(train_config, model_config, optimizer_config)
            time_model = TimeCostModel(train_config, model_config)
            for layer in pipeline_stage_layer:
                memory_cost = memory_model.get_peak_memory(layer)
                tensor_model_parallel_size = get_policy(pipeline_stage_device, memory_cost, \
                                                        cluster_config.single_npu_max_available_memory)
                time_cost = time_model.get_iteration_time(pipeline_stage_layer)
                if time_cost < time_min_cost:
                    time_min_cost = time_cost
                    parallel_policy['pipeline_stage_device'] = pipeline_stage_device
                    parallel_policy['num_layer_list'] = pipeline_stage_layer
                    parallel_policy['tensor_model_parallel_size'] = tensor_model_parallel_size
                    parallel_policy['pipeline_model_parallel_size'] = len(pipeline_stage_device)
        cur_best_parallel_policy.append(parallel_policy)
        best_parallel_policy.append(cur_best_parallel_policy)

    for policy in best_parallel_policy:
        logger.info(f"pipeline_stage_device:{policy[0]['pipeline_stage_device']}"
                    + f"num_layer_list:{policy[0]['num_layer_list']}"
                    + f"tensor_model_parallel_size:{policy[0]['tensor_model_parallel_size']}")
