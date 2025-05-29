"""utils for parallel policy shard"""
from itertools import product

def get_all_merges(nproc_per_node):
    """get all merges of each node"""
    if len(nproc_per_node) == 1:
        return [nproc_per_node]

    merges = []
    # check all possible splits where the sum of the left part is a multiplt of 8
    for i in range(1, len(nproc_per_node)):
        left = nproc_per_node[:i]
        right = nproc_per_node[i:]
        if sum(left) % 8 == 0:
            for l_merge in get_all_merges(left):
                for r_merge in get_all_merges(right):
                    merges.append(l_merge + r_merge)

    # consider the case where the entire list is merged if the sum is multiple of 8
    if sum(nproc_per_node) % 8 == 0:
        merges.append([sum(nproc_per_node)])
    return merges if merges else [nproc_per_node]

def get_all_combincations(nproc_per_node):
    """get combincations of each node"""
    def split_number(n):
        if n <= 4:
            num_result = [[n]]
        elif 4 < n < 8:
            num_result = [[4, n - 4]]
        elif n == 8:
            num_result = [[8], [4, 4]]
        else:
            num_result = [[n]]
        return num_result

    all_merges = get_all_merges(nproc_per_node)
    unique_all_merges = set()
    for merge in all_merges:
        merge = tuple(merge)
        if merge not in unique_all_merges:
            unique_all_merges.add(merge)
    unique_all_merges = list(unique_all_merges)
    result = []
    for merge in unique_all_merges:
        all_splits = [split_number(num) for num in merge]
        combinations = product(*all_splits)
        for combo in combinations:
            flattened = []
            for part in combo:
                flattened.extend(part)
            result.append(flattened)
    return result

def calculate_layer_allocations(total_layers, gpus_per_stage):
    """calculate the number of layers for each pipeline stage"""
    num_stages = len(gpus_per_stage)

    def generate_allocations(remaining_layers, remaining_stages, current_allocation):
        """ generate all layers allocation"""
        if remaining_stages == 1:
            yield current_allocation + [remaining_layers]
        else:
            min_layers = 1
            max_layers = remaining_layers- (remaining_stages - 1)
            for layers in range(min_layers, max_layers + 1):
                yield from generate_allocations(
                    remaining_layers - layers,
                    remaining_stages - 1,
                    current_allocation + [layers]
                )

    allocations = []
    for alloc in generate_allocations(total_layers, num_stages, []):
        per_gpu_layers = [alloc[i] / gpus_per_stage[i] for i in range(num_stages)]

        # calculate difference of num_layers of each npu
        max_diff = max(per_gpu_layers) - min(per_gpu_layers)
        avg_diff = sum(abs(p - sum(per_gpu_layers)/num_stages) for p in per_gpu_layers) / num_stages
        allocations.append({
            'allocation': alloc,
            'per_gpu_layers': per_gpu_layers,
            'max_diff': max_diff,
            'avg_diff': avg_diff
        })
    # sort
    allocations.sort(key=lambda x: (x['max_diff'], x['avg_diff']))

    return allocations
