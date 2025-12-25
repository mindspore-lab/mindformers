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
"""
Expert mapping utilities for dynamic expert relocation in mixture of experts (MoE) models.

This module provides algorithms for redistributing experts across devices based on load balancing
and affinity metrics to optimize training efficiency.
"""
import copy


class ExpertDynamicRelocation:
    """Dynamic expert relocation algorithms for load balancing in MoE models."""
    @staticmethod
    def expert_relocation_greedy(experts, devices):
        """
        Greedy algorithm for expert relocation based on load balancing.

        This method implements a greedy strategy to redistribute experts across available devices
        to achieve load balancing. Experts are sorted by their computational cost (number of samples)
        in descending order and assigned to devices with the least current load.

        Args:
            experts (Dict[int, int]): Mapping from expert ID to the number of samples/tokens it processes.
                           The keys are expert identifiers (typically integers), and values are
                           the computational load metrics (e.g., sample counts or token counts).
            devices (List[int]): List of available device IDs where experts can be assigned.
                           These are typically GPU device identifiers or rank numbers.

        Returns:
            tuple: A tuple containing two mappings:
                - expert_device_map (dict): Maps each expert ID to the assigned device ID.
                  Format: {expert_id: [device_id]}
                - device_expert_map (dict): Maps each device ID to the list of expert IDs
                  assigned to it. Format: {device_id: [expert_id1, expert_id2, ...]}
        """
        num_experts = len(experts)
        num_devices = len(devices)
        expert_keys = list(experts.keys())
        # Initialize the current number of samples per device
        samples_each_device = dict.fromkeys(devices, 0)
        # Initialize the current number of experts per device
        experts_each_device = dict.fromkeys(devices, 0)
        expert_load_each_device = [[] for _ in range(num_devices)]
        # Initialize the placement of experts
        expert_device_map = {key: [] for key in expert_keys}

        # Sort the experts in descending order of their costs and get the sorted indices
        sorted_indices = sorted(
            expert_keys, key=lambda i: experts[i], reverse=True)

        for i in sorted_indices:
            samples_i = experts[i]
            # Initialize the minimum number of samples to infinity
            samples_min = float('inf')
            target_device = -1
            # Decide the placement of the current expert
            for device in devices:
                if experts_each_device[device] < num_experts / num_devices and \
                        samples_each_device[device] < samples_min:
                    samples_min = samples_each_device[device]
                    target_device = device
            # Place the current expert on the selected device
            expert_device_map[i].append(target_device)
            # Update the number of samples on the selected device
            samples_each_device[target_device] += samples_i
            # Update the number of experts on the selected device
            experts_each_device[target_device] += 1

            expert_load_each_device[target_device].append(samples_i)

        device_expert_map = {key: [] for key in devices}
        expert_device_map_duplicate = copy.deepcopy(expert_device_map)
        for i in expert_keys:
            device = expert_device_map_duplicate[i].pop()
            device_expert_map[device].append(i)

        return expert_device_map, device_expert_map
