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
import numpy as np
from mindformers.tools.logger import logger


class ExpertDynamicRelocation:
    """Dynamic expert relocation algorithms for load balancing in MoE models."""
    @staticmethod
    def compute_jaccard_matrix(expert_token_map):
        """Compute Jaccard similarity matrix between experts based on token overlap.

        Args:
            expert_token_map: Mapping from expert ID to set of token IDs it processes.

        Returns:
            tuple: (jaccard_matrix, expert_ids) where jaccard_matrix is the similarity
                   matrix and expert_ids is the list of expert IDs.
        """
        expert_ids = list(expert_token_map.keys())
        num_experts = len(expert_ids)
        matrix = np.ones((num_experts, num_experts))

        for i in range(num_experts):
            for j in range(num_experts):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    set_i = expert_token_map[expert_ids[i]]
                    set_j = expert_token_map[expert_ids[j]]
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    matrix[i][j] = intersection / union if union > 0 else 0.0

        return matrix, expert_ids

    @staticmethod
    def evaluate_groups(expert_device_map, expert_token_map, jaccard_matrix, expert_ids):
        """Evaluate expert group placement quality based on load and affinity.

        Args:
            expert_device_map: Mapping from expert ID to device assignment.
            expert_token_map: Mapping from expert ID to set of token IDs.
            jaccard_matrix: Pre-computed Jaccard similarity matrix.
            expert_ids: List of expert IDs.

        Returns:
            dict: Device statistics including token sum, average affinity, and expert count.
        """
        device_stats = {}
        device_experts = {}
        total_avg_affinity = 0

        for eid, did in expert_device_map.items():
            device_experts.setdefault(did[0], []).append(eid)

        for did, experts in device_experts.items():
            token_sum = sum(len(expert_token_map[eid]) for eid in experts)

            indices = [expert_ids.index(eid) for eid in experts]
            pairwise_scores = []
            for i in indices:
                for j in indices:
                    if i < j:
                        pairwise_scores.append(jaccard_matrix[i][j])
            avg_affinity = np.mean(pairwise_scores) if pairwise_scores else 0.0
            total_avg_affinity += avg_affinity
            device_stats[did] = {
                "token_sum": token_sum,
                "avg_affinity": avg_affinity,
                "num_experts": len(experts)
            }

        for device_id, info in device_stats.items():
            logger.info(f"Device {device_id}:")
            logger.info(f"  Token Sum      : {info['token_sum']}")
            logger.info(f"  Avg Affinity   : {info['avg_affinity']:.4f}")
            logger.info(f"  Num of Experts : {info['num_experts']}")
        logger.info(f"Total Avg Affinity : {total_avg_affinity}")
        return device_stats

    @staticmethod
    def expert_relocation_greedy_with_affinity(expert_token_map, device_ids, beta=1.0):
        """
        Greedy expert relocation with affinity-based placement.

        Args:
            expert_token_map: Mapping from expert ID to the set of token IDs it
                processes. Used to compute expert affinity and estimate per-expert load.
            device_ids: List of available device IDs. Determines how many groups
                to form and where experts will be assigned.
            beta: Weight factor for load imbalance penalty in the scoring function.
                Higher values prioritize load balancing over expert affinity.

        Returns:
            tuple: (expert_device_map, device_expert_map) mappings for expert relocation.
        """
        jaccard_matrix, _ = ExpertDynamicRelocation.compute_jaccard_matrix(
            expert_token_map)

        experts = {expert_id: len(tokens)
                   for expert_id, tokens in expert_token_map.items()}
        num_experts = len(experts)
        num_devices = len(device_ids)
        expert_keys = list(experts.keys())
        experts_per_device = num_experts // num_devices
        avg_tokens_per_device = sum(experts.values()) // num_devices
        # Initialize the current number of tokens per device
        tokens_each_device = dict.fromkeys(device_ids, 0)
        # Initialize the current number of experts per device
        experts_each_device = dict.fromkeys(device_ids, 0)
        # Initialize the placement of experts
        expert_device_map = {key: [] for key in expert_keys}
        device_expert_map = {key: [] for key in device_ids}

        # Sort the experts in descending order of their costs and get the sorted indices
        sorted_indices = sorted(
            expert_keys, key=lambda i: experts[i], reverse=True)

        for expert_id in sorted_indices:
            best_device = None
            # Initialize the minimum number of tokens to infinity
            best_score = float('-inf')
            # Decide the placement of the current expert
            for device_id in device_ids:
                if experts_each_device[device_id] >= experts_per_device:
                    continue
                affinity_score = 0
                for assigned_expert in device_expert_map[device_id]:
                    affinity_score += jaccard_matrix[expert_id][assigned_expert]
                token_load = tokens_each_device[device_id]
                score = beta * affinity_score - token_load / avg_tokens_per_device
                if score > best_score:
                    best_score = score
                    best_device = device_id
            if best_device is not None:
                # Place the current expert on the selected device
                expert_device_map[expert_id].append(best_device)
                device_expert_map[best_device].append(expert_id)
                # Update the number of tokens on the selected device
                tokens_each_device[best_device] += experts[expert_id]
                # Update the number of experts on the selected device
                experts_each_device[best_device] += 1

        return expert_device_map, device_expert_map

    @staticmethod
    def expert_relocation_greedy(experts, devices):
        """
        Greedy algorithm for expert relocation based on load balancing.

        This method implements a greedy strategy to redistribute experts across available devices
        to achieve load balancing. Experts are sorted by their computational cost (number of samples)
        in descending order and assigned to devices with the least current load.

        Args:
            experts (dict): Mapping from expert ID to the number of samples/tokens it processes.
                           The keys are expert identifiers (typically integers), and values are
                           the computational load metrics (e.g., sample counts or token counts).
            devices (list): List of available device IDs where experts can be assigned.
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
                if (experts_each_device[device] < num_experts / num_devices and
                        samples_each_device[device] < samples_min):
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
