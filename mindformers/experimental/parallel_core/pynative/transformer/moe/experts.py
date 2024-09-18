"""define experts"""

from mindspore import nn, ops

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.mlp import ParallelMLP


class GroupedMLP(Module):
    """
    Implement of Grouped MLP
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(GroupedMLP, self).__init__()

        self.num_local_experts = num_local_experts
        self.config = config

        raise NotImplementedError("GroupedMLP function not implemented.")


class SequentialMLP(Module):
    """define SequentialMLP module"""
    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(SequentialMLP, self).__init__()
        if submodules is not None:
            raise NotImplementedError("`submodules` is not supported for now.")
        self.config = config
        self.num_local_experts = num_local_experts
        self.local_experts = nn.SequentialCell()
        for _ in range(self.num_local_experts):
            expert = ParallelMLP(self.config, is_expert=True)
            self.local_experts.append(expert)
        self.cast = ops.Cast()

    def construct(self, permuted_local_hidden_states, tokens_per_expert):
        """forward process"""
        if not permuted_local_hidden_states.shape:
            return permuted_local_hidden_states
        output_local = ops.zeros_like(permuted_local_hidden_states)

        start_idx = 0
        end_idx = tokens_per_expert[0]
        # every expert rank expert_id starts from `0`
        for expert_id, expert in enumerate(self.local_experts):
            hidden_expert = permuted_local_hidden_states[start_idx:end_idx]
            if hidden_expert.shape[0] == 0:
                output = hidden_expert
            else:
                output, _ = expert(hidden_expert)
            output_local[start_idx:end_idx] = output
            if expert_id != len(self.local_experts) - 1:
                start_idx = end_idx
                end_idx += tokens_per_expert[expert_id + 1]
        return output_local, None
