"""system config"""
from dataclasses import dataclass
import mindspore as ms

@dataclass
class ClusterConfig:
    r"""cluster config"""
    nnodes: int

    nproc_per_node: list

    single_npu_max_available_memory: float = 0.0

    world_size: int = 0

    def __post_init__(self):
        self.world_size = sum(self.nproc_per_node)
        device_info = ms.hal.get_device_properties(0, "Ascend")
        self.single_npu_max_available_memory = device_info.total_memory / (1024**2)
