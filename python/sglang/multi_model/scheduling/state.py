import dataclasses
import enum
from typing import Optional

import torch

from sglang.srt.managers.io_struct import MemoryUsage


class ModelState(enum.Enum):
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()
    ACTIVATING = enum.auto()  # transitioning to active state
    DEACTIVATING = enum.auto()  # transitioning to inactive state


@dataclasses.dataclass
class ModelInstanceState:
    model_name: str
    model_path: str
    instance_idx: int
    gpu_ids: list[int]  # for tensor parallelism > 1
    state: ModelState
    memory_usage: MemoryUsage
    init_memory_pool_size: float
    num_vio_reqs: int = 0  # num of violated request in current window

    # for priority scheduling
    req_freq: float = float("-inf")
    urgency: float = float("-inf")
    priority: float = float("-inf")

    def __post_init__(self):
        # TODO: tune the transmition speed
        speed = 6.0  # GB/s
        self.swap_in_time = self.memory_usage.model_weights_memory / speed
        self.swap_out_time = 0.0
        self.last_memory_pool_size: float = self.init_memory_pool_size

    def __str__(self):
        total_memory = (
            self.memory_usage.model_weights_memory
            + self.memory_usage.memory_pool_memory
        )
        return f"Instance {self.instance_idx}: {self.state.name}, memory usage: {total_memory:.2f} GB, gpu_ids: {self.gpu_ids}, init_memory_pool_size: {self.init_memory_pool_size if self.init_memory_pool_size is not None else 'N/A'} GB"

    def on_activate(self, memory_usage: MemoryUsage, gpu_id: Optional[int] = None):
        self.state = ModelState.ACTIVE
        self.memory_usage = memory_usage
        if gpu_id is not None:
            self.gpu_ids = [gpu_id]

    def on_deactivate(self, memory_usage: MemoryUsage):
        self.state = ModelState.INACTIVE
        self.memory_usage = memory_usage


def get_gpu_memory_usage(gpu_id: int) -> float:
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)
    return free_gpu_memory / (1 << 30)
