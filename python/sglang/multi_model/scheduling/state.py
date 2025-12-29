import enum
import torch
import dataclasses
from typing import Optional
from sglang.srt.managers.io_struct import MemoryUsage


class ModelState(enum.Enum):
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()
    ACTIVATING = enum.auto()  # transitioning to active state
    DEACTIVATING = enum.auto()  # transitioning to inactive state


@dataclasses.dataclass
class ModelInstanceState:
    """ 维护实例激活状态/显存占用数据 """
    model_name: str
    model_path: str
    instance_idx: int
    gpu_ids: list[int] # TP > 1时，维护全部使用的GPU
    state: ModelState
    memory_usage: MemoryUsage
    init_memory_pool_size: float
    num_vio_reqs: int = 0 # num of violated request in current window
    # for priority scheduling
    req_freq: float = float("-inf")
    urgency: float = float("-inf")
    priority: float = float("-inf")

    def __post_init__(self):
        # TODO: tune the transmition speed
        speed = 6.0  # GB/s
        self.swap_in_time = self.memory_usage.model_weights_memory / speed
        self.swap_out_time = 0.0
        self.memory_pool_size = self.init_memory_pool_size

    def __str__(self):
        total_memory = (self.memory_usage.model_weights_memory + self.memory_usage.memory_pool_memory)
        return (f"Instance {self.instance_idx}: {self.state.name}, memory usage: {total_memory:.2f} GB, gpu_ids: {self.gpu_ids}, "
               f"memory_pool_size: {self.memory_pool_size} GB, "
               f"req to token Mem: {self.memory_usage.req_to_token_pool_memory:.2f} GB, "
               f"token to kv mem: {self.memory_usage.token_to_kv_pool_memory:.2f} GB")

    def on_activate(self, memory_usage: MemoryUsage, gpu_id: Optional[int] = None):
        self.state = ModelState.ACTIVE
        self.memory_usage = memory_usage
        if gpu_id is not None:
            self.gpu_ids = [gpu_id]

    def on_deactivate(self, memory_usage: MemoryUsage):
        self.state = ModelState.INACTIVE
        self.memory_usage = memory_usage
        
    def update_memory_usage(self, memory_usage: MemoryUsage):
        memory_usage.token_to_kv_pool_memory *= self.memory_pool_size
        memory_usage.memory_pool_memory += memory_usage.token_to_kv_pool_memory
        memory_usage.total_used_memory += memory_usage.token_to_kv_pool_memory
        self.memory_usage = memory_usage


def get_gpu_memory_usage(gpu_id: int) -> float:
    """ 获取指定GPU剩余显存（GB） """
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)
    return free_gpu_memory / (1 << 30)
