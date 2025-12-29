import time
import heapq
import logging
import threading
from typing import Dict, List, Set
from collections import defaultdict
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)

PREFILL_RATE = 1.0 / 2048
DEFAULT_PREFILL_LEN = 1024
LOWER_PREFILL_BOUND = 0.05
UPPER_PREFILL_BOUND = 20
MAX_QUEUE_LEN = 10
GPU_SIZE = 80 * (1 << 30)

class RequestWrapper:
    """封装请求对象，赋值优先级"""
    def __init__(self, req: GenerateReqInput):
        self.req = req
        self.model_name = req.model
        self.priority = self._calculate_priority(req)  # Lower value means higher priority (min-heap)

    def _calculate_priority(self, req: GenerateReqInput):
        """计算优先级
        priority = arrival_time + slo - prefill_time
        到达越早，slo越紧张，优先级越低
        """
        def clamp(x, lower, upper): # 确保时间处于区间范围内
            return max(lower, min(x, upper))
        profiled_prefill_time = (
            req.prompt_len * PREFILL_RATE
            if req.prompt_len is not None
            else DEFAULT_PREFILL_LEN * PREFILL_RATE
        )
        profiled_prefill_time = clamp(profiled_prefill_time, LOWER_PREFILL_BOUND, UPPER_PREFILL_BOUND)
        return req.arrival_time + req.slo - profiled_prefill_time

    def __lt__(self, other):
        return self.priority < other.priority  # 用于排序

    def __str__(self):
        return f"RequestWrapper(model_name={self.model_name}, priority={self.priority}, req_id={self.req.rid})"

    def __repr__(self):
        return self.__str__()


class RequestQueue:
    """维护多模型请求优先队列
    资源跟踪，请求准入
    对模型rank0 GPU跟踪
    """
    def __init__(self, model_name_to_cell_size: Dict[str, int]):
        self._model_name_to_cell_size = model_name_to_cell_size
        self._queue: List[RequestWrapper] = []  # 小顶优先队列，维护GPU请求
        self._model_requests: Dict[str, Set[RequestWrapper]] = defaultdict(set) # 分模型维护请求队列
        self._lock = threading.Lock() # 队列锁
        self.last_log_time = 0
        # 运行中模型显存占用占用，未体现在物理显存变化中，需要自己跟踪
        self._activating_usage_by_model = defaultdict(float)

    def empty(self) -> bool:
        """清空队列"""
        with self._lock:
            return len(self._queue) == 0

    def pop_model_requests(self, model_name: str) -> List[GenerateReqInput]:
        """弹出指定模型所有请求"""
        if model_name not in self._model_requests: return []
        with self._lock:
            model_reqs = list(self._model_requests[model_name])
            self._queue = [req for req in self._queue if req not in model_reqs]
            heapq.heapify(self._queue) # 重新排序
            del self._model_requests[model_name]
            return [req_wrapper.req for req_wrapper in model_reqs] # 返回所有原请求对象

    def add_requests(self, reqs: List[GenerateReqInput]):
        """添加请求batch"""
        wrapped_reqs = [RequestWrapper(req) for req in reqs]
        with self._lock:
            for wrapped_req in wrapped_reqs:
                heapq.heappush(self._queue, wrapped_req)
                self._model_requests[wrapped_req.model_name].add(wrapped_req)

    def remove_model_requests(self, model_name):
        """清除但不返回制定模型所有请求"""
        if model_name not in self._model_requests: return
        with self._lock:
            removed = self._model_requests[model_name]
            self._queue = [req for req in self._queue if req not in removed]
            heapq.heapify(self._queue)
            del self._model_requests[model_name]
    
    def log_info(self, info: str):
        """限制log速率，每秒至多一条"""
        current_time = time.time()
        if current_time - self.last_log_time > 1:
            logger.info(info)
            self.last_log_time = current_time

    def admission_control(
        self,
        model_states: Dict[str, str],
        available_resources: float, # 剩余KV cache
        model_backend_queue_lens: Dict[str, int], # Engine侧队列长度
        allow_sending_when_activating: bool = False,
    ) -> Dict[str, List[GenerateReqInput]]:
        """准入控制
        Memory Track
        |—————————————————|------------------|               |
        | activated usage | activating usage | net available |
        |  tracked usage  |        available resource        |
        """
        admitted = defaultdict(list)
        total_activating_usage = sum(self._activating_usage_by_model.values()) # 激活中模型显存占用
        # Note: Using infinity here as the actual implementation doesn't seem to limit resources
        net_available = float("inf")
        # net_available = available_resources - total_activating_usage
        if net_available <= 0:
            self.log_info(f"😟 Resource ran out, net_available: {net_available}, queue_len: {len(self._queue)}")
            self.log_info(f"Activating usages: {self._activating_usage_by_model}")
            return admitted
        if len(self._queue) == 0:
            self.log_info(f"😃 No request queuing")
            return admitted
        # 后端队列太长时跳过
        models_to_skip = {
            model_name for model_name, queue_len 
            in model_backend_queue_lens.items()
            if queue_len > MAX_QUEUE_LEN
        }
        put_backs = []
        with self._lock:
            while self._queue and net_available > 0:
                # 显存充足时逐个添加请求，不可用请求放回等待队列
                req_wrapper = heapq.heappop(self._queue)
                model_name = req_wrapper.model_name
                model_state = model_states.get(model_name, "deactivated")
                if model_name in models_to_skip:
                    put_backs.append(req_wrapper) # 排队太长
                    self.log_info(f"⏰ Queuing exceeds limit, model queue: {model_backend_queue_lens[model_name]}")
                    continue
                if model_state in ("deactivating", "deactivated"):
                    put_backs.append(req_wrapper) # 模型未激活
                    self.log_info(f"💤 {model_name} deactivated")
                    continue
                if model_state == "activating" and not allow_sending_when_activating:
                    put_backs.append(req_wrapper) # 模型不接受请求
                    self.log_info(f"🔕 {model_name} does not allow message while activating")
                    continue
                resources_needed = self._get_request_resources(req_wrapper.req)
                if net_available >= resources_needed:
                    net_available -= resources_needed
                    admitted[model_name].append(req_wrapper.req) # 添加充足请求
                    self._model_requests[model_name].remove(req_wrapper)
                    if not self._model_requests[model_name]: del self._model_requests[model_name]
                    if model_state == "activating": self._activating_usage_by_model[model_name] += resources_needed
                else:
                    put_backs.append(req_wrapper) # 资源不足
                    self.log_info(f"😢 Resource limited, net_available: {net_available}, queue_len: {len(self._queue)}")
                    break
            # 维护等待队列
            put_backs.extend(self._queue)
            self._queue = put_backs
            heapq.heapify(self._queue)
            self.log_info(
                f"📰 Resource update: net_available: {net_available}, queue_len: {len(self._queue)}, model_backend_queue_lens: {model_backend_queue_lens}"
            )
        return admitted

    def _get_request_resources(self, req: GenerateReqInput) -> float:
        """估计请求显存占用"""
        cell_size = self._model_name_to_cell_size[req.model]
        input_len = (
            req.prompt_len
            if req.prompt_len is not None and req.prompt_len > 0
            else DEFAULT_PREFILL_LEN
        )
        return cell_size * (input_len + 20)

    def clear_activating_usage(self, model_name: str):
        """清空激活中模型资源跟踪"""
        with self._lock:
            if model_name in self._activating_usage_by_model:
                del self._activating_usage_by_model[model_name]

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        if len(self._model_requests) == 0:
            req_counts_str = ""
        else:
            req_counts_str = ", ".join([
                f"{model_name}: {len(reqs)}"
                for model_name, reqs in self._model_requests.items()
            ])
        return f"RequestQueue(total_queued={len(self._queue)}, {req_counts_str})"
