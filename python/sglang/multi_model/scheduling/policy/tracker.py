import time
from collections import defaultdict
from typing import Dict, List, Tuple
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker, Req

class RequestViolationTracker:
    """ 维护窗口追踪每个模型的请求，并统计窗口内违约（remaining_time_budget <= 0）的数量/比例/总数 """
    def __init__(self, window_size_seconds=10):
        self.window_size = window_size_seconds
        self.model_request_history = defaultdict(list)  # Dict[model_name, List[Tuple[timestamp, Req]]]
    
    def update_request_history(self, model_queues: Dict[str, ModelQueueTracker]):
        """更新时间窗口内的请求"""
        current_time = time.time()
        for model_name, queue in model_queues.items():
            all_reqs = queue.running_reqs + list(queue.waiting_reqs)
            # 清理过期数据
            self.model_request_history[model_name] = [
                (ts, req) for ts, req in self.model_request_history[model_name] 
                if current_time - ts <= self.window_size
            ]
            # 将当前等待/运行请求批量加入窗口
            for req in all_reqs: self.model_request_history[model_name].append((current_time, req))
    
    def get_model_violation_stats(self, model_name: str) -> Tuple[int, float, int]:
        """返回指定模型 (violated_reqs_count, violation_proportion, total_reqs)"""
        current_time = time.time()
        # 清理过期数据
        history: List[Tuple[float, Req]] = self.model_request_history.get(model_name)
        recent_history = [
            (ts, req) for ts, req in history
            if current_time - ts <= self.window_size
        ]
        if not recent_history: return (0, 0.0, 0)
        # 计算统计数据
        violated_reqs = sum(1 for _, req in recent_history if req.remaining_time_budget is not None and req.remaining_time_budget <= 0)
        total_reqs = len(recent_history)
        violation_proportion = violated_reqs / total_reqs if total_reqs > 0 else 0.0
        return (violated_reqs, violation_proportion, total_reqs)


class RequestMemoryTracker:
    """ 维护窗口内每个请求的显存占用，并统计窗口内请求平均剩余显存 """
    def __init__(self, gpu_mem: float, model_weights_info: Dict[str, Dict[str, float]], window_size_seconds=10):
        self.gpu_mem = gpu_mem
        self.model_weights_info = model_weights_info # Dict[model_name, Dict[model_size, weight_size]]
        self.window_size = window_size_seconds
        self.gpu_memory_request_history = defaultdict(list) # Dict[gpu_id, List[Tuple[timestamp, memory_per_request, total_reqs]]]
    
    def update_memory_history(
        self,
        gpu_to_model_mapping: Dict[int, List[str]],
        model_queues: Dict[str, ModelQueueTracker],
        model_weights_info: Dict[str, Dict[str, float]],
        total_gpu_memory: float
    ):
        """更新显存数据"""
        current_time = time.time()
        # 清理过期数据
        for gpu_id in self.gpu_memory_request_history:
            self.gpu_memory_request_history[gpu_id] = [
                entry for entry in self.gpu_memory_request_history[gpu_id]
                if current_time - entry[0] <= self.window_size
            ]
        # 计算剩余显存
        for gpu_id, model_names in gpu_to_model_mapping.items():
            total_model_memory = 0
            total_reqs = 0
            for model_name in model_names:
                total_model_memory += model_weights_info[model_name]["model_size"]
                queue = model_queues.get(model_name)
                if queue: total_reqs += len(queue.running_reqs) + queue.get_num_waiting_reqs()
            memory_available_for_requests = total_gpu_memory - total_model_memory
            memory_per_request = memory_available_for_requests / max(1, total_reqs)
            self.gpu_memory_request_history[gpu_id].append((current_time, memory_per_request, total_reqs))


class ModelRequestTracker:
    """ 维护窗口内每个模型的请求历史，并统计窗口内每时刻平均请求数 """    
    def __init__(self, window_size_seconds=10):
        self.window_size = window_size_seconds
        self.model_request_history = defaultdict(list)  # Dict[model_name, List[Tuple[timestamp, num_requests]]]
    
    def update_request_history(self, model_queues: Dict[str, ModelQueueTracker]):
        """更新时间窗口内的请求"""
        current_time = time.time()
        for model_name, queue in model_queues.items():
            cur_num_requests = queue.get_num_unfinished_reqs()
            # 清理过期数据
            self.model_request_history[model_name] = [
                (ts, reqs) for ts, reqs in self.model_request_history[model_name] 
                if current_time - ts <= self.window_size
            ]
            # 添加未完成请求，可能为0
            self.model_request_history[model_name].append((current_time, cur_num_requests))
    
    def get_model_request_stats(self, model_name: str) -> float:
        """返回指定模型窗口内每时刻平均请求数（平滑均值）"""
        current_time = time.time()
        history = self.model_request_history.get(model_name, [])        
        recent_history = [(ts, reqs) for ts, reqs in history if current_time - ts <= self.window_size]
        if not recent_history: return 0.0
        total_requests = sum(reqs for _, reqs in recent_history)
        return total_requests / len(recent_history)