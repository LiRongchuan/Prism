import logging
import time
from itertools import chain
from typing import Dict, List, Set, Tuple, Optional, DefaultDict
from collections import defaultdict, deque
import torch
from sglang.multi_model.scheduling.action import (
    ActivateAction,
    BaseAction,
    DeactivateAction,
    ResizeAction,
)
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker, Req, ReqState
from sglang.multi_model.scheduling.policy.base_global import GlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.multi_model.scheduling.gpu.request_queue import RequestQueue
from sglang.srt.utils import get_available_gpu_memory
from sglang.multi_model.scheduling.policy.tracker import RequestViolationTracker, RequestMemoryTracker, ModelRequestTracker
import itertools

logger = logging.getLogger(__name__)

"""
保留当前ModelInstanceState定义
迁移时
"""

class TPGlobalPolicy(GlobalPolicy):
    def __init__(
        self,
        num_gpus: int,
        gpu_mem: float,
        model_weights_info: Dict[str, Dict[str, float]], # Dict[model_name, Dict[model_size, weight_size]]模型原始信息
        workers_per_gpu: int,
    ):
        self.num_gpus = num_gpus
        # assume all GPUs have the same maximal usable memory size
        self.gpu_mem = gpu_mem
        self.model_weights_info = model_weights_info
        self.workers_per_gpu = workers_per_gpu
        self.first_request_time = None
        self.model_last_active_time = {}
        
        self.request_history = RequestViolationTracker(window_size_seconds=3)
        self.memory_history = RequestMemoryTracker(gpu_mem, model_weights_info, window_size_seconds=30)
        self.model_request_tracker = ModelRequestTracker(window_size_seconds=30)

        self.MEMORY_POOL_BUDGET = 6
        self.MODEL_IDLE_THRESHOLD = 50  # seconds
        self.VIOLATION_PROPORTION_THRESHOLD = 0.1 # 10%
        self.MEMORY_PER_REQUEST_RATIO_THRESHOLD = 15

        self.migrate_policy = "memory_per_request"
        assert self.migrate_policy in ["violation", "memory_per_request"]

    def _update_req_remaining_time_budget(
        self,
        model_queues: Dict[str, ModelQueueTracker]
    ):
        """更新所有请求剩余budget"""
        # budget = slo - (current - arrival)
        cur_time = time.time()
        for model_name, queue in model_queues.items():
            for req in queue.running_reqs + list(queue.waiting_reqs):
                if req.arrival_time is None or req.slo is None:
                    continue
                elapsed_time = cur_time - req.arrival_time
                req.remaining_time_budget = req.slo - elapsed_time

    def _calculate_model_avg_remaining_time_budget(
        self,
        model_name: str,
        model_queues: Dict[str, ModelQueueTracker]
    ) -> float:
        """计算指定模型请求队列的平均budget"""
        queue = model_queues.get(model_name)
        if not queue:
            return 0.0
            
        total_budget = 0.0
        all_reqs = list(queue.running_reqs) + list(queue.waiting_reqs)
        
        for req in all_reqs:
            if req.remaining_time_budget is not None:
                total_budget += req.remaining_time_budget
                
        return total_budget / len(all_reqs) if len(all_reqs) > 0 else float('inf')

    def _get_gpu_to_active_instances(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Dict[int, List[ModelInstanceState]]:
        """
        获取 GPU id-活跃模型实例 映射 \\
        TP > 1时，每个模型只被首个GPU统计一次 \\
        返回{gpu_id: [activated_model_instance]}
        """
        gpu_to_active_instances: Dict[int, List[ModelInstanceState]] = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        for model_name, instances in model_instance_state_dict.items():
            model_active_instances = {}  # GPU ID -> list of active instances
            for instance in instances:
                if instance.state == ModelState.ACTIVE:
                    for gpu_id in instance.gpu_ids: # TP > 1时可能加入多个
                        if gpu_id not in model_active_instances:
                            model_active_instances[gpu_id] = []
                        model_active_instances[gpu_id].append(instance)
            # assert len(model_active_instances) <= 1
            # TP > 1时，仅保留第一个GPU id标识
            if model_active_instances:
                gpu_id = list(model_active_instances.keys())[0]
                gpu_to_active_instances[gpu_id].extend(model_active_instances[gpu_id])
        return gpu_to_active_instances
    
    def _check_idle_instance_eviction(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        gpu_to_active_instances: Dict[int, List[ModelInstanceState]]
    ) -> Dict[int, List[ModelInstanceState]]:
        """
        检查并清理空闲实例 \\
        返回{gpu_id, [ModelInstanceState]}
        """
        evictable_instances: Dict[int, List[ModelInstanceState]] = dict() # {gpu_id, [ModelInstanceState]}
        if self.first_request_time is None:
            for model_name, queue in model_queues.items():
                if queue.get_last_arrival_time() != float("-inf"):
                    self.first_request_time = queue.get_last_arrival_time() # 首批请求到达后，初始化全局激活时间
                    logger.info(f"first request time: {self.first_request_time}")
                    self.model_last_active_time = {model_name: self.first_request_time for model_name in model_queues.keys()}
                    break
        if self.first_request_time is None: # 无到达请求
            return evictable_instances
        idle_models = set()
        current_time = time.time()
        for model_name, queue in model_queues.items():
            if queue.get_last_arrival_time() > self.first_request_time:
                self.model_last_active_time[model_name] = queue.get_last_arrival_time()
            last_request_time = self.model_last_active_time.get(model_name)
            if last_request_time is not None:
                idle_time = current_time - last_request_time
                has_requests = queue.get_num_waiting_reqs() > 0 or queue.get_num_running_reqs() > 0
                if model_name in [instance.model_name for instances in gpu_to_active_instances.values() for instance in instances]:
                    logger.info(f"Model {model_name} has been idle for {idle_time:.2f} seconds, its own last request or global first request time is {last_request_time}")
                    logger.info(f"Model {model_name} has waiting/running requests: {has_requests}")
                # 长期空闲且无未完成请求
                if idle_time > self.MODEL_IDLE_THRESHOLD and not has_requests:
                    idle_models.add(model_name)
        # 筛选可驱逐实例
        for gpu_id, active_instances in gpu_to_active_instances.items():
            if not active_instances:
                continue
            evictable_instances[gpu_id] = []
            for model_instance in active_instances:
                if model_instance.model_name in idle_models:
                    evictable_instances[gpu_id].append(model_instance)        
        return evictable_instances

    def _calculate_model_violation_stats(
        self,
        model_queues: Dict[str, ModelQueueTracker],
    ) -> Dict[str, Tuple[int, float, int]]:
        """
        统计并获取 模型名-服务违规数据 映射 \\
        返回{model_name: (violated_reqs_count, violation_proportion, total_reqs)}
        """
        # 更新时间窗口
        self.request_history.update_request_history(model_queues)
        model_violation_stats = {}
        for model_name in model_queues.keys():
            model_violation_stats[model_name] = self.request_history.get_model_violation_stats(model_name)
        return model_violation_stats

    def _calculate_gpu_violation_stats(
        self,
        gpu_to_model_mapping: Dict[int, List[str]],
        model_violation_stats: Dict[str, Tuple[int, float, int]]
    ) -> Dict[int, Tuple[int, float, int]]:
        """
        统计并获取 GPU id-服务违规数据 映射 \\
        TP > 1时，每个模型违规数据只被首个GPU统计一次 \\
        返回{gpu_id: (violated_reqs_count, violation_proportion, total_reqs)}
        """
        gpu_violation_stats = {}
        for gpu_id, model_names in gpu_to_model_mapping.items():
            violated_reqs = 0
            total_reqs = 0
            for model_name in model_names:
                if model_name in model_violation_stats:
                    violated_count, _, req_count = model_violation_stats[model_name]
                    violated_reqs += violated_count
                    total_reqs += req_count
            violation_proportion = violated_reqs / total_reqs if total_reqs > 0 else 0.0
            gpu_violation_stats[gpu_id] = (violated_reqs, violation_proportion, total_reqs)
        return gpu_violation_stats
    
    def _calculate_memory_per_request(
        self,
        gpu_to_model_mapping: Dict[int, List[str]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float]
    ) -> Dict[int, Tuple[float, int]]:
        """统计请求平均剩余显存，返回{gpu_id: (memory_per_request, total_reqs)}"""
        # Update model request history
        self.model_request_tracker.update_request_history(model_queues)
        gpu_memory_per_request = {}
        
        for gpu_id, model_names in gpu_to_model_mapping.items():
            total_model_memory = 0
            total_reqs = 0
            
            for model_name in model_names:
                total_model_memory += self.model_weights_info[model_name]["model_size"]
                # 平滑均值（时间窗口内均值）
                total_reqs += self.model_request_tracker.get_model_request_stats(model_name)
            
            total_gpu_memory = self.gpu_mem
            memory_available_for_requests = total_gpu_memory - total_model_memory
            
            if total_reqs == 0:
                memory_per_request = memory_available_for_requests / 1
            else:
                memory_per_request = memory_available_for_requests / total_reqs
                
            gpu_memory_per_request[gpu_id] = (memory_per_request, total_reqs)
        
        return gpu_memory_per_request

    def _is_placement_stable_by_violation(self, gpu_violation_stats: Dict[int, Tuple[int, float, int]]) -> bool:
        """是否违规率不均衡"""
        for high_gpu_id, (_, high_violation, _) in gpu_violation_stats.items():
            for low_gpu_id, (_, low_violation, _) in gpu_violation_stats.items():
                if high_gpu_id != low_gpu_id and high_violation - low_violation > self.VIOLATION_PROPORTION_THRESHOLD:
                    return False
        return True

    def _is_placement_stable_by_memory(self, gpu_memory_per_request: Dict[int, Tuple[float, int]]) -> bool:
        """是否剩余平均显存不均衡"""
        for high_gpu_id, (high_memory, _) in gpu_memory_per_request.items():
            for low_gpu_id, (low_memory, _) in gpu_memory_per_request.items():
                if high_gpu_id != low_gpu_id and high_memory / low_memory > self.MEMORY_PER_REQUEST_RATIO_THRESHOLD:
                    return False
        return True

    def _get_gpu_to_model_mapping(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Dict[int, List[str]]:
        """
        获取 GPU id-活跃模型名称 映射 \\
        TP > 1时，每个模型只被首个GPU统计一次 \\
        返回{gpu_id: [activated_model_name]}
        """
        gpu_to_models = {gpu_id: [] for gpu_id in range(self.num_gpus)}
        for model_name, instances in model_instance_state_dict.items():
            for instance in instances:
                if instance.state == ModelState.ACTIVE:
                    gpu_to_models[instance.gpu_ids[0]].append(model_name)
        return gpu_to_models
    
    def _find_optimal_migrations_by_violation(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        model_violation_stats: Dict[str, Tuple[int, float, int]],
        gpu_to_model_mapping: Dict[int, List[str]]
    ) -> List[Tuple[str, int, int]]:
        """
        根据违规结果确定迁移方案
            将高违规率GPU的低请求模型迁移至低违规率GPU
            返回迁移方案[(model_name, instance_idx, target_gpu_id)]
        """
        gpu_violation_stats = self._calculate_gpu_violation_stats(
            gpu_to_model_mapping, model_violation_stats
        )
        # 逐GPU、模型统计违规数据
        for gpu_id, stats in gpu_violation_stats.items():
            logger.info(f"GPU {gpu_id}, violation_proportion: {stats[1]}, total_reqs: {stats[2]}")
            for model_name in gpu_to_model_mapping[gpu_id]:
                logger.info(f"  {model_name}, violation_proportion: {model_violation_stats[model_name][1]}, total_reqs: {model_violation_stats[model_name][2]}")
        # 违规率不均衡时迁移
        if self._is_placement_stable_by_violation(gpu_violation_stats):
            logger.info("Placement is stable, no migrations needed")
            return []
        # 按违规率降序排列GPU
        sorted_gpus = sorted(
            [(gpu_id, stats[1]) for gpu_id, stats in gpu_violation_stats.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        # 初始化{gpu_id: (model_name, violation_proportion, avg_budget)}
        gpu_models_with_violations: Dict[int, List[Tuple[str, float, float]]] = {}
        for gpu_id in range(self.num_gpus):
            gpu_models_with_violations[gpu_id] = []
            for model_name in gpu_to_model_mapping.get(gpu_id, []):
                if model_name in model_violation_stats:
                    avg_budget = self._calculate_model_avg_remaining_time_budget(model_name, model_queues)
                    gpu_models_with_violations[gpu_id].append(
                        (model_name, model_violation_stats[model_name][1], avg_budget)
                    )
            # 按violation_proportion升序，avg_budget降序排列
            gpu_models_with_violations[gpu_id].sort(key=lambda x: (x[1], -x[2]))
        # 遍历查找迁移方案
        for high_idx in range(len(sorted_gpus)):
            high_gpu_id, high_violation = sorted_gpus[high_idx]
            for low_idx in range(len(sorted_gpus)-1, high_idx, -1):
                low_gpu_id, low_violation = sorted_gpus[low_idx]
                # 获取违规率最高/最低的GPU
                if high_violation - low_violation < self.VIOLATION_PROPORTION_THRESHOLD:
                    break
                if not gpu_models_with_violations[high_gpu_id]:
                    continue
                # 按未完成请求数升序排列，优先迁移剩余请求少的模型
                models_with_req_count = []
                for model_name, violation, avg_budget in gpu_models_with_violations[high_gpu_id]:
                    queue = model_queues.get(model_name)
                    model_reqs = 0
                    if queue:
                        model_reqs = len(queue.running_reqs) + queue.get_num_waiting_reqs()
                    models_with_req_count.append((model_name, violation, avg_budget, model_reqs))
                sorted_models = sorted(models_with_req_count, key=lambda x: x[3])
                # 逐模型尝试迁移
                for model_name, violation, avg_budget, model_reqs in sorted_models:
                    if model_reqs == 0:
                        continue
                    instance_idx = None
                    for instance in model_instance_state_dict.get(model_name):
                        if instance.state == ModelState.ACTIVE and high_gpu_id in instance.gpu_ids:
                            instance_idx = instance.instance_idx
                            break
                    if instance_idx is None:
                        continue
                    # 目标GPU显存不足则放弃迁移
                    model_memory = self.model_weights_info[model_name]["model_size"]
                    if gpu_available_memory[low_gpu_id] < model_memory:
                        continue
                    # 模拟迁移结果
                    simulated_mapping = {gpu_id: models.copy() for gpu_id, models in gpu_to_model_mapping.items()}
                    simulated_mapping[high_gpu_id].remove(model_name)
                    simulated_mapping[low_gpu_id].append(model_name)
                    # 模拟迁移后违规情况
                    simulated_stats = self._calculate_gpu_violation_stats(
                        simulated_mapping, model_violation_stats
                    )
                    logger.info(f"Migration Placement: model {model_name} (with {model_reqs} requests) from {high_gpu_id} to {low_gpu_id}")
                    for gpu_id, stats in simulated_stats.items():
                        logger.info(f"  Simulated GPU {gpu_id}, violation_proportion: {stats[1]}, total_reqs: {stats[2]}")
                    # 迁移后违规率平衡则返回结果
                    if self._is_placement_stable_by_violation(simulated_stats):
                        return [(model_name, instance_idx, low_gpu_id)]
                    logger.info("Migration placement is not stable, continue")
        logger.info("No optimal migrations found")
        return []

    def _count_unstable_pairs(self, gpu_memory_per_request: Dict[int, Tuple[float, int]]) -> int:
        """统计请求平均显存比例相差过大的GPU对"""
        unstable_pairs = 0
        gpu_ids = list(gpu_memory_per_request.keys())
        for i in range(len(gpu_ids)):
            for j in range(i+1, len(gpu_ids)):
                gpu_i = gpu_ids[i]
                gpu_j = gpu_ids[j]
                mem_i, reqs_i = gpu_memory_per_request[gpu_i]
                mem_j, reqs_j = gpu_memory_per_request[gpu_j]
                # 同时空闲记作不稳定
                if reqs_i == 0 and reqs_j == 0:
                    unstable_pairs += 1
                    continue
                if reqs_i == 0:
                    mem_i = float('inf')
                if reqs_j == 0:
                    mem_j = float('inf')
                ratio = max(mem_i, mem_j) / min(mem_i, mem_j) if min(mem_i, mem_j) > 0 else float('inf')
                # 请求平均显存比值超过阈值视为不平衡
                if ratio > self.MEMORY_PER_REQUEST_RATIO_THRESHOLD:
                    unstable_pairs += 1
        return unstable_pairs

    def _get_gpu_active_instance_count(
        self,
        gpu_to_active_instances: Dict[int, List[ModelInstanceState]]
    ) -> Dict[int, int]:
        """获取激活实例数{gpu_id: activated_instance_num}"""
        return {gpu_id: len(instances) for gpu_id, instances in gpu_to_active_instances.items()}

    def _find_optimal_migrations_by_memory(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        gpu_to_model_mapping: Dict[int, List[str]]
    ) -> List[Tuple[str, int, int]]:
        """
        根据显存分布确定迁移方案
            将显存紧张GPU的低请求模型迁移至显存宽松GPU
            返回迁移方案[(model_name, instance_idx, target_gpu_id)]
        """
        self.memory_history.update_memory_history(
            gpu_to_model_mapping, 
            model_queues, 
            gpu_available_memory,
            self.model_weights_info,
            self.gpu_mem
        )
        self.model_request_tracker.update_request_history(model_queues)
        current_memory_per_request = self._calculate_memory_per_request(
            gpu_to_model_mapping, model_queues, gpu_available_memory
        )
        # 逐GPU、模型统计显存数据
        for gpu_id, (memory_per_req, total_reqs) in current_memory_per_request.items():
            logger.info(f"GPU {gpu_id}:")
            logger.info(f"  Memory per request (smoothed requests): {memory_per_req:.2f} GB")
            logger.info(f"  Total requests (smoothed): {total_reqs:.2f}")
            logger.info(f"  Available memory: {gpu_available_memory[gpu_id]:.2f} GB")
            for model_name in gpu_to_model_mapping[gpu_id]:
                model_reqs = self.model_request_tracker.get_model_request_stats(model_name)
                model_memory = self.model_weights_info[model_name]["model_size"]
                logger.info(f"    Model {model_name}:")
                logger.info(f"      Requests (smoothed): {model_reqs:.2f}")
                logger.info(f"      Model memory: {model_memory:.2f} GB")
        # 显存不均衡时迁移
        current_unstable_pairs = self._count_unstable_pairs(current_memory_per_request)
        if current_unstable_pairs == 0:
            logger.info("Memory placement is already stable, no migrations needed")
            return []
        logger.info(f"Current unstable pairs: {current_unstable_pairs}")
        # 获取激活实例数量
        gpu_to_active_instances = self._get_gpu_to_active_instances(model_instance_state_dict)
        gpu_active_instance_count = self._get_gpu_active_instance_count(gpu_to_active_instances)
        # 按请求平均显存升序排列GPU
        sorted_gpus = sorted(
            [(gpu_id, memory, reqs) for gpu_id, (memory, reqs) in current_memory_per_request.items()],
            key=lambda x: x[1]
        )
        # 遍历查找迁移方案
        for low_idx in range(len(sorted_gpus)):
            # 优先从显存最紧张的GPU迁移
            low_gpu_id, low_memory, low_reqs = sorted_gpus[low_idx]
            # 仅有一个实例，无法迁移
            if gpu_active_instance_count[low_gpu_id] == 1:
                continue
            for high_idx in range(len(sorted_gpus)-1, low_idx, -1):
                # 优先迁移到显存最宽松的GPU
                high_gpu_id, high_memory, high_reqs = sorted_gpus[high_idx]
                if high_memory / low_memory <= self.MEMORY_PER_REQUEST_RATIO_THRESHOLD:
                    continue
                # 按未完成请求数升序排列，优先迁移剩余请求少的模型
                models_with_req_count = []
                for model_name in gpu_to_model_mapping[low_gpu_id]:
                    model_reqs = self.model_request_tracker.get_model_request_stats(model_name)
                    models_with_req_count.append((model_name, model_reqs))
                sorted_models = sorted(models_with_req_count, key=lambda x: x[1])
                # 逐模型尝试迁移
                for model_name, model_reqs in sorted_models:
                    if model_reqs == 0:
                        continue
                    instance_idx = None
                    for instance in model_instance_state_dict.get(model_name):
                        if instance.state == ModelState.ACTIVE and low_gpu_id in instance.gpu_ids:
                            instance_idx = instance.instance_idx
                            break
                    if instance_idx is None:
                        continue
                    # 目标GPU显存不足则放弃迁移
                    model_memory = self.model_weights_info[model_name]["model_size"]
                    if gpu_available_memory[high_gpu_id] < model_memory:
                        continue
                    # 模拟迁移结果
                    simulated_mapping = {gpu_id: models.copy() for gpu_id, models in gpu_to_model_mapping.items()}
                    simulated_mapping[low_gpu_id].remove(model_name)
                    simulated_mapping[high_gpu_id].append(model_name)
                    # 模拟迁移后平均显存
                    simulated_memory_per_request = self._calculate_memory_per_request(
                        simulated_mapping, model_queues, gpu_available_memory
                    )
                    simulated_unstable_pairs = self._count_unstable_pairs(simulated_memory_per_request)
                    logger.info(f"Migration Placement: model {model_name} (with {model_reqs:.2f} smoothed requests) from {low_gpu_id} to {high_gpu_id}")
                    logger.info(f"  Current unstable pairs: {current_unstable_pairs}, Simulated unstable pairs: {simulated_unstable_pairs}")
                    for gpu_id, (memory, reqs) in simulated_memory_per_request.items():
                        logger.info(f"  Simulated GPU {gpu_id}, memory_per_request: {memory:.2f} GB, total_reqs: {reqs:.2f}")
                    # 迁移后显存平衡则返回结果
                    if simulated_unstable_pairs < current_unstable_pairs:
                        logger.info(f"Migration reduces unstable pairs from {current_unstable_pairs} to {simulated_unstable_pairs}")
                        return [(model_name, instance_idx, high_gpu_id)]
                    logger.info("Migration does not reduce unstable pairs, continue")
        logger.info("No migrations found that reduce unstable pairs")
        return []

    def _find_optimal_migrations(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float],
        model_violation_stats: Dict[str, Tuple[int, float, int]],
        gpu_to_model_mapping: Dict[int, List[str]]
    ) -> List[Tuple[str, int, int]]:
        """
        根据全局迁移策略查找迁移方案
            回迁移方案[(model_name, instance_idx, target_gpu_id)]
        """
        if self.migrate_policy == "violation":
            return self._find_optimal_migrations_by_violation(
                model_instance_state_dict,
                model_queues,
                gpu_available_memory, 
                model_violation_stats,
                gpu_to_model_mapping
            )
        elif self.migrate_policy == "memory_per_request":
            return self._find_optimal_migrations_by_memory(
                model_instance_state_dict,
                model_queues,
                gpu_available_memory,
                gpu_to_model_mapping
            )

    def _find_inactive_models_with_requests(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> List[str]:
        """获取有未完成请求但无活跃实例的模型名称列表"""
        inactive_models_with_requests = []
        for model_name, queue_tracker in model_queues.items():
            if queue_tracker.get_num_waiting_reqs() > 0 or queue_tracker.get_num_running_reqs() > 0:
                has_active_instance = False
                for instance in model_instance_state_dict.get(model_name):
                    if instance.state == ModelState.ACTIVE:
                        has_active_instance = True
                        break
                if not has_active_instance:
                    inactive_models_with_requests.append(model_name)
        return inactive_models_with_requests
    
    def _place_inactive_model(
        self,
        model_name: str,
        model_memory_required: float,
        gpu_available_memory: Dict[int, float],
        sorted_clusters: List[List[int]],
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Optional[int]:
        """
        放置非活跃模型
            优先选择剩余显存大、请求负载轻松的GPU
            能放置则返回目标gpu_id，否则返回None
        """
        for cluster in sorted_clusters: # 按平均剩余显存降序排列
            gpu_time_budgets = []
            for gpu_id in cluster:
                # 剩余显存不足时跳过
                if gpu_available_memory[gpu_id] < model_memory_required:
                    continue
                # 统计目标GPU平均剩余budget
                total_budget = 0
                total_reqs = 0
                for model_name, queue in model_queues.items():
                    instance_states = model_instance_state_dict.get(model_name)
                    for instance in instance_states:
                        if instance.state == ModelState.ACTIVE and gpu_id in instance.gpu_ids:
                            for req in queue.running_reqs + list(queue.waiting_reqs):
                                total_budget += req.remaining_time_budget if req.remaining_time_budget is not None else 0
                                total_reqs += 1
                avg_budget = total_budget / total_reqs if total_reqs > 0 else float('inf')
                gpu_time_budgets.append((gpu_id, avg_budget))
            # 按平均剩余budget降序排列GPU
            sorted_gpus_by_budget = sorted(gpu_time_budgets, key=lambda x: x[1], reverse=True)
            # 优先加入平均budget最多且能容纳的GPU
            for gpu_id, _ in sorted_gpus_by_budget:
                if gpu_available_memory[gpu_id] >= model_memory_required:
                    gpu_available_memory[gpu_id] -= model_memory_required
                    return gpu_id
        return None
    
    def _prepare_gpu_clusters(
        self, 
        gpu_available_memory: Dict[int, float]
    ) -> List[List[int]]:
        """按剩余显存对GPU分组，每组极差不超过5GB，返回列表按剩余平均显存降序排列"""
        # 按剩余显存降序排列GPU
        sorted_gpus = sorted(
            [(gpu_id, gpu_available_memory[gpu_id]) for gpu_id in range(self.num_gpus)],
            key=lambda x: x[1],
            reverse=True
        )
        gpu_clusters = [] # List[List[int]]
        current_cluster = [] # List[int]
        for gpu_id, mem in sorted_gpus:
            if not current_cluster:
                current_cluster.append(gpu_id) # 划分新cluster
            else:
                max_mem_in_current_cluster = max(gpu_available_memory[gpu_id] for gpu_id in current_cluster)
                if max_mem_in_current_cluster - mem <= 5: # 同一cluster剩余显存极差不超过5GB
                    current_cluster.append(gpu_id)
                else:
                    gpu_clusters.append(current_cluster)
                    current_cluster = [gpu_id]
        if current_cluster:
            gpu_clusters.append(current_cluster)
        # 求cluster平均显存，按平均剩余显存倒序排列cluster
        cluster_avg_memory = []
        for i, cluster in enumerate(gpu_clusters):
            avg_mem = sum(gpu_available_memory[gpu_id] for gpu_id in cluster) / len(cluster)
            cluster_avg_memory.append((i, avg_mem))
        return [gpu_clusters[idx] for idx, _ in sorted(cluster_avg_memory, key=lambda x: x[1], reverse=True)]
    
    def gen_actions(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
    ) -> List[BaseAction]:
        start_total = time.time()
        # {(model_name, instance_id): action}
        model_instance_to_action_dict: Dict[Tuple[str, int], List[BaseAction]] = {}
        # 计算常用数据
        start_precompute = time.time()
        self._update_req_remaining_time_budget(model_queues)
        model_violation_stats = self._calculate_model_violation_stats(model_queues)
        gpu_to_model_mapping = self._get_gpu_to_model_mapping(model_instance_state_dict) # TP > 1时只有首GPU对模型有映射
        gpu_to_active_instances = self._get_gpu_to_active_instances(model_instance_state_dict)
        end_precompute = time.time()
        logger.info(f"Time for preprocessing data structures: {end_precompute - start_precompute:.4f}s")
        # 打印实例信息
        for model_name, model_instances in model_instance_state_dict.items():
            logger.info(f"Model {model_name}:")
            for model_instance in model_instances:
                logger.info(f"  {model_instance}")

        # 1. 清理空闲实例
        start_idle_check = time.time()
        idle_instance_keys = set()
        idle_instances = self._check_idle_instance_eviction(
            model_queues, gpu_to_active_instances
        )
        for gpu_id, evictable_instances in idle_instances.items():
            for model_instance in evictable_instances:
                instance_key = (model_instance.model_name, model_instance.instance_idx)
                idle_instance_keys.add(instance_key)
                model_instance_to_action_dict.setdefault(
                    instance_key, []
                ).append(
                    DeactivateAction(
                        model_name=model_instance.model_name,
                        instance_idx=model_instance.instance_idx,
                        preempt=False,
                        preempt_mode="RECOMPUTE",
                        evict_waiting_requests=True,
                        gpu_id=gpu_id,
                    )
                )
                logger.info(f"ACTION: deactivate {model_instance.model_name}:{model_instance.instance_idx} on GPU {gpu_id}. Reason: idle instance eviction")
        end_idle_check = time.time()
        logger.info(f"Time for idle instance eviction: {end_idle_check - start_idle_check:.4f}s")
        # 打印显存占用
        start_mem_check = time.time()
        gpu_available_memory = {}
        for gpu_id in range(self.num_gpus):
            torch.cuda.set_device(gpu_id)
            gpu_available_memory[gpu_id] = get_available_gpu_memory("cuda", gpu_id)
            logger.info(f"GPU {gpu_id}, mem {gpu_available_memory[gpu_id]}")
        end_mem_check = time.time()
        logger.info(f"Time for checking GPU memory: {end_mem_check - start_mem_check:.4f}s")
        
        # 2. 根据全局策略迁移活跃模型
        start_migrations = time.time()
        migrations = self._find_optimal_migrations(
            model_instance_state_dict,
            model_queues,
            gpu_available_memory,
            model_violation_stats,
            gpu_to_model_mapping
        )
        valid_migrations = []
        for model_name, instance_idx, target_gpu_id in migrations:
            instances = gpu_to_active_instances[target_gpu_id]
            if len(instances) < self.workers_per_gpu: # 迁移后需要可容纳
                valid_migrations.append((model_name, instance_idx, target_gpu_id))
            else:
                logger.info(f"Skipping migration instance {instance_idx} of {model_name} to {target_gpu_id} because target GPU {target_gpu_id} has {len(instances)} active instances")
        migrations = valid_migrations
        for model_name, instance_idx, target_gpu_id in migrations:
            model_memory_required = self.model_weights_info[model_name]["model_size"]
            gpu_available_memory[target_gpu_id] -= model_memory_required
            logger.info(f"PLANNING: migrate instance {instance_idx} of {model_name} to GPU {target_gpu_id}")
        end_migrations = time.time()
        logger.info(f"Time for finding migrations: {end_migrations - start_migrations:.4f}s")
        
        # 3. 激活有请求的休眠模型
        start_inactive = time.time()
        inactive_models_with_requests = self._find_inactive_models_with_requests(model_queues, model_instance_state_dict)
        # 按violation_proportion降序排列
        sorted_inactive_models = sorted(
            inactive_models_with_requests,
            key=lambda model_name: model_violation_stats[model_name][1],
            reverse=True
        )
        sorted_clusters = self._prepare_gpu_clusters(gpu_available_memory)
        activation_plan = {}
        for model_name in sorted_inactive_models:
            model_memory_required = self.model_weights_info[model_name]["model_size"]
            target_gpu = self._place_inactive_model(
                model_name,
                model_memory_required,
                gpu_available_memory,
                sorted_clusters,
                model_queues,
                model_instance_state_dict
            )
            if target_gpu is not None:
                activation_plan[(model_name, target_gpu)] = target_gpu
            else:
                logger.warning(f"No suitable GPU found for inactive model {model_name}")
        end_inactive = time.time()
        for model_name, gpu_id in activation_plan.keys():
            logger.info(f"PLANNING: activate inactive model {model_name} on GPU {gpu_id}. Reason: inactive models but with requests")
        logger.info(f"Time for planning inactive model activation: {end_inactive - start_inactive:.4f}s")
        
        # 4. Apply migrations from step 2
        start_apply_actions = time.time()
        for model_name, instance_idx, target_gpu_id in migrations:
            # Deactivate the instance on source GPU
            source_instance_key = (model_name, instance_idx)
            if source_instance_key not in idle_instance_keys:
                model_instance_to_action_dict.setdefault(source_instance_key, []).append(
                    DeactivateAction(
                        model_name=model_name,
                        instance_idx=instance_idx,
                        preempt=False,
                        preempt_mode="RECOMPUTE",
                        evict_waiting_requests=True,
                        gpu_id=instance_idx,
                    )
                )
                # Activate the instance on target GPU
                target_instance_key = (model_name, target_gpu_id)
                model_instance_to_action_dict.setdefault(target_instance_key, []).append(
                    ActivateAction(
                        model_name=model_name,
                        instance_idx=target_gpu_id,
                        memory_pool_size=self.MEMORY_POOL_BUDGET,
                        gpu_id=target_gpu_id,
                    )
                )
                logger.info(f"ACTION: deactivate {model_name} on GPU {instance_idx} and activate {model_name} on GPU {target_gpu_id}. Reason: migrate model")

        # 5. Activate inactive models
        for (model_name, instance_idx), gpu_id in activation_plan.items():
            instance_key = (model_name, instance_idx)
            model_instance_to_action_dict.setdefault(instance_key, []).append(
                ActivateAction(
                    model_name=model_name,
                    instance_idx=instance_idx,
                    memory_pool_size=self.MEMORY_POOL_BUDGET,
                    gpu_id=gpu_id,
                )
            )
            logger.info(f"ACTION: activate inactive model {model_name} on GPU {gpu_id}. Reason: inactive models but with requests")
        end_apply_actions = time.time()
        logger.info(f"Time for creating action list: {end_apply_actions - start_apply_actions:.4f}s")
        
        end_total = time.time()
        logger.info(f"Total time for global scheduling: {end_total - start_total:.4f}s")

        all_actions = []

        for actions in model_instance_to_action_dict.values():
            sorted_actions = sorted(actions, key=lambda action: 0 if isinstance(action, DeactivateAction) else 1)
            all_actions.extend(sorted_actions)

        return all_actions