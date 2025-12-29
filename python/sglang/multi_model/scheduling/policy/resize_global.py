import time
import torch
import logging
from typing import Dict, List, Tuple, Optional
from sglang.multi_model.scheduling.action import (
    BaseAction,
    ActivateAction,
    DeactivateAction,
    ResizeAction
)
from sglang.srt.utils import get_available_gpu_memory
from sglang.multi_model.scheduling.policy.base_global import GlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.policy.tracker import RequestViolationTracker

logger = logging.getLogger(__name__)

"""
保留当前ModelInstanceState定义
TP > 1时无法进行迁移操作，所有activate/deactivate保证GPU不变
"""

class ResizeGlobalPolicy(GlobalPolicy):
    def __init__(
        self,
        num_gpus: int,
        gpu_mem: float,
        model_weights_info: Dict[str, Dict[str, float]], # Dict[model_name, Dict[model_size, weight_size]]模型原始信息
        workers_per_gpu: int, # enable_worker_pool时1，否则-1
    ):
        self.num_rank0_gpus = num_gpus # 所有作为rank0的GPU
        self.gpu_mem = gpu_mem # 内存大小必须相同
        self.model_weights_info = model_weights_info
        self.workers_per_gpu = workers_per_gpu
        self.enable_worker_pool = (workers_per_gpu != -1)
        self.first_request_time = None
        self.model_last_active_time = {}
        self.all_gpus = None
        self.check_resize = False
        
        self.violation_tracker = RequestViolationTracker(window_size_seconds=3)

        self.MEMORY_POOL_BUDGET = 5
        self.RESIZE_THRESHOLD = 2

    def _get_all_gpus(self, model_instance_state_dict: Dict[str, List[ModelInstanceState]]) -> int:
        """获取全部GPU id"""
        all_gpus = set()
        for _, instances in model_instance_state_dict.items():
            for instance in instances:
                for gpu_id in instance.gpu_ids:
                    all_gpus.add(gpu_id)
        self.all_gpus = list(all_gpus)
        logger.info(f"All used GPUs: {self.all_gpus}")

    def _update_req_remaining_time_budget(
        self,
        model_queues: Dict[str, ModelQueueTracker]
    ):
        """更新队列中所有请求的剩余时间预算"""
        # budget = slo - (current - arrival)
        current_time = time.time()
        for _, queue in model_queues.items():
            for req in queue.running_reqs + list(queue.waiting_reqs):
                if req.arrival_time is None or req.slo is None: continue
                elapsed_time = current_time - req.arrival_time
                req.remaining_time_budget = req.slo - elapsed_time

    def _calculate_model_violation_stats(
        self,
        model_queues: Dict[str, ModelQueueTracker],
    ) -> Dict[str, Tuple[int, float, int]]:
        """
        统计并获取 模型名-服务违规数据 映射 \\
        返回{model_name: (violated_reqs_count, violation_proportion, total_reqs)}
        """
        # 更新时间窗口
        self.violation_tracker.update_request_history(model_queues)
        model_violation_stats = {}
        for model_name in model_queues.keys():
            model_violation_stats[model_name] = self.violation_tracker.get_model_violation_stats(model_name)
        return model_violation_stats

    def _get_gpu_to_model_mapping(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Dict[int, List[str]]:
        """
        获取 GPU id-活跃模型名称 映射 \\
        TP > 1的实例只被rank0 GPU统计一次 \\
        返回{gpu_id: [activated_model_name]}
        """
        gpu_to_models = {gpu_id: [] for gpu_id in self.all_gpus}
        for model_name, instances in model_instance_state_dict.items():
            for instance in instances: # 有多个实例可能被加入多次
                if instance.state == ModelState.ACTIVE:
                    gpu_to_models[instance.gpu_ids[0]].append(model_name)
        return gpu_to_models

    def _get_gpu_to_instances_mapping(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> Dict[int, List[ModelInstanceState]]:
        """
        获取 GPU id-活跃模型实例 映射 \\
        TP > 1的模型只被rank0 GPU统计一次 \\
        返回{gpu_id: [activated_model_instance]}
        """
        gpu_to_active_instances = {gpu_id: [] for gpu_id in self.all_gpus}
        for _, instances in model_instance_state_dict.items():
            for instance in instances: # 有多个实例可能被加入多次
                if instance.state == ModelState.ACTIVE:
                    gpu_to_active_instances[instance.gpu_ids[0]].append(instance)
        return gpu_to_active_instances
    
    def _find_inactive_models_with_requests(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ) -> List[str]:
        """获取有未完成请求但无活跃实例的模型名称列表"""
        inactive_models_with_requests = []
        for model_name, queue in model_queues.items():
            if queue.get_num_unfinished_reqs() > 0:
                has_active_instance = False
                for instance in model_instance_state_dict.get(model_name):
                    if instance.state == ModelState.ACTIVE:
                        has_active_instance = True
                        break
                if not has_active_instance:
                    inactive_models_with_requests.append(model_name)
        return inactive_models_with_requests
    
    def _prepare_gpu_clusters(
        self, 
        gpu_available_memory: Dict[int, float]
    ) -> List[List[int]]:
        """按剩余显存对GPU分组，每组极差不超过5GB，返回列表按剩余平均显存降序排列"""
        # 按剩余显存降序排列GPU
        sorted_gpus = sorted(
            [(gpu_id, gpu_available_memory[gpu_id]) for gpu_id in self.all_gpus],
            key=lambda x: x[1],
            reverse=True
        )
        gpu_clusters = [] # List[List[int]]
        current_cluster = [] # List[int]
        for gpu_id, mem in sorted_gpus:
            if not current_cluster: current_cluster.append(gpu_id) # 划分新cluster
            else:
                max_mem_in_current_cluster = max(gpu_available_memory[gpu_id] for gpu_id in current_cluster)
                # 同一cluster剩余显存极差不超过5GB
                if max_mem_in_current_cluster - mem <= 5:
                    current_cluster.append(gpu_id)
                else:
                    gpu_clusters.append(current_cluster)
                    current_cluster = [gpu_id]
        if current_cluster: gpu_clusters.append(current_cluster)
        # 求cluster平均显存，按平均剩余显存倒序排列cluster
        cluster_avg_memory = []
        for i, cluster in enumerate(gpu_clusters):
            avg_mem = sum(gpu_available_memory[gpu_id] for gpu_id in cluster) / len(cluster)
            cluster_avg_memory.append((i, avg_mem))
        return [gpu_clusters[idx] for idx, _ in sorted(cluster_avg_memory, key=lambda x: x[1], reverse=True)]    
           
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
                if gpu_available_memory[gpu_id] < model_memory_required: continue
                # 统计目标GPU平均剩余budget
                total_budget, total_reqs = 0, 0
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
    
    def _find_resize_actions(
        self,
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        gpu_to_instance_mapping: Dict[int, List[ModelInstanceState]],
        model_queues: Dict[str, ModelQueueTracker],
        gpu_available_memory: Dict[int, float]
    ) -> Dict[Tuple[str, int], float]:
        instance_target_memory: Dict[Tuple[str, int], float] = {}
        for gpu_id, instances in gpu_to_instance_mapping.items():
            workloads = []
            available_memory = gpu_available_memory[gpu_id]
            total_weight_memory = 0
            # 计算工作负载，统计显存占用
            for instance in instances:
                model_name = instance.model_name
                # 细化负载计算，使用token数
                workload = model_queues[model_name].get_num_unfinished_tokens()
                # NOTE: 假设平分负载
                workload /= sum(
                    1 if model_instance.state == ModelState.ACTIVE else 0
                    for model_instance in model_instance_state_dict[model_name]
                )
                workloads.append(workload)
                available_memory += instance.memory_usage.token_to_kv_pool_memory # 假设单实例
                total_weight_memory += instance.memory_usage.model_weights_memory
            # 两次显存分配
            total_workload = sum(workloads)
            available_memory = min(available_memory, self.gpu_mem - total_weight_memory) # 避免分配溢出
            if total_workload == 0: continue
            # 分配 1
            for i, instance in enumerate(instances): # 避免缩减后运行显存不足导致无效
                current_memory_usage = instance.memory_usage.token_to_kv_pool_memory
                if available_memory * workloads[i] / total_workload < current_memory_usage:
                    total_workload -= workloads[i]
                    available_memory -= current_memory_usage
            # 分配 2
            for i, instance in enumerate(instances): # 生成计划
                key = (instance.model_name, instance.instance_idx)
                current_memory_usage = instance.memory_usage.token_to_kv_pool_memory
                target_memory = min(
                    available_memory * workloads[i] / total_workload,
                    instance_target_memory.get(key, available_memory)
                )
                target_memory = max(target_memory, current_memory_usage)
                logger.info(f"Model {key[0]} Instance{key[1]}, taking workload {workloads[i]} of {total_workload}, "
                            f"current boundary: {instance.memory_pool_size:.2f}, memory usage: {current_memory_usage:.2f}, "
                            f"target boundary: {target_memory:.2f}")
                if abs(target_memory - instance.memory_pool_size) > self.RESIZE_THRESHOLD:
                    logger.info(f"ACTION: Resize {key[0]} instance{key[1]} {instance.memory_pool_size:.2f} to {target_memory:.2f}. Reason: to balance workload")
                    instance_target_memory[key] = target_memory
                    instance.memory_pool_size = target_memory
        return instance_target_memory
      
    def gen_actions(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
    ) -> List[BaseAction]:
        start_total = time.time()
        # {(model_name, instance_id): action}，worker pool时为{(model_name, gpu_id): action}
        model_instance_to_action_dict: Dict[Tuple[str, int], List[BaseAction]] = {}
        for model_queue in model_queues.values(): logger.info(f"{model_queue}") 
        # 0. 计算常用数据
        start_precompute = time.time()
        if self.all_gpus is None: self._get_all_gpus(model_instance_state_dict)
        self._update_req_remaining_time_budget(model_queues)
        model_violation_stats = self._calculate_model_violation_stats(model_queues)
        gpu_to_model_mapping = self._get_gpu_to_model_mapping(model_instance_state_dict)
        gpu_to_active_instances = self._get_gpu_to_instances_mapping(model_instance_state_dict)
        end_precompute = time.time()
        logger.info(f"Time for preprocessing data structures: {end_precompute - start_precompute:.4f}s")
        # 打印实例信息，更新显存占用
        for model_name, instances in model_instance_state_dict.items():
            logger.info(f"Model {model_name}:")
            for instance in instances:
                logger.info(f"  {instance}")
        # 打印显存占用
        start_mem_check = time.time()
        gpu_available_memory = {}
        for gpu_id in self.all_gpus:
            torch.cuda.set_device(gpu_id)
            gpu_available_memory[gpu_id] = get_available_gpu_memory("cuda", gpu_id)
            logger.info(f"GPU {gpu_id}, mem {gpu_available_memory[gpu_id]}")
        end_mem_check = time.time()
        logger.info(f"Time for checking GPU memory: {end_mem_check - start_mem_check:.4f}s")
        
        # 2. 激活有请求的休眠模型
        start_inactive = time.time()
        inactive_models_with_requests = self._find_inactive_models_with_requests(model_queues, model_instance_state_dict)
        # 按违规率降序排列
        sorted_inactive_models = sorted(
            inactive_models_with_requests,
            key=lambda model_name: model_violation_stats[model_name][1],
            reverse=True
        )
        sorted_clusters = self._prepare_gpu_clusters(gpu_available_memory)
        activation_plan = {}
        for model_name in sorted_inactive_models:
            model_memory_required = self.model_weights_info[model_name]["model_size"]
            # 只对TP = 1的实例选择GPU
            last_gpu_ids = model_instance_state_dict.get(model_name)[0].gpu_ids
            target_gpu = self._place_inactive_model(
                model_name,
                model_memory_required,
                gpu_available_memory,
                sorted_clusters,
                model_queues,
                model_instance_state_dict
            ) if len(last_gpu_ids) == 1 else last_gpu_ids[0]
            if target_gpu is not None:
                activation_plan[model_name] = target_gpu
                gpu_to_active_instances[target_gpu].append(None)
            else: logger.warning(f"No suitable GPU found for inactive model {model_name}")
        end_inactive = time.time()
        for model_name, gpu_id in activation_plan.items():
            logger.info(f"PLANNING: activate inactive model {model_name} on GPU {gpu_id}. Reason: inactive models but with requests")
        logger.info(f"Time for planning inactive model activation: {end_inactive - start_inactive:.4f}s")
        
        # 3. 生成激活指令
        start_apply_actions = time.time()
        for model_name, gpu_id in activation_plan.items():
            instance_idx = gpu_id if self.enable_worker_pool else 0 # 非worker pool时instance编号定为0
            instance_key = (model_name, instance_idx)
            model_instance_to_action_dict.setdefault(instance_key, []).append(
                ActivateAction(
                    model_name=model_name,
                    instance_idx=instance_idx,
                    gpu_id=gpu_id,
                    memory_pool_size=self.MEMORY_POOL_BUDGET
                )
            )
            logger.info(f"ACTION: activate inactive model {model_name}: {instance_idx} on GPU {gpu_id}. Reason: inactive models but with requests")
            self.check_resize = False
        end_apply_actions = time.time()
        logger.info(f"Time for creating action list: {end_apply_actions - start_apply_actions:.4f}s")
        
        # 4. 弹性分配显存
        start_alloc = time.time()
        instance_target_memory = self._find_resize_actions(model_instance_state_dict, gpu_to_active_instances, model_queues, gpu_available_memory)        
        for key, target_size in instance_target_memory.items():
            model_instance_to_action_dict.setdefault(key, []).append(
                ResizeAction(
                    model_name=key[0],
                    instance_idx=key[1],
                    memory_pool_size=target_size
                )
            )
        end_alloc = time.time()
        logger.info(f"Time for checking resize: {end_alloc - start_alloc:.4f}s")

        end_total = time.time()
        logger.info(f"Total time for global scheduling: {end_total - start_total:.4f}s")
        all_actions = []
        for actions in model_instance_to_action_dict.values():
            # 先清理再激活
            sorted_actions = sorted(actions, key=lambda action: 0 if isinstance(action, DeactivateAction) else 1)
            all_actions.extend(sorted_actions)
        return all_actions