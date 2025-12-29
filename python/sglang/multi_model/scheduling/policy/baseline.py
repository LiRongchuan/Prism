import time
import torch
import logging
from typing import Dict, List, Tuple, Optional
from sglang.multi_model.scheduling.action import (
    BaseAction,
    ActivateAction
)
from sglang.srt.utils import get_available_gpu_memory
from sglang.multi_model.scheduling.policy.base_global import GlobalPolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker

logger = logging.getLogger(__name__)


class BaselinePolicy(GlobalPolicy):
    def __init__(
        self,
        num_gpus: int,
        gpu_mem: float,
        model_weights_info: Dict[str, Dict[str, float]], # Dict[model_name, Dict[model_size, weight_size]]模型原始信息
        workers_per_gpu: int, # enable_worker_pool时1，否则-1
    ):
        self.enable_worker_pool = (workers_per_gpu != -1)
        self.first_request_time = None
        self.model_last_active_time = {}
        self.all_gpus = None

        self.MEMORY_POOL_BUDGET = 6

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

    def _get_all_gpus(self, model_instance_state_dict: Dict[str, List[ModelInstanceState]]) -> int:
        """获取全部GPU id"""
        all_gpus = set()
        for _, instances in model_instance_state_dict.items():
            for instance in instances:
                for gpu_id in instance.gpu_ids:
                    all_gpus.add(gpu_id)
        self.all_gpus = list(all_gpus)
        logger.info(f"All used GPUs: {self.all_gpus}")

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
        end_precompute = time.time()
        logger.info(f"Time for preprocessing data structures: {end_precompute - start_precompute:.4f}s")
        # 打印实例信息
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
        
        # 1. 激活有请求的休眠模型
        start_inactive = time.time()
        inactive_models_with_requests = self._find_inactive_models_with_requests(model_queues, model_instance_state_dict)
        activation_plan = {}
        for model_name in inactive_models_with_requests:
            target_gpu = model_instance_state_dict.get(model_name)[0].gpu_ids[0]
            if target_gpu is not None: activation_plan[model_name] = target_gpu
            else: logger.warning(f"No suitable GPU found for inactive model {model_name}")
        end_inactive = time.time()
        for model_name, gpu_id in activation_plan.items():
            logger.info(f"PLANNING: activate inactive model {model_name} on GPU {gpu_id}. Reason: inactive models but with requests")
        logger.info(f"Time for planning inactive model activation: {end_inactive - start_inactive:.4f}s")
        
        # 2. 生成激活指令
        start_apply_actions = time.time()
        for model_name, gpu_id in activation_plan.items():
            instance_idx = gpu_id if self.enable_worker_pool else 0 # 非worker pool时instance编号从0开始
            instance_key = (model_name, instance_idx)
            model_instance_to_action_dict.setdefault(instance_key, []).append(
                ActivateAction(
                    model_name=model_name,
                    instance_idx=instance_idx,
                    memory_pool_size=self.MEMORY_POOL_BUDGET,
                    gpu_id=gpu_id,
                )
            )
            logger.info(f"ACTION: activate inactive model {model_name}: {instance_idx} on GPU {gpu_id}. Reason: inactive models but with requests")
        end_apply_actions = time.time()
        logger.info(f"Time for creating action list: {end_apply_actions - start_apply_actions:.4f}s")
        
        end_total = time.time()
        logger.info(f"Total time for global scheduling: {end_total - start_total:.4f}s")
        all_actions = []
        for actions in model_instance_to_action_dict.values(): all_actions.extend(actions)
        return all_actions