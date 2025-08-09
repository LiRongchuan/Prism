import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import requests

from sglang.multi_model.scheduling.constants import REQUEST_TIMEOUT
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState
from sglang.srt.managers.io_struct import MemoryUsage
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseAction(ABC):
    model_name: str
    instance_idx: Optional[int] = None
    gpu_id: Optional[int] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @abstractmethod
    def execute(
        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ):
        pass

    def __str__(self):
        action_name = self.__class__.__name__.replace("Action", "")
        return f"{action_name} instance {self.instance_idx} of model {self.model_name}"


@dataclasses.dataclass
class ActivateAction(BaseAction):
    memory_pool_size: Optional[int] = None
    gpu_id: Optional[int] = None

    def execute(
        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ):
        try:
            logger.info(
                f"Sending activate request to {self.model_name} (instance {self.instance_idx})"
            )
            response = requests.post(
                f"{url}/activate", json=self.to_dict(), timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                if response.json()["memory_usage"] is None:
                    memory_usage = MemoryUsage(
                        total_used_memory=0.0,
                        model_weights_memory=0.0,
                        memory_pool_memory=0.0,
                        req_to_token_pool_memory=0.0,
                        token_to_kv_pool_memory=0.0,
                    )
                else:
                    memory_usage = MemoryUsage.from_dict(
                        response.json()["memory_usage"]
                    )
                model_instance_state_dict[self.model_name][
                    self.instance_idx
                ].on_activate(memory_usage, gpu_id=self.gpu_id)
                success = True
            else:
                success = False
                response.raise_for_status()
        except Exception as e:
            logger.error(
                f"Failed to activate model {self.model_name} (instance {self.instance_idx}): {get_exception_traceback()}"
            )
            raise e
        return success


@dataclasses.dataclass
class DeactivateAction(BaseAction):
    preempt: bool = True
    preempt_mode: str = "RECOMPUTE"
    evict_waiting_requests: bool = False

    def execute(
        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ):
        try:
            logger.info(
                f"Sending deactivate request to {self.model_name} (instance {self.instance_idx})"
            )
            response = requests.post(
                f"{url}/deactivate", json=self.to_dict(), timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                if response.json()["memory_usage"] is None:
                    memory_usage = MemoryUsage(
                        total_used_memory=0.0,
                        model_weights_memory=0.0,
                        memory_pool_memory=0.0,
                        req_to_token_pool_memory=0.0,
                        token_to_kv_pool_memory=0.0,
                    )
                else:
                    memory_usage = MemoryUsage.from_dict(
                        response.json()["memory_usage"]
                    )
                model_instance_state_dict[self.model_name][
                    self.instance_idx
                ].on_deactivate(memory_usage)
                success = True
            else:
                success = False
                response.raise_for_status()
        except Exception as e:
            logger.error(
                f"Failed to deactivate model {self.model_name} (instance {self.instance_idx}): {get_exception_traceback()}"
            )
            raise e
        return success


@dataclasses.dataclass
class ResizeAction(BaseAction):
    memory_pool_size: Optional[int] = None

    def execute(
        self, url: str, model_instance_state_dict: Dict[str, List[ModelInstanceState]]
    ):
        try:
            logger.info(
                f"Sending resize request to {self.model_name} (instance {self.instance_idx})"
            )
            response = requests.post(
                f"{url}/resize_mem_pool", json=self.to_dict(), timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                success = True
            else:
                success = False
                response.raise_for_status()
        except Exception as e:
            logger.error(
                f"Failed to resize model {self.model_name} (instance {self.instance_idx}): {get_exception_traceback()}"
            )
            raise e
        return success
