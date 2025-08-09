import logging
from abc import abstractmethod
from typing import Dict, List

from sglang.multi_model.scheduling.action import BaseAction
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker, Req
from sglang.multi_model.scheduling.policy.base import BasePolicy
from sglang.multi_model.scheduling.state import ModelInstanceState, ModelState

logger = logging.getLogger(__name__)


class GlobalPolicy(BasePolicy):
    @abstractmethod
    def gen_actions(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
        simulated_gpu_queues: Dict[int, List[Req]],
    ) -> List[BaseAction]:
        pass
