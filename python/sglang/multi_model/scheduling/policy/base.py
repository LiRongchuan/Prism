from abc import ABC, abstractmethod
from typing import Dict, List

from sglang.multi_model.scheduling.action import BaseAction
from sglang.multi_model.scheduling.model_queue_tracker import ModelQueueTracker
from sglang.multi_model.scheduling.state import ModelInstanceState


class BasePolicy(ABC):

    @abstractmethod
    def gen_actions(
        self,
        model_queues: Dict[str, ModelQueueTracker],
        model_instance_state_dict: Dict[str, List[ModelInstanceState]],
    ) -> List[BaseAction]:
        pass
