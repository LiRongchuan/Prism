from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ScheduleStats:
    time: str
    mod_req_per_sec: Dict[str, int]
    mod_on_gpu: Dict[str, List[int]]
    mod_priority: Dict[str, float]
    mod_q_waiting_len: Dict[str, int]
    mod_total_req_len: Dict[str, int]
    mod_req_vio: Optional[Dict[str, int]] = None
    gpu_metrics: Optional[Dict[str, Dict[str, any]]] = None

    @staticmethod
    def from_dict(data: dict) -> "ScheduleStats":
        # If time needs to be converted back to datetime:
        data["time"] = datetime.fromisoformat(data["time"])
        return ScheduleStats(**data)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  time={self.time},\n"
            f"  mod_req_per_sec={self.mod_req_per_sec},\n"
            f"  mod_on_gpu={self.mod_on_gpu},\n"
            f"  mod_priority={self.mod_priority},\n"
            f"  mod_q_waiting_len={self.mod_q_waiting_len},\n"
            f"  mod_total_req_len={self.mod_total_req_len},\n"
            f"  gpu_metrics={self.gpu_metrics}\n"
            ")\n"
        )
