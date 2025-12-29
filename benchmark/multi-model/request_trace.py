import os
import json
import pickle
import dataclasses
import numpy as np
from typing import Optional
from collections import Counter
from collections import defaultdict

@dataclasses.dataclass
class Request:
    """ 单个请求对象 """
    req_id: str
    prompt: str
    prompt_len: int
    output_len: int
    arrival_time: float
    model: str
    slo: Optional[float] = None
    slo_ttft: Optional[float] = None
    slo_tpot: Optional[float] = None


@dataclasses.dataclass
class TraceConfig:
    micro_benchmark: bool = False
    e2e_benchmark: bool = False
    synthetic: bool = False
    replication: int = 1
    time_scale: float = 1

    model_paths: list[str] = dataclasses.field(default_factory=list)
    req_rate: float = 2
    duration: float = 60
    workload_scale: float = 1
    rate_scale: int = 1
    input_range: tuple[int, int] = (8, 512)
    output_range: tuple[int, int] = (8, 512)
    seed: int = 42
    tokenizer_paths: Optional[list[str]] = None

    # 生成请求间隔
    alpha: Optional[float] = None
    cv: Optional[float] = None
    sequential: Optional[bool] = False
    start_index: Optional[int] = 0
    
    # on-off模式选项
    on_off_cycle_len: Optional[int] = -1 # the time of each cycle (on+off)
    off_ratio: Optional[float] = 0.5 # the ratio of off time in each cycle
    on_off_model_percentage: Optional[float] = 0

    # slo
    slo: Optional[float] = 10
    slo_ttft: Optional[float] = 5
    slo_tpot: Optional[float] = 0.05
    tpot_slo_scale: Optional[float] = 1
    ttft_slo_scale: Optional[float] = 1

    # for baseline slo
    dedicated_model: Optional[str] = None # model_1...

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = 0.1
        if self.cv is None:
            self.cv = 1
        if self.tokenizer_paths is None:
            self.tokenizer_paths = self.model_paths


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_synthetic_reqs(
    config: TraceConfig,
    print_model_req_rate: bool = False,
) -> list[Request]:
    """ 生成合成请求 """
    np.random.seed(config.seed)
    num_reqs = int(config.req_rate * config.duration)
    if config.sequential:
        # 顺序平均负载分配
        num_models = len(config.model_paths)
        model_indices = np.repeat(np.arange(num_models), num_reqs // num_models)
        remaining_reqs = num_reqs % num_models
        if remaining_reqs > 0:
            model_indices = np.concatenate([model_indices, [num_models - 1] * remaining_reqs])
    else:
        # 请求负载指数分布
        probs = np.random.power(config.alpha, num_reqs)
        num_models = len(config.model_paths)
        model_indices = (probs * num_models).astype(int)
    # 生成请求间隔
    shape = 1 / (config.cv**2)
    scale = config.cv**2 / config.req_rate
    intervals = np.random.gamma(shape, scale, num_reqs)
    timestamps = np.cumsum(intervals)
    # 生成输入/输出负载
    input_lens = np.random.randint(*config.input_range, num_reqs) * config.workload_scale
    output_lens = np.random.randint(*config.output_range, num_reqs) * config.workload_scale
    if print_model_req_rate:
        print("request_rates before on_off")
        for i in range(num_models - 1, -1, -1):
            print(f"{config.model_paths[i]}: {np.sum(model_indices == i) / config.duration:.2f}") # 初始速率
        print("\n")
    # on-off选项下，尾部模型为on-off模式
    if config.on_off_model_percentage > 0:
        on_off_models = config.model_paths[-int(len(config.model_paths) * config.on_off_model_percentage):]
    else: on_off_models = []
    if print_model_req_rate: print(f"on_off_models: {on_off_models}")
    # 生成逻辑
    requests = []
    model_num_requests = Counter()
    for i in range(num_reqs):
        model = config.model_paths[model_indices[i]]
        if config.on_off_cycle_len > 0:
            if model in on_off_models:
                arrival_time = timestamps[i]
                phase_shift = (on_off_models.index(model) / max((len(on_off_models) // 2), 2)) * config.on_off_cycle_len
                cycle_time = (arrival_time + phase_shift) % config.on_off_cycle_len
                if cycle_time < config.on_off_cycle_len * config.off_ratio:
                    continue
        req = Request(
            req_id=str(i),
            prompt=dummy_prompt(input_lens[i]),
            prompt_len=int(input_lens[i]),
            output_len=int(output_lens[i]),
            arrival_time=timestamps[i],
            model=model,
            slo=config.slo,
            slo_ttft=config.slo_ttft,
            slo_tpot=config.slo_tpot,
        )
        requests.append(req)
        model_num_requests[model] += 1
    if print_model_req_rate:
        print("actual request rate each model:")
        for model, num_reqs in model_num_requests.items():
            print(f"{model}: {num_reqs / config.duration:.2f}")
    return requests


def generate_synthetic_reqs_sequential(
    config: TraceConfig,
) -> list[Request]:
    return None


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Request":
            return Request
        return super().find_class(module, name)


class RealWorldTrace:
    """ 从pkl读取请求trace """
    def __init__(self, pkl_file_path: Optional[str] = None):
        if pkl_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.pkl_file_path = os.path.join(current_dir, "real_trace.pkl")
        else:
            self.pkl_file_path = pkl_file_path
        self.requests = []
        self._load_requests()

    def _load_requests(self):
        if os.path.exists(self.pkl_file_path):
            with open(self.pkl_file_path, "rb") as pkl_file:
                obj = CustomUnpickler(pkl_file).load()
            adapter_dirs, self.requests = obj[0], obj[1]
        else:
            raise FileNotFoundError(f"File {self.pkl_file_path} not found")

    def generate_real_reqs(self, config: TraceConfig, model_ids: list[int]):
        # 选择LoRA rank
        all_ranks = [0, 10, 2, 5]
        selected_ranks = [all_ranks[i] for i in model_ids]
        model_list = [f"model_{i+1}" for i in range(len(model_ids))]
        rank_model_mapping = dict(zip(selected_ranks, model_list))
        # 定义slo
        ttft_slos = {"model_1": 0.03867427111, "model_2": 0.03222719193, "model_3": 0.03141047955, "model_4": 0.05097425461}
        tpot_slos = {"model_1": 6.938047111, "model_2": 6.70813661, "model_3": 6.111749992, "model_4": 10.81588957}
        model_ttft_slo = {k: v * config.ttft_slo_scale for k, v in ttft_slos.items()}
        model_tpot_slo = {k: v * config.tpot_slo_scale / 1000 for k, v in tpot_slos.items()}
        # 筛选请求
        requests = []
        count = 0
        arrival_offset = 0
        for req in self.requests:
            try:
                rank = int(req.adapter_dir.split("-")[-1])
                if rank not in rank_model_mapping.keys(): continue
                model = rank_model_mapping[rank]
                prompt_len = int(config.workload_scale * req.prompt_len)
                output_len = int(config.workload_scale * req.output_len)
                if arrival_offset == 0 and len(model_ids) == 1: # 整体提前至0.2秒时开始
                    arrival_offset = req.req_time - 0.2
                for _ in range(config.rate_scale):
                    requests.append(Request(
                        req_id=str(count),
                        prompt=req.prompt if config.workload_scale == 1 else dummy_prompt(prompt_len),
                        prompt_len=prompt_len,
                        output_len=output_len,
                        arrival_time=req.req_time - arrival_offset,
                        model=model,
                        slo=model_ttft_slo[model],
                        slo_ttft=model_ttft_slo[model],
                        slo_tpot=model_tpot_slo[model],
                    ))
                    count += 1
            except Exception as e:
                print(f"ERROR in processing request: {e}")
                continue
        return requests
    
class SharedgptTrace:
    def __init__(self, json_file_path: Optional[str] = None):
        if json_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.json_file_path = os.path.join(current_dir, "sharedgpt", "sharedgpt_n3_rate_2.json")
        else:
            self.json_file_path = json_file_path
        if json_file_path is not None:
            self._load_requests()
    
    def _load_requests(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as f:
                trace = json.load(f)
                self.arrivals = trace.get("arrivals", [])
                self.requests = trace.get("requests", [])
                
    def get_reqs(self, config: TraceConfig, model_ids: list[int]):
        model_name_mapping = {f"llm-{id}": f"model_{i+1}" for i, id in enumerate(model_ids)}
        # 定义slo
        ttft_slos = {"model_1": 0.1, "model_2": 0.1,}
        tpot_slos = {"model_1": 10, "model_2": 10,}
        model_ttft_slo = {k: v * config.ttft_slo_scale for k, v in ttft_slos.items()}
        model_tpot_slo = {k: v * config.tpot_slo_scale / 1000 for k, v in tpot_slos.items()}
        # 筛选请求
        requests = []
        count = 0
        arrival_offset = self.arrivals[0] - 0.2 if len(model_ids) == 1 else 0
        for i, req in enumerate(self.requests):
            try:
                # model = model_name_mapping[req["model_name"]]
                model = model_name_mapping.get(req["model_name"], None)
                if model is None: continue
                prompt_len = int(config.workload_scale * req["data"][-2])
                output_len = int(config.workload_scale * req["data"][-1])
                for _ in range(config.rate_scale):
                    requests.append(Request(
                        req_id=str(count),
                        prompt=dummy_prompt(prompt_len),
                        prompt_len=prompt_len,
                        output_len=output_len,
                        arrival_time=self.arrivals[i] - arrival_offset,
                        model=model,
                        slo=model_ttft_slo[model],
                        slo_ttft=model_ttft_slo[model],
                        slo_tpot=model_tpot_slo[model],
                    ))
                    count += 1
            except Exception as e:
                print(f"ERROR in processing request: {e}")
                continue
        return requests