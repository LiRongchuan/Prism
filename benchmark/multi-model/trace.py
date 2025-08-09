import dataclasses
import os
import pickle
from collections import Counter
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Request:
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
    # for micro benchmark and e2e benchmark
    replication: int = 1
    time_scale: float = 1

    model_paths: list[str] = dataclasses.field(default_factory=list)
    req_rate: float = 2
    duration: float = 60
    input_range: tuple[int, int] = (8, 512)
    output_range: tuple[int, int] = (8, 512)
    seed: int = 42
    tokenizer_paths: Optional[list[str]] = None

    # for synthetic requests
    alpha: Optional[float] = None
    cv: Optional[float] = None
    sequential: Optional[bool] = False
    start_index: Optional[int] = 0
    # on off
    on_off_cycle_len: Optional[int] = -1  # the time of each cycle (on+off)
    off_ratio: Optional[float] = 0.5  # the ratio of off time in each cycle
    on_off_model_percentage: Optional[float] = (
        0  # the percentage of models that follow the on_off pattern
    )

    # slo
    slo: Optional[float] = 10
    slo_ttft: Optional[float] = -1
    slo_tpot: Optional[float] = -1
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
    np.random.seed(config.seed)

    num_reqs = int(config.req_rate * config.duration)

    # generate model path and tokenizer path
    if config.sequential:
        # generate model path and tokenizer path sequentially, i.e. first num_reqs // num_models reqs use model 0, next num_reqs // num_models reqs use model 1, etc.
        num_models = len(config.model_paths)
        model_indices = np.repeat(np.arange(num_models), num_reqs // num_models)

        remaining_reqs = num_reqs % num_models
        if remaining_reqs > 0:
            # assign the remaining reqs to the last model
            model_indices = np.concatenate(
                [model_indices, [num_models - 1] * remaining_reqs]
            )
    else:
        probs = np.random.power(config.alpha, num_reqs)
        num_models = len(config.model_paths)
        model_indices = (probs * num_models).astype(int)

    tokenizer_indices = model_indices

    # generate timestamps, with gamma distributed intervals
    # cv is the coefficient of variation, which is the ratio of the standard deviation to
    # the mean of the gamma distribution
    shape = 1 / (config.cv**2)
    scale = config.cv**2 / config.req_rate
    intervals = np.random.gamma(shape, scale, num_reqs)
    timestamps = np.cumsum(intervals)

    # generate input and output lengths
    input_lens = np.random.randint(*config.input_range, num_reqs)
    output_lens = np.random.randint(*config.output_range, num_reqs)

    if print_model_req_rate:
        print("request_rates before on_off")
        for i in range(num_models - 1, -1, -1):
            print(
                f"{config.model_paths[i]}: {np.sum(model_indices == i) / config.duration:.2f}"
            )
        print("\n")

    # for on_off pattern, default on_off models is the last on_off_model_percentage models
    if config.on_off_model_percentage > 0:
        on_off_models = config.model_paths[
            -int(len(config.model_paths) * config.on_off_model_percentage) :
        ]
    else:
        on_off_models = []

    if print_model_req_rate:
        print(f"on_off_models: {on_off_models}")

    requests = []
    model_num_requests = Counter()
    for i in range(num_reqs):
        model = config.model_paths[model_indices[i]]

        # on_off pattern enabled
        if config.on_off_cycle_len > 0:
            if model in on_off_models:
                arrival_time = timestamps[i]
                # add a phase_shift to control the start time of on_cycle for different models
                phase_shift = (
                    on_off_models.index(model) / max((len(on_off_models) // 2), 2)
                ) * config.on_off_cycle_len

                # current time in the on-off cycle
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
    np.random.seed(config.seed)

    num_reqs = int(config.req_rate * config.duration)

    # generate model path and tokenizer path sequentially, i.e. first num_reqs // num_models reqs use model 0, next num_reqs // num_models reqs use model 1, etc.
    num_models = len(config.model_paths)
    model_indices = np.repeat(np.arange(num_models), num_reqs // num_models)

    remaining_reqs = num_reqs % num_models
    if remaining_reqs > 0:
        # assign the remaining reqs to the last model
        model_indices = np.concatenate(
            [model_indices, [num_models - 1] * remaining_reqs]
        )
    tokenizer_indices = model_indices

    # generate timestamps, with gamma distributed intervals
    # cv is the coefficient of variation, which is the ratio of the standard deviation to
    # the mean of the gamma distribution
    shape = 1 / (config.cv**2)
    scale = config.cv**2 / config.req_rate
    intervals = np.random.gamma(shape, scale, num_reqs)
    timestamps = np.cumsum(intervals)

    # generate input and output lengths
    input_lens = np.random.randint(*config.input_range, num_reqs)
    output_lens = np.random.randint(*config.output_range, num_reqs)

    requests = []
    for i in range(num_reqs):
        req = Request(
            req_id=str(i),
            prompt=dummy_prompt(input_lens[i]),
            prompt_len=input_lens[i],
            output_len=output_lens[i],
            arrival_time=timestamps[i],
            model=config.model_paths[model_indices[i]],
            slo=config.slo,
        )
        requests.append(req)
    return requests


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Request":
            return Request
        return super().find_class(module, name)


class RealWorldTrace:
    def __init__(self, pkl_file_path: Optional[str] = None):
        if pkl_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.pkl_file_path = os.path.join(current_dir, "real_trace.pkl")
        else:
            self.pkl_file_path = pkl_file_path
        self.requests = []
        if pkl_file_path is not None:
            self._load_requests()

    def _load_requests(self):
        if os.path.exists(self.pkl_file_path):
            with open(self.pkl_file_path, "rb") as pkl_file:
                obj = CustomUnpickler(pkl_file).load()
            adapter_dirs, self.requests = obj[0], obj[1]
        else:
            raise FileNotFoundError(f"File {self.pkl_file_path} not found")

    def generate_e2e_benchmark_reqs_18m(self, config, num_models=18):
        time_scale = config.time_scale
        replication = config.replication

        selected_ranks = [5,10,4,17,2,10,23,2,10,14,19,14,18,14,21,2,6,19]
        model_list = [f"model_{i+1}" for i in range(len(selected_ranks))]
        model_to_rank = dict(zip(model_list, selected_ranks))

        from collections import defaultdict
        rank_to_models = defaultdict(list)
        for model, rank in model_to_rank.items():
            rank_to_models[rank].append(model)

        # Offset direction settings
        forward_ranks = {14, 19}  # shift earlier
        backward_ranks = {2, 10}  # shift later

        # Count each model's position per rank
        rank_model_position = defaultdict(dict)
        for rank, models in rank_to_models.items():
            for index, model in enumerate(models):
                rank_model_position[rank][model] = index

        # SLO table definitions
        # actually we use p99 there instead of p95
        model_ttft_slo_baseline_p95 = {
            "model_1": 0.03867427111, "model_2": 0.03222719193, "model_3": 0.03141047955, "model_4": 0.05097425461,
            "model_5": 0.03247243166, "model_6": 0.03926961422, "model_7": 0.04668841362, "model_8": 0.04194666147,
            "model_9": 0.03449705124, "model_10": 0.0606341815, "model_11": 0.04554865837, "model_12": 0.05023122549,
            "model_13": 0.03568530083, "model_14": 0.03531268835, "model_15": 0.04233850479, "model_16": 0.02106617689,
            "model_17": 0.03062166691, "model_18": 0.04841831684,
        }

        model_tpot_slo_baseline_p95 = {
            "model_1": 6.938047111, "model_2": 6.70813661, "model_3": 6.111749992, "model_4": 10.81588957,
            "model_5": 6.990488146, "model_6": 6.845023713, "model_7": 12.60465384, "model_8": 6.939990897,
            "model_9": 7.523151769, "model_10": 6.19979826, "model_11": 10.21128654, "model_12": 5.780082814,
            "model_13": 7.62432434, "model_14": 5.667404957, "model_15": 8.504742742, "model_16": 5.001050144,
            "model_17": 5.441103017, "model_18": 8.66446155,
        }

        tpot_slo_scale = config.tpot_slo_scale
        ttft_slo_scale = config.ttft_slo_scale

        model_ttft_slo = {k: v * ttft_slo_scale for k, v in model_ttft_slo_baseline_p95.items()}
        model_tpot_slo = {k: v * tpot_slo_scale for k, v in model_tpot_slo_baseline_p95.items()}

        selected_models = set(model_list)
        if config.dedicated_model is not None:
            selected_models = {config.dedicated_model}
            print(f"DEBUG: Filtering with dedicated model: {config.dedicated_model}")
            if config.dedicated_model not in model_to_rank:
                print(f"ERROR: Dedicated model {config.dedicated_model} not found.")
                return []

        selected_requests = []
        req_count = 0
        matched_ranks_count = 0

        for req in self.requests:
            try:
                req_rank = int(req.adapter_dir.split("-")[-1])
                if req_rank not in rank_to_models:
                    continue

                for model in rank_to_models[req_rank]:
                    if model not in selected_models:
                        continue

                    matched_ranks_count += 1

                    shift_index = rank_model_position[req_rank][model]

                    for _ in range(replication):
                        arrival_time = req.req_time * time_scale

                        if req_rank in forward_ranks:
                            arrival_time -= 10 * shift_index
                        elif req_rank in backward_ranks:
                            arrival_time += 10 * shift_index

                        slo_ttft = model_ttft_slo[model]
                        slo_tpot = model_tpot_slo[model]

                        processed_request = Request(
                            req_id=str(req_count),
                            prompt=req.prompt,
                            prompt_len=req.prompt_len,
                            output_len=req.output_len,
                            arrival_time=arrival_time,
                            model=model,
                            slo=slo_ttft,
                            slo_ttft=slo_ttft,
                            slo_tpot=slo_tpot,
                        )
                        selected_requests.append(processed_request)
                        req_count += 1

            except Exception as e:
                print(f"ERROR in processing request: {e}")
                continue

        print(f"DEBUG: Total selected models: {selected_models}")
        print(f"DEBUG: Total matched ranks: {matched_ranks_count}")
        print(f"DEBUG: Total selected requests: {len(selected_requests)}")

        if not selected_requests:
            print("ERROR: No requests generated. Possible reasons:")
            print("- No requests match selected ranks.")
            print("- Model filter removed all requests.")
            print("- Data format issue.")

        return selected_requests
    
    def generate_tp_reqs(self, config, num_models=2, req_count=0):
        """Generate requests for tp benchmark from trace_1.py"""
        time_scale = config.time_scale
        replication = config.replication
        model_paths = config.model_paths

        selected_ranks = [4, 14, 3, 10]
        model_mapping = {rank: f"model_{i+1}" for i, rank in enumerate(selected_ranks)}
        
        model_ttft_slo_baseline_p95 = {
            "model_1": 1.17849572, # "Qwen/Qwen2.5-32B"
            "model_2": 0.04077239752, # "meta-llama/Llama-3.3-70B-Instruct"
            "model_3": 0.02226704121, # "Qwen/Qwen2.5-32B"
            "model_4": 1.3508948, # "meta-llama/Llama-3.3-70B-Instruct"
        }

        model_tpot_slo_baseline_p95 = {
            "model_1": 302.7183027103021, # "Qwen/Qwen2.5-32B"
            "model_2": 9.607242716, # "meta-llama/Llama-3.3-70B-Instruct""meta-llama/Llama-3.3-70B-Instruct"
            "model_3": 6.097079518, # model_3_meta-llama/Llama-3.2-1B
            "model_4": 251.14, # "meta-llama/Llama-3.3-70B-Instruct"
        }

        tpot_slo_scale = config.tpot_slo_scale
        ttft_slo_scale = config.ttft_slo_scale
        model_ttft_slo = {
            k: v * ttft_slo_scale for k, v in model_ttft_slo_baseline_p95.items()
        }
        model_tpot_slo = {
            k: v * tpot_slo_scale for k, v in model_tpot_slo_baseline_p95.items()
        }

        selected_requests = []
        for req in self.requests:
            req_rank = int(req.adapter_dir.split("-")[-1])
            if req_rank in selected_ranks:
                this_model = model_mapping[req_rank]
                if model_paths is not None and this_model not in model_paths:
                    continue

                for i in range(replication):
                    arrival_time = req.req_time * time_scale
                    slo, slo_ttft, slo_tpot = model_ttft_slo[model_mapping[req_rank]], model_ttft_slo[model_mapping[req_rank]], model_tpot_slo[model_mapping[req_rank]]
                    if not arrival_time or not slo or not slo_ttft or not slo_tpot:
                        print(f"arrival_time: {arrival_time}, slo: {slo}, slo_ttft: {slo_ttft}, slo_tpot: {slo_tpot}, req_rank: {req_rank}, model_mapping[req_rank]: {model_mapping[req_rank]}")
                    processed_request = Request(
                        req_id=str(req_count),
                        prompt=req.prompt,
                        prompt_len=req.prompt_len,
                        output_len=req.output_len,
                        arrival_time=arrival_time,
                        model=model_mapping[req_rank],
                        slo=slo_ttft,
                        slo_ttft=slo_ttft,
                        slo_tpot=slo_tpot,
                    )
                    selected_requests.append(processed_request)
                    req_count += 1
        return selected_requests

    def generate_e2e_benchmark_reqs(self, config, num_models=8, req_count=0):
        """Generate e2e benchmark requests from trace_1.py"""
        time_scale = config.time_scale
        replication = config.replication
        model_paths = config.model_paths

        selected_ranks = [2, 14, 3, 10, 5, 19, 23, 24]
        model_mapping = {rank: f"model_{i+1}" for i, rank in enumerate(selected_ranks)}

        model_ttft_slo_baseline_p95 = {
            "model_1": 0.04286971092, # model_1_meta-llama/Llama-3.1-8B
            "model_2": 0.04077239752, # model_2_meta-llama/Llama-3.2-3B
            "model_3": 0.02226704121, # model_3_meta-llama/Llama-3.2-1B
            "model_4": 0.04649914742, # model_4_meta-llama/Llama-3.1-8B
            "model_5": 0.04855853081, # model_5_meta-llama/Llama-3.1-8B
            "model_6": 0.03988536596, # model_6_meta-llama/Llama-3.2-1B
            "model_7": 0.03698700905, # model_7_meta-llama/Llama-3.2-1B
            "model_8": 0.02187654972, # model_8_meta-llama/Llama-3.2-1B
        }

        model_tpot_slo_baseline_p95 = {
            "model_1": 11.46289839, # model_1_meta-llama/Llama-3.1-8B
            "model_2": 9.607242716, # model_2_meta-llama/Llama-3.2-3B
            "model_3": 6.097079518, # model_3_meta-llama/Llama-3.2-1B
            "model_4": 11.23110303, # model_4_meta-llama/Llama-3.1-8B
            "model_5": 11.08433425, # model_5_meta-llama/Llama-3.1-8B
            "model_6": 8.795845509, # model_6_meta-llama/Llama-3.2-1B
            "model_7": 6.114510298, # model_7_meta-llama/Llama-3.2-1B
            "model_8": 5.484097324, # model_8_meta-llama/Llama-3.2-1B
        }

        tpot_slo_scale = config.tpot_slo_scale
        ttft_slo_scale = config.ttft_slo_scale
        model_ttft_slo = {
            k: v * ttft_slo_scale for k, v in model_ttft_slo_baseline_p95.items()
        }
        model_tpot_slo = {
            k: v * tpot_slo_scale for k, v in model_tpot_slo_baseline_p95.items()
        }

        # we only keep the mapping of config.model_id
        if config.dedicated_model is not None:
            model_mapping = {k: v for k, v in model_mapping.items() if v == config.dedicated_model}
            selected_ranks = list(model_mapping.keys())
        
        selected_requests = []
        for req in self.requests:
            req_rank = int(req.adapter_dir.split("-")[-1])
            if req_rank in selected_ranks:
                # assert num_models == 6
                if num_models == 6 and model_mapping[req_rank] in {
                    "model_7",
                    "model_8",
                }:
                    print(f"skipping {req_rank}")
                else:
                    this_model = model_mapping[req_rank]
                    if model_paths is not None and this_model not in model_paths:
                        continue

                    for i in range(replication):
                        arrival_time = req.req_time * time_scale
                        slo, slo_ttft, slo_tpot = model_ttft_slo[model_mapping[req_rank]], model_ttft_slo[model_mapping[req_rank]], model_tpot_slo[model_mapping[req_rank]]
                        if not arrival_time or not slo or not slo_ttft or not slo_tpot:
                            print(f"arrival_time: {arrival_time}, slo: {slo}, slo_ttft: {slo_ttft}, slo_tpot: {slo_tpot}, req_rank: {req_rank}, model_mapping[req_rank]: {model_mapping[req_rank]}")
                        processed_request = Request(
                            req_id=str(req_count),
                            prompt=req.prompt,
                            prompt_len=req.prompt_len,
                            output_len=req.output_len,
                            arrival_time=arrival_time,
                            model=model_mapping[req_rank],
                            slo=slo_ttft,
                            slo_ttft=slo_ttft,
                            slo_tpot=slo_tpot,
                        )
                        selected_requests.append(processed_request)
                        req_count += 1
        return selected_requests