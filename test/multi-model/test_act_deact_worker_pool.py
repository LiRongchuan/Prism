import json
import subprocess
import time
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)


def popen_launch_server(
    model_config_file: str,
    server_log_file: str,
    base_url: str,
    timeout: float,
    enable_controller: bool = True,
    enable_elastic_memory: bool = True,
    enable_cpu_share_memory: bool = True,
    use_kvcached_v0: bool = True,
    enable_model_service: bool = True,
    num_model_service_workers: int = 1,
    policy: str = "simple-global",
    enable_gpu_scheduler: bool = True,
    workers_per_gpu: int = 1,
    num_gpus: int = 1,
    other_args: tuple = (),
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang.launch_multi_model_server",
        "--model-config-file",
        model_config_file,
        "--host",
        host,
        "--port",
        port,
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--enable-worker-pool",
        "--workers-per-gpu",
        str(workers_per_gpu),
        "--num-gpus",
        str(num_gpus),
        # "--disable-regex-jump-forward",
        "--log-file",
        server_log_file,
        *other_args,
    ]
    if enable_elastic_memory:
        command.append("--enable-elastic-memory")
    if enable_cpu_share_memory:
        command.append("--enable-cpu-share-memory")
    if use_kvcached_v0:
        command.append("--use-kvcached-v0")
    if enable_controller:
        command.append("--enable-controller")
    if policy:
        command.append(f"--policy")
        command.append(policy)
    if enable_gpu_scheduler:
        command.append("--enable-gpu-scheduler")
    if enable_model_service:
        command.append("--enable-model-service")
        command.append(f"--num-model-service-workers")
        command.append(str(num_model_service_workers))

    printed_cmd = " ".join(command)

    print(f"Launching server with command: {printed_cmd}")

    process = subprocess.Popen(command, stdout=None, stderr=None)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                # "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/get_model_names", headers=headers)
            if response.status_code == 200:
                time.sleep(2)
                return process
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


class TestActivateDeactivate(unittest.TestCase):
    def setUp(self):
        self.model = DEFAULT_MODEL_NAME_FOR_TEST
        self.base_url = DEFAULT_URL_FOR_TEST
        model_config_file = "/sgl-workspace/sglang-multi-model/test/multi-model/model_config.json"
        server_log_file = "server-logs/test_act_deact_worker_pool.log"
        enable_elastic_memory = True
        enable_cpu_share_memory = True
        use_kvcached_v0 = True
        enable_controller = False
        enable_gpu_scheduler = True
        policy = "simple-global"
        self.process = popen_launch_server(
            model_config_file=model_config_file,
            server_log_file=server_log_file,
            base_url=self.base_url,
            timeout=1200,
            enable_elastic_memory=enable_elastic_memory,
            enable_cpu_share_memory=enable_cpu_share_memory,
            use_kvcached_v0=use_kvcached_v0,
            enable_controller=enable_controller,
            enable_gpu_scheduler=enable_gpu_scheduler,
            enable_model_service=True,
            num_model_service_workers=2,
            policy=policy,
            workers_per_gpu=2,
            num_gpus=2,
        )

    def tearDown(self):
        kill_child_process(self.process.pid)

    def run_generate(self, model_name):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "model": model_name,
                "slo": 10,
                "prompt_len": 10,
                "output_len": 32,
            },
        )
        text = response.json()["text"]
        print(f"text: {text}")
        return text

    def get_model_names(self):
        response = requests.get(self.base_url + "/get_model_names")
        model_names = response.json()
        return model_names

    def activate_model(self, model_name, gpu_id):
        response = requests.post(
            self.base_url + "/activate",
            json={
                "model_name": model_name,
                "gpu_id": gpu_id,
            },
        )
        return response.json()

    def deactivate_model(self, model_name, gpu_id):
        deactivate_url = self.base_url + "/deactivate"
        pload = {
            "model_name": model_name,
            "instance_idx": 0,
            "preempt": False,
            "gpu_id": gpu_id,
        }
        response = requests.post(deactivate_url, json=pload)
        return response.json()

    def test_activate_deactivate_diff_worker(self):
        model_name_1 = "model_1"
        model_name_2 = "model_2"
        model_names = self.get_model_names()
        print(f"model_names: {model_names}")
        origin_response = self.run_generate(model_name_1)

        # deactivate model
        self.deactivate_model(model_name_1, gpu_id=0)

        self.activate_model(model_name="model_1", gpu_id=0)  # worker 1
        response_1 = self.run_generate(model_name="model_1")
        assert response_1[:32] == origin_response[:32]

        # activate model
        self.activate_model(model_name="model_2", gpu_id=1)  # worker 0
        response_2 = self.run_generate(model_name="model_2")

        self.activate_model(model_name="model_3", gpu_id=1)  # worker 1
        response_3 = self.run_generate(model_name="model_3")

        self.activate_model(model_name="model_4", gpu_id=1)  # error

        self.deactivate_model(model_name="model_3", gpu_id=1)

        self.activate_model(model_name="model_4", gpu_id=1)  # worker 1
        response_4 = self.run_generate(model_name="model_4")

    # def test_activate_deactivate_same_worker(self):
    #     model_name_1 = "model_1"
    #     model_name_2 = "model_2"
    #     origin_model_names = self.get_model_names()
    #     print(f"origin_model_names: {origin_model_names}")
    #     origin_response = self.run_generate(model_name_1)

    #     # deactivate model
    #     self.deactivate_model(model_name_1, gpu_id=0)

    #     # activate model
    #     self.activate_model(model_name_2, gpu_id=0, worker_id=0)

    #     # generate
    #     response_2 = self.run_generate(model_name_2)


if __name__ == "__main__":
    unittest.main()
