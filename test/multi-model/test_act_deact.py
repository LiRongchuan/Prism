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
    policy: str = "simple-global",
    enable_gpu_scheduler: bool = True,
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
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        model_config_file = "model_config.json"
        server_log_file = "server-logs/test_act_deact.log"
        enable_elastic_memory = True
        enable_cpu_share_memory = True
        use_kvcached_v0 = True
        enable_controller = True
        enable_gpu_scheduler = True
        policy = "simple-global"
        cls.process = popen_launch_server(
            model_config_file=model_config_file,
            server_log_file=server_log_file,
            base_url=cls.base_url,
            timeout=1200,
            enable_elastic_memory=enable_elastic_memory,
            enable_cpu_share_memory=enable_cpu_share_memory,
            use_kvcached_v0=use_kvcached_v0,
            enable_controller=enable_controller,
            enable_gpu_scheduler=enable_gpu_scheduler,
            policy=policy,
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

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

    def activate_model(self, model_name):
        response = requests.post(
            self.base_url + "/activate",
            json={"model_name": model_name, "instance_idx": 0, "gpu_id": 0},
        )
        return response.json()

    def deactivate_model(self, model_name):
        deactivate_url = self.base_url + "/deactivate"
        pload = {
            "model_name": model_name,
            "instance_idx": 0,
            "preempt": False,
            "gpu_id": 0,
        }
        response = requests.post(deactivate_url, json=pload)
        return response.json()

    def test_activate_deactivate(self):
        model_name = "model_1"
        origin_model_names = self.get_model_names()
        print(f"origin_model_names: {origin_model_names}")
        origin_response = self.run_generate(model_name)

        # deactivate model
        self.deactivate_model(model_name)

        # activate model
        self.activate_model(model_name)

        # generate
        response = self.run_generate(model_name)
        assert response[:32] == origin_response[:32]


if __name__ == "__main__":
    unittest.main()
