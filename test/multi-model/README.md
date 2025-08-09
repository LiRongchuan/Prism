## Run multi-model tests

There are two ways to run multi-model tests:

### 1. Start a single multi-model server with a initial model placement file.

Launch a multi-model server.
```bash
python -m sglang.launch_multi_model_server --model-config-file path/to/your_model_config.json --disable-cuda-graph --enable-overlap-schedule
```

Run tests.
```bash
python test.py -n {num_models}
```

### 2. For single model test, you can launch the multi-model server with model_path.

Launch a multi-model server with model_path.
```bash
python -m sglang.launch_multi_model_server --model-path path/to/your_model --port 30000 --disable-cuda-graph --enable-overlap-schedule --model-name model_1
```

Run tests.
```bash
python test.py -n 1
```

<!-- ### 2. Start sperate endpoints and engines
Launch the endpoint with any `--model-path`. The actural model will be loaded by the engines.
```bash
python -m sglang.launch_endpoint --model-path meta-llama/Meta-Llama-3.1-8B --port 30000 --disable-cuda-graph --enable-overlap-schedule
```
Launch one engine
```bash
python -m sglang.launch_engine --model-path meta-llama/Meta-Llama-3.1-8B --port 30000 --disable-cuda-graph --enable-overlap-schedule
```
You can also launch another engine with different model.
```bash
python -m sglang.launch_engine --model-path mistralai/Mistral-7B-Instruct-v0.3  --port 30000 --disable-cuda-graph --enable-overlap-schedule
```

Run tests.
```bash 
python test.py
``` -->
