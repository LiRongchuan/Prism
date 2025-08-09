if [ ! -d "../benchmark/multi-model/server-logs" ]; then
    mkdir -p ../benchmark/multi-model/server-logs
fi

python -m sglang.launch_multi_model_server \
    --model-config-file ../benchmark/multi-model/model_configs/collocate_3.json \
    --port 30000 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --log-file ../benchmark/multi-model/server-logs/collocate-static-req_rate_10_input_256_512_output_256_512.log
