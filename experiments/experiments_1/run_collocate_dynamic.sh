if [ ! -d "../benchmark/multi-model/server-logs" ]; then
    mkdir -p ../benchmark/multi-model/server-logs
fi

python -m sglang.launch_multi_model_server \
    --model-config-file ./model_configs/collocate_dynamic_4.json \
    --port 31456 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-cpu-share-memory \
    --log-file ./server-logs/collocate-dynamic-req_rate_10_input_256_512_output_256_512.log