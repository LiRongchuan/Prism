if [ ! -d "../benchmark/multi-model/server-logs" ]; then
    mkdir -p ../benchmark/multi-model/server-logs
fi

python -m sglang.launch_multi_model_server \
    --model-config-file ./model_configs/collocate_static_4.json \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-cpu-share-memory \
    --port 31456 \
    --log-file ./server-logs/collocate-static-req_rate_10_input_256_512_output_256_512.log