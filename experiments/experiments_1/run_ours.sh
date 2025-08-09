if [ ! -d "./server-logs" ]; then
    mkdir -p ./server-logs
fi

python -m sglang.launch_multi_model_server \
    --model-config-file ./model_configs/swap_4.json \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-cpu-share-memory \
    --port 31456 \
    --enable-controller \
    --policy priority \
    --use-kvcached-v0 \
    --log-file ./server-logs/ours-req_rate_10_input_256_512_output_256_512.log
