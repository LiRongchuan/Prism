if [ ! -d "../benchmark/multi-model/server-logs" ]; then
    mkdir -p ../benchmark/multi-model/server-logs
fi

python -m sglang.launch_multi_model_server \
    --model-config-file ../benchmark/multi-model/model_configs/swap_2.json \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-cpu-share-memory \
    --enable-controller \
    --policy priority \
    --use-kvcached-v0 \
    --port 31456 \
    --log-file ../benchmark/multi-model/server-logs/ours-priority-swap_2.log
