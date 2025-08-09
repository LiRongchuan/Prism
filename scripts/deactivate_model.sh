curl http://localhost:30000/deactivate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "model_1",
    "evict_waiting_requests": true,
    "preempt": false
  }'
