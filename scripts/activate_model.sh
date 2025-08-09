curl http://localhost:30000/activate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "model_1",
    "memory_pool_size": 25
  }'