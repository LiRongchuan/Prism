import asyncio
import pickle

import redis


class AsyncRedisClient:
    def __init__(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db
        self.client = redis.asyncio.Redis(host=host, port=port, db=db)

    async def reconnect(self):
        """Reconnect to Redis server if connection is lost"""
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass  # Ignore errors during close
        self.client = redis.asyncio.Redis(host=self.host, port=self.port, db=self.db)
        # Test connection
        await self.client.ping()

    async def send_pyobj(self, key, obj):
        obj_bytes = pickle.dumps(obj)
        await self.client.rpush(key, obj_bytes)

    async def recv_pyobj_non_block(self, key, count=1):
        obj_bytes = await self.client.lpop(key, count=count)
        if isinstance(obj_bytes, list):
            return [pickle.loads(obj) for obj in obj_bytes]
        if obj_bytes is None:
            return []
        return pickle.loads(obj_bytes)

    async def recv_pyobj_block(self, key):
        _, obj_bytes = await self.client.blpop(key)
        return pickle.loads(obj_bytes)

    async def close(self):
        await self.client.close()


class RedisClient:
    def __init__(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db
        self.client = redis.Redis(host=host, port=port, db=db)

    def reconnect(self):
        """Reconnect to Redis server if connection is lost"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors during close
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
        # Test connection
        self.client.ping()

    def clear_queue(self):
        self.client.flushdb()

    def send_pyobj(self, key, obj):
        obj_bytes = pickle.dumps(obj)
        self.client.rpush(key, obj_bytes)

    def recv_pyobj_non_block(self, key, count=1):
        obj_bytes = self.client.lpop(key, count=count)
        if isinstance(obj_bytes, list):
            return [pickle.loads(obj) for obj in obj_bytes]
        if obj_bytes is None:
            return []
        return pickle.loads(obj_bytes)

    def get_queue_length(self, key):
        return self.client.llen(key)

    def recv_pyobj_block(self, key):
        _, obj_bytes = self.client.blpop(key)
        return pickle.loads(obj_bytes)

    def pop_all(self, key):
        length = self.client.llen(key)
        if length == 0:
            return []

        pipe = self.client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = pipe.execute()
        return [pickle.loads(obj) for obj in results[0]]

    def close(self):
        self.client.close()


if __name__ == "__main__":
    import asyncio

    from sglang.srt.managers.io_struct import GenerateReqInput

    model_names = [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
    num_reqs = 5
    reqs = []
    for i in range(num_reqs):
        reqs.append(
            GenerateReqInput(
                model=model_names[i % len(model_names)],
                text=f"{i} What is the meaning of life?",
                sampling_params={"temperature": 0.5, "top_p": 0.9, "top_k": 50},
                slo=10,
            )
        )

    generate_prefix = "generate"
    response_prefix = "response"

    async def test():
        client = AsyncRedisClient("localhost", 6379, 0)
        await client.recv_pyobj_non_block(f"{generate_prefix}:{model_names[0]}")
        # for req in reqs:
        #     await client.send_pyobj(f"{generate_prefix}:{req.model}", req)

        # for model_name in model_names:
        #     while True:
        #         try:
        #             response = await client.recv_pyobj_non_block(f"{generate_prefix}:{model_name}")
        #             print(response)
        #             print("\n")
        #         except:
        #             break

    asyncio.run(test())
