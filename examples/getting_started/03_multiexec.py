import asyncio

import aioredis
from aioredis.utils import pipeline


async def main():
    redis = await aioredis.from_url("redis://localhost")
    async with redis.pipeline(transaction=True) as pipe:
        ok1, ok2 = await (pipe.set("key1", "value1").set("key2", "value2").execute())
    assert ok1
    assert ok2


async def main2():
    redis = await aioredis.Redis.from_url("redis://localhost", decode_responses=True)
    async with pipeline(redis) as pipe:
        pipe.set("key1", "value1").set("key2", "value2")
    assert await redis.get("key1", "value1")
    assert await redis.get("key2", "value2")


asyncio.run(main())
asyncio.run(main2())
