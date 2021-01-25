import asyncio

import aioredis


async def main():
    redis = await aioredis.Redis.from_url("redis://localhost")

    await redis.hmset_dict("hash", key1="value1", key2="value2", key3=123)

    result = await redis.hgetall("hash", encoding="utf-8")
    assert result == {
        "key1": "value1",
        "key2": "value2",
        "key3": "123",  # note that Redis returns int as string
    }

    await redis.close()


asyncio.run(main())
