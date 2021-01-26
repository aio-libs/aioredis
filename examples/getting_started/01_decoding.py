import asyncio

import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost")
    await redis.set("key", "string-value")
    bin_value = await redis.get("key")
    assert bin_value == b"string-value"
    redis = aioredis.from_url("redis://localhost", decode_responses=True)
    str_value = await redis.get("key")
    assert str_value == "string-value"

    await redis.close()


asyncio.run(main())
