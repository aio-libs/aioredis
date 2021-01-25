import asyncio

import aioredis


async def main():
    redis = await aioredis.Redis.from_url("redis://localhost")
    await redis.set("key", "string-value")
    bin_value = await redis.get("key")
    assert bin_value == b"string-value"

    str_value = await redis.get("key", encoding="utf-8")
    assert str_value == "string-value"

    await redis.close()


asyncio.run(main())
