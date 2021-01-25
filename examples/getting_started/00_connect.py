import asyncio

import aioredis


async def main():
    redis = await aioredis.Redis.from_url("redis://localhost")
    await redis.set("my-key", "value")
    value = await redis.get("my-key", encoding="utf-8")
    print(value)

    await redis.close()


asyncio.run(main())
