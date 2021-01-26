import asyncio

import aioredis


async def main():
    redis = await aioredis.from_url("redis://localhost")
    await redis.set("my-key", "value")
    value = await redis.get("my-key", encoding="utf-8")
    print(value)


asyncio.run(main())
