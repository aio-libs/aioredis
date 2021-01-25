import asyncio

import aioredis


async def main():
    redis = aioredis.Redis.from_url("redis://localhost", max_connections=10)
    async with redis as r:
        await r.execute_command("set", "my-key", "value")
        val = await r.execute_command("get", "my-key")
    print("raw value:", val)
    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())
