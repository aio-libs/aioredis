import asyncio

import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost", max_connections=10)
    await redis.execute_command("set", "my-key", "value")
    val = await redis.execute_command("get", "my-key")
    print("raw value:", val)


if __name__ == "__main__":
    asyncio.run(main())
