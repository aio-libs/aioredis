import asyncio

import aioredis


async def main():
    # Redis client bound to single connection (no auto reconnection).
    redis = aioredis.Redis(host="localhost", single_connection_client=True)
    await redis.set("my-key", "value")
    val = await redis.get("my-key")
    print(val)

    # gracefully closing underlying connection
    await redis.close()


async def redis_pool():
    # Redis client bound to pool of connections (auto-reconnecting).
    redis = aioredis.Redis.from_url("redis://localhost")
    await redis.set("my-key", "value")
    val = await redis.get("my-key")
    print(val)

    # gracefully closing underlying connection
    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(redis_pool())
