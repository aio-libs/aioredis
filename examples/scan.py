import asyncio

import aioredis


async def main():
    """Scan command example."""
    redis = aioredis.Redis.from_url("redis://localhost")

    await redis.mset({"key:1": "value1", "key:2": "value2"})
    cur = b"0"  # set initial cursor to 0
    while cur:
        cur, keys = await redis.scan(cur, match="key:*")
        print("Iteration results:", keys)

    await redis.close()


if __name__ == "__main__":
    import os

    if "redis_version:2.6" not in os.environ.get("REDIS_VERSION", ""):
        asyncio.run(main())
