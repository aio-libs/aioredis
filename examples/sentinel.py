import asyncio

import aioredis.sentinel


async def main():
    sentinel_client = aioredis.sentinel.Sentinel([("localhost", 26379)])

    master_redis: aioredis.Redis = sentinel_client.master_for("mymaster")
    info = await master_redis.sentinel_master("mymaster")
    print("Master role:", info)


if __name__ == "__main__":
    asyncio.run(main())
