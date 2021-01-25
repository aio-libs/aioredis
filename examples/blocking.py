import asyncio

import aioredis


async def blocking_commands():
    # Redis client bound to pool of connections (auto-reconnecting).
    redis = aioredis.Redis.from_url("redis://localhost")

    async def get_message():
        # Redis blocking commands block the connection they are on
        # until they complete. For this reason, the connection must
        # not be returned to the connection pool until we've
        # finished waiting on future created by brpop(). To achieve
        # this, 'await redis' acquires a dedicated connection from
        # the connection pool and creates a new Redis command object
        # using it. This object is a context manager and the
        # connection will be released back to the pool at the end of
        # the with block."
        with await redis as r:
            return await r.brpop("my-key")

    future = asyncio.create_task(get_message())
    await redis.lpush("my-key", "value")
    await future
    print(future.result())

    # gracefully closing underlying connection
    await redis.close()


if __name__ == "__main__":
    asyncio.run(blocking_commands())
