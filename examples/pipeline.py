import asyncio

import aioredis


async def main():
    redis = aioredis.Redis.from_url("redis://localhost")

    # No pipelining;
    async def wait_each_command():
        val = await redis.get("foo")  # wait until `val` is available
        cnt = await redis.incr("bar")  # wait until `cnt` is available
        return val, cnt

    # Sending multiple commands and then gathering results
    async def concurrent():
        fut1 = redis.get("foo")  # issue command and return future
        fut2 = redis.incr("bar")  # issue command and return future
        # block until results are available
        val, cnt = await asyncio.gather(fut1, fut2)
        return val, cnt

    # Explicit pipeline
    async def explicit_pipeline():
        pipe = redis.pipeline()
        pipe.get("foo").incr("bar")
        result = await pipe.execute()
        return result

    async def context_pipeline():
        async with redis.pipeline() as pipe:
            pipe.get("foo").incr("bar")
            result = await pipe.execute()
        return result

    res = await wait_each_command()
    print(res)
    res = await concurrent()
    print(res)
    res = await explicit_pipeline()
    print(res)
    res = await context_pipeline()
    print(res)

    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())
