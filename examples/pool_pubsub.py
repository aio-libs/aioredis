import asyncio

import async_timeout

import aioredis

STOPWORD = "STOP"


async def pubsub():
    redis = aioredis.Redis.from_url(
        "redis://localhost", max_connections=10, decode_responses=True
    )
    psub = redis.pubsub()

    async def reader(channel: aioredis.client.PubSub):
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        print(f"(Reader) Message Received: {message}")
                        if message["data"] == STOPWORD:
                            print("(Reader) STOP")
                            break
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass

    async with psub as p:
        await p.subscribe("channel:1")
        await reader(p)  # wait for reader to complete
        await p.unsubscribe("channel:1")

    # closing all open connections
    await psub.close()


async def main():
    tsk = asyncio.create_task(pubsub())

    async def publish():
        pub = aioredis.Redis.from_url("redis://localhost", decode_responses=True)
        while not tsk.done():
            # wait for clients to subscribe
            while True:
                subs = dict(await pub.pubsub_numsub("channel:1"))
                if subs["channel:1"] == 1:
                    break
                await asyncio.sleep(0)
            # publish some messages
            for msg in ["one", "two", "three"]:
                print(f"(Publisher) Publishing Message: {msg}")
                await pub.publish("channel:1", msg)
            # send stop word
            await pub.publish("channel:1", STOPWORD)
        await pub.close()

    await publish()


if __name__ == "__main__":
    import os

    if "redis_version:2.6" not in os.environ.get("REDIS_VERSION", ""):
        asyncio.run(main())
