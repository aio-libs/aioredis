import asyncio

import aioredis


async def main():
    redis = aioredis.from_url(
        "redis://localhost", encoding="utf-8", decode_responses=True
    )
    # get a redis client bound to a single connection.
    async with redis.client() as conn:
        ok = await conn.execute_command("set", "my-key", "some value")
        assert ok is True

        str_value = await conn.execute_command("get", "my-key")
        assert str_value == "some value"

        print("str value:", str_value)


if __name__ == "__main__":
    asyncio.run(main())
