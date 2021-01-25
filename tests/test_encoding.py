import pytest

import aioredis
from aioredis.connection import Connection

from .conftest import _get_client

pytestmark = pytest.mark.asyncio


class TestEncoding:
    @pytest.fixture()
    async def r(self, request, event_loop):
        return await _get_client(
            aioredis.Redis,
            request=request,
            event_loop=event_loop,
            decode_responses=True,
        )

    @pytest.fixture()
    async def r_no_decode(self, request, event_loop):
        return await _get_client(
            aioredis.Redis,
            request=request,
            event_loop=event_loop,
            decode_responses=False,
        )

    async def test_simple_encoding(self, r_no_decode: aioredis.Redis):
        unicode_string = chr(3456) + "abcd" + chr(3421)
        await r_no_decode.set("unicode-string", unicode_string.encode("utf-8"))
        cached_val = await r_no_decode.get("unicode-string")
        assert isinstance(cached_val, bytes)
        assert unicode_string == cached_val.decode("utf-8")

    async def test_simple_encoding_and_decoding(self, r: aioredis.Redis):
        unicode_string = chr(3456) + "abcd" + chr(3421)
        await r.set("unicode-string", unicode_string)
        cached_val = await r.get("unicode-string")
        assert isinstance(cached_val, str)
        assert unicode_string == cached_val

    async def test_memoryview_encoding(self, r_no_decode: aioredis.Redis):
        unicode_string = chr(3456) + "abcd" + chr(3421)
        unicode_string_view = memoryview(unicode_string.encode("utf-8"))
        await r_no_decode.set("unicode-string-memoryview", unicode_string_view)
        cached_val = await r_no_decode.get("unicode-string-memoryview")
        # The cached value won't be a memoryview because it's a copy from Redis
        assert isinstance(cached_val, bytes)
        assert unicode_string == cached_val.decode("utf-8")

    async def test_memoryview_encoding_and_decoding(self, r: aioredis.Redis):
        unicode_string = chr(3456) + "abcd" + chr(3421)
        unicode_string_view = memoryview(unicode_string.encode("utf-8"))
        await r.set("unicode-string-memoryview", unicode_string_view)
        cached_val = await r.get("unicode-string-memoryview")
        assert isinstance(cached_val, str)
        assert unicode_string == cached_val

    async def test_list_encoding(self, r: aioredis.Redis):
        unicode_string = chr(3456) + "abcd" + chr(3421)
        result = [unicode_string, unicode_string, unicode_string]
        await r.rpush("a", *result)
        assert await r.lrange("a", 0, -1) == result


class TestEncodingErrors:
    async def test_ignore(self, request, event_loop):
        r = await _get_client(
            aioredis.Redis,
            request=request,
            event_loop=event_loop,
            decode_responses=True,
            encoding_errors="ignore",
        )
        await r.set("a", b"foo\xff")
        assert await r.get("a") == "foo"

    async def test_replace(self, request, event_loop):
        r = await _get_client(
            aioredis.Redis,
            request=request,
            event_loop=event_loop,
            decode_responses=True,
            encoding_errors="replace",
        )
        await r.set("a", b"foo\xff")
        assert await r.get("a") == "foo\ufffd"


class TestMemoryviewsAreNotPacked:
    def test_memoryviews_are_not_packed(self):
        c = Connection()
        arg = memoryview(b"some_arg")
        arg_list = ["SOME_COMMAND", arg]
        cmd = c.pack_command(*arg_list)
        assert cmd[1] is arg
        cmds = c.pack_commands([arg_list, arg_list])
        assert cmds[1] is arg
        assert cmds[3] is arg


class TestCommandsAreNotEncoded:
    @pytest.fixture()
    async def r(self, request, event_loop):
        return await _get_client(
            aioredis.Redis, request=request, event_loop=event_loop, encoding="utf-16"
        )

    async def test_basic_command(self, r: aioredis.Redis):
        await r.set("hello", "world")


class TestInvalidUserInput:
    async def test_boolean_fails(self, r: aioredis.Redis):
        with pytest.raises(aioredis.DataError):
            await r.set("a", True)

    async def test_none_fails(self, r: aioredis.Redis):
        with pytest.raises(aioredis.DataError):
            await r.set("a", None)

    async def test_user_type_fails(self, r: aioredis.Redis):
        class Foo:
            def __str__(self):
                return "Foo"

        with pytest.raises(aioredis.DataError):
            await r.set("a", Foo())
