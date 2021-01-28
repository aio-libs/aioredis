from typing import TYPE_CHECKING
from unittest import mock

import pytest

from aioredis.exceptions import InvalidResponse
from aioredis.utils import HIREDIS_AVAILABLE

if TYPE_CHECKING:
    from aioredis.connection import PythonParser


@pytest.mark.skipif(HIREDIS_AVAILABLE, reason="PythonParser only")
@pytest.mark.asyncio
async def test_invalid_response(r):
    raw = b"x"
    parser: "PythonParser" = r.connection._parser
    with mock.patch.object(parser._buffer, "readline", return_value=raw):
        with pytest.raises(InvalidResponse) as cm:
            await parser.read_response()
    assert str(cm.value) == "Protocol Error: %r" % raw
