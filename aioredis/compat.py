# flake8: noqa
try:
    from typing import Protocol, TypedDict  # lgtm [py/unused-import]
except ImportError:
    from typing_extensions import Protocol, TypedDict  # lgtm [py/unused-import]
