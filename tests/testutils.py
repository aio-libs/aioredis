import pytest

REDIS_6_VERSION = (5, 9, 0)


def redis_version(*version: int, reason: str = None):
    reason = reason or f"Redis version {version} required."
    assert 1 < len(version) <= 3, version
    assert all(isinstance(v, int) for v in version), version
    return pytest.mark.redis_version(version=version, reason=reason)
