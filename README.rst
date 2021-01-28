aioredis
========

asyncio (PEP 3156) Redis client library.

.. image:: https://travis-ci.com/aio-libs/aioredis.svg?branch=master
   :target: https://travis-ci.com/aio-libs/aioredis


.. image:: https://codecov.io/gh/aio-libs/aioredis/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/aio-libs/aioredis

Features
--------

================================  ==============================
hiredis_ parser                     Yes
Pure-python parser                  Yes
Low-level & High-level APIs         Yes
Connections Pool                    Yes
Pipelining support                  Yes
Pub/Sub support                     Yes
SSL/TLS support                     Yes
Sentinel support                    Yes
Redis Cluster support               WIP
Trollius (python 2.7)               No
Tested CPython versions             `3.6, 3.7, 3.8 <travis_>`_ [1]_ [2]_
Tested PyPy3 versions               `pypy3.6-7.1.1 <travis_>`_
Tested for Redis server             `2.6, 2.8, 3.0, 3.2, 4.0 5.0 <travis_>`_
Support for dev Redis server        through low-level API
================================  ==============================

.. [1] For Python 3.3, 3.4 support use aioredis v0.3.
.. [2] For Python 3.5 support use aioredis v1.2.

Documentation
-------------

http://aioredis.readthedocs.io/

Usage example
-------------

Simple high-level interface with connections pool:

.. code:: python

    import asyncio
    import aioredis

    async def go():
        redis = await aioredis.create_redis_pool(
            'redis://localhost')
        await redis.set('my-key', 'value')
        val = await redis.get('my-key', encoding='utf-8')
        print(val)
        redis.close()
        await redis.wait_closed()

    asyncio.run(go())
    # will print 'value'

Using a pool among multiple coroutine calls:

.. code:: python

    import asyncio
    import string

    import aioredis

    async def go(r, key, value):
        await r.set(key, value)
        val = await r.get(key)
        print(f"Got {key} -> {val}")

    async def main(loop):
        r = await aioredis.create_redis_pool(
            "redis://localhost", minsize=5, maxsize=10, loop=loop
        )
        try:
            return await asyncio.gather(
                *(
                    go(r, i, j)
                    for i, j in zip(string.ascii_uppercase, string.ascii_lowercase)
                ),
                return_exceptions=True,
            )
        finally:
            r.close()
            await r.wait_closed()

    if __name__ == "__main__":
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(main(loop))

Requirements
------------

* Python_ 3.6+
* hiredis_

.. note::

    hiredis is preferred requirement.
    Pure-python protocol parser is implemented as well and can be used
    through ``parser`` parameter.

Benchmarks
----------

Benchmarks can be found here: https://github.com/popravich/python-redis-benchmark

Discussion list
---------------

*aio-libs* google group: https://groups.google.com/forum/#!forum/aio-libs

Or gitter room: https://gitter.im/aio-libs/Lobby

License
-------

The aioredis is offered under MIT license.

.. _Python: https://www.python.org
.. _hiredis: https://pypi.python.org/pypi/hiredis
.. _travis: https://travis-ci.com/aio-libs/aioredis
