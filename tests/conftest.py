import argparse
import asyncio
import atexit
import contextlib
import os
import random
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from distutils.version import StrictVersion
from unittest.mock import AsyncMock
from urllib.parse import urlencode, urlparse, urlunparse

import pytest

import aioredis
import aioredis.sentinel
from aioredis.client import Monitor

TCPAddress = namedtuple("TCPAddress", "host port")

RedisServer = namedtuple("RedisServer", "name tcp_address unixsocket version password")

SentinelServer = namedtuple(
    "SentinelServer", "name tcp_address unixsocket version masters"
)


# Public fixtures


@pytest.fixture(scope="session")
def event_loop():
    """Creates new event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def _unused_tcp_port():
    """Find an unused localhost TCP port from 1024-65535 and return it."""
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def tcp_port_factory():
    """A factory function, producing different unused TCP ports."""
    produced = set()

    def factory():
        """Return an unused port."""
        port = _unused_tcp_port()

        while port in produced:
            port = _unused_tcp_port()

        produced.add(port)

        return port

    return factory


@pytest.fixture(params=[True, False], ids=["single", "pool"])
def create_redis(request, event_loop):
    """Wrapper around aioredis.create_redis."""
    single_connection = request.param

    async def f(address: TCPAddress, **kwargs):
        redis = aioredis.Redis(host=address.host, port=address.port, **kwargs)
        if single_connection:
            redis = await redis.client()

        def teardown():
            async def ateardown():
                if "username" in kwargs:
                    return
                try:
                    await redis.flushdb()
                except aioredis.ConnectionError:
                    # handle cases where a test disconnected a client
                    # just manually retry the flushdb
                    await redis.flushdb()
                await redis.close()
                await redis.connection_pool.disconnect()

            if event_loop.is_running():
                event_loop.create_task(ateardown())
            else:
                event_loop.run_until_complete(ateardown())

        request.addfinalizer(teardown)
        return redis

    return f


@pytest.fixture
def create_sentinel():
    """Helper instantiating RedisSentinel client."""

    async def f(*args, **kw):
        # make it fail fast on slow CIs (if timeout argument is omitted)
        kw.setdefault("timeout", 0.001)
        client = aioredis.sentinel.Sentinel(*args, **kw)
        return client

    return f


@pytest.fixture
async def redis(create_redis, server):
    """Returns Redis client instance."""
    redis = await create_redis(server.tcp_address)
    return redis


@pytest.fixture
def r(redis) -> aioredis.Redis:
    return redis


@pytest.fixture
async def r2(create_redis, server) -> aioredis.Redis:
    """Returns Redis client instance, for tests which need multiple."""
    redis = await create_redis(server.tcp_address)
    yield redis


@pytest.fixture
async def redis_sentinel(create_sentinel, sentinel):
    """Returns Redis Sentinel client instance."""
    redis_sentinel = await create_sentinel([sentinel.tcp_address], timeout=2)
    return redis_sentinel


@pytest.fixture(scope="session")
def server(start_server):
    """Starts redis-server instance."""
    return start_server("A")


@pytest.fixture(scope="session")
def serverB(start_server):
    """Starts redis-server instance."""
    return start_server("B")


@pytest.fixture(scope="session")
def sentinel(start_sentinel, request, start_server):
    """Starts redis-sentinel instance with one master -- masterA."""
    # Adding main+replica for normal (no failover) tests:
    main_no_fail = start_server("main-no-fail")
    start_server("replica-no-fail", slaveof=main_no_fail)
    # Adding master+slave for failover test;
    mainA = start_server("mainA")
    start_server("replicaA", slaveof=mainA)
    return start_sentinel("main", mainA, main_no_fail)


@pytest.fixture(params=["path", "query"])
def server_tcp_url(server, request):
    def make(**kwargs):
        netloc = "{0.host}:{0.port}".format(server.tcp_address)
        path = ""
        if request.param == "path":
            if "password" in kwargs:
                netloc = ":{0}@{1.host}:{1.port}".format(
                    kwargs.pop("password"), server.tcp_address
                )
            if "db" in kwargs:
                path = "/{}".format(kwargs.pop("db"))
        query = urlencode(kwargs)
        return urlunparse(("redis", netloc, path, "", query, ""))

    return make


@pytest.fixture
def server_unix_url(server):
    def make(**kwargs):
        query = urlencode(kwargs)
        return urlunparse(("unix", "", server.unixsocket, "", query, ""))

    return make


# Internal stuff #

# Taken from python3.9
class BooleanOptionalAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith("--"):
                option_string = "--no-" + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None:
            help += f" (default: {default})"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    def format_usage(self):
        return " | ".join(self.option_strings)


def pytest_addoption(parser):
    parser.addoption(
        "--redis-server",
        default=[],
        action="append",
        help="Path to redis-server executable," " defaults to `%(default)s`",
    )
    parser.addoption(
        "--uvloop", action=BooleanOptionalAction, help="Run tests with uvloop"
    )


def _read_server_version(redis_bin):
    args = [redis_bin, "--version"]
    with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
        version = proc.stdout.readline().decode("utf-8")
    for part in version.split():
        if part.startswith("v="):
            break
    else:
        raise RuntimeError(f"No version info can be found in {version}")
    return tuple(map(int, part[2:].split(".")))


@contextlib.contextmanager
def config_writer(path):
    with open(path, "wt") as f:

        def write(*args):
            print(*args, file=f)

        yield write


REDIS_SERVERS = []
VERSIONS = {}


def format_version(srv):
    return "redis_v{}".format(".".join(map(str, VERSIONS[srv])))


@pytest.fixture(scope="session")
def start_server(_proc, request, tcp_port_factory, server_bin):
    """Starts Redis server instance.
    Caches instances by name.
    ``name`` param -- instance alias
    ``config_lines`` -- optional list of config directives to put in config
        (if no config_lines passed -- no config will be generated,
         for backward compatibility).
    """

    version = _read_server_version(server_bin)
    verbose = request.config.getoption("-v") > 3

    servers = {}

    def timeout(t):
        end = time.time() + t
        while time.time() <= end:
            yield True
        raise RuntimeError("Redis startup timeout expired")

    def maker(name, config_lines=None, *, slaveof=None, password=None):
        print("Start REDIS", name)
        assert slaveof is None or isinstance(slaveof, RedisServer), slaveof
        if name in servers:
            return servers[name]

        port = tcp_port_factory()
        tcp_address = TCPAddress("localhost", port)
        if sys.platform == "win32":
            unixsocket = None
        else:
            unixsocket = f"/tmp/aioredis.{port}.sock"
        dumpfile = f"dump-{port}.rdb"
        data_dir = tempfile.gettempdir()
        dumpfile_path = os.path.join(data_dir, dumpfile)
        stdout_file = os.path.join(data_dir, f"aioredis.{port}.stdout")
        tmp_files = [dumpfile_path, stdout_file]
        if config_lines:
            config = os.path.join(data_dir, f"aioredis.{port}.conf")
            with config_writer(config) as write:
                write("daemonize no")
                write('save ""')
                write("dir ", data_dir)
                write("dbfilename", dumpfile)
                write("port", port)
                if unixsocket:
                    write("unixsocket", unixsocket)
                    tmp_files.append(unixsocket)
                if password:
                    write(f'requirepass "{password}"')
                write("# extra config")
                for line in config_lines:
                    write(line)
                if slaveof is not None:
                    write(
                        "slaveof {0.tcp_address.host} {0.tcp_address.port}".format(
                            slaveof
                        )
                    )
                    if password:
                        write(f'masterauth "{password}"')
            args = [config]
            tmp_files.append(config)
        else:
            args = [
                "--daemonize",
                "no",
                "--save",
                '""',
                "--dir",
                data_dir,
                "--dbfilename",
                dumpfile,
                "--port",
                str(port),
            ]
            if unixsocket:
                args += [
                    "--unixsocket",
                    unixsocket,
                ]
            if password:
                args += [f'--requirepass "{password}"']
            if slaveof is not None:
                args += [
                    "--slaveof",
                    str(slaveof.tcp_address.host),
                    str(slaveof.tcp_address.port),
                ]
                if password:
                    args += [f'--masterauth "{password}"']
        f = open(stdout_file, "w")
        atexit.register(f.close)
        proc = _proc(
            server_bin,
            *args,
            stdout=f,
            stderr=subprocess.STDOUT,
            _clear_tmp_files=tmp_files,
        )
        with open(stdout_file) as f:
            for _ in timeout(10):
                assert proc.poll() is None, ("Process terminated", proc.returncode)
                log = f.readline()
                if log and verbose:
                    print(name, ":", log, end="")
                if "The server is now ready to accept connections " in log:
                    break
            if slaveof is not None:
                for _ in timeout(10):
                    log = f.readline()
                    if log and verbose:
                        print(name, ":", log, end="")
                    if "sync: Finished with success" in log:
                        break
        info = RedisServer(name, tcp_address, unixsocket, version, password)
        servers.setdefault(name, info)
        print("Ready REDIS", name)
        return info

    return maker


@pytest.fixture(scope="session")
def start_sentinel(_proc, request, tcp_port_factory, server_bin):
    """Starts Redis Sentinel instances."""
    version = _read_server_version(server_bin)
    verbose = request.config.getoption("-v") > 3

    sentinels = {}

    def timeout(t):
        end = time.time() + t
        while time.time() <= end:
            yield True
        raise RuntimeError("Redis startup timeout expired")

    def maker(
        name,
        *masters,
        quorum=1,
        noslaves=False,
        down_after_milliseconds=3000,
        failover_timeout=1000,
    ):
        key = (name,) + masters
        if key in sentinels:
            return sentinels[key]
        port = tcp_port_factory()
        tcp_address = TCPAddress("localhost", port)
        data_dir = tempfile.gettempdir()
        config = os.path.join(data_dir, f"aioredis-sentinel.{port}.conf")
        stdout_file = os.path.join(data_dir, f"aioredis-sentinel.{port}.stdout")
        tmp_files = [config, stdout_file]
        if sys.platform == "win32":
            unixsocket = None
        else:
            unixsocket = os.path.join(data_dir, f"aioredis-sentinel.{port}.sock")
            tmp_files.append(unixsocket)

        with config_writer(config) as write:
            write("daemonize no")
            write('save ""')
            write("port", port)
            if unixsocket:
                write("unixsocket", unixsocket)
            write("loglevel debug")
            for master in masters:
                write(
                    "sentinel monitor",
                    master.name,
                    "127.0.0.1",
                    master.tcp_address.port,
                    quorum,
                )
                write(
                    "sentinel down-after-milliseconds",
                    master.name,
                    down_after_milliseconds,
                )
                write("sentinel failover-timeout", master.name, failover_timeout)
                write("sentinel auth-pass", master.name, master.password)

        f = open(stdout_file, "w")
        atexit.register(f.close)
        proc = _proc(
            server_bin,
            config,
            "--sentinel",
            stdout=f,
            stderr=subprocess.STDOUT,
            _clear_tmp_files=tmp_files,
        )
        # XXX: wait sentinel see all masters and slaves;
        all_masters = {m.name for m in masters}
        if noslaves:
            all_slaves = {}
        else:
            all_slaves = {m.name for m in masters}
        with open(stdout_file) as f:
            for _ in timeout(30):
                assert proc.poll() is None, ("Process terminated", proc.returncode)
                log = f.readline()
                if log and verbose:
                    print(name, ":", log, end="")
                for m in masters:
                    if f"# +monitor master {m.name}" in log:
                        all_masters.discard(m.name)
                    if "* +slave slave" in log and f"@ {m.name}" in log:
                        all_slaves.discard(m.name)
                if not all_masters and not all_slaves:
                    break
            else:
                raise RuntimeError("Could not start Sentinel")

        masters = {m.name: m for m in masters}
        info = SentinelServer(name, tcp_address, unixsocket, version, masters)
        sentinels.setdefault(key, info)
        return info

    return maker


@pytest.fixture(scope="session")
def ssl_proxy(_proc, request, tcp_port_factory):
    by_port = {}

    cafile = os.path.abspath(request.config.getoption("--ssl-cafile"))
    certfile = os.path.abspath(request.config.getoption("--ssl-cert"))
    dhfile = os.path.abspath(request.config.getoption("--ssl-dhparam"))
    assert os.path.exists(
        cafile
    ), "Missing SSL CA file, run `make certificate` to generate new one"
    assert os.path.exists(
        certfile
    ), "Missing SSL CERT file, run `make certificate` to generate new one"
    assert os.path.exists(
        dhfile
    ), "Missing SSL DH params, run `make certificate` to generate new one"

    ssl_ctx = ssl.create_default_context(cafile=cafile)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    ssl_ctx.load_dh_params(dhfile)

    def sockat(unsecure_port):
        if unsecure_port in by_port:
            return by_port[unsecure_port]

        secure_port = tcp_port_factory()
        _proc(
            "/usr/bin/socat",
            "openssl-listen:{port},"
            "dhparam={param},"
            "cert={cert},verify=0,fork".format(
                port=secure_port, param=dhfile, cert=certfile
            ),
            f"tcp-connect:localhost:{unsecure_port}",
        )
        time.sleep(1)  # XXX
        by_port[unsecure_port] = secure_port, ssl_ctx
        return secure_port, ssl_ctx

    return sockat


@pytest.fixture(scope="session")
def _proc():
    processes = []
    tmp_files = set()

    def run(*commandline, _clear_tmp_files=(), **kwargs):
        proc = subprocess.Popen(commandline, **kwargs)
        processes.append(proc)
        tmp_files.update(_clear_tmp_files)
        return proc

    try:
        yield run
    finally:
        while processes:
            proc = processes.pop(0)
            proc.terminate()
            proc.wait()
        for path in tmp_files:
            try:
                os.remove(path)
            except OSError:
                pass


def pytest_collection_modifyitems(session, config, items):
    skip_by_version = []
    for item in items[:]:
        marker = item.get_closest_marker("redis_version")
        if marker is not None:
            try:
                version = VERSIONS[item.callspec.getparam("server_bin")]
            except (KeyError, ValueError, AttributeError):
                # TODO: throw noisy warning
                continue
            if version < marker.kwargs["version"]:
                skip_by_version.append(item)
                item.add_marker(pytest.mark.skip(reason=marker.kwargs["reason"]))
        if "ssl_proxy" in item.fixturenames:
            item.add_marker(
                pytest.mark.skipif(
                    "not os.path.exists('/usr/bin/socat')",
                    reason="socat package required (apt-get install socat)",
                )
            )
    if len(items) != len(skip_by_version):
        for i in skip_by_version:
            items.remove(i)


def pytest_configure(config):
    bins = config.getoption("--redis-server")[:]
    cmd = "which redis-server"
    if not bins:
        with os.popen(cmd) as pipe:
            path = pipe.read().rstrip()
        assert path, (
            "There is no redis-server on your computer." " Please install it first"
        )
        REDIS_SERVERS[:] = [path]
    else:
        REDIS_SERVERS[:] = bins

    VERSIONS.update({srv: _read_server_version(srv) for srv in REDIS_SERVERS})
    assert VERSIONS, ("Expected to detect redis versions", REDIS_SERVERS)

    class DynamicFixturePlugin:
        @pytest.fixture(scope="session", params=REDIS_SERVERS, ids=format_version)
        def server_bin(self, request):
            """Common for start_server and start_sentinel
            server bin path parameter.
            """
            return request.param

    config.pluginmanager.register(DynamicFixturePlugin(), "server-bin-fixture")

    if config.getoption("--uvloop"):
        try:
            import uvloop
        except ImportError:
            raise RuntimeError("Can not import uvloop, make sure it is installed")
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def _gen_cluster_mock_resp(r, response):
    connection = AsyncMock()
    connection.read_response.return_value = response
    r.connection = connection
    return r


@pytest.fixture()
async def mock_cluster_resp_ok(r):
    return _gen_cluster_mock_resp(r, "OK")


@pytest.fixture()
async def mock_cluster_resp_int(r):
    return _gen_cluster_mock_resp(r, "2")


@pytest.fixture()
async def mock_cluster_resp_info(r, event_loop, **kwargs):
    response = (
        "cluster_state:ok\r\ncluster_slots_assigned:16384\r\n"
        "cluster_slots_ok:16384\r\ncluster_slots_pfail:0\r\n"
        "cluster_slots_fail:0\r\ncluster_known_nodes:7\r\n"
        "cluster_size:3\r\ncluster_current_epoch:7\r\n"
        "cluster_my_epoch:2\r\ncluster_stats_messages_sent:170262\r\n"
        "cluster_stats_messages_received:105653\r\n"
    )
    return _gen_cluster_mock_resp(r, response)


@pytest.fixture()
async def mock_cluster_resp_nodes(r):
    response = (
        "c8253bae761cb1ecb2b61857d85dfe455a0fec8b 172.17.0.7:7006 "
        "slave aa90da731f673a99617dfe930306549a09f83a6b 0 "
        "1447836263059 5 connected\n"
        "9bd595fe4821a0e8d6b99d70faa660638a7612b3 172.17.0.7:7008 "
        "master - 0 1447836264065 0 connected\n"
        "aa90da731f673a99617dfe930306549a09f83a6b 172.17.0.7:7003 "
        "myself,master - 0 0 2 connected 5461-10922\n"
        "1df047e5a594f945d82fc140be97a1452bcbf93e 172.17.0.7:7007 "
        "slave 19efe5a631f3296fdf21a5441680f893e8cc96ec 0 "
        "1447836262556 3 connected\n"
        "4ad9a12e63e8f0207025eeba2354bcf4c85e5b22 172.17.0.7:7005 "
        "master - 0 1447836262555 7 connected 0-5460\n"
        "19efe5a631f3296fdf21a5441680f893e8cc96ec 172.17.0.7:7004 "
        "master - 0 1447836263562 3 connected 10923-16383\n"
        "fbb23ed8cfa23f17eaf27ff7d0c410492a1093d6 172.17.0.7:7002 "
        "master,fail - 1447829446956 1447829444948 1 disconnected\n"
    )
    return _gen_cluster_mock_resp(r, response)


@pytest.fixture()
async def mock_cluster_resp_slaves(create_redis, server, **kwargs):
    r = await create_redis(server.tcp_address, **kwargs)
    response = (
        "['1df047e5a594f945d82fc140be97a1452bcbf93e 172.17.0.7:7007 "
        "slave 19efe5a631f3296fdf21a5441680f893e8cc96ec 0 "
        "1447836789290 3 connected']"
    )
    return _gen_cluster_mock_resp(r, response)


@pytest.fixture(scope="session")
def master_host(server):
    yield server.tcp_address.host


async def wait_for_command(client: aioredis.Redis, monitor: Monitor, command: str):
    # issue a command with a key name that's local to this process.
    # if we find a command with our key before the command we're waiting
    # for, something went wrong
    info = await client.info()
    redis_version = (await client.info())["redis_version"]
    if StrictVersion(redis_version) >= StrictVersion("5.0.0"):
        id_str = str(await client.client_id())
    else:
        id_str = "%08x" % random.randrange(2 ** 32)
    key = "__REDIS-PY-%s__" % id_str
    await client.get(key)
    while True:
        monitor_response = await monitor.next_command()
        if command in monitor_response["command"]:
            return monitor_response
        if key in monitor_response["command"]:
            return None
