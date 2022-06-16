# -*- coding: utf-8 -*-
"""PipeSwitch Redis Servers

This module implements classes for Redis servers to be used
by the manager, runners, and clients.

Todo:
    * None
"""

from threading import Event, Thread
from abc import ABC, abstractmethod
from torch.multiprocessing import Queue
from typing import Any, List, OrderedDict, Tuple
from redis import Redis
from pprint import pformat

from pipeswitch.common.consts import Timers
from pipeswitch.common.logger import logger
from pipeswitch.profiling.timer import timer


class RedisServer(ABC, Thread):
    """Redis Server abstract base class.

    It has two main functions:
        - Subscribes to one channel and receives messages.
        - Publishes to another channel.

    Attributes:
        _server_name (`str`): Name of the server.
        _host (`str`): Redis host IP address.
        _port (`int`): Redis port number.
        _redis_pub (`redis.Redis`): Redis client for publishing.
        pub_stream (`str`): Redis channel to publish to.
        _pub_queue (`Queue[str]`): Queue for publishing messages.
        _redis_sub (`redis.Redis`): Redis client for subscribing.
        sub_stream (`str`): Redis channel to subscribe to.
        _sub_queue (`Queue[str]`): Queue for receiving messages.
        _module_id (`int`, optional): ID of the module the server is under.
        _worker_id (`int`, optional): ID of the worker the server is under.
        _client_id (`str`, optional): ID of the client.
    """

    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        host: str,
        port: int,
        sub_queue: "Queue[OrderedDict[str, Any]]" = Queue(),
        client_id: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._stop: Event = Event()
        self._host: str = host
        self._port: int = port
        self._client_id: str = client_id
        self._sub_queue: "Queue[OrderedDict[str, Any]]" = sub_queue
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )
        self._msg_id: str = ""
        self._msg_count: int = 0
        self._msg_limit: int = 10

    def run(self) -> None:
        if self._redis.ping():
            self._ready = True
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
        else:
            logger.error(f"{self._server_name}: connection failed!")
        while not self._stop.is_set():
            msg: Tuple[str, OrderedDict[str, Any]] = self._redis.xread(
                streams={
                    self.sub_stream: self._msg_id if self._msg_id else "$"
                },
                count=None,
                block=0,
            )
            if msg is not None:
                self._process_msg(msg)

    @timer(Timers.PERF_COUNTER)
    def publish(self, msg: OrderedDict[str, Any]) -> None:
        if self.pub_stream == "":
            return
        logger.debug(
            f"{self._server_name}: Publishing msg to stream"
            f" {self.pub_stream}\n{pformat(object=msg, indent=1, width=1)}"
        )
        self._redis.xadd(self.pub_stream, msg)

    @timer(Timers.PERF_COUNTER)
    def delete_streams(self) -> None:
        logger.debug(f"{self._server_name}: Deleting stream: {self.pub_stream}")
        self._redis.delete(self.pub_stream)
        logger.debug(f"{self._server_name}: Deleting stream: {self.sub_stream}")
        self._redis.delete(self.sub_stream)

    @timer(Timers.THREAD_TIMER)
    def _process_msg(self, msg: List[Any]) -> None:
        logger.debug(
            f"{self._server_name}: Message"
            f" received:\n{pformat(object=msg, indent=1, width=1)}"
        )
        for msg_item in msg:
            entry_id, entry = msg_item[1][0]
            self._sub_queue.put(entry)
            self._msg_id = entry_id
        self._msg_count += 1
        if self._msg_count > self._msg_limit:
            logger.debug(
                f"{self._server_name}: Clearing old messages from stream"
                f" {self.sub_stream}"
            )
            self._redis.xtrim(name=self.sub_stream, minid=self._msg_id)
            self._msg_count = 0

    def shutdown(self) -> None:
        self._stop.set()
        self.delete_streams()

    @property
    def _server_name(self) -> str:
        if self._client_id != "":
            return f"{self.__class__.__name__}-{self._client_id}"
        else:
            return self.__class__.__name__

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    @abstractmethod
    def pub_stream(self) -> str:
        pass

    @property
    @abstractmethod
    def sub_stream(self) -> str:
        pass


class ManagerClientConnectionServer(RedisServer):
    """Server in the manager that:
    - Receives connection requests from clients.
    - Sends handshake back to clients to confirm connection.
    """

    @property
    def pub_stream(self) -> str:
        return "HANDSHAKES"

    @property
    def sub_stream(self) -> str:
        return "CONNECTIONS"


class ManagerClientRequestsServer(RedisServer):
    """Server in the manager that:
    - Receives inference requests from a specific client.
    - Sends inference results to the specific client.
    """

    @property
    def pub_stream(self) -> str:
        if self._client_id != "":
            return f"RESPONSES-{self._client_id}"
        else:
            return "RESPONSES"

    @property
    def sub_stream(self) -> str:
        if self._client_id != "":
            return f"REQUESTS-{self._client_id}"
        else:
            return "REQUESTS"


class ClientConnectionServer(RedisServer):
    """Dummy client server that:
    - Sends connection requests to the manager.
    - Receives handshakes from the manager with the name of the requests server.
    """

    @property
    def pub_stream(self) -> str:
        return "CONNECTIONS"

    @property
    def sub_stream(self) -> str:
        return "HANDSHAKES"


class ClientRequestsServer(RedisServer):
    """Dummy client server that:
    - Sends inference requests to the manager.
    - Receives inference results from the manager.
    """

    @property
    def pub_stream(self) -> str:
        if self._client_id != "":
            return f"REQUESTS-{self._client_id}"
        else:
            return "REQUESTS"

    @property
    def sub_stream(self) -> str:
        if self._client_id != "":
            return f"RESPONSES-{self._client_id}"
        else:
            return "RESPONSES"
