# -*- coding: utf-8 -*-
"""PipeSwitch Redis Servers

This module implements classes for Redis servers to be used
by the manager, runners, and clients.

Todo:
    * None
"""

import threading
from abc import ABC, abstractmethod
import multiprocessing as mp
from queue import Queue
import redis

from pipeswitch.common.logger import logger


class RedisServer(ABC, threading.Thread):
    """Redis Server abstract base class.

    It has two main functions:
        - Subscribes to one channel and receives messages
        - Publishes to another channel

    Attributes:
        _server_name (`str`): Name of the server.
        _host (`str`): Redis host IP address.
        _port (`int`): Redis port number.
        _redis_pub (`redis.Redis`): Redis client for publishing.
        _pub_channel (`str`): Redis channel to publish to.
        _pub_queue (`Queue[str]`): Queue for publishing messages.
        _redis_sub (`redis.Redis`): Redis client for subscribing.
        _sub_channel (`str`): Redis channel to subscribe to.
        _sub_queue (`Queue[str]`): Queue for receiving messages.
        _module_id (`int`, optional): ID of the module the server is under.
        _worker_id (`int`, optional): ID of the worker the server is under.
    """

    def __init__(
        self, host: str, port: int, module_id: int = -1, worker_id: int = -1
    ) -> None:
        super().__init__()
        self._host: str = host
        self._port: int = port
        self._module_id: int = module_id
        self._worker_id: int = worker_id
        self._pub_queue: Queue[str] = Queue()
        self._sub_queue: Queue[str] = Queue()

    def run(self) -> None:
        self._create_pub()
        self._create_sub()
        self._listen()

    def publish(self) -> bool:
        if self._pub_channel == "":
            return False
        if not self.pub_queue.empty():
            item = self.pub_queue.get(block=False)
            logger.debug(
                f"{self._server_name}: publishing msg to channel"
                f" {self._pub_channel}"
            )
            self._redis_pub.publish(self._pub_channel, item)
            return True
        return False

    def _create_pub(self) -> None:
        self._redis_pub = redis.Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
        )
        assert (
            self._redis_pub.ping()
        ), f"{self._server_name}-pub: connection failed!"

    def _create_sub(self) -> None:
        self._redis_sub = redis.Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
        )
        assert (
            self._redis_sub.ping()
        ), f"{self._server_name}-sub: connection failed!"
        self._pubsub: redis.client.PubSub = self._redis_sub.pubsub()

    def _listen(self) -> None:
        if self._sub_channel == "":
            return
        self._pubsub.subscribe(self._sub_channel)
        for msg in self._pubsub.listen():  # type: ignore
            if msg is not None:
                if msg["data"] == 1:
                    logger.info(
                        f"{self._server_name}: subscribed to channel"
                        f" {msg['channel']}"
                    )
                else:
                    logger.debug(
                        f"{self._server_name}: msg received from channel"
                        f" {msg['channel']}"
                    )
                    self.sub_queue.put(msg["data"])

    @property
    @abstractmethod
    def _server_name(self) -> str:
        pass

    @property
    @abstractmethod
    def _pub_channel(self) -> str:
        pass

    @property
    @abstractmethod
    def _sub_channel(self) -> str:
        pass

    @property
    def module_id(self) -> int:
        return self._module_id

    @property
    def worker_id(self) -> int:
        return self._worker_id

    @property
    def sub_queue(self) -> "Queue[str]":
        return self._sub_queue

    @property
    def pub_queue(self) -> "Queue[str]":
        return self._pub_queue


class SchedulerCommsServer(RedisServer):
    """Server in the manager scheduler that records status of runners."""

    def __init__(self, host: str, port: int, module_id: int = -1) -> None:
        super().__init__(host, port, module_id)

    @property
    def _server_name(self) -> str:
        return "SchedulerCommsServer"

    @property
    def _pub_channel(self) -> str:
        return ""

    @property
    def _sub_channel(self) -> str:
        return "STATUS"


class ManagerRequestsServer(RedisServer):
    """Server in the manager that:
    - Receives inference requests from clients
    - Sends inference requests in the form of tasks to the runners
    """

    @property
    def _server_name(self) -> str:
        return "ManagerRequestsServer"

    @property
    def _pub_channel(self) -> str:
        return "TASKS"

    @property
    def _sub_channel(self) -> str:
        return "REQUESTS"


class ManagerResultsServer(RedisServer):
    """Server in the manager that:
    - Receives task results from runners
    - Sends inference results to the clients
    """

    @property
    def _server_name(self) -> str:
        return "ManagerResultsServer"

    @property
    def _pub_channel(self) -> str:
        return "RESPONSES"

    @property
    def _sub_channel(self) -> str:
        return "RESULTS"


class RunnerCommsServer(RedisServer):
    """Server in the runner that updates runner status."""

    @property
    def _server_name(self) -> str:
        return f"RunnerCommsServer-{self._module_id}"

    @property
    def _pub_channel(self) -> str:
        return "STATUS"

    @property
    def _sub_channel(self) -> str:
        return "STATUS"


class RunnerTaskServer(RedisServer):
    """Server in the runner that:
    - Receives tasks from the manager
    - Sends task results to the manager
    """

    @property
    def _server_name(self) -> str:
        return f"RunnerTaskServer-{self._module_id}"

    @property
    def _pub_channel(self) -> str:
        return "RESULTS"

    @property
    def _sub_channel(self) -> str:
        return "TASKS"


class WorkerCommsServer:
    """Server in the worker that updates worker status.

    Attributes:
        _server_name (`str`): Name of the server.
        _host (`str`): Redis host IP address.
        _port (`int`): Redis port number.
        _redis_pub (`redis.Redis`): Redis client for publishing.
        _pub_channel (`str`): Redis channel to publish to.
        _pub_queue (`Queue[str]`): Queue for publishing messages.
        _module_id (`int`, optional): ID of the module the server is under.
        _worker_id (`int`, optional): ID of the worker the server is under.
    """

    def __init__(
        self, host: str, port: int, module_id: int = -1, worker_id: int = -1
    ) -> None:
        super().__init__()
        self._host: str = host
        self._port: int = port
        self._module_id: int = module_id
        self._worker_id: int = worker_id
        self._pub_queue: mp.Queue = mp.Queue()

    def publish(self) -> bool:
        if self._pub_channel == "":
            return False
        if not self.pub_queue.empty():
            item = self.pub_queue.get(block=False)
            logger.debug(
                f"{self._server_name}: publishing msg to channel"
                f" {self._pub_channel}"
            )
            self._redis_pub.publish(self._pub_channel, item)
            return True
        return False

    def create_pub(self) -> None:
        self._redis_pub = redis.Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
        )
        assert (
            self._redis_pub.ping()
        ), f"{self._server_name}-pub: connection failed!"

    @property
    def _server_name(self) -> str:
        return f"WorkerCommsServer-{self._module_id}"

    @property
    def _pub_channel(self) -> str:
        return "STATUS"

    @property
    def pub_queue(self) -> mp.Queue:
        return self._pub_queue


class ClientServer(RedisServer):
    """Dummy client server that:
    - Sends inference requests to the manager
    - Receives inference results from the manager
    """

    @property
    def _server_name(self) -> str:
        return "ClientServer"

    @property
    def _pub_channel(self) -> str:
        return "REQUESTS"

    @property
    def _sub_channel(self) -> str:
        return "RESPONSES"
