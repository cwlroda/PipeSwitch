# -*- coding: utf-8 -*-
"""PipeSwitch Redis Servers

This module implements classes for Redis servers to be used
by the manager, runners, and clients.

Todo:
    * None
"""

import time
from threading import Thread
from abc import ABC, abstractmethod
from queue import Queue  # pylint: disable=unused-import
import redis

from pipeswitch.common.logger import logger


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
        pub_channel (`str`): Redis channel to publish to.
        _pub_queue (`Queue[str]`): Queue for publishing messages.
        _redis_sub (`redis.Redis`): Redis client for subscribing.
        sub_channel (`str`): Redis channel to subscribe to.
        _sub_queue (`Queue[str]`): Queue for receiving messages.
        _module_id (`int`, optional): ID of the module the server is under.
        _worker_id (`int`, optional): ID of the worker the server is under.
        _client_id (`str`, optional): ID of the client.
    """

    def __init__(
        self,
        host: str,
        port: int,
        pub_queue: "Queue[str]",
        sub_queue: "Queue[str]",
        module_id: int = -1,
        worker_id: int = -1,
        client_id="",
    ) -> None:
        super().__init__()
        self.do_run: bool = True
        self._host: str = host
        self._port: int = port
        self._module_id: int = module_id
        self._worker_id: int = worker_id
        self._client_id: str = client_id
        self._pub_queue: "Queue[str]" = pub_queue
        self._sub_queue: "Queue[str]" = sub_queue

    def run(self) -> None:
        try:
            self._create_pub()
            self._create_sub()
            listen = Thread(target=self._listen)
            listen.daemon = True
            listen.start()
            while self.do_run:
                time.sleep(100)
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    def publish(self) -> bool:
        if self.pub_channel == "":
            return True
        if not self.pub_queue.empty():
            item = self.pub_queue.get()
            logger.debug(
                f"{self._server_name}: publishing msg to channel"
                f" {self.pub_channel}"
            )
            self._redis_pub.publish(self.pub_channel, item)
            return True
        return False

    def _create_pub(self) -> None:
        self._redis_pub = redis.Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
        )
        if not self._redis_pub.ping():
            logger.error(f"{self._server_name}-pub: connection failed!")

    def _create_sub(self) -> None:
        self._redis_sub = redis.Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
        )
        if not self._redis_sub.ping():
            logger.error(f"{self._server_name}-sub: connection failed!")
        self._pubsub: redis.client.PubSub = self._redis_sub.pubsub()

    def _listen(self) -> None:
        if self.sub_channel == "":
            return
        try:
            self._pubsub.subscribe(self.sub_channel)
            while True:
                for msg in self._pubsub.listen():  # type: ignore
                    if msg is not None:
                        if msg["data"] == 0:
                            logger.error(
                                f"{self._server_name}: error subscribing to"
                                f" channel {msg['channel']}"
                            )
                        elif msg["data"] == 1:
                            logger.info(
                                f"{self._server_name}: subscribed to channel"
                                f" {msg['channel']}"
                            )
                        else:
                            logger.debug(
                                f"{self._server_name}: msg received from"
                                f" channel {msg['channel']}"
                            )
                            self.sub_queue.put(msg["data"])
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    @property
    @abstractmethod
    def _server_name(self) -> str:
        pass

    @property
    @abstractmethod
    def pub_channel(self) -> str:
        pass

    @property
    @abstractmethod
    def sub_channel(self) -> str:
        pass

    @property
    def module_id(self) -> int:
        return self._module_id

    @property
    def worker_id(self) -> int:
        return self._worker_id

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def sub_queue(self) -> "Queue[str]":
        return self._sub_queue

    @property
    def pub_queue(self) -> "Queue[str]":
        return self._pub_queue


class SchedulerCommsServer(RedisServer):
    """Server in the manager scheduler that records status of runners."""

    @property
    def _server_name(self) -> str:
        return f"SchedulerCommsServer-{self.module_id}"

    @property
    def pub_channel(self) -> str:
        return ""

    @property
    def sub_channel(self) -> str:
        return f"RUNNER_STATUS_{self.module_id}"


class ManagerClientConnectionServer(RedisServer):
    """Server in the manager that:
    - Receives connection requests from clients.
    - Sends handshake back to clients to confirm connection.
    """

    @property
    def _server_name(self) -> str:
        return "ManagerClientConnectionServer"

    @property
    def pub_channel(self) -> str:
        return "HANDSHAKES"

    @property
    def sub_channel(self) -> str:
        return "CONNECTIONS"


class ManagerClientRequestsServer(RedisServer):
    """Server in the manager that:
    - Receives inference requests from a specific client.
    - Sends inference results to the specific client.
    """

    @property
    def _server_name(self) -> str:
        if self.client_id != "":
            return f"ManagerClientRequestsServer-{self.client_id}"
        else:
            return "ManagerClientRequestsServer"

    @property
    def pub_channel(self) -> str:
        if self.client_id != "":
            return f"RESPONSES-{self.client_id}"
        else:
            return "RESPONSES"

    @property
    def sub_channel(self) -> str:
        if self.client_id != "":
            return f"REQUESTS-{self.client_id}"
        else:
            return "REQUESTS"


class ManagerTasksServer(RedisServer):
    """Server in the manager that:
    - Sends tasks to a specific runner.
    - Receives task results from the specific runner.
    """

    @property
    def _server_name(self) -> str:
        if self.module_id != -1:
            return f"ManagerTasksServer-{self.module_id}"
        else:
            return "ManagerTasksServer"

    @property
    def pub_channel(self) -> str:
        if self.module_id != -1:
            return f"TASKS_{self.module_id}"
        else:
            return "TASKS"

    @property
    def sub_channel(self) -> str:
        if self.module_id != -1:
            return f"RESULTS_{self.module_id}"
        else:
            return "RESULTS"


class RunnerCommsServer(RedisServer):
    """Server in the runner that updates runner status."""

    @property
    def _server_name(self) -> str:
        if self.module_id != -1:
            return f"RunnerCommsServer-{self._module_id}"
        else:
            return "RunnerCommsServer"

    @property
    def pub_channel(self) -> str:
        if self.module_id != -1:
            return f"RUNNER_STATUS_{self._module_id}"
        else:
            return "RUNNER_STATUS"

    @property
    def sub_channel(self) -> str:
        if self.module_id != -1:
            return f"WORKER_STATUS_{self._module_id}"
        else:
            return "WORKER_STATUS"


class RunnerTaskServer(RedisServer):
    """Server in the runner that:
    - Receives tasks from the manager
    - Sends task results to the manager
    """

    @property
    def _server_name(self) -> str:
        if self.module_id != -1:
            return f"RunnerTaskServer-{self.module_id}"
        else:
            return "RunnerTaskServer"

    @property
    def pub_channel(self) -> str:
        if self.module_id != -1:
            return f"RESULTS_{self.module_id}"
        else:
            return "RESULTS"

    @property
    def sub_channel(self) -> str:
        if self.module_id != -1:
            return f"TASKS_{self.module_id}"
        else:
            return "TASKS"


class WorkerCommsServer(RedisServer):
    @property
    def _server_name(self) -> str:
        if self._module_id != -1:
            return f"WorkerCommsServer-{self._module_id}"
        else:
            return "WorkerCommsServer"

    @property
    def pub_channel(self) -> str:
        if self._module_id != -1:
            return f"WORKER_STATUS_{self._module_id}"
        else:
            return "WORKER_STATUS"

    @property
    def sub_channel(self) -> str:
        return ""


# class WorkerCommsServer:
#     """Server in the worker that updates worker status.
#     Attributes:
#         _server_name (`str`): Name of the server.
#         _host (`str`): Redis host IP address.
#         _port (`int`): Redis port number.
#         _redis_pub (`redis.Redis`): Redis client for publishing.
#         _pub_channel (`str`): Redis channel to publish to.
#         _pub_queue (`Queue[str]`): Queue for publishing messages.
#         _module_id (`int`, optional): ID of the module the server is under.
#         _worker_id (`int`, optional): ID of the worker the server is under.
#     """

#     def __init__(
#         self, host: str, port: int, module_id: int = -1, worker_id: int = -1
#     ) -> None:
#         super().__init__()
#         self._host: str = host
#         self._port: int = port
#         self._module_id: int = module_id
#         self._worker_id: int = worker_id
#         self._pub_queue: mp.Queue = mp.Queue()

#     def publish(self) -> bool:
#         if self.pub_channel == "":
#             return True
#         if not self.pub_queue.empty():
#             item = self.pub_queue.get()
#             logger.debug(
#                 f"{self._server_name}: publishing msg to channel"
#                 f" {self.pub_channel}"
#             )
#             self._redis_pub.publish(self.pub_channel, item)
#             return True
#         return False

#     def create_pub(self) -> None:
#         self._redis_pub = redis.Redis(
#             host=self._host,
#             port=self._port,
#             encoding="utf-8",
#             decode_responses=True,
#         )
#         assert (
#             self._redis_pub.ping()
#         ), f"{self._server_name}-pub: connection failed!"

#     @property
#     def _server_name(self) -> str:
#         if self._module_id != -1:
#             return f"WorkerCommsServer-{self._module_id}"
#         else:
#             return "WorkerCommsServer"

#     @property
#     def pub_channel(self) -> str:
#         if self._module_id != -1:
#             return f"WORKER_STATUS_{self._module_id}"
#         else:
#             return "WORKER_STATUS"

#     @property
#     def pub_queue(self) -> mp.Queue:
#         return self._pub_queue


class ClientConnectionServer(RedisServer):
    """Dummy client server that:
    - Sends connection requests to the manager.
    - Receives handshakes from the manager with the name of the requests server.
    """

    @property
    def _server_name(self) -> str:
        if self.client_id != "":
            return f"ClientConnectionServer{self.client_id}"
        else:
            return "ClientConnectionServer"

    @property
    def pub_channel(self) -> str:
        return "CONNECTIONS"

    @property
    def sub_channel(self) -> str:
        return "HANDSHAKES"


class ClientRequestsServer(RedisServer):
    """Dummy client server that:
    - Sends inference requests to the manager.
    - Receives inference results from the manager.
    """

    @property
    def _server_name(self) -> str:
        if self.client_id != "":
            return f"ClientRequestsServer-{self.client_id}"
        else:
            return "ClientRequestsServer"

    @property
    def pub_channel(self) -> str:
        if self.client_id != "":
            return f"REQUESTS-{self.client_id}"
        else:
            return "REQUESTS"

    @property
    def sub_channel(self) -> str:
        if self.client_id != "":
            return f"RESPONSES-{self.client_id}"
        else:
            return "RESPONSES"
