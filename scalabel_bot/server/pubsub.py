from multiprocessing import Event
from threading import Thread
from abc import ABC, abstractmethod
from typing import List
from redis import Redis
import json

from scalabel_bot.common.consts import Timers
from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.profiling.timer import timer


class PubSub(Thread, ABC):
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

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        stop_run: Event,
        host: str,
        port: int,
        sub_queue: List[TaskMessage],
        idx: str = "",
    ) -> None:
        super().__init__()
        self._stop_run: Event = stop_run
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._idx: str = idx
        self._sub_queue: List[TaskMessage] = sub_queue
        self._msg_id: str = ""
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )
        self._pubsub = self._redis.pubsub()

    def run(self) -> None:
        if self._redis.ping():
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
            self._ready = True
        else:
            logger.error(f"{self._server_name}: connection failed!")
        self._listen()

    @timer(Timers.PERF_COUNTER)
    def publish(self, channel: str, msg: str) -> None:
        self._redis.publish(channel, msg)

    def _listen(self) -> None:
        self._pubsub.subscribe(self.sub_stream)
        try:
            while not self._stop_run.is_set():
                for msg in self._pubsub.listen():
                    if msg is not None:
                        if msg["data"] == 1:
                            logger.info(
                                f"{self._server_name}: subscribed to channel"
                                f" {msg['channel']}"
                            )
                        else:
                            logger.debug(
                                f"{self._server_name}: msg received from"
                                f" channel {msg['channel']}"
                            )
                            self._process_msg(msg)
        except KeyboardInterrupt:
            return

    def _process_msg(self, msg: str) -> None:
        data: TaskMessage = json.loads(msg["data"])
        self._sub_queue.append(data)

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

    @property
    def _server_name(self) -> str:
        if self._idx != "":
            return f"{self.__class__.__name__}-{self._idx}"
        return self.__class__.__name__


class ManagerRequestsPubSub(PubSub):
    @property
    def pub_stream(self) -> str:
        return super().pub_stream

    @property
    def sub_stream(self) -> str:
        return "REQUESTS"


class ClientRequestsPubSub(PubSub):
    @property
    def pub_stream(self) -> str:
        return "REQUESTS"

    @property
    def sub_stream(self) -> str:
        return f"RESPONSES_{self._server_name}"
