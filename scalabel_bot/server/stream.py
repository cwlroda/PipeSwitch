from threading import Thread
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from redis import Redis
from pprint import pformat
import json

from scalabel_bot.common.consts import ESTCT, MODELS, Timers
from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import Message
from scalabel_bot.profiling.timer import timer


class Stream(ABC):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        host: str,
        port: int,
        sub_queue: List[Message],
        idx: str = "",
    ) -> None:
        super().__init__()
        self._ready: bool = False
        self._host: str = host
        self._port: int = port
        self._idx: str = idx
        self._sub_queue: List[Message] = sub_queue
        self._msg_id: str = ""
        self._redis = Redis(
            host=self._host,
            port=self._port,
            encoding="utf-8",
            decode_responses=True,
            retry_on_timeout=True,
        )

    def run(self) -> None:
        if self._redis.ping():
            logger.info(
                f"{self._server_name}: Listening to stream {self.sub_stream}"
            )
            self._ready = True
        else:
            logger.error(f"{self._server_name}: connection failed!")
        listen = Thread(target=self._listen)
        listen.daemon = True
        listen.start()

    def _listen(self):
        while True:
            msg: List[
                List[str | List[Tuple[str, Dict[str, str]]]]
            ] = self._redis.xread(
                streams={
                    self.sub_stream: self._msg_id if self._msg_id else "$"
                },
                count=None,
                block=0,
            )
            if len(msg) > 0:
                self._process_msg(msg)

    @timer(Timers.PERF_COUNTER)
    def publish(self, stream: str, msg: str) -> None:
        logger.spam(
            f"{self._server_name}: Publishing msg to stream"
            f" {stream}\n{pformat(msg)}"
        )
        self._redis.xadd(stream, msg)

    @timer(Timers.PERF_COUNTER)
    def delete_streams(self) -> None:
        logger.debug(f"{self._server_name}: Deleting stream: {self.sub_stream}")
        self._redis.delete(self.sub_stream)

    @timer(Timers.PERF_COUNTER)
    @abstractmethod
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        pass

    def shutdown(self) -> None:
        self.delete_streams()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def _server_name(self) -> str:
        if self._idx != "":
            return f"{self.__class__.__name__}-{self._idx}"
        return self.__class__.__name__

    @property
    @abstractmethod
    def pub_stream(self) -> str:
        pass

    @property
    @abstractmethod
    def sub_stream(self) -> str:
        pass


class ManagerRequestsStream(Stream):
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
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.debug(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            if "message" in entry.keys():
                data: Message = json.loads(entry["message"])
                data["ect"] = (
                    ESTCT[data["type"]][MODELS[data["taskType"]]]
                    * data["dataSize"]
                )
                data["wait"] = 0
                self._sub_queue.append(data)
            self._msg_id = entry_id
        logger.spam(f"{self._server_name}: Deleting message\n{pformat(msg)}")
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return super().pub_stream

    @property
    def sub_stream(self) -> str:
        return "REQUESTS"


class ClientRequestsStream(Stream):
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
    def _process_msg(
        self, msg: List[List[str | List[Tuple[str, Dict[str, str]]]]]
    ) -> None:
        logger.spam(f"{self._server_name}: Message received:\n{pformat(msg)}")
        for msg_item in msg:
            if isinstance(msg_item[1], str):
                continue
            entry_id, entry = msg_item[1][0]
            data: Message = json.loads(entry["message"])
            self._sub_queue.put(data)
            self._msg_id = entry_id
        logger.spam(f"{self._server_name}: Deleting message\n{pformat(msg)}")
        self._redis.xtrim(self.sub_stream, minid=self._msg_id)

    @property
    def pub_stream(self) -> str:
        return "REQUESTS"

    @property
    def sub_stream(self) -> str:
        return f"RESPONSES_{self._server_name}"
