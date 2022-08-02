# -*- coding: utf-8 -*-
import json
from time import perf_counter, sleep
from typing import Dict, List
from multiprocessing import Process
from threading import Thread

from scalabel_bot.common.consts import (
    ConnectionRequest,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    Timers,
)
from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import Message
from scalabel_bot.profiling.timer import timer
from scalabel_bot.server.stream import (
    ManagerConnectionsStream,
)


class ClientManager(Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self, clients: Dict[str, int], clients_queue: List[Message]
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self._clients: Dict[str, int] = clients
        self._clients_queue: List[Message] = clients_queue
        self._connection_timeout = 10

    def run(self) -> None:
        try:
            self._conn_server: ManagerConnectionsStream = (
                ManagerConnectionsStream(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    sub_queue=self._clients_queue,
                )
            )
            self._conn_server.run()
            self._clients_checker = Thread(target=self._check_clients)
            self._clients_checker.daemon = True
            self._clients_checker.start()
            while self._do_run:
                if self._clients_queue:
                    self._manage_connections(self._clients_queue.pop(0))

        except KeyboardInterrupt as _:
            return

    @timer(Timers.PERF_COUNTER)
    def _manage_connections(self, conn: Message) -> None:
        logger.info(
            "ClientManager: Received connection request from"
            f" client {conn['clientId']}"
        )
        if conn["request"] == str(ConnectionRequest.CONNECT):
            self._connect(conn)
        elif conn["request"] == str(ConnectionRequest.DISCONNECT):
            self._disconnect(conn)
        else:
            logger.debug(
                "ClientManager: Unknown connection request from"
                f" client {conn['clientId']}"
            )

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: Message) -> None:
        self._clients[conn["clientId"]] = self._connection_timeout
        resp: str = {
            "clientId": conn["clientId"],
            "status": str(ResponseStatus.OK),
            "requests_stream": "REQUESTS",
            "host": REDIS_HOST,
            "port": REDIS_PORT,
        }
        msg: Dict[str, str] = {"message": json.dumps(resp)}
        self._conn_server.publish(conn["channel"], msg)

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self, conn: Message) -> None:
        self._clients.pop(conn["clientId"])
        resp = {
            "clientId": conn["clientId"],
            "status": str(ResponseStatus.OK),
        }
        msg: Dict[str, str] = {"message": json.dumps(resp)}
        self._conn_server.publish(conn["channel"], msg)

    def _check_clients(self) -> None:
        while self._do_run:
            sleep(1)
            if self._clients:
                for client_id in self._clients.keys():
                    if self._clients[client_id] == 0:
                        self._clients.pop(client_id)
                    else:
                        self._clients[client_id] -= 1

    def ready(self) -> bool:
        return self._conn_server.ready
