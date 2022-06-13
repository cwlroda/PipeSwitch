# -*- coding: utf-8 -*-
from threading import Thread
from typing import Any, OrderedDict
from torch.multiprocessing import Process, Queue

from pipeswitch.common.consts import (
    ConnectionRequest,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    State,
    timer,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ManagerClientConnectionServer,
    ManagerClientRequestsServer,
    RedisServer,
)


class ClientManager(Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        requests_queue: Queue,
        results_queue: Queue,
        client_blacklist: OrderedDict[str, int],
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._do_run: bool = True
        self._max_clients: int = 10
        self._clients: OrderedDict[str, RedisServer] = OrderedDict()
        self._conn_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._requests_queue: "Queue[OrderedDict[str, Any]]" = requests_queue
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._tasks_complete: int = 0
        self._tasks_failed: int = 0
        self._client_blacklist: OrderedDict[str] = client_blacklist
        self.conn_server: RedisServer = ManagerClientConnectionServer(
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._conn_queue,
        )

    def run(self) -> None:
        try:
            self.conn_server.daemon = True
            self.conn_server.start()
            check_results = Thread(target=self._check_results)
            check_results.daemon = True
            check_results.start()
            while self._do_run:
                self._manage_connections(self._conn_queue.get())
        except KeyboardInterrupt as _:
            return

    @timer(Timers.PERF_COUNTER)
    def _manage_connections(self, conn: OrderedDict[str, Any]) -> None:
        logger.info(
            f"{self._name}: Received connection request from"
            f" client {conn['client_id']}"
        )
        if conn["request"] == ConnectionRequest.CONNECT.value:
            resp = self._connect(conn)
            if resp:
                logger.debug(
                    f"{self._name}: Connection to client"
                    f" {conn['client_id']} established"
                )
            else:
                logger.debug(
                    f"{self._name}: Connection to client"
                    f" {conn['client_id']} rejected"
                )
        elif conn["request"] == ConnectionRequest.DISCONNECT.value:
            resp = self._disconnect(conn)
            if resp:
                logger.debug(
                    f"{self._name}: Connection to client"
                    f" {conn['client_id']} closed"
                )
            else:
                logger.debug(
                    f"{self._name}: Unknown connection request from"
                    f" client {conn['client_id']}"
                )
        else:
            logger.debug(
                f"{self._name}: Unknown connection request from"
                f" client {conn['client_id']}"
            )

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: OrderedDict[str, Any]) -> None:
        if len(self._clients) < self._max_clients:
            if conn["client_id"] in self._client_blacklist.keys():
                self._client_blacklist.pop(conn["client_id"])
            req_server: RedisServer = ManagerClientRequestsServer(
                client_id=conn["client_id"],
                host=REDIS_HOST,
                port=REDIS_PORT,
                sub_queue=self._requests_queue,
            )
            req_server.daemon = True
            req_server.start()
            self._clients[conn["client_id"]] = req_server
            resp: str = {
                "client_id": conn["client_id"],
                "status": ResponseStatus.OK.value,
                "requests_stream": req_server.sub_stream,
                "results_stream": req_server.pub_stream,
                "host": REDIS_HOST,
                "port": REDIS_PORT,
            }
            ok = True
        else:
            resp = {
                "client_id": conn["client_id"],
                "status": ResponseStatus.ERROR.value,
                "err_msg": "Max clients reached",
            }
            ok = False
        self.conn_server.publish(resp)
        return ok

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self, conn: OrderedDict[str, Any]) -> None:
        self._clients[conn["client_id"]].delete_streams()
        self._client_blacklist[conn["client_id"]] = 1
        logger.info(f"{self._name}: Blacklisted client {conn['client_id']}")
        self._clients.pop(conn["client_id"])
        resp = {
            "client_id": conn["client_id"],
            "status": ResponseStatus.OK.value,
        }
        ok = True
        self.conn_server.publish(resp)
        return ok

    def _check_results(self) -> None:
        while self._do_run:
            result: OrderedDict[str, Any] = self._results_queue.get()
            self._check_result(result)

    @timer(Timers.PERF_COUNTER)
    def _check_result(self, result: OrderedDict[str, Any]) -> None:
        if result["client_id"] not in self._client_blacklist.keys():
            if result["status"] == State.SUCCESS:
                self._tasks_complete += 1
            else:
                self._tasks_failed += 1
            logger.success(
                f"{self._name}: {self._tasks_complete} task(s) complete!"
            )
            if self._tasks_failed > 0:
                logger.error(
                    f"{self._name}: {self._tasks_failed} task(s) failed!"
                )
            result["status"] = result["status"].value
            self._clients[result["client_id"]].publish(result)
        else:
            logger.warning(
                f"{self._name}: Discarding results of stale task"
                f" {result['model_name']} {result['task_type']} with id"
                f" {result['task_id']} from client {result['client_id']}"
            )

    def ready(self) -> bool:
        return self.conn_server.ready
