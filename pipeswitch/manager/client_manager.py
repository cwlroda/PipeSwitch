# -*- coding: utf-8 -*-
from threading import Thread
from typing import Any, OrderedDict
from torch.multiprocessing import Event, Process, Queue

from pipeswitch.common.consts import (
    ConnectionRequest,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    State,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ManagerClientConnectionServer,
    ManagerClientRequestsServer,
    RedisServer,
)
from pipeswitch.profiling.timer import timer


class ClientManager(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        requests_queue: Queue,
        results_queue: Queue,
        client_blacklist: OrderedDict[str, int],
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop: Event = Event()
        self._max_clients: int = 10
        self._clients: OrderedDict[str, RedisServer] = OrderedDict()
        self._conn_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._requests_queue: "Queue[OrderedDict[str, Any]]" = requests_queue
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._tasks_complete: int = 0
        self._tasks_failed: int = 0
        self._client_blacklist: OrderedDict[str] = client_blacklist

    def run(self) -> None:
        logger.debug(f"{self._name}: Start")
        self._conn_server: RedisServer = ManagerClientConnectionServer(
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._conn_queue,
        )
        self._conn_server.daemon = True
        self._conn_server.start()
        self._results_checker = Thread(target=self._check_results)
        self._results_checker.daemon = True
        self._results_checker.start()
        while not self._stop.is_set():
            conn: OrderedDict[str, Any] = self._conn_queue.get()
            self._manage_connections(conn)

    @timer(Timers.THREAD_TIMER)
    def _manage_connections(self, conn: OrderedDict[str, Any]) -> None:
        if conn["request"] == ConnectionRequest.CONNECT.value:
            logger.info(
                f"{self._name}: Received connection request from"
                f" client {conn['client_id']}"
            )
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
            logger.info(
                f"{self._name}: Received disconnection request from"
                f" client {conn['client_id']}"
            )
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

    @timer(Timers.THREAD_TIMER)
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
        self._conn_server.publish(resp)
        return ok

    @timer(Timers.THREAD_TIMER)
    def _disconnect(self, conn: OrderedDict[str, Any]) -> None:
        self._clients[conn["client_id"]].delete_streams()
        self._client_blacklist[conn["client_id"]] = 1
        logger.info(f"{self._name}: Disconnected client {conn['client_id']}")
        self._clients.pop(conn["client_id"])
        resp = {
            "client_id": conn["client_id"],
            "status": ResponseStatus.OK.value,
        }
        ok = True
        self._conn_server.publish(resp)
        return ok

    def _check_results(self) -> None:
        while self._stop:
            result: OrderedDict[str, Any] = self._results_queue.get()
            self._check_result(result)

    @timer(Timers.THREAD_TIMER)
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
        if hasattr(self, "_conn_server"):
            return self._conn_server.ready

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop.set()
        if hasattr(self, "_results_checker"):
            if self._results_checker.is_alive():
                self._results_checker.terminate()
        for client_server in self._clients.values():
            if client_server.is_alive():
                client_server.shutdown()
                client_server.terminate()
        if hasattr(self, "_conn_server"):
            if self._conn_server.is_alive():
                self._conn_server.shutdown()
                self._conn_server.terminate()
        logger.debug(f"{self._name}: stopped!")
