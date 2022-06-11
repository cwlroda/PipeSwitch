# -*- coding: utf-8 -*-
import time
from threading import Thread
from typing import Any, OrderedDict
import multiprocessing as mp

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


class ClientManager(mp.Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self, requests_queue: mp.Queue, results_queue: mp.Queue
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self.max_clients: int = 10
        self.clients: OrderedDict[str, RedisServer] = OrderedDict()
        self.requests_queue: mp.Queue = requests_queue
        self.results_queue: mp.Queue = results_queue
        self.num_tasks_complete: int = 0
        self.num_tasks_failed: int = 0

    def run(self) -> None:
        try:
            self.conn_server: RedisServer = ManagerClientConnectionServer(
                host=REDIS_HOST,
                port=REDIS_PORT,
            )
            self.conn_server.daemon = True
            self.conn_server.start()
            check_results = Thread(target=self._check_results)
            check_results.daemon = True
            check_results.start()
            while self._do_run:
                self._manage_connections(self.conn_server.sub_queue.get())
        except KeyboardInterrupt as _:
            return

    @timer(Timers.PERF_COUNTER)
    def _manage_connections(self, conn: OrderedDict[str, Any]) -> None:
        logger.info(
            "ClientManager: Received connection request from"
            f" client {conn['client_id']}"
        )
        if conn["request"] == str(ConnectionRequest.CONNECT):
            resp = self._connect(conn)
            if resp:
                logger.debug(
                    "ClientManager: Connection to client"
                    f" {conn['client_id']} established"
                )
            else:
                logger.debug(
                    "ClientManager: Connection to client"
                    f" {conn['client_id']} rejected"
                )
        elif conn["request"] == str(ConnectionRequest.DISCONNECT):
            resp = self._disconnect(conn)
            if resp:
                logger.debug(
                    "ClientManager: Connection to client"
                    f" {conn['client_id']} closed"
                )
            else:
                logger.debug(
                    "ClientManager: Unknown connection request from"
                    f" client {conn['client_id']}"
                )
        else:
            logger.debug(
                "ClientManager: Unknown connection request from"
                f" client {conn['client_id']}"
            )

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: OrderedDict[str, Any]) -> None:
        if len(self.clients) < self.max_clients:
            req_server: RedisServer = ManagerClientRequestsServer(
                client_id=conn["client_id"],
                host=REDIS_HOST,
                port=REDIS_PORT,
            )
            req_server.daemon = True
            req_server.start()
            check_client_reqs: Thread = Thread(
                target=self._check_client_reqs, args=(conn["client_id"],)
            )
            check_client_reqs.daemon = True
            check_client_reqs.start()
            self.clients[conn["client_id"]] = req_server
            msg: str = {
                "client_id": conn["client_id"],
                "status": str(ResponseStatus.OK),
                "requests_stream": req_server.sub_stream,
                "results_stream": req_server.pub_stream,
                "host": REDIS_HOST,
                "port": REDIS_PORT,
            }
            ok = True
        else:
            msg = {
                "client_id": conn["client_id"],
                "status": str(ResponseStatus.ERROR),
                "err_msg": "Max clients reached",
            }
            ok = False
        self.conn_server.pub_queue.put(msg)
        self.conn_server.publish()
        return ok

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self, conn: OrderedDict[str, Any]) -> None:
        self.clients.pop(conn["client_id"])
        resp = {
            "client_id": conn["client_id"],
            "status": str(ResponseStatus.OK),
        }
        ok = True
        self.conn_server.pub_queue.put(resp)
        self.conn_server.publish()
        return ok

    def _check_client_reqs(self, client_id) -> None:
        while client_id not in self.clients:
            time.sleep(0.001)
        client: RedisServer = self.clients[client_id]

        while self._do_run:
            msg: OrderedDict[str, Any] = client.sub_queue.get()
            self._check_client_req(msg)

    @timer(Timers.PERF_COUNTER)
    def _check_client_req(self, task: OrderedDict[str, Any]) -> None:
        logger.info(
            "Client Manager: Received task"
            f" {task['model_name']} {task['task_type']} with id"
            f" {task['task_id']} from client {task['client_id']}"
        )
        self.requests_queue.put(task)

    def _check_results(self) -> None:
        while self._do_run:
            result: OrderedDict[str, Any] = self.results_queue.get()
            self._check_result(result)

    @timer(Timers.PERF_COUNTER)
    def _check_result(self, result: OrderedDict[str, Any]) -> None:
        if result["status"] == str(State.SUCCESS):
            self.num_tasks_complete += 1
        else:
            self.num_tasks_failed += 1
        logger.success(f"{self.num_tasks_complete} task(s) complete!")
        if self.num_tasks_failed > 0:
            logger.error(f"{self.num_tasks_failed} task(s) failed!")
        req_server: RedisServer = self.clients[result["client_id"]]
        req_server.pub_queue.put(result)
        req_server.publish()

    def ready(self) -> bool:
        return self.conn_server.ready
