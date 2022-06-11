# -*- coding: utf-8 -*-
import time
import json
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
        self.do_run: bool = True
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
            while self.do_run:
                msg: str = self.conn_server.sub_queue.get()
                self._manage_connections(msg)
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    @timer(Timers.PERF_COUNTER)
    def _manage_connections(self, msg) -> None:
        try:
            conn: OrderedDict[str, Any] = json.loads(msg)
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
        except TypeError as json_decode_err:
            logger.debug(json_decode_err)
            logger.debug(f"Ignoring invalid client connection request: {msg}")
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _connect(self, conn: OrderedDict[str, Any]) -> None:
        try:
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
                    "requests_channel": req_server.sub_channel,
                    "results_channel": req_server.pub_channel,
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
            self.conn_server.pub_queue.put(json.dumps(msg))
            self.conn_server.publish()
            return ok
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self, conn: OrderedDict[str, Any]) -> None:
        try:
            self.clients.pop(conn["client_id"])
            resp = {
                "client_id": conn["client_id"],
                "status": str(ResponseStatus.OK),
            }
            ok = True
            self.conn_server.pub_queue.put(json.dumps(resp))
            self.conn_server.publish()
            return ok
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _check_client_reqs(self, client_id) -> None:
        try:
            while client_id not in self.clients:
                time.sleep(0.001)
            client: RedisServer = self.clients[client_id]
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

        while True:
            msg: str = client.sub_queue.get()
            self._check_client_req(msg)

    @timer(Timers.PERF_COUNTER)
    def _check_client_req(self, msg: str) -> None:
        try:
            task: OrderedDict[str, Any] = json.loads(msg)
            logger.info(
                "Client Manager: Received task"
                f" {task['model_name']} {task['task_type']} with id"
                f" {task['task_id']} from client {task['client_id']}"
            )
            self.requests_queue.put(task)
        except TypeError as json_decode_err:
            logger.debug(json_decode_err)
            logger.debug(f"Ignoring msg {msg}")
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _check_results(self) -> None:
        while True:
            result: OrderedDict[str, Any] = self.results_queue.get()
            self._check_result(result)

    @timer(Timers.PERF_COUNTER)
    def _check_result(self, result: OrderedDict[str, Any]) -> None:
        try:
            if result["status"] == str(State.SUCCESS):
                self.num_tasks_complete += 1
            else:
                self.num_tasks_failed += 1
            logger.success(f"{self.num_tasks_complete} task(s) complete!")
            if self.num_tasks_failed > 0:
                logger.error(f"{self.num_tasks_failed} task(s) failed!")
            req_server: RedisServer = self.clients[result["client_id"]]
            req_server.pub_queue.put(json.dumps(result))
            req_server.publish()

        except TypeError as json_decode_err:
            logger.debug(json_decode_err)
            logger.debug(f"Ignoring msg {result}")
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def ready(self) -> bool:
        return self.conn_server.ready
