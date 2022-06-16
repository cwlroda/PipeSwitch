import sys
import time
from argparse import ArgumentParser
from uuid import uuid4
from queue import Queue
from typing import Any, OrderedDict
from threading import Thread

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
    ClientConnectionServer,
    ClientRequestsServer,
    RedisServer,
)
from pipeswitch.profiling.timer import timer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="FsDet demo for builtin models")

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet152",
        help="Name of model for processing requests. Default is 'resnet152'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size of data. Default is 8.",
    )
    parser.add_argument(
        "--it",
        type=int,
        default=1,
        help="Number of iterations of requests. Default is 1.",
    )
    return parser


class Client:
    @timer(Timers.PERF_COUNTER)
    def __init__(self, model_name, batch_size, num_it) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._client_id: str = str(uuid4())
        self._conn_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._conn_server: RedisServer = ClientConnectionServer(
            client_id=self._client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._conn_queue,
        )
        self._model_name: str = model_name
        self._batch_size: int = batch_size
        self._num_it: int = num_it
        self._task_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._pending_tasks: OrderedDict[str, str] = OrderedDict()
        self._results_queue: "Queue[OrderedDict[str, Any]]" = Queue()

    @timer(Timers.PERF_COUNTER)
    def run(self) -> None:
        try:
            self._conn_server.daemon = True
            self._conn_server.start()
            self._connect()
            self._prepare_requests()
            send_requests = Thread(target=self._send_requests)
            send_requests.daemon = True
            send_requests.start()
            timeout = Thread(target=self._timeout)
            timeout.daemon = True
            timeout.start()
            self._receive_results()
            self._disconnect()
        except KeyboardInterrupt:
            self._shutdown()

    @timer(Timers.PERF_COUNTER)
    def _connect(self) -> None:
        msg: OrderedDict[str, Any] = {
            "client_id": self._client_id,
            "request": ConnectionRequest.CONNECT.value,
        }
        self._conn_server.publish(msg)

        conn: OrderedDict[str, Any] = self._conn_queue.get()
        if conn["client_id"] == self._client_id:
            logger.success(
                f"{self._name} {self._client_id}: Received handshake from"
                " manager"
            )
            if conn["status"] == ResponseStatus.OK.value:
                self.req_server = ClientRequestsServer(
                    client_id=self._client_id,
                    host=conn["host"],
                    port=conn["port"],
                    sub_queue=self._results_queue,
                )
                if self.req_server.pub_stream != conn["requests_stream"]:
                    logger.error(
                        f"{self._name} {self._client_id}: Stream mismatch:"
                        f" {self.req_server.pub_stream} !="
                        f" {conn['requests_stream']}"
                    )
                if self.req_server.sub_stream != conn["results_stream"]:
                    logger.error(
                        f"{self._name} {self._client_id}: Stream mismatch:"
                        f" {self.req_server.sub_stream} !="
                        f" {conn['results_stream']}"
                    )
                self.req_server.daemon = True
                self.req_server.start()

            elif conn["status"] == ResponseStatus.ERROR.value:
                logger.error(
                    f"{self._name} {self._client_id}: {conn['err_msg']}"
                )
                sys.exit(1)
        else:
            logger.debug(
                f"{self._name} {self._client_id}: Ignoring invalid handshake"
                f" {msg}"
            )

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self) -> None:
        msg = {
            "client_id": self._client_id,
            "request": ConnectionRequest.DISCONNECT.value,
        }
        self._conn_server.publish(msg)

        conn: OrderedDict[str, Any] = self._conn_queue.get()
        if conn["client_id"] == self._client_id:
            logger.success(
                f"{self._name} {self._client_id}: Received handshake from"
                " manager"
            )
            if conn["status"] == ResponseStatus.OK.value:
                logger.info(f"{self._name} {self._client_id}: Disconnecting...")
                return

            elif conn["status"] == ResponseStatus.ERROR.value:
                logger.error(
                    f"{self._name} {self._client_id} error msg:"
                    f" {conn['err_msg']}"
                )
                sys.exit(1)
        else:
            logger.debug(
                f"{self._name} {self._client_id}: Ignoring invalid handshake"
                f" {msg}"
            )

    @timer(Timers.PERF_COUNTER)
    def _prepare_requests(self) -> None:
        for _ in range(self._num_it):
            task_id: str = str(uuid4())
            task_type: str = "inference"
            task_key: str = "projects/image/saved/000025"
            msg = {
                "client_id": self._client_id,
                "task_id": task_id,
                "task_type": task_type,
                "task_key": task_key,
                "model_name": self._model_name,
            }
            self._task_queue.put(msg)

    def _send_requests(self) -> None:
        while True:
            msg = self._task_queue.get()
            self._send_request(msg)

    @timer(Timers.THREAD_TIMER)
    def _send_request(self, msg) -> None:
        logger.info(
            f"{self._name} {self._client_id}: sending task"
            f" {msg['model_name']} {msg['task_type']} with id"
            f" {msg['task_id']}"
        )
        self._pending_tasks[msg["task_id"]] = msg
        self.req_server.publish(msg)

    @timer(Timers.PERF_COUNTER)
    def _receive_results(self) -> None:
        task_count = 0
        while task_count != self._num_it:
            result: OrderedDict[str, Any] = self._results_queue.get()
            if result["task_id"] in self._pending_tasks.keys():
                if result["status"] == State.SUCCESS.value:
                    logger.success(
                        f"{self._name} {self._client_id}: received task"
                        f" {result['model_name']} {result['task_type']} with"
                        f" id {result['task_id']} result"
                    )
                    logger.spam(f"{result['output']}")
                    task_count += 1
                    logger.info(
                        f"{self._name} {self._client_id}: completed task(s)"
                        f" {task_count}/{self._num_it}"
                    )
                    self._pending_tasks.pop(result["task_id"])
                    # TODO: store results in the client
                else:
                    # TODO: handle failed task
                    # TODO: push failed task back to the task queue and resend
                    logger.error(
                        f"{self._name} {self._client_id}: task"
                        f" {result['model_name']} {result['task_type']} with"
                        f" id {result['task_id']} failed"
                    )
                    logger.debug(
                        f"{self._name} {self._client_id}: retrying task"
                        f" {result['model_name']} {result['task_type']} with"
                        f" id {result['task_id']}"
                    )
                    self._task_queue.put(self._pending_tasks[result["task_id"]])

    def _timeout(self) -> None:
        start_time = time.perf_counter()
        timeout = 30  # 10000 * self._num_it
        while True:
            if (time.perf_counter() - start_time) > timeout:
                logger.warning(
                    f"{self._name} {self._client_id}: requests timed out after"
                    f" {timeout}s"
                )
                logger.warning(
                    f"{self._name} {self._client_id}: resending"
                    f" {len(self._pending_tasks)} pending task(s)..."
                )
                for _, task in self._pending_tasks.items():
                    self._task_queue.put(task)
                start_time = time.perf_counter()

    def _shutdown(self) -> None:
        logger.warning(f"{self._name} {self._client_id}: shutting down")


def launch():
    args: ArgumentParser = get_parser().parse_args()
    client = Client(args.model_name, args.batch_size, args.it)
    client.run()


if __name__ == "__main__":
    launch()
