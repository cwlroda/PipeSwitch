import sys
import time
import uuid
from queue import Queue
from typing import Any, OrderedDict
from threading import Thread

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
    ClientConnectionServer,
    ClientRequestsServer,
    RedisServer,
)


class Client:
    @timer(Timers.PERF_COUNTER)
    def __init__(self, _model_name, _batch_size, num_it) -> None:
        super().__init__()
        self.client_id: str = str(uuid.uuid4())
        self.conn_server: RedisServer = ClientConnectionServer(
            client_id=self.client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
        )
        self.model_name: str = _model_name
        self.batch_size: int = _batch_size
        self.num_it: int = num_it
        self.task_queue: Queue = Queue()
        self.pending_tasks: OrderedDict[str, str] = OrderedDict()

    @timer(Timers.PERF_COUNTER)
    def run(self) -> None:
        try:
            self.conn_server.daemon = True
            self.conn_server.start()
            self._connect()
            self._prepare_requests()
            send_requests = Thread(target=self._send_requests)
            send_requests.daemon = True
            send_requests.start()
            self._receive_results()
            self._disconnect()
        except KeyboardInterrupt as _:
            self._shutdown()

    @timer(Timers.PERF_COUNTER)
    def _connect(self) -> None:
        msg = {
            "client_id": self.client_id,
            "request": str(ConnectionRequest.CONNECT),
        }
        self.conn_server.pub_queue.put(msg)
        self.conn_server.publish()

        conn: OrderedDict[str, Any] = self.conn_server.sub_queue.get()
        if conn["client_id"] == self.client_id:
            logger.success(
                f"Client {self.client_id}: Received handshake from manager"
            )
            if conn["status"] == str(ResponseStatus.OK):
                self.req_server = ClientRequestsServer(
                    client_id=self.client_id,
                    host=conn["host"],
                    port=conn["port"],
                )
                if self.req_server.pub_stream != conn["requests_stream"]:
                    logger.error(
                        "Stream mismatch:"
                        f" {self.req_server.pub_stream} !="
                        f" {conn['requests_stream']}"
                    )
                if self.req_server.sub_stream != conn["results_stream"]:
                    logger.error(
                        "Stream mismatch:"
                        f" {self.req_server.sub_stream} !="
                        f" {conn['results_stream']}"
                    )
                self.req_server.daemon = True
                self.req_server.start()

            elif conn["status"] == str(ResponseStatus.ERROR):
                logger.error(
                    f"Client {self.client_id} error msg: {conn['err_msg']}"
                )
                sys.exit(1)
        else:
            logger.debug(
                f"Client {self.client_id}: Ignoring invalid handshake {msg}"
            )

    @timer(Timers.PERF_COUNTER)
    def _disconnect(self) -> None:
        msg = {
            "client_id": self.client_id,
            "request": str(ConnectionRequest.DISCONNECT),
        }
        self.conn_server.pub_queue.put(msg)
        self.conn_server.publish()

        conn: OrderedDict[str, Any] = self.conn_server.sub_queue.get()
        if conn["client_id"] == self.client_id:
            logger.success(
                f"Client {self.client_id}: Received handshake from manager"
            )
            if conn["status"] == str(ResponseStatus.OK):
                logger.info("Client: Disconnecting...")
                return

            elif conn["status"] == str(ResponseStatus.ERROR):
                logger.error(
                    f"Client {self.client_id} error msg: {conn['err_msg']}"
                )
                sys.exit(1)
        else:
            logger.debug(
                f"Client {self.client_id}: Ignoring invalid handshake {msg}"
            )

    @timer(Timers.PERF_COUNTER)
    def _prepare_requests(self) -> None:
        for _ in range(self.num_it):
            task_id = str(uuid.uuid4())
            task_type = "inference"
            task_key = "projects/image/saved/000025"
            msg = {
                "client_id": self.client_id,
                "task_id": task_id,
                "task_type": task_type,
                "task_key": task_key,
                "model_name": self.model_name,
            }
            self.task_queue.put(msg)

    def _send_requests(self) -> None:
        while True:
            msg = self.task_queue.get()
            self._send_request(msg)

    @timer(Timers.PERF_COUNTER)
    def _send_request(self, msg) -> None:
        logger.info(
            f"Client {self.client_id}: sending task"
            f" {msg['model_name']} {msg['task_type']} with id"
            f" {msg['task_id']}"
        )
        self.pending_tasks[msg["task_id"]] = msg
        self.req_server.pub_queue.put(msg)
        self.req_server.publish()
        time.sleep(0.001)  # Necessary to avoid missing messages

    @timer(Timers.PERF_COUNTER)
    def _receive_results(self) -> None:
        count = 0
        while count != self.num_it:
            result: OrderedDict[str, Any] = self.req_server.sub_queue.get()
            if result["status"] == str(State.SUCCESS):
                logger.success(
                    f"Client {self.client_id}: received task"
                    f" {result['model_name']} {result['task_type']} with"
                    f" id {result['task_id']} result {result['output']}"
                )
                count += 1
                logger.info(
                    f"Client {self.client_id}: completed tasks"
                    f" {count}/{self.num_it}"
                )
                self.pending_tasks.pop(result["task_id"])
                # TODO: store results in the client
            else:
                # TODO: handle failed task
                # TODO: push failed task back to the task queue and resend
                logger.error(
                    f"Client {self.client_id}: task"
                    f" {result['model_name']} {result['task_type']} with"
                    f" id {result['task_id']} failed"
                )
                logger.debug(
                    f"Client {self.client_id}: retrying task"
                    f" {result['model_name']} {result['task_type']} with"
                    f" id {result['task_id']}"
                )
                self.task_queue.put(self.pending_tasks[result["task_id"]])

    def _shutdown(self) -> None:
        logger.warning(f"Client {self.client_id}: shutting down")


if __name__ == "__main__":
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    it = int(sys.argv[3])
    client = Client(model_name, batch_size, it)
    client.run()
