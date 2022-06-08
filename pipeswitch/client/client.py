import json
import sys
import uuid
import time
from queue import Queue
from typing import Any, OrderedDict

from pipeswitch.common.consts import (
    ConnectionRequest,
    REDIS_HOST,
    REDIS_PORT,
    ResponseStatus,
    State,
)
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ClientConnectionServer,
    ClientRequestsServer,
)

# from pipeswitch.task.helper import get_data


class Client:
    def __init__(self, _model_name, _batch_size, num_it) -> None:
        super().__init__()
        self.client_id = str(uuid.uuid4())
        self.conn_server = ClientConnectionServer(
            client_id=self.client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            pub_queue=Queue(),
            sub_queue=Queue(),
        )
        self.req_server = None
        self.model_name = _model_name
        self.batch_size = _batch_size
        self.num_it = num_it

    def run(self) -> None:
        self.conn_server.daemon = True
        self.conn_server.start()
        self._connect()
        self._send_requests()
        self._receive_results()
        self._disconnect()

    def _connect(self) -> None:
        msg = {
            "client_id": self.client_id,
            "request": str(ConnectionRequest.CONNECT),
        }
        json_msg = json.dumps(msg)
        self.conn_server.pub_queue.put(json_msg)
        while not self.conn_server.publish():
            continue

        while True:
            try:
                if not self.conn_server.sub_queue.empty():
                    msg: str = self.conn_server.sub_queue.get()
                    conn = json.loads(msg)
                    if conn["client_id"] == self.client_id:
                        logger.info(
                            f"Client {self.client_id}: Received handshake from"
                            " manager"
                        )
                        if conn["status"] == str(ResponseStatus.OK):
                            self.req_server = ClientRequestsServer(
                                client_id=self.client_id,
                                host=conn["host"],
                                port=conn["port"],
                                pub_queue=Queue(),
                                sub_queue=Queue(),
                            )
                            if (
                                self.req_server.pub_channel
                                != conn["requests_channel"]
                            ):
                                logger.error(
                                    "Channel mismatch:"
                                    f" {self.req_server.pub_channel} !="
                                    f" {conn['requests_channel']}"
                                )
                            if (
                                self.req_server.sub_channel
                                != conn["results_channel"]
                            ):
                                logger.error(
                                    "Channel mismatch:"
                                    f" {self.req_server.sub_channel} !="
                                    f" {conn['results_channel']}"
                                )
                            self.req_server.daemon = True
                            self.req_server.start()
                            return

                        elif conn["status"] == str(ResponseStatus.ERROR):
                            logger.error(
                                f"Client {self.client_id} error msg:"
                                f" {conn['err_msg']}"
                            )
                            sys.exit(1)
                    else:
                        logger.debug(
                            f"Client {self.client_id}: Ignoring invalid"
                            f" handshake {msg}"
                        )
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(
                    f"Client {self.client_id}: Ignoring invalid handshake {msg}"
                )

    def _disconnect(self) -> None:
        msg = {
            "client_id": self.client_id,
            "request": str(ConnectionRequest.DISCONNECT),
        }
        json_msg = json.dumps(msg)
        self.conn_server.pub_queue.put(json_msg)
        while not self.conn_server.publish():
            continue

        while True:
            try:
                if not self.conn_server.sub_queue.empty():
                    msg: str = self.conn_server.sub_queue.get()
                    conn = json.loads(msg)
                    if conn["client_id"] == self.client_id:
                        logger.info(
                            f"Client {self.client_id}: Received handshake from"
                            " manager"
                        )
                        if conn["status"] == str(ResponseStatus.OK):
                            logger.info("Client: Disconnecting...")
                            return

                        elif conn["status"] == str(ResponseStatus.ERROR):
                            logger.error(
                                f"Client {self.client_id} error msg:"
                                f" {conn['err_msg']}"
                            )
                            sys.exit(1)
                    else:
                        logger.debug(
                            f"Client {self.client_id}: Ignoring invalid"
                            f" handshake {msg}"
                        )
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(
                    f"Client {self.client_id}: Ignoring invalid handshake {msg}"
                )

    def _send_requests(self) -> None:
        # data = get_data(
        #     self.model_name, self.batch_size
        # ).cpu().numpy().tolist()
        for _ in range(self.num_it):
            task_id = str(uuid.uuid4())
            task_name = f"{self.model_name}_inference"
            msg = {
                "client_id": self.client_id,
                "task_id": task_id,
                "task_name": task_name,
            }  # , "data": data}
            json_msg = json.dumps(msg)
            logger.info(
                f"Client {self.client_id}: sending task {task_name} with id"
                f" {task_id}"
            )
            self.req_server.pub_queue.put(json_msg)
            while not self.req_server.publish():
                continue
            time.sleep(1)

    def _receive_results(self) -> None:
        count = 0
        while count != self.num_it:
            try:
                if not self.req_server.sub_queue.empty():
                    msg: str = self.req_server.sub_queue.get()
                    result: OrderedDict[str, Any] = json.loads(msg)
                    if result["status"] == str(State.SUCCESS):
                        logger.info(
                            f"Client {self.client_id}: received task"
                            f" {result['task_name']} with id"
                            f" {result['task_id']} result {result['output']}"
                        )
                        count += 1
                        # TODO: store results in the client
                    else:
                        # TODO: handle failed task
                        # TODO: push failed task back to the task queue and resend
                        logger.error(
                            f"Client {self.client_id}: task"
                            f" {result['task_name']} with id"
                            f" {result['task_id']} failed"
                        )
                        logger.debug(
                            f"Client {self.client_id}: retrying task"
                            f" {result['task_name']} with id"
                            f" {result['task_id']}"
                        )
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(
                    f"Client {self.client_id}: Ignoring invalid handshake {msg}"
                )


if __name__ == "__main__":
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    it = int(sys.argv[3])
    client = Client(model_name, batch_size, it)
    client.run()
