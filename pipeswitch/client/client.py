import json
import sys
import struct
import threading
import torch
import uuid

from pipeswitch.task.helper import get_data
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import ClientServer


class ClientThd(threading.Thread):
    def __init__(self, model_name, batch_size):
        self.id = str(uuid.uuid4())
        self.reqServer = ClientServer(self.id, "127.0.0.1", 6379)
        self.model_name = model_name
        self.batch_size = batch_size

    def run(self):
        self.reqServer.start()
        data = get_data(self.model_name, self.batch_size).cpu().numpy().tolist()

        for _ in range(1):
            task_name = f"{self.model_name}_inference"
            msg = {"client_id": self.id, "task_name": task_name, "data": data}
            json_msg = json.dumps(msg)
            logger.info(f"Client {self.id}: sending task {task_name}")
            self.reqServer.pub_queue.put(json_msg)
            self.reqServer.publish()


if __name__ == "__main__":
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    client = ClientThd(model_name, batch_size)
    client.run()
