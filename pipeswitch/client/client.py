import json
import sys
import threading
import uuid
import time

from pipeswitch.common.logger import logger
from pipeswitch.common.servers import ClientServer
from pipeswitch.task.helper import get_data


class ClientThd(threading.Thread):
    def __init__(self, _model_name, _batch_size, num_it):
        super(ClientThd, self).__init__()
        self.id = str(uuid.uuid4())
        self.reqServer = ClientServer(
            module_id=self.id, host="127.0.0.1", port=6379
        )
        self.model_name = _model_name
        self.batch_size = _batch_size
        self.num_it = num_it

    def run(self):
        self.reqServer.start()
        data = get_data(self.model_name, self.batch_size).cpu().numpy().tolist()

        for _ in range(self.num_it):
            task_name = f"{self.model_name}_inference"
            msg = {
                "client_id": self.id,
                "task_name": task_name,
            }  # , "data": data}
            json_msg = json.dumps(msg)
            logger.info(f"Client {self.id}: sending task {task_name}")
            self.reqServer.pub_queue.put(json_msg)
            while not self.reqServer.publish():
                continue
            time.sleep(1)


if __name__ == "__main__":
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    it = int(sys.argv[3])
    client = ClientThd(model_name, batch_size, it)
    client.start()
    client.join()
