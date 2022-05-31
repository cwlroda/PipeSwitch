import redis
import threading
from abc import ABC, abstractmethod
from queue import Queue
from pipeswitch.common.logger import logger


class RedisServer(ABC, threading.Thread):
    def __init__(self, host, port):
        super(RedisServer, self).__init__()
        self.host = host
        self.port = port
        self.queue = Queue()

    @property
    @abstractmethod
    def server_name(self):
        return ""

    @property
    @abstractmethod
    def pub_channel(self):
        return ""

    @property
    @abstractmethod
    def sub_channel(self):
        return ""

    def run(self):
        self.create_pub()
        self.create_sub()
        self.listen()

    def create_pub(self):
        self.pub = redis.Redis(
            host=self.host, port=self.port, charset="utf-8", decode_responses=True
        )
        assert self.pub.ping()

    def create_sub(self):
        self.sub = redis.Redis(
            host=self.host, port=self.port, charset="utf-8", decode_responses=True
        )
        assert self.sub.ping()
        self.sub = self.sub.pubsub()

    def listen(self):
        logger.info(f"{self.server_name}: Listening")
        while True:
            self.sub.subscribe(self.sub_channel)
            for msg in self.sub.listen():
                if msg is not None:
                    data = msg.get("data")
                    logger.info(f"Message received, id: {data}")
                    self.queue.put(data)

    def publish(self):
        while not self.queue.empty():
            item = self.queue.get(block=False)
            self.pub.publish(self.pub_channel, item)


# server that records idle runners
class CommsServer(RedisServer):
    def __init__(self, host, port):
        super(CommsServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "CommsServer"

    @property
    def pub_channel(self):
        pass

    @property
    def sub_channel(self):
        return "runners"


# server that:
# 1. receives inference requests from clients
# 2. forwards them to the runners
class RequestsServer(RedisServer):
    def __init__(self, host, port):
        super(RequestsServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "RequestsServer"

    @property
    def pub_channel(self):
        return "tasks"

    @property
    def sub_channel(self):
        return "requests"


# server that:
# 1. receives inference results from runners
# 2. forwards them to the clients
class RunnersServer(RedisServer):
    def __init__(self, host, port):
        super(RunnersServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "RunnersServer"

    @property
    def pub_channel(self):
        return "responses"

    @property
    def sub_channel(self):
        return "results"


if __name__ == "__main__":
    reqServer = RequestsServer("127.0.0.1", 6379)
    if reqServer.pub.ping() and reqServer.sub.ping():
        print("reqServer: PONG")
    else:
        print("reqServer: Connection failed!")

    runnServer = RunnersServer("127.0.0.1", 6379)
    if runnServer.pub.ping() and runnServer.sub.ping():
        print("runnServer: PONG")
    else:
        print("runnServer: Connection failed!")
