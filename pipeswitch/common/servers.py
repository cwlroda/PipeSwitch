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
        self.pub_queue = Queue()
        self.sub_queue = Queue()

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
            host=self.host, port=self.port, encoding="utf-8", decode_responses=True
        )
        assert self.pub.ping(), f"{self.server_name}-pub: connection failed!"

    def create_sub(self):
        self.sub = redis.Redis(
            host=self.host, port=self.port, encoding="utf-8", decode_responses=True
        )
        assert self.sub.ping(), f"{self.server_name}-sub: connection failed!"
        self.sub = self.sub.pubsub()

    def listen(self):
        if self.sub_channel == "":
            return

        self.sub.subscribe(self.sub_channel)
        logger.debug(f"{self.server_name}: listening to channel {self.sub_channel}")

        for msg in self.sub.listen():
            if msg is not None:
                logger.debug(
                    f"{self.server_name}: msg received from channel {msg['channel']}"
                )
                self.sub_queue.put(msg["data"])

    def publish(self):
        if self.pub_channel == "":
            return

        while not self.pub_queue.empty():
            item = self.pub_queue.get(block=False)
            self.pub.publish(self.pub_channel, item)
            logger.debug(
                f"{self.server_name}: publishing msg to channel {self.pub_channel}"
            )


# server in the manager scheduler that records status of runners
class SchedulerCommsServer(RedisServer):
    def __init__(self, host, port):
        super(SchedulerCommsServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "SchedulerCommsServer"

    @property
    def pub_channel(self):
        return ""

    @property
    def sub_channel(self):
        return "STATUS"


# server in the manager that:
# 1. receives inference requests from clients
# 2. sends inference requests in the form of tasks to the runners
class ManagerRequestsServer(RedisServer):
    def __init__(self, host, port):
        super(ManagerRequestsServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "ManagerRequestsServer"

    @property
    def pub_channel(self):
        return "TASKS"

    @property
    def sub_channel(self):
        return "REQUESTS"


# server in the manager that:
# 1. receives task results from runners
# 2. sends inference results to the clients
class ManagerResultsServer(RedisServer):
    def __init__(self, host, port):
        super(ManagerResultsServer, self).__init__(host, port)

    @property
    def server_name(self):
        return "ManagerResultsServer"

    @property
    def pub_channel(self):
        return "RESPONSES"

    @property
    def sub_channel(self):
        return "RESULTS"


# server in the runner that updates runner status
class RunnerCommsServer(RedisServer):
    def __init__(self, host_runner, host, port):
        super(RunnerCommsServer, self).__init__(host, port)
        self._host_runner = host_runner

    @property
    def host_runner(self):
        return self._host_runner

    @property
    def server_name(self):
        return f"RunnerCommsServer-{self.host_runner}"

    @property
    def pub_channel(self):
        return "STATUS"

    @property
    def sub_channel(self):
        return ""


# server in the runner that:
# 1. receives tasks from the manager
# 2. sends task results to the manager
class RunnerTaskServer(RedisServer):
    def __init__(self, host_runner, host, port):
        super(RunnerTaskServer, self).__init__(host, port)
        self._host_runner = host_runner

    @property
    def host_runner(self):
        return self._host_runner

    @property
    def server_name(self):
        return f"RunnerTaskServer-{self.host_runner}"

    @property
    def pub_channel(self):
        return "RESULTS"

    @property
    def sub_channel(self):
        return "TASKS"


# dummy client server that:
# 1. sends inference requests to the manager
# 2. receives inference results from the manager
class ClientServer(RedisServer):
    def __init__(self, id, host, port):
        super(ClientServer, self).__init__(host, port)
        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def server_name(self):
        return "ClientServer"

    @property
    def pub_channel(self):
        return "REQUESTS"

    @property
    def sub_channel(self):
        return "RESPONSES"
