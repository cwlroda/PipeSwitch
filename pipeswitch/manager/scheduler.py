from abc import ABC, abstractmethod

from pipeswitch.common.logger import logger
from pipeswitch.manager.servers import CommsServer


class Policy(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def selectNext(self):
        pass


class RoundRobinPolicy(Policy):
    def __init__(self):
        super(RoundRobinPolicy, self).__init__()

    def select_next(self, list):
        return list[0]


class Scheduler:
    def __init__(self):
        self.policy = RoundRobinPolicy()
        self.comms = CommsServer("127.0.0.1", 6379)
        self.comms.start()

    def schedule(self, free_gpus):
        next_available_gpu = self.policy.select_next(free_gpus)
