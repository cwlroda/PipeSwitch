import jsonpickle
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

from pipeswitch.common.logger import logger
from pipeswitch.common.servers import SchedulerCommsServer
from pipeswitch.common.consts import State


class Policy(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def selectNext(self):
        pass


class RoundRobinPolicy(Policy):
    def __init__(self):
        super(RoundRobinPolicy, self).__init__()

    def select_next(self, runners_list):
        if len(runners_list) == 0:
            return None
        return runners_list[0]


class Scheduler(threading.Thread):
    def __init__(self):
        super(Scheduler, self).__init__()
        self.policy = RoundRobinPolicy()
        self.commsServer = SchedulerCommsServer("127.0.0.1", 6379)
        self.runner_status = OrderedDict()

    def run(self):
        self.commsServer.start()

        while True:
            if not self.commsServer.sub_queue.empty():
                msg = self.commsServer.sub_queue.get()
                try:
                    status_update = jsonpickle.decode(msg)
                    logger.debug(
                        f"Scheduler status update: Runner {status_update.device} {status_update.status}"
                    )
                    if status_update.device not in self.runner_status:
                        self.runner_status[status_update.device] = status_update.status
                except:
                    logger.debug(f"Ignoring message: {msg}")
                    continue

    def get_free_runners(self):
        free_runners = []
        for id, status in self.runner_status.items():
            if status == State.IDLE:
                free_runners.append(id)
        return free_runners

    def schedule(self):
        free_runners = self.get_free_runners()
        next_available_runner = self.policy.select_next(free_runners)
        if next_available_runner is not None:
            self.runner_status[next_available_runner] = State.BUSY
            return next_available_runner
