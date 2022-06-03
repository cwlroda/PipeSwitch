import threading
import multiprocessing as mp
from abc import ABC, abstractmethod
import jsonpickle  # type: ignore
from typing import List, OrderedDict

from pipeswitch.common.consts import State
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import SchedulerCommsServer, RedisServer
from pipeswitch.runner.status import RunnerStatus


class Policy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_next(self, runners_list: List[int]) -> int:
        pass


class RoundRobinPolicy(Policy):
    def select_next(self, runners_list: List[int]) -> int:
        if len(runners_list) == 0:
            return -1
        return runners_list[0]


class Scheduler(threading.Thread):
    def __init__(self) -> None:
        super(Scheduler, self).__init__()
        self.policy: Policy = RoundRobinPolicy()
        self.comms_server: RedisServer = SchedulerCommsServer(
            host="127.0.0.1", port=6379
        )
        self.runner_status: OrderedDict[int, State] = OrderedDict()

    def run(self) -> None:
        self.comms_server.start()

        while True:
            if not self.comms_server.sub_queue.empty():
                msg: str = self.comms_server.sub_queue.get()
                try:
                    status_update: RunnerStatus = jsonpickle.decode(msg)
                    if status_update.worker_id == -1:
                        logger.debug(
                            "Scheduler status update: Runner"
                            f" {status_update.device} {status_update.status}"
                        )
                        self.runner_status[
                            status_update.device
                        ] = status_update.status
                    else:
                        logger.debug(f"Ignoring non-runner update msg: {msg}")
                except TypeError as json_decode_err:
                    logger.debug(json_decode_err)
                    logger.debug(f"Ignoring msg: {msg}")
                    continue

    def _get_free_runners(self) -> List[int]:
        free_runners: List[int] = []
        for runner_id, status in self.runner_status.items():
            if status == State.IDLE:
                free_runners.append(runner_id)
        return free_runners

    def schedule(self) -> int:
        free_runners: List[int] = self._get_free_runners()
        next_available_runner: int = self.policy.select_next(free_runners)
        if next_available_runner != -1:
            self.runner_status[next_available_runner] = State.RESERVED
            logger.debug(
                f"Scheduler: Next available runner is {next_available_runner}"
            )
            return next_available_runner
        return -1
