import time
from threading import Thread
from queue import Queue
from abc import ABC, abstractmethod
import jsonpickle  # type: ignore
from typing import List, OrderedDict

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, State
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


class Scheduler(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.do_run: bool = True
        self._policy: Policy = RoundRobinPolicy()
        self._runner_idx: List[int] = []
        self._runner_status: OrderedDict[int, State] = OrderedDict()

    def run(self) -> None:
        try:
            check_runners_status: List[Thread] = [
                Thread(target=self._check_runner_status, args=(runner_id,))
                for runner_id in self.runner_idx
            ]
            for check_runner_status in check_runners_status:
                check_runner_status.daemon = True
                check_runner_status.start()
            while self.do_run:
                time.sleep(100)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def schedule(self) -> int:
        try:
            free_runners: List[int] = self._get_free_runners()
            next_available_runner: int = self._policy.select_next(free_runners)
            if next_available_runner != -1:
                self.runner_status[next_available_runner] = State.RESERVED
                logger.debug(
                    "Scheduler: Next available runner is"
                    f" {next_available_runner}"
                )
                return next_available_runner
            return -1
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @property
    def runner_idx(self) -> List[int]:
        return self._runner_idx

    @runner_idx.setter
    def runner_idx(self, runner_idx: List[int]) -> None:
        self._runner_idx = runner_idx

    @property
    def runner_status(self) -> OrderedDict[int, State]:
        return self._runner_status

    def _check_runner_status(self, runner_id: int) -> None:
        try:
            comms_server: RedisServer = SchedulerCommsServer(
                module_id=runner_id,
                host=REDIS_HOST,
                port=REDIS_PORT,
                pub_queue=Queue(),
                sub_queue=Queue(),
            )
            comms_server.daemon = True
            comms_server.start()
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
        while True:
            try:
                if not comms_server.sub_queue.empty():
                    msg: str = comms_server.sub_queue.get()
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
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    def _get_free_runners(self) -> List[int]:
        try:
            free_runners: List[int] = []
            for runner_id, status in self.runner_status.items():
                if status == State.IDLE:
                    free_runners.append(runner_id)
            return free_runners
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
