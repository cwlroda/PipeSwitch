import time
from abc import ABC, abstractmethod
from typing import List, OrderedDict
import multiprocessing as mp

from pipeswitch.common.consts import State, timer, Timers
from pipeswitch.common.logger import logger
from pipeswitch.runner.status import RunnerStatus


class Policy(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_next(self, runners_list: List[int]) -> int:
        pass


class RoundRobinPolicy(Policy):
    @timer(Timers.PERF_COUNTER)
    def select_next(self, runners_list: List[int]) -> int:
        return runners_list[0]


class Scheduler(mp.Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        num_runners: List[int],
        runner_status,
        runner_status_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self._policy: Policy = RoundRobinPolicy()
        self._runner_idx: List[int] = num_runners
        self._runner_status = runner_status
        self._runner_status_queue: mp.Queue = runner_status_queue

    def run(self) -> None:
        while self._do_run:
            try:
                status: RunnerStatus = self._runner_status_queue.get()
                self._runner_status[status.device] = status.status
                logger.debug(
                    "Scheduler status update: Runner"
                    f" {status.device} {status.status}"
                )
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    @timer(Timers.PROCESS_TIMER)
    def schedule(self) -> int:
        try:
            free_runners = self._get_free_runners()
            while len(free_runners) < 1:
                time.sleep(0.25)
                free_runners = self._get_free_runners()

            next_available_runner: int = self._policy.select_next(free_runners)
            self.runner_status[next_available_runner] = State.RESERVED
            logger.debug(
                f"Scheduler: Next available runner is {next_available_runner}"
            )
            return next_available_runner
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @property
    def runner_idx(self) -> List[int]:
        return self._runner_idx

    @property
    def runner_status(self) -> OrderedDict[int, State]:
        return self._runner_status

    def _get_free_runners(self) -> List[int]:
        try:
            free_runners: List[int] = []
            for runner_id, status in self.runner_status.items():
                if status == State.IDLE:
                    free_runners.append(runner_id)
            return free_runners
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
