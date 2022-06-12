import time
from abc import ABC, abstractmethod
from typing import List, OrderedDict
from torch.multiprocessing import (
    Process,
    Queue,
)  # pylint: disable=unused-import

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


class Scheduler(Process):
    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        num_runners: List[int],
        runner_status: OrderedDict[int, RunnerStatus],
        runner_status_queue: "Queue[RunnerStatus]",
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self._policy: Policy = RoundRobinPolicy()
        self._runner_idx: List[int] = num_runners
        self._runner_status: OrderedDict[int, RunnerStatus] = runner_status
        self._runner_status_queue: "Queue[RunnerStatus]" = runner_status_queue

    def run(self) -> None:
        try:
            while self._do_run:
                status: RunnerStatus = self._runner_status_queue.get()
                self._runner_status[status.device] = status.status
                logger.debug(
                    "Scheduler status update: Runner"
                    f" {status.device} {status.status}"
                )
        except KeyboardInterrupt as _:
            return

    @timer(Timers.PROCESS_TIMER)
    def schedule(self) -> int:
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

    def _get_free_runners(self) -> List[int]:
        return [
            runner_id
            for runner_id, status in self.runner_status.items()
            if status == State.IDLE
        ]

    @property
    def runner_idx(self) -> List[int]:
        return self._runner_idx

    @property
    def runner_status(self) -> OrderedDict[int, State]:
        return self._runner_status
