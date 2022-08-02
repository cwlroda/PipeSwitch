import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple  # pylint: disable=unused-import
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Process,
    Queue,
)

from scalabel_bot.common.consts import State, Timers
from scalabel_bot.common.func import cantor_pairing
from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import Message
from scalabel_bot.profiling.timer import timer


class SchedulingPolicy(ABC):
    @abstractmethod
    def select_next(
        self, runner_ect: Dict[int, int], runner_count: int = None
    ) -> Tuple[int, int]:
        pass


class RoundRobin(SchedulingPolicy):
    @timer(Timers.THREAD_TIMER)
    def select_next(
        self, runner_ect: Dict[int, int], runner_count: int
    ) -> Tuple[int, int]:
        next_runner_id = (runner_count + 1) % len(runner_ect)
        return next_runner_id, list(runner_ect.keys())[next_runner_id]


class LoadBalancing(SchedulingPolicy):
    @timer(Timers.THREAD_TIMER)
    def select_next(
        self, runner_ect: Dict[int, int], runner_count: int = None
    ) -> Tuple[int, int]:
        next_runner_id = min(runner_ect, key=runner_ect.get)
        return runner_count, next_runner_id


class TaskScheduler(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        runner_status: Dict[int, State],
        runner_status_queue: "Queue[Tuple[int, int, State]]",
        runner_ect: Dict[int, int],
        runner_ect_queue: "Queue[Tuple[int, int, int]]",
        requests_queue: List[Message],
        clients: Dict[str, int],
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = Event()
        self._policy: SchedulingPolicy = LoadBalancing()
        self._runner_status: Dict[int, State] = runner_status
        self._runner_status_queue: "Queue[Tuple[int, int, State]]" = (
            runner_status_queue
        )
        self._runner_ect_queue: "Queue[Tuple[int, int, int]]" = runner_ect_queue
        self._runner_ect: Dict[int, int] = runner_ect
        self._runner_count: int = -1
        self._requests_queue: List[Message] = requests_queue
        self._clients: Dict[str, int] = clients

    def run(self) -> None:
        while not self._stop_run.is_set():
            device, runner_id, status = self._runner_status_queue.get()
            logger.debug(f"{self._name}: Runner {device}-{runner_id} {status}")
            self._runner_status[cantor_pairing(device, runner_id)] = status

    def _update_ect(self):
        while not self._stop_run.is_set():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect

    def choose_task(self) -> Message:
        while True:
            if not self._requests_queue:
                return None
            next_task = self._requests_queue[0]
            # next_task = min(self._requests_queue, key=lambda x: x["ect"])
            # next_task = min(
            #     self._requests_queue, key=lambda x: np.log(x["ect"]) - x["wait"]
            # )
            self._requests_queue.remove(next_task)
            if next_task["clientId"] in self._clients.keys():
                break
        # for i, task in enumerate(self._requests_queue):
        #     task.update(
        #         {
        #             "wait": task["wait"]
        #             + (task["ect"] / 2000 / len(self._runner_ect))
        #         }
        #     )
        #     self._requests_queue[i] = task
        return next_task

    @timer(Timers.THREAD_TIMER)
    def schedule(self) -> int:
        while not self._runner_ect_queue.empty():
            device, runner_id, ect = self._runner_ect_queue.get()
            self._runner_ect[cantor_pairing(device, runner_id)] = ect
        self._runner_count, next_runner_id = self._policy.select_next(
            self._runner_ect, self._runner_count
        )
        return next_runner_id

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop_run.set()
        logger.debug(f"{self._name}: stopped!")

    @property
    def runner_status(self) -> Dict[int, State]:
        return self._runner_status
