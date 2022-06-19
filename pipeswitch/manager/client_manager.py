# -*- coding: utf-8 -*-
from typing import Any, OrderedDict
from torch.multiprocessing import Event, Process, Queue

from pipeswitch.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ManagerRequestsServer,
    RedisServer,
)
from pipeswitch.profiling.timer import timer


class ClientManager(Process):
    @timer(Timers.THREAD_TIMER)
    def __init__(
        self,
        requests_queue: Queue,
        results_queue: Queue,
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = Event()
        self._requests_queue: "Queue[OrderedDict[str, Any]]" = requests_queue
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._tasks_complete: int = 0
        self._tasks_failed: int = 0

    def run(self) -> None:
        logger.debug(f"{self._name}: Start")
        self._req_server: RedisServer = ManagerRequestsServer(
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._req_server.daemon = True
        self._req_server.start()
        while not self._stop_run.is_set():
            result: OrderedDict[str, Any] = self._results_queue.get()
            self._check_result(result)

    @timer(Timers.THREAD_TIMER)
    def _check_result(self, result: OrderedDict[str, Any]) -> None:
        if result["status"] == State.SUCCESS:
            self._tasks_complete += 1
        else:
            self._tasks_failed += 1
        logger.success(
            f"{self._name}: {self._tasks_complete} task(s) complete!"
        )
        if self._tasks_failed > 0:
            logger.error(f"{self._name}: {self._tasks_failed} task(s) failed!")
        result["status"] = result["status"].value
        stream = f"RESPONSES-{result['client_id']}"
        self._req_server.publish(stream, result)

    def ready(self) -> bool:
        if hasattr(self, "_conn_server"):
            return self._req_server.ready

    @timer(Timers.THREAD_TIMER)
    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name}: stopping...")
        self._stop_run.set()
        if hasattr(self, "_conn_server"):
            if self._req_server.is_alive():
                self._req_server.shutdown()
                self._req_server.join()
        logger.debug(f"{self._name}: stopped!")
