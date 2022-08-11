# -*- coding: utf-8 -*-
"""PipeSwitch Manager.

This module is the main entry point for the PipeSwitch Manager.

Example:
    Run this file from the main directory of the project::

        $ python3 manager/manager.py

Todo:
    * None
"""


from time import sleep
import os
import importlib
from typing import List, Dict, Tuple
from threading import Thread
from pprint import pformat
import json
from queue import Queue
from multiprocessing.managers import SyncManager
from multiprocessing import Event, Manager, Queue as MPQueue

from scalabel_bot.common.consts import (
    MODEL_LIST_FILE,
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)
from scalabel_bot.common.func import cantor_pairing
from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import TaskMessage
from scalabel_bot.manager.gpu_resource_allocator import GPUResourceAllocator
from scalabel_bot.scheduler.task_scheduler import TaskScheduler
from scalabel_bot.manager.client_manager import ClientManager
from scalabel_bot.profiling.timer import timer
from scalabel_bot.runner.runner import Runner
from scalabel_bot.server.stream import ManagerRequestsStream
from scalabel_bot.server.pubsub import ManagerRequestsPubSub


class BotManager:
    """Manager thread that acts as a middleman between clients and runners.

    It has two main functions:
        1. It receives requests from clients and sends them
           to the appropriate runner.
        2. It receives results from runners and sends them
           to the appropriate client.

    Attributes:
        do_run (`bool`): Manager run flag.
        gra (`GPUResourceAllocator`):
            Thread that polls for available GPUs and reserves them.
        req_server (`ManagerRequestsServer`):
            Redis server for receiving requests from clients
            and sending tasks to runners.
        res_server (ManagerResultsServer):
            Redis server for receiving results from runners
            and sending them to clients.
        model_list (`List[str]`): List of ML models to be loaded into memory.
        runners (`OrderedDict[int, RunnerThd]`): Database of runners.
        scheduler (`Scheduler`): Thread that schedules tasks to runners.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        stop_run: Event,
        mode: str = "gpu",
        num_gpus: int = 0,
        gpu_ids: List[int] = None,
    ) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._stop_run: Event = stop_run
        self._mode: str = mode
        self._num_gpus: int = num_gpus
        self._gpu_ids: List[int] = gpu_ids if gpu_ids else []
        self._manager: SyncManager = Manager()
        self._clients: Dict[str, int] = self._manager.dict()
        self._client_manager: ClientManager = ClientManager(
            stop_run=self._stop_run,
            clients=self._clients,
        )
        self._runner_status: Dict[int, State] = self._manager.dict()
        self._runner_status_queue: Queue[Tuple[int, int, State]] = MPQueue()
        self._runner_ect: Dict[int, int] = self._manager.dict()
        self._runner_ect_queue: Queue[Tuple[int, int, int]] = MPQueue()
        self._requests_queue: List[TaskMessage] = self._manager.list()
        self._req_stream: ManagerRequestsStream = ManagerRequestsStream(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._req_pubsub: ManagerRequestsPubSub = ManagerRequestsPubSub(
            stop_run=self._stop_run,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._requests_queue,
        )
        self._results_queue: Queue[TaskMessage] = MPQueue()
        self._model_list: List[str] = []
        self._model_classes: Dict = self._manager.dict()
        self._models: Dict[str, object] = {}
        self._runners: Dict[int, Runner] = {}
        self._tasks_complete: int = 0
        self._tasks_failed: int = 0
        if self._mode == "gpu":
            self._gra: GPUResourceAllocator = GPUResourceAllocator()
            self._allocated_gpus: List[int] = self._gra.reserve_gpus(
                self._num_gpus, self._gpu_ids
            )
            logger.info(f"{self._name}: Allocated GPUs {self._allocated_gpus}")
            self._gra.warmup_gpus(gpus=self._allocated_gpus)
        else:
            self._allocated_gpus: List[int] = [*range(self._num_gpus)]
        self._task_scheduler: TaskScheduler = TaskScheduler(
            stop_run=self._stop_run,
            num_runners=len(self._allocated_gpus),
            runner_status=self._runner_status,
            runner_status_queue=self._runner_status_queue,
            runner_ect=self._runner_ect,
            runner_ect_queue=self._runner_ect_queue,
            requests_queue=self._requests_queue,
            clients=self._clients,
        )
        self._results_checker: Thread = Thread(
            target=self._check_result,
            args=(
                self._results_queue,
                self._req_stream,
                self._req_pubsub,
            ),
        )

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        self._client_manager.daemon = True
        self._client_manager.start()
        self._req_stream.daemon = True
        self._req_stream.start()
        self._req_pubsub.daemon = True
        self._req_pubsub.start()
        logger.success(
            "\n*********************************************\n"
            f"{self._name}: Ready to receive requests!\n"
            "*********************************************"
        )
        self._load_models()
        self._task_scheduler.daemon = True
        self._task_scheduler.start()
        self._create_runners()
        self._results_checker.daemon = True
        self._results_checker.start()
        logger.debug(
            f"{self._name}: Waiting for at least one runner to be ready"
        )
        while (len(self._runner_status) < len(self._runners)) and (
            len(self._runner_ect) < len(self._runners)
        ):
            sleep(1)
        logger.success(
            "\n******************************************\n"
            f"{self._name}: Ready to execute tasks!\n"
            "******************************************"
        )
        try:
            while not self._stop_run.is_set():
                if self._requests_queue:
                    self._allocate_task()
        except KeyboardInterrupt:
            self._stop_run.set()
            self._req_stream.shutdown()

    @timer(Timers.PERF_COUNTER)
    def _load_model_list(self, file_name: str) -> None:
        """Load a list of models to be used by the manager.

        Args:
            file_name (`str`): Path to the file containing the list of models.

        Raises:
            `AssertionError`: If the file does not exist.
        """
        if not os.path.exists(file_name):
            logger.error(f"{self._name}: Model list file {file_name} not found")
            return

        with open(file=file_name, mode="r", encoding="utf-8") as f:
            self._model_list: List[str] = [
                line.strip() for line in f.readlines()
            ]

    @timer(Timers.PERF_COUNTER)
    def _load_models(self) -> None:
        self._load_model_list(file_name=MODEL_LIST_FILE)
        for model_name in self._model_list:
            model_module = importlib.import_module(
                "scalabel_bot.task." + model_name
            )
            model_class: object = model_module.MODEL_CLASS
            self._model_classes[model_name] = model_class

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Create runner for each available GPU."""
        for device in self._allocated_gpus:
            for runner_id in range(1):
                runner: Runner = Runner(
                    stop_run=self._stop_run,
                    mode=self._mode,
                    device=device,
                    runner_id=runner_id,
                    runner_status_queue=self._runner_status_queue,
                    runner_ect_queue=self._runner_ect_queue,
                    model_list=self._model_list,
                    model_classes=self._model_classes,
                    results_queue=self._results_queue,
                    clients=self._clients,
                )
                runner.daemon = True
                runner.start()
                self._runners[cantor_pairing(device, runner_id)] = runner
                logger.debug(
                    f"{self._name}: Created runner {runner_id} in GPU {device}"
                )

    @timer(Timers.PERF_COUNTER)
    def _allocate_task(self) -> None:
        task, runner_id = self._task_scheduler.schedule()
        logger.debug(pformat(task))
        runner: Runner = self._runners[runner_id]
        runner.task_in.send(task)

    @timer(Timers.PERF_COUNTER)
    def _check_result(
        self,
        results_queue: Queue[TaskMessage],
        req_stream: ManagerRequestsStream,
        req_pubsub: ManagerRequestsPubSub,
    ) -> None:
        tasks_complete = 0
        tasks_failed = 0
        while True:
            result: TaskMessage = results_queue.get()
            if result["status"] == State.SUCCESS:
                tasks_complete += 1
            else:
                tasks_failed += 1
            logger.success(f"{self._name}: {tasks_complete} task(s) complete!")
            if self._tasks_failed > 0:
                logger.error(f"{self._name}: {tasks_failed} task(s) failed!")
            result["status"] = result["status"].value
            msg: Dict[str, str] = {"message": json.dumps(result)}
            req_stream.publish(result["channel"], msg)
            req_pubsub.publish(result["channel"], json.dumps(result))
