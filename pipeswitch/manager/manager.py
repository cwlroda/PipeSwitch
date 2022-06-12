# -*- coding: utf-8 -*-
"""PipeSwitch Manager

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
from threading import Thread
from typing import Any, List, OrderedDict
import torch
from torch.multiprocessing import Manager, Process, Queue
from pprint import pformat

from pipeswitch.common.consts import timer, Timers
from pipeswitch.common.logger import logger
from pipeswitch.manager.client_manager import ClientManager
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.manager.scheduler import Scheduler
from pipeswitch.runner.runner import Runner
from pipeswitch.runner.status import (  # pylint: disable=unused-import
    RunnerStatus,
)


class PipeSwitchManager(Thread):
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

    @timer(Timers.THREAD_TIMER)
    def __init__(self, mode: str = "gpu", num_gpus: int = -1) -> None:
        super().__init__()
        self._do_run: bool = True
        self._mode: str = mode
        self._num_gpus: int = num_gpus
        self._manager: Manager = Manager()
        if self._mode == "gpu":
            self._gra: GPUResourceAllocator = GPUResourceAllocator()
        self._runner_status: OrderedDict[
            int, RunnerStatus
        ] = self._manager.dict()
        self._runner_status_queue: "Queue[RunnerStatus]" = Queue()
        self._requests_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._results_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._client_blacklist: OrderedDict[str, int] = self._manager.dict()
        self._client_manager: ClientManager = ClientManager(
            requests_queue=self._requests_queue,
            results_queue=self._results_queue,
            client_blacklist=self._client_blacklist,
        )
        self._model_classes: OrderedDict[str, object] = self._manager.dict()
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._runners: OrderedDict[int, Runner] = OrderedDict()

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            self._setup()
            logger.info("Manager setup done")
            while not self._client_manager.ready:
                sleep(0.001)
            logger.success(
                "\n***********************************\n"
                "Manager: Ready to receive requests!\n"
                "***********************************"
            )
            logger.debug("Manager: Waiting for at least one runner to be ready")
            while len(self.scheduler.runner_status) < 1 or len(
                self._models
            ) != len(self._model_list):
                sleep(0.001)
            logger.success("Manager: Ready to execute tasks!")
            while self._do_run:
                task: OrderedDict[str, Any] = self._requests_queue.get()
                if task["client_id"] in self._client_blacklist.keys():
                    logger.warning(
                        "Manager: Ignoring stale task"
                        f" {task['model_name']} {task['task_type']} with id"
                        f" {task['task_id']} from client {task['client_id']}"
                    )
                else:
                    logger.info(
                        "Manager: Received task"
                        f" {task['model_name']} {task['task_type']} with id"
                        f" {task['task_id']} from client {task['client_id']}"
                    )
                    logger.info(
                        f"Manager: {self._requests_queue.qsize()} tasks in"
                        " requests queue"
                    )
                    self._allocate_tasks(task)
        except KeyboardInterrupt as _:
            return

    @timer(Timers.PERF_COUNTER)
    def shutdown(self) -> None:
        logger.warning("Manager: Shutting down")
        self._client_manager.terminate()
        logger.warning("Terminated client manager")
        self.scheduler.terminate()
        logger.warning("Terminated scheduler")
        for runner in self._runners.values():
            runner.terminate()
        logger.warning("Terminated runners")
        self.do_run = False
        if self._mode == "gpu":
            self._gra.release_gpus()
        logger.info("Manager: Successfully shut down")

    @timer(Timers.PROCESS_TIMER)
    def _setup(self) -> None:
        """Run setup tasks in the manager."""
        logger.info("Manager: Start")
        self._client_manager.daemon = True
        self._client_manager.start()
        if self._mode == "gpu":
            self.allocated_gpus = self._gra.reserve_gpus(self._num_gpus)
            logger.info(f"Allocated GPUs: {self.allocated_gpus}")
            self._gra.warmup_gpus(gpus=self.allocated_gpus)
        else:
            self.allocated_gpus = [*range(self._num_gpus)]
        self._load_models()
        self.scheduler: Scheduler = Scheduler(
            num_runners=self.allocated_gpus,
            runner_status=self._runner_status,
            runner_status_queue=self._runner_status_queue,
        )
        self.scheduler.daemon = True
        self.scheduler.start()
        self._create_runners()

    @timer(Timers.PERF_COUNTER)
    def _load_model_list(self, file_name: str) -> None:
        """Load a list of models to be used by the manager.

        Args:
            file_name (`str`): Path to the file containing the list of models.

        Raises:
            `AssertionError`: If the file does not exist.
        """
        assert os.path.exists(file_name)
        with open(file=file_name, encoding="utf-8") as f:
            self._model_list: List[str] = [
                line.strip() for line in f.readlines()
            ]

    def _load_models(self) -> None:
        self._load_model_list(file_name="model_list.txt")
        for model_name in self._model_list:
            model_module = importlib.import_module(
                "pipeswitch.task." + model_name
            )
            model_class: object = model_module.MODEL_CLASS()
            self._model_classes[model_name] = model_class
        self.load_models = [
            Thread(
                target=self._load_model,
                args=(
                    model_name,
                    model_class,
                ),
            )
            for model_name, model_class in self._model_classes.items()
        ]
        for load_model in self.load_models:
            load_model.daemon = True
            load_model.start()

    @timer(Timers.THREAD_TIMER)
    def _load_model(self, model_name, model_class) -> None:
        # Import parameters
        logger.debug(f"Manager: loading model {model_name}")
        logger.debug(f"Manager: importing {model_name} parameters")
        batched_parameter_list: List[Any] = model_class.import_parameters()

        # Preprocess batches
        logger.debug(f"Manager: preprocessing {model_name} parameters")

        self._models[model_name] = [
            (None, mod_list)
            if param is None
            else (param.pin_memory(), mod_list)
            for param, mod_list in batched_parameter_list
        ]
        logger.spam(
            f"\n{pformat(object=self._models[model_name], indent=1, width=1)}"
        )

        logger.debug(f"Manager: loaded model {model_name}")

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Create runner for each available GPU.

        Args:
            num_workers (`int`, optional):
                number of workers to create per runner, default 2.
        """
        for runner_id in self.allocated_gpus:
            if self._mode == "gpu":
                torch.cuda.set_device(device=runner_id)
            runner: Process = Runner(
                mode=self._mode,
                device=runner_id,
                runner_status_queue=self._runner_status_queue,
                model_list=self._model_list,
                model_classes=self._model_classes,
                results_queue=self._results_queue,
            )
            runner.daemon = True
            runner.start()
            self._runners[runner_id] = runner
            logger.info(f"Created runner in GPU {runner_id}")
            # break

    @timer(Timers.PROCESS_TIMER)
    def _allocate_tasks(self, task: OrderedDict[str, Any]) -> None:
        runner_id = self.scheduler.schedule()
        runner = self._runners[runner_id]
        msg: OrderedDict[str, Any] = {
            "client_id": task["client_id"],
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_key": task["task_key"],
            "model_name": task["model_name"],
            # "model": self._models[task["model_name"]]
        }
        logger.info(
            "Assigning task"
            f" {task['model_name']} {task['task_type']} with id"
            f" {task['task_id']} from client {task['client_id']} to"
            f" runner {runner_id}"
        )
        runner.task_in.send(msg)

        if self._mode == "gpu":
            # Create CUDA stream
            cuda_stream_for_parameter = torch.cuda.Stream(runner_id)
            logger.debug(f"Manager: create CUDA stream for runner {runner_id}")

            # Allocate cache to streams
            with torch.cuda.stream(cuda_stream_for_parameter):
                torch.cuda.insert_shared_cache_for_parameter(runner_id)
            logger.debug(
                "Manager: insert shared cache for parameters for runner"
                f" {runner_id}"
            )

            # Transfer parameters to GPU
            batched_parameter_list = self._models[
                f"{task['model_name']}_{task['task_type']}"
            ]
            self._transfer_parameter(
                batched_parameter_list,
                cuda_stream_for_parameter,
                runner.model_in,
            )
            logger.debug(f"Manager: transfer parameters to runner {runner_id}")

            # Clear status
            with torch.cuda.stream(cuda_stream_for_parameter):
                torch.cuda.clear_shared_cache(runner_id)
            logger.debug(f"Manager: clear shared cache for runner {runner_id}")
