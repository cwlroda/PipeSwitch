# -*- coding: utf-8 -*-
"""PipeSwitch Manager

This module is the main entry point for the PipeSwitch Manager.

Example:
    Run this file from the main directory of the project::

        $ python3 manager/manager.py

Todo:
    * None
"""

import time
import os
import importlib
from threading import Thread
from typing import Any, List, OrderedDict
import torch
import torch.multiprocessing as mp

from pipeswitch.common.consts import timer, Timers
from pipeswitch.common.logger import logger
from pipeswitch.manager.client_manager import ClientManager
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.manager.scheduler import Scheduler
from pipeswitch.runner.runner import Runner


class Manager(Thread):
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
        self.do_run: bool = True
        self.mode: str = mode
        self.num_gpus = num_gpus
        self.manager = mp.Manager()
        if self.mode == "gpu":
            self.gra: GPUResourceAllocator = GPUResourceAllocator()
        self.runner_status = self.manager.dict()
        self.runner_status_queue: mp.Queue = mp.Queue()
        self.requests_queue: mp.Queue = mp.Queue()
        self.results_queue: mp.Queue = mp.Queue()
        self.client_manager: ClientManager = ClientManager(
            requests_queue=self.requests_queue,
            results_queue=self.results_queue,
        )
        self.model_list: List(str) = []

        self.models: OrderedDict[str, Any] = OrderedDict()
        self.runners: OrderedDict[int, Runner] = OrderedDict()

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            self._setup()
            logger.info("Manager setup done")
            while not self.client_manager.ready:
                time.sleep(0.001)
            logger.success(
                "\n***********************************\n"
                "Manager: Ready to receive requests!\n"
                "***********************************"
            )
            logger.debug("Manager: Waiting for at least one runner to be ready")
            while len(self.scheduler.runner_status) < 1:
                time.sleep(0.001)
            logger.success("Manager: Ready to execute tasks!")
            while self.do_run:
                task: OrderedDict[str, Any] = self.requests_queue.get()
                self._allocate_tasks(task)
        except KeyboardInterrupt as _:
            logger.warning("Manager: Shutting down")
            if self.mode == "gpu":
                self.gra.release_gpus()
            self.do_run = False
            logger.info("Manager: Successfully shut down")

    @timer(Timers.PROCESS_TIMER)
    def _setup(self) -> None:
        """Run setup tasks in the manager."""
        logger.info("Manager: Start")
        try:
            self.client_manager.daemon = True
            self.client_manager.start()
            if self.mode == "gpu":
                self.gra.cuda_init()
                self.allocated_gpus = self.gra.reserve_gpus(self.num_gpus)
                logger.info(f"Allocated GPUs: {self.allocated_gpus}")
                self.gra.warmup_gpus(gpus=self.allocated_gpus)
            else:
                self.allocated_gpus = [*range(self.num_gpus)]
            self._load_models()
            self.scheduler: Scheduler = Scheduler(
                num_runners=self.allocated_gpus,
                runner_status=self.runner_status,
                runner_status_queue=self.runner_status_queue,
            )
            self.scheduler.daemon = True
            self.scheduler.start()
            self._create_runners()
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

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
            for line in f.readlines():
                self.model_list.append(line.strip())

    def _load_models(self) -> None:
        self._load_model_list(file_name="model_list.txt")
        load_models = [
            Thread(
                target=self._load_model,
                args=(model_name,),
            )
            for model_name in self.model_list
        ]
        for load_model in load_models:
            load_model.daemon = True
            load_model.start()

    @timer(Timers.THREAD_TIMER)
    def _load_model(self, model_name) -> None:
        # Import parameters
        logger.debug(f"Manager: loading model {model_name}")
        model_module = importlib.import_module("pipeswitch.task." + model_name)
        # logger.debug(f"Manager: importing {model_name} parameters")
        model = model_module.import_model()
        logger.spam(model)
        logger.debug(f"Manager: loaded model {model_name}")

        # Preprocess batches
        # logger.debug(f"Manager: preprocessing {model_name} parameters")
        # processed_batched_parameter_list: List[Any] = []
        # for param, mod_list in batched_parameter_list:
        #     if param is None:
        #         processed_batched_parameter_list.append((None, mod_list))
        #     else:
        #         processed_batched_parameter_list.append(
        #             (param.pin_memory(), mod_list)
        #         )
        self.models[model_name] = model
        # return processed_batched_parameter_list

    @timer(Timers.PERF_COUNTER)
    def _create_runners(self) -> None:
        """Create runner for each available GPU.

        Args:
            num_workers (`int`, optional):
                number of workers to create per runner, default 2.
        """
        try:
            for runner_id in self.allocated_gpus:
                if self.mode == "gpu":
                    torch.cuda.set_device(device=runner_id)
                runner: mp.Process = Runner(
                    mode=self.mode,
                    device=runner_id,
                    runner_status_queue=self.runner_status_queue,
                    results_queue=self.results_queue,
                )
                runner.daemon = True
                runner.start()
                if runner.exception is not None:
                    logger.error(
                        f"Manager exception caught: {runner.exception[0]}"
                    )
                    raise KeyboardInterrupt from runner.exception[0]
                self.runners[runner_id] = runner
                logger.info(f"Created runner in GPU {runner_id}")
                # break
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    @timer(Timers.PROCESS_TIMER)
    def _allocate_tasks(self, task: OrderedDict[str, Any]) -> None:
        try:
            runner_id = self.scheduler.schedule()
            msg: OrderedDict[str, Any] = {
                "client_id": task["client_id"],
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "task_key": task["task_key"],
                "model_name": task["model_name"],
                # "model": self.models[task["model_name"]]
            }
            logger.info(
                "Assigning task"
                f" {task['model_name']} {task['task_type']} with id"
                f" {task['task_id']} from client {task['client_id']} to"
                f" runner {runner_id}"
            )
            self.runners[runner_id].task_queue.put(msg)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
