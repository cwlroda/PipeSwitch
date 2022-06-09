# -*- coding: utf-8 -*-
"""PipeSwitch Manager

This module is the main entry point for the PipeSwitch Manager.

Example:
    Run this file from the main directory of the project::

        $ python3 manager/manager.py

Todo:
    * None
"""

import os
from threading import Thread
from typing import Any, List, OrderedDict
import torch
import torch.multiprocessing as mp

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

    def __init__(self, mode: str = "gpu", num_gpus: int = -1) -> None:
        super().__init__()
        self.do_run: bool = True
        self.mode: str = mode
        self.num_gpus = num_gpus
        if self.mode == "gpu":
            self.gra: GPUResourceAllocator = GPUResourceAllocator()
        self.results_queue: mp.Queue = mp.Queue()
        self.client_manager: ClientManager = ClientManager(
            results_queue=self.results_queue,
        )
        self.model_list: List(str) = []
        self.runners: OrderedDict[int, Runner] = OrderedDict()

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        try:
            self._setup()
            logger.info("Manager setup done")
            logger.info("Manager: Waiting for all runners to be ready")
            while len(self.scheduler.runner_status) < 1:
                pass
            logger.info(
                "\n***********************************\n"
                "Manager: Ready to receive requests!\n"
                "***********************************"
            )
            self._allocate_tasks()
        except KeyboardInterrupt as _:
            logger.warning("Manager: Shutting down")
            if self.mode == "gpu":
                self.gra.release_gpus()
            self.do_run = False
            logger.info("Manager: Successfully shut down")

    def _setup(self) -> None:
        """Run setup tasks in the manager."""
        logger.info("Manager: Start")
        try:
            self.client_manager.daemon = True
            self.client_manager.start()
            self._load_model_list(file_name="model_list.txt")
            if self.mode == "gpu":
                self.gra.cuda_init()
                self.allocated_gpus = self.gra.reserve_gpus(self.num_gpus)
                logger.info(f"Allocated GPUs: {self.allocated_gpus}")
                self.gra.warmup_gpus(gpus=self.allocated_gpus)
            else:
                self.allocated_gpus = [*range(self.num_gpus)]
            self.runner_status_queue: mp.Queue = mp.Queue()
            self.scheduler: Scheduler = Scheduler(
                num_runners=self.allocated_gpus,
                runner_status_queue=self.runner_status_queue,
            )
            self.scheduler.daemon = True
            self.scheduler.start()
            self._create_runners()
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

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
                    model_list=self.model_list,
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

    def _allocate_tasks(self) -> None:
        while True:
            try:
                if not self.client_manager.requests_queue.empty():
                    task: OrderedDict[
                        str, Any
                    ] = self.client_manager.requests_queue.get()
                    runner_id: int = -1
                    while runner_id == -1:
                        runner_id = self.scheduler.schedule()
                    msg: OrderedDict[str, Any] = {
                        "runner_id": runner_id,
                        "client_id": task["client_id"],
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        # "data": task["data"],
                    }
                    logger.info(
                        f"Assigning task {task['task_name']} from client"
                        f" {task['client_id']} to runner {runner_id}"
                    )
                    self.runners[runner_id].task_queue.put(msg)
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err
