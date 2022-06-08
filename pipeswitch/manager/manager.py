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
import json
from threading import Thread
from queue import Queue
import time

from typing import Any, List, OrderedDict, Tuple
import torch
import torch.multiprocessing as mp
from multiprocessing import connection

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ManagerTasksServer,
    RedisServer,
)
from pipeswitch.manager.client_manager import ClientManager
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.manager.scheduler import Scheduler
from pipeswitch.runner.runner import Runner
from pipeswitch.runner.worker import WorkerProc


class Manager:
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

    def __init__(self, num_gpus: int = -1) -> None:
        super().__init__()
        self.do_run: bool = True
        self.num_gpus = num_gpus
        self.gra: GPUResourceAllocator = GPUResourceAllocator()
        self.client_manager: Thread = ClientManager()
        self.model_list: List(str) = []
        self.runners: OrderedDict[
            int, OrderedDict[Runner, RedisServer]
        ] = OrderedDict()
        self.scheduler: Scheduler = Scheduler()
        self.num_tasks_complete: int = 0

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
                time.sleep(0.1)
            logger.info(
                "\n***********************************\n"
                "Manager: Ready to receive requests!\n"
                "***********************************"
            )
            check_num_tasks_complete: Thread = Thread(
                target=self._check_num_tasks_complete
            )
            allocate_tasks: Thread = Thread(target=self._allocate_tasks)
            check_results: List[Thread] = [
                Thread(target=self._check_results, args=(runner_id,))
                for runner_id in self.runners.keys()
            ]
            check_num_tasks_complete.daemon = True
            check_num_tasks_complete.start()
            allocate_tasks.daemon = True
            allocate_tasks.start()
            for check_result in check_results:
                check_result.daemon = True
                check_result.start()
            while self.do_run:
                time.sleep(100)
        except KeyboardInterrupt as _:
            logger.warning("Manager: Shutting down")
            self.gra.release_gpus()
            self.do_run = False
            logger.info("Manager: Successfully shut down")

    def _setup(self) -> None:
        """Run setup tasks in the manager."""
        logger.info("Manager: Start")
        try:
            self.gra.cuda_init()
            self.client_manager.daemon = True
            self.client_manager.start()
            self._load_model_list(file_name="model_list.txt")
            self.allocated_gpus = self.gra.reserve_gpus(self.num_gpus)
            logger.info(f"Allocated GPUs: {self.allocated_gpus}")
            self.gra.warmup_gpus(gpus=self.allocated_gpus)
            self._create_runners()
            self.scheduler.daemon = True
            self.scheduler.runner_idx = self.runners.keys()
            self.scheduler.start()
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
                torch.cuda.set_device(device=runner_id)
                runner: Thread = Runner(
                    device=runner_id,
                    model_list=self.model_list,
                    worker_list=self._create_workers(runner_id),
                )
                runner.daemon = True
                runner.start()
                task_server = ManagerTasksServer(
                    module_id=runner_id,
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    pub_queue=Queue(),
                    sub_queue=Queue(),
                )
                task_server.daemon = True
                task_server.start()
                self.runners[runner_id] = {
                    "runner": runner,
                    "task_server": task_server,
                }
                logger.info(f"Created runner in GPU {runner_id}")
                break
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    def _create_workers(
        self, runner_id: int, num_workers: int = 2
    ) -> List[
        Tuple[
            connection.Connection,
            WorkerProc,
            connection.Connection,
            connection.Connection,
        ]
    ]:
        try:
            worker_list: List[
                Tuple[
                    connection.Connection,
                    WorkerProc,
                    connection.Connection,
                    connection.Connection,
                ]
            ] = []
            for i in range(num_workers):
                p_parent: connection.Connection
                p_child: connection.Connection
                param_trans_parent: connection.Connection
                param_trans_child: connection.Connection
                term_parent: connection.Connection
                term_child: connection.Connection

                p_parent, p_child = mp.Pipe()
                param_trans_parent, param_trans_child = mp.Pipe()
                term_parent, term_child = mp.Pipe()

                worker: mp.Process = WorkerProc(
                    device=runner_id,
                    worker_id=i,
                    model_list=self.model_list,
                    pipe=p_child,
                    param_trans_pipe=param_trans_child,
                    term_pipe=term_child,
                )
                worker.daemon = True
                worker.start()
                if worker.exception is not None:
                    logger.error(
                        f"Manager exception caught: {worker.exception[0]}"
                    )
                    raise KeyboardInterrupt from worker.exception[0]
                torch.cuda.send_cache(device=runner_id)
                worker_list.append(
                    (p_parent, worker, param_trans_parent, term_parent)
                )
                logger.debug(f"Created worker {i+1} in GPU {runner_id}")
            return worker_list
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _check_num_tasks_complete(self) -> None:
        while True:
            try:
                num_tasks_complete = sum(
                    val["runner"].num_tasks_complete
                    for val in self.runners.values()
                )
                if num_tasks_complete > self.num_tasks_complete:
                    self.num_tasks_complete = num_tasks_complete
                    logger.info(f"{num_tasks_complete} task(s) complete!")
                    time.sleep(5)
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    def _allocate_tasks(self) -> None:
        while True:
            try:
                if not self.client_manager.requests_queue.empty():
                    task: OrderedDict[
                        str, Any
                    ] = self.client_manager.requests_queue.get()
                    runner_id: int = -1
                    while runner_id == -1:
                        time.sleep(1)
                        runner_id = self.scheduler.schedule()
                    msg: OrderedDict[str, Any] = {
                        "runner_id": runner_id,
                        "client_id": task["client_id"],
                        "task_name": task["task_name"],
                        # "data": task["data"],
                    }
                    logger.info(
                        f"Assigning task {task['task_name']} from client"
                        f" {task['client_id']} to runner {runner_id}"
                    )
                    json_msg: str = json.dumps(msg)
                    self.runners[runner_id]["task_server"].pub_queue.put(
                        json_msg
                    )
                    while not self.runners[runner_id]["task_server"].publish():
                        continue
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    def _check_results(self, runner_id: int) -> None:
        task_server: RedisServer = self.runners[runner_id]["task_server"]
        while True:
            try:
                if not task_server.sub_queue.empty():
                    msg: str = task_server.sub_queue.get()
                    try:
                        result: OrderedDict[str, Any] = json.loads(msg)
                        logger.info(
                            f"Received {result['task_name']} result from runner"
                            f" {runner_id}"
                        )
                        self.client_manager.results_queue.put(msg)
                    except TypeError as json_decode_err:
                        logger.debug(json_decode_err)
                        logger.debug(f"Ignoring msg {msg}")
                        continue
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err
