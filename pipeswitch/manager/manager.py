# -*- coding: utf-8 -*-
"""PipeSwitch Manager

This module is the main entry point for the PipeSwitch Manager.

Example:
    Run this file from the main directory of the project::

        $ python3 manager/manager.py

Todo:
    * None
"""

import json
import os
import threading
import time

from typing import Any, List, OrderedDict, Tuple

import torch
import torch.multiprocessing as mp
from multiprocessing import connection

from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    ManagerRequestsServer,
    ManagerResultsServer,
    RedisServer,
)
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.manager.scheduler import Scheduler
from pipeswitch.runner.runner import RunnerThd
from pipeswitch.runner.worker import WorkerProc


class ManagerThd(threading.Thread):
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

    def __init__(self) -> None:
        super().__init__()
        self.do_run: bool = True
        self.gra: GPUResourceAllocator = GPUResourceAllocator()
        self.req_server: RedisServer = ManagerRequestsServer(
            host="127.0.0.1", port=6379
        )
        self.res_server: RedisServer = ManagerResultsServer(
            host="127.0.0.1", port=6379
        )
        self.model_list: List(str) = []
        self.runners: OrderedDict[int, RunnerThd] = OrderedDict()
        self.scheduler: Scheduler = Scheduler()
        self.num_tasks_complete: int = 0

    def run(self) -> None:
        """Main manager function that sets up the manager and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
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
        while True:
            num_tasks_complete = 0
            for runner_id, runner in self.runners.items():
                num_tasks_complete += runner.num_tasks_complete
            if num_tasks_complete > self.num_tasks_complete:
                self.num_tasks_complete = num_tasks_complete
                logger.info(f"{num_tasks_complete} task(s) complete!")
            if not self.req_server.sub_queue.empty():
                in_msg: str = self.req_server.sub_queue.get()
                try:
                    task: OrderedDict[str, Any] = json.loads(in_msg)
                    logger.info(f"Manager: Received task {task['task_name']}")
                except TypeError as json_decode_err:
                    logger.debug(json_decode_err)
                    logger.debug(f"Ignoring msg {in_msg}")
                    continue
                runner_id: int = -1
                while runner_id == -1:
                    time.sleep(1)
                    runner_id = self.scheduler.schedule()
                out_msg: OrderedDict[str, Any] = {
                    "runner_id": runner_id,
                    "client_id": task["client_id"],
                    "task_name": task["task_name"],
                    # "data": task["data"],
                }
                logger.info(
                    f"Assigning task {task['task_name']} from client"
                    f" {task['client_id']} to runner {runner_id}"
                )
                json_msg: str = json.dumps(out_msg)
                self.req_server.pub_queue.put(json_msg)
                self.req_server.publish()

    def _setup(self) -> None:
        """Run setup tasks in the manager."""

        logger.info("Manager: Start")
        self.req_server.start()
        self.res_server.start()
        self.scheduler.start()
        self._load_model_list(file_name="model_list.txt")
        logger.info(f"Available GPUs: {self.gra.reserve_gpus()}")
        self.gra.warmup_gpus()
        self._create_runners()

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

    def _create_runners(self, num_workers: int = 2) -> None:
        """Create runner for each available GPU.

        Args:
            num_workers (`int`, optional):
                number of workers to create per runner, default 2.
        """
        for runner_id in self.gra.get_free_gpus():
            torch.cuda.set_device(device=runner_id)
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
                worker.start()
                torch.cuda.send_cache(device=runner_id)
                worker_list.append(
                    (p_parent, worker, param_trans_parent, term_parent)
                )
                logger.debug(f"Created worker {i+1} in GPU {runner_id}")

            runner: threading.Thread = RunnerThd(
                device=runner_id,
                model_list=self.model_list,
                worker_list=worker_list,
            )
            runner.start()

            self.runners[runner_id] = runner
            logger.info(f"Created runner in GPU {runner_id}")
            # break


if __name__ == "__main__":
    os.system("redis-server redis.conf")
    mp.set_start_method("forkserver")
    manager: ManagerThd = ManagerThd()

    try:
        manager.run()
    except RuntimeError as runtime_err:
        logger.exception(f"Manager Exception: {runtime_err}")
        manager.do_run = False
        manager.gra.release_gpus()
