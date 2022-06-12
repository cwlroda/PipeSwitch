# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import os
import time
from typing import Any, OrderedDict
from torch.multiprocessing import Process, Queue
import torch
from redis import Redis, exceptions
import json
from urllib import request
from PIL import Image
from pprint import pformat

from pipeswitch.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
    State,
    timer,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.runner.status import RunnerStatus


class Runner(Process):
    """Runner thread that receives tasks from the manager
    and allocates them to the available workers.
    It collects results from the workers and sends them back to the manager.

    Attributes:
        device (int): ID of the GPU that the runner is running on.
        model_list (List[str]): List of model names.

        worker_list
            (List[
                Tuple[
                    connection.Connection,
                    WorkerProc,
                    connection.Connection,
                    connection.Connection
                ]
            ]): List of worker processes.

        worker_status (OrderedDict[int, WorkerStatus]):
            Dictionary containing the status of the workers
        cur_w_idx (int): Index of the current worker.
        comms_server (RedisServer):
            Redis server for receiving status updates from the workers
            and updating own status to the manager.
        task_server (RedisServer):
            Redis server for receiving tasks from the manager
            and sending results to the manager.
    """

    @timer(Timers.PERF_COUNTER)
    def __init__(
        self,
        mode: str,
        device: int,
        runner_status_queue: "Queue[RunnerStatus]",
        results_queue: "Queue[OrderedDict[str, Any]]",
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self._mode: str = mode
        self._device: int = device
        self._runner_status_queue: "Queue[RunnerStatus]" = runner_status_queue
        self._task_queue: "Queue[OrderedDict[str, Any]]" = Queue()
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._status: State = State.STARTUP
        self._models_queue: Queue = Queue()

    def run(self) -> None:
        """Main runner function that sets up the runner and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        logger.debug(f"Runner {self._device}: start")
        try:
            self._data_loader = Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                encoding="utf-8",
                decode_responses=True,
            )
            if self._mode == "gpu":
                # Create CUDA stream
                self.cuda_stream_for_parameter = torch.cuda.Stream(  # type: ignore
                    self._device
                )
                logger.debug(f"Runner {self._device}: create stream")
            else:
                self.cuda_stream_for_parameter = None
            self._update_status(State.IDLE)
            while self._do_run:
                self.task = self._task_queue.get()
                self._manage_task()
        except KeyboardInterrupt as _:
            return

    @timer(Timers.PERF_COUNTER)
    def _update_status(self, status: State) -> None:
        """Updates own runner status based on worker statuses"""
        try:
            logger.debug(f"Updating runner {self._device} status")
            self._status = status
            logger.debug(f"Runner {self._device}: status {self._status}")
            runner_status = RunnerStatus(device=self._device, status=status)
            self._runner_status_queue.put(runner_status)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _manage_task(self) -> None:
        try:
            logger.info(
                f"Runner {self._device}: received task"
                f" {self.task['model_name']} {self.task['task_type']} with id"
                f" {self.task['task_id']} from client {self.task['client_id']}"
            )
            self._update_status(State.BUSY)
            output = self._execute_task()
            msg: OrderedDict[str, Any] = {
                "worker_id": self._device,
                "client_id": self.task["client_id"],
                "task_type": self.task["task_type"],
                "task_id": self.task["task_id"],
                "model_name": self.task["model_name"],
                "status": str(State.SUCCESS),
                "output": output,
            }
            self._results_queue.put(msg)
            # model_summary.reset_initialized(model_summary.model)
            self._update_status(State.IDLE)
        except RuntimeError as runtime_err:
            logger.error(runtime_err)
            logger.error(f"Runner {self._device}: task failed!")
            msg: OrderedDict[str, Any] = {
                "worker_id": self._device,
                "client_id": self.task["client_id"],
                "task_type": self.task["task_type"],
                "task_id": self.task["task_id"],
                "model_name": self.task["model_name"],
                "status": str(State.FAILED),
            }
            self._results_queue.put(msg)
            self._update_status(State.IDLE)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _execute_task(self) -> Any:
        """Executes a task."""
        logger.debug(f"Runner {self._device}: CPU debug mode task execution")
        # task_type = task["task_type"]
        task_key = self.task["task_key"]
        # model = task["model"]

        data = self._load_data(task_key)
        logger.spam(data)
        # if task_type == "inference":
        #     model.eval()
        # elif task_type == "train":
        #     model.train()
        # else:
        #     logger.error(
        #         f"Runner {self._device}: unknown task type"
        #         f" {task_type}"
        #     )

        # TODO: run inference on a proper model
        # TODO: load data from redis store
        # output = self._run_inference()
        # start doing inference
        # frontend_scheduler will directly put
        # mod_list[0] in to self._results_queue_trans

        # if self._mode == "gpu":
        #     with torch.cuda.stream(
        #         model_summary.cuda_stream_for_computation
        #     ):
        #         logger.debug(
        #             f"Worker {self._device}-{self.worker_id}: Dummy"
        #             " inference execution"
        #         )
        #         output = "some random data"
        #         time.sleep(5)
        # output = model_summary.execute(task["data"])
        # logger.info(f"Get output: {output}")
        # del output
        # else:
        output = "some random data"
        time.sleep(5)
        return output

    @timer(Timers.PERF_COUNTER)
    def _run_inference(self) -> None:
        try:
            data = torch.cuda.FloatTensor(self.data)
            input_batch = data.view(-1, 3, 224, 224).cuda(
                device=self._device, non_blocking=True
            )
            with torch.no_grad():
                output = self._model(input_batch)
            return output.sum().item()
        except TypeError as type_err:
            logger.error(type_err)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _load_data(self, task_key: str) -> OrderedDict[str, Any]:
        """Loads data from redis store"""
        try:
            while not self._data_loader.ping():
                logger.error(
                    f"Runner {self._device} data loader: connection failed!"
                )
                logger.error(
                    f"Runner {self._device} data loader: reconnecting in 5s..."
                )
                time.sleep(5)
            data_str = self._data_loader.get(task_key)
            data = json.loads(data_str)
            logger.debug(
                f"Runner {self._device}: retrieved data for task"
                f" {self.task['model_name']} {self.task['task_type']} with id"
                f" {self.task['task_id']} from client {self.task['client_id']}"
            )
            logger.spam(f"\n{pformat(object=data, indent=1, width=1)}")
            img_url = data["task"]["items"][0]["urls"]["-1"]
            img_name = img_url.split("/")[-1]
            if not os.path.exists(img_name):
                logger.debug(
                    f"Runner {self._device}: Downloading image {img_name}..."
                )
                request.urlretrieve(img_url, img_name)
            img = Image.open(img_name)
            return img
        except exceptions.ConnectionError as conn_err:
            logger.error(conn_err)
            raise exceptions.ConnectionError from conn_err
        except exceptions.RedisError as redis_err:
            logger.error(redis_err)
            raise exceptions.RedisError from redis_err
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @property
    def task_queue(self) -> Queue:
        return self._task_queue
