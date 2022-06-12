# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import os
from time import sleep
from typing import Any, List, OrderedDict
from torch.multiprocessing import (  # pylint: disable=unused-import
    Pipe,
    Process,
    Queue,
)
import torch
from redis import Redis, exceptions
import json
from urllib import request
from PIL import Image
from pprint import pformat
from threading import Thread

from pipeswitch.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
    State,
    timer,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.runner.status import RunnerStatus
from pipeswitch.runner.runner_common import ModelSummary


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
        model_list: List[str],
        model_classes: OrderedDict[str, object],
        results_queue: "Queue[OrderedDict[str, Any]]",
    ) -> None:
        super().__init__()
        self._do_run: bool = True
        self._mode: str = mode
        self._device: int = device
        self._status: State = State.STARTUP
        self._runner_status_queue: "Queue[RunnerStatus]" = runner_status_queue
        self._model_in, self._model_out = Pipe()
        self._task_in, self._task_out = Pipe()
        self._results_queue: "Queue[OrderedDict[str, Any]]" = results_queue
        self._model_list = model_list
        self._model_classes: OrderedDict[str, object] = model_classes
        self._models: OrderedDict[str, Any] = OrderedDict()

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
                torch.cuda.set_device(device=self._device)
                torch.cuda.recv_cache(device=self._device)
                logger.debug(f"Runner {self._device}: share GPU memory")
                load_jobs = []
                for model_name, model_class in self._model_classes.items():
                    model_summary: ModelSummary = ModelSummary(
                        mode=self._mode,
                        device=self._device,
                        model_name=model_name,
                        model_class=model_class,
                        param_trans_pipe=self._model_out,
                    )
                    load_model = Thread(target=model_summary.load_model)
                    load_model.daemon = True
                    load_model.start()
                    load_model.join()
                    load_jobs.append(load_model)
                    self._models[model_name] = model_summary

            logger.debug(f"Runner {self._device}: import models")
            self._update_status(State.IDLE)
            while self._do_run:
                task: OrderedDict[str, Any] = self._task_out.recv()
                self._manage_task(task)
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
    def _manage_task(self, task: OrderedDict[str, Any]) -> None:
        try:
            logger.info(
                f"Runner {self._device}: received task"
                f" {task['model_name']} {task['task_type']} with id"
                f" {task['task_id']} from client {task['client_id']}"
            )
            self._update_status(State.BUSY)
            if self._mode == "gpu":
                model_summary: ModelSummary = self._models[
                    f"{task['model_name']}_{task['task_type']}"
                ]
            else:
                model_summary = None
            output = self._execute_task(task, model_summary)
            msg: OrderedDict[str, Any] = {
                "worker_id": self._device,
                "client_id": task["client_id"],
                "task_type": task["task_type"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": str(State.SUCCESS),
                "output": output,
            }
            self._results_queue.put(msg)
            logger.debug(
                f"Runner {self._device}: task"
                f" {task['task_id']} {task['task_type']} with id"
                f" {task['task_id']} complete"
            )
            if self._mode == "gpu":
                model_summary.reset_initialized(model_summary.model)
            self._update_status(State.IDLE)
        except RuntimeError as runtime_err:
            logger.error(runtime_err)
            logger.error(f"Runner {self._device}: task failed!")
            msg: OrderedDict[str, Any] = {
                "worker_id": self._device,
                "client_id": task["client_id"],
                "task_type": task["task_type"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": str(State.FAILED),
            }
            self._results_queue.put(msg)
            self._update_status(State.IDLE)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _execute_task(
        self, task: OrderedDict[str, Any], model_summary: ModelSummary
    ) -> Any:
        """Executes a task."""
        # TODO: run inference on a proper model
        data = self._load_data(task)
        logger.spam(
            "Runner"
            f" {self._device} data:\n{pformat(object=data, indent=1, width=1)}"
        )

        if self._mode == "gpu":
            with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                output = model_summary.execute(data)
                logger.spam(
                    f"Runner {self._device} output:"
                    f" \n{pformat(object=output, indent=1, width=1)}"
                )
        else:
            logger.debug(
                f"Runner {self._device}: CPU debug mode task execution"
            )
            output = "some random data"
            sleep(5)
        return output

    @timer(Timers.PERF_COUNTER)
    def transfer_parameter(
        self,
        batched_parameter_list,
        cuda_stream_for_parameter,
        param_trans_pipe,
    ):
        param_cuda_list = []
        for param, mod_list in batched_parameter_list:
            with torch.cuda.stream(cuda_stream_for_parameter):
                if param is not None:
                    param_cuda = param.cuda(non_blocking=True)
                    param_cuda_list.append(param_cuda)
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    e.synchronize()
                param_trans_pipe.send(mod_list[0])

    @timer(Timers.PERF_COUNTER)
    def _load_data(self, task: str) -> OrderedDict[str, Any]:
        """Loads data from redis store"""
        try:
            while not self._data_loader.ping():
                logger.error(
                    f"Runner {self._device} data loader: connection failed!"
                )
                logger.error(
                    f"Runner {self._device} data loader: reconnecting in 5s..."
                )
                sleep(5)
            data_str = self._data_loader.get(task["task_key"])
            data = json.loads(data_str)
            logger.debug(
                f"Runner {self._device}: retrieved data for task"
                f" {task['model_name']} {task['task_type']} with id"
                f" {task['task_id']} from client {task['client_id']}"
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
    def model_in(self):
        return self._model_in

    @property
    def task_in(self):
        return self._task_in
