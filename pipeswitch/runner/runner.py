# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import os
from time import sleep
from typing import (  # pylint: disable=unused-import
    Any,
    List,
    OrderedDict,
    Tuple,
)
from torch.multiprocessing import (  # pylint: disable=unused-import
    Event,
    Pipe,
    Process,
    Queue,
)
import torch
from redis import Redis, exceptions
import jsonpickle
from urllib import request
from PIL import Image
from pprint import pformat
from threading import Thread

from pipeswitch.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)
from pipeswitch.common.logger import logger
from pipeswitch.profiling.timer import timer
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
        runner_status_queue: "Queue[Tuple[int, State]]",
        model_list: List[str],
        model_classes: OrderedDict[str, object],
        results_queue: "Queue[OrderedDict[str, Any]]",
    ) -> None:
        super().__init__()
        self._name = self.__class__.__name__
        self._stop: Event = Event()
        self._mode: str = mode
        self._device: int = device
        self._status: State = State.STARTUP
        self._runner_status_queue: "Queue[Tuple[int, State]]" = (
            runner_status_queue
        )
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
        logger.debug(f"{self._name} {self._device}: start")
        self._data_loader = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            encoding="utf-8",
            decode_responses=True,
        )
        if self._mode == "gpu":
            torch.cuda.recv_cache(device=self._device)
            logger.debug(f"{self._name} {self._device}: share GPU memory")
            self._load_jobs = []
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
                self._load_jobs.append(load_model)
                self._models[model_name] = model_summary

        logger.debug(f"{self._name} {self._device}: import models")
        self._update_status(State.IDLE)
        while not self._stop.is_set():
            task: OrderedDict[str, Any] = self._task_out.recv()
            self._manage_task(task)

    @timer(Timers.PERF_COUNTER)
    def _update_status(self, status: State) -> None:
        """Updates own runner status based on worker statuses"""
        try:

            self._status = status
            logger.debug(
                f"{self._name} {self._device}: Updating status to"
                f" {self._status}"
            )
            self._runner_status_queue.put((self._device, self._status))
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @timer(Timers.PERF_COUNTER)
    def _manage_task(self, task: OrderedDict[str, Any]) -> None:
        try:
            logger.debug(
                f"{self._name} {self._device}: received task"
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
                "client_id": task["client_id"],
                "task_type": task["task_type"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": State.SUCCESS,
                "output": jsonpickle.encode(output),
            }
            self._results_queue.put(msg)
            logger.debug(
                f"{self._name} {self._device}: task"
                f" {task['task_id']} {task['task_type']} with id"
                f" {task['task_id']} complete"
            )
            if self._mode == "gpu":
                model_summary.reset_initialized(model_summary.model)
            self._update_status(State.IDLE)
        except RuntimeError as runtime_err:
            logger.error(runtime_err)
            logger.error(f"{self._name} {self._device}: task failed!")
            msg: OrderedDict[str, Any] = {
                "client_id": task["client_id"],
                "task_type": task["task_type"],
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": State.FAILED,
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
        # data = self._load_data(task)
        # data = self._load_data()
        data = model_summary.load_data(task["task_key"])
        logger.debug(
            f"{self._name} {self._device}: retrieved data for task"
            f" {task['model_name']} {task['task_type']} with id"
            f" {task['task_id']} from client {task['client_id']}"
        )
        logger.spam(
            f"{self._name} {self._device} data:"
            f" \n{pformat(object=data, indent=1, width=1)}"
        )

        if self._mode == "gpu":
            with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                output = model_summary.execute(data)
                logger.spam(
                    f"{self._name} {self._device} output:"
                    f" \n{pformat(object=output, indent=1, width=1)}"
                )
        else:
            logger.debug(
                f"{self._name} {self._device}: CPU debug mode task execution"
            )
            output = "some random data"
            sleep(5)
        return output

    # @timer(Timers.PERF_COUNTER)
    # def _load_data(self, task: str) -> Image:
    #     """Loads data from redis store"""
    #     try:
    #         while not self._data_loader.ping():
    #             logger.error(
    #                 f"{self._name} {self._device} data loader: connection"
    #                 " failed!"
    #             )
    #             logger.error(
    #                 f"{self._name} {self._device} data loader: reconnecting in"
    #                 " 5s..."
    #             )
    #             sleep(5)
    #         data_str = self._data_loader.get(task["task_key"])
    #         data = json.loads(data_str)
    #         logger.debug(
    #             f"{self._name} {self._device}: retrieved data for task"
    #             f" {task['model_name']} {task['task_type']} with id"
    #             f" {task['task_id']} from client {task['client_id']}"
    #         )
    #         logger.spam(f"\n{pformat(object=data, indent=1, width=1)}")
    #         img_url = data["task"]["items"][0]["urls"]["-1"]
    #         img_name = img_url.split("/")[-1]
    #         if not os.path.exists(img_name):
    #             logger.debug(
    #                 f"{self._name} {self._device}: Downloading image"
    #                 f" {img_name}..."
    #             )
    #             request.urlretrieve(img_url, img_name)
    #         img = Image.open(img_name)
    #         return img
    #     except exceptions.ConnectionError as conn_err:
    #         logger.error(conn_err)
    #         raise exceptions.ConnectionError from conn_err
    #     except exceptions.RedisError as redis_err:
    #         logger.error(redis_err)
    #         raise exceptions.RedisError from redis_err
    #     except KeyboardInterrupt as kb_err:
    #         raise KeyboardInterrupt from kb_err

    # @timer(Timers.PERF_COUNTER)
    # def _load_data(self):
    #     filename = "dog.jpg"

    #     # Download an example image from the pytorch website
    #     if not os.path.isfile(filename):
    #         import urllib

    #         url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    #         try:
    #             urllib.URLopener().retrieve(url, filename)
    #         except:
    #             urllib.request.urlretrieve(url, filename)

    #     # sample execution (requires torchvision)
    #     from torchvision import transforms

    #     input_image = Image.open(filename)
    #     preprocess = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),
    #         ]
    #     )
    #     input_tensor = preprocess(input_image)
    #     image = input_tensor.unsqueeze(
    #         0
    #     )  # create a mini-batch as expected by the model

    #     images = torch.cat([image] * 8)
    #     return images

    def shutdown(self):
        """Shutdown the runner."""
        logger.debug(f"{self._name} {self._device}: stopping...")
        self._stop.set()
        if hasattr(self, "_load_jobs"):
            for load_job in self._load_jobs:
                load_job.terminate()
        logger.debug(f"{self._name} {self._device}: stopped!")

    @property
    def model_in(self):
        return self._model_in

    @property
    def task_in(self):
        return self._task_in
