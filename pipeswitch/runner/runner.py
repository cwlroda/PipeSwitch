# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import time
from threading import Thread
from typing import Any, OrderedDict
import multiprocessing as mp
import torch
import traceback
import redis
import json
import urllib
from PIL import Image

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, State
from pipeswitch.common.logger import logger
from pipeswitch.runner.status import (  # pylint: disable=unused-import
    RunnerStatus,
)


class Runner(mp.Process):
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

    def __init__(
        self,
        mode: str,
        device: int,
        runner_status_queue: mp.Queue,
        results_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self.do_run: bool = True
        self.mode: str = mode
        self.device: int = device
        self.runner_status_queue: mp.Queue = runner_status_queue
        self.task_queue: mp.Queue = mp.Queue()
        self.results_queue: mp.Queue = results_queue
        self.status: State = State.STARTUP
        self.models_queue: mp.Queue = mp.Queue()
        self.num_tasks_complete: int = 0
        self.num_tasks_failed: int = 0
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self) -> None:
        """Main runner function that sets up the runner and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        logger.debug(f"Runner {self.device}: start")
        try:
            self._data_loader = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                encoding="utf-8",
                decode_responses=True,
            )
            get_tasks: Thread = Thread(target=self._get_tasks)
            get_tasks.daemon = True
            get_tasks.start()

            if self.mode == "gpu":
                # Create CUDA stream
                self.cuda_stream_for_parameter = torch.cuda.Stream(  # type: ignore
                    self.device
                )
                logger.debug(f"Runner {self.device}: create stream")
            else:
                self.cuda_stream_for_parameter = None

            while self.do_run:
                time.sleep(100000)
        except RuntimeError as runtime_err:
            logger.error(runtime_err)
            tb = traceback.format_exc()
            self._cconn.send((runtime_err, tb))
        except BrokenPipeError as bp_error:
            logger.error(bp_error)
            tb = traceback.format_exc()
            self._cconn.send((bp_error, tb))
        except KeyboardInterrupt as kb_int:
            tb = traceback.format_exc()
            self._cconn.send((kb_int, tb))

    def _update_status(self, status: State) -> None:
        """Updates own runner status based on worker statuses"""
        try:
            logger.debug(f"Updating runner {self.device} status")
            self.status = status
            logger.debug(f"Runner {self.device}: status {self.status}")
            runner_status = RunnerStatus(device=self.device, status=status)
            self.runner_status_queue.put(runner_status)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _get_tasks(self) -> None:
        self._update_status(State.IDLE)
        while True:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    logger.info(
                        f"Runner {self.device}: received task"
                        f" {task['model_name']} {task['task_type']} with id"
                        f" {task['task_id']} from client {task['client_id']}"
                    )
                    self._update_status(State.BUSY)
                    logger.debug(
                        f"Runner {self.device}: CPU debug mode task execution"
                    )
                    # task_type = task["task_type"]
                    task_key = task["task_key"]
                    # model = task["model"]

                    data = self._load_data(task_key)
                    logger.spam(data)
                    # if task_type == "inference":
                    #     model.eval()
                    # elif task_type == "train":
                    #     model.train()
                    # else:
                    #     logger.error(
                    #         f"Runner {self.device}: unknown task type"
                    #         f" {task_type}"
                    #     )

                    # TODO: run inference on a proper model
                    # TODO: load data from redis store
                    # output = self._run_inference()
                    # start doing inference
                    # frontend_scheduler will directly put
                    # mod_list[0] in to self.results_queue_trans

                    # if self.mode == "gpu":
                    #     with torch.cuda.stream(
                    #         model_summary.cuda_stream_for_computation
                    #     ):
                    #         logger.debug(
                    #             f"Worker {self.device}-{self.worker_id}: Dummy"
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
                    msg: OrderedDict[str, Any] = {
                        "worker_id": self.device,
                        "client_id": task["client_id"],
                        "task_type": task["task_type"],
                        "task_id": task["task_id"],
                        "model_name": task["model_name"],
                        "status": str(State.SUCCESS),
                        "output": output,
                    }
                    self.results_queue.put(msg)
                    # model_summary.reset_initialized(model_summary.model)
                    self._update_status(State.IDLE)
            except RuntimeError as runtime_err:
                logger.error(runtime_err)
                logger.error(f"Runner {self.device}: task failed!")
                msg: OrderedDict[str, Any] = {
                    "worker_id": self.device,
                    "client_id": task["client_id"],
                    "task_type": task["task_type"],
                    "task_id": task["task_id"],
                    "model_name": task["model_name"],
                    "status": str(State.FAILED),
                }
                self.results_queue.put(msg)
                self._update_status(State.IDLE)
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(f"Runner {self.device}: invalid task message")
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    def _run_inference(self) -> None:
        try:
            data = torch.cuda.FloatTensor(self.data)
            input_batch = data.view(-1, 3, 224, 224).cuda(
                device=self.device, non_blocking=True
            )
            with torch.no_grad():
                output = self.model(input_batch)
            return output.sum().item()
        except TypeError as type_err:
            logger.error(type_err)
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _load_data(self, task_key: str) -> OrderedDict[str, Any]:
        """Loads data from redis store"""
        try:
            while not self._data_loader.ping():
                logger.error(
                    f"Runner {self.device} data loader: connection failed!"
                )
                logger.error(
                    f"Runner {self.device} data loader: reconnecting in 5s..."
                )
                time.sleep(5)
            task_str = self._data_loader.get(task_key)
            task = json.loads(task_str)
            logger.debug(f"Runner {self.device}: retrieved task data {task}")
            img_url = task["task"]["items"][0]["urls"]["-1"]
            urllib.request.urlretrieve(img_url, "img.png")
            img = Image.open("img.png")
            return img
        except redis.exceptions.ConnectionError as conn_err:
            logger.error(conn_err)
            raise redis.exceptions.ConnectionError from conn_err
        except redis.exceptions.RedisError as redis_err:
            logger.error(redis_err)
            raise redis.exceptions.RedisError from redis_err
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    @property
    def exception(self):
        """Returns (exception, traceback) if run raises one."""
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
