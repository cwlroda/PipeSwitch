# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import time
import importlib
import json
from threading import Thread
from queue import Queue
from typing import Any, OrderedDict, List, Tuple
from multiprocessing import connection
import jsonpickle  # type: ignore
import torch

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, State
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import (
    RunnerCommsServer,
    RunnerTaskServer,
    RedisServer,
)
from pipeswitch.runner.status import (  # pylint: disable=unused-import
    RunnerStatus,
    WorkerStatus,
)
from pipeswitch.runner.worker import WorkerProc


class Runner(Thread):
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
        device: int,
        model_list: List[str],
        worker_list: List[
            Tuple[
                connection.Connection,
                WorkerProc,
                connection.Connection,
                connection.Connection,
            ]
        ],
    ) -> None:
        super().__init__()
        self.do_run: bool = True
        self.device: int = device
        self.model_list: List[str] = model_list
        self.status: State = State.STARTUP
        self.worker_list: List[
            Tuple[
                connection.Connection,
                WorkerProc,
                connection.Connection,
                connection.Connection,
            ]
        ] = worker_list
        self.worker_status: OrderedDict[int, State] = OrderedDict()
        self.cur_w_idx: int = 0
        self.models = {}
        self.models_queue: Queue = Queue()
        self.num_tasks_complete: int = 0
        self.num_tasks_failed: int = 0

    def run(self) -> None:
        """Main runner function that sets up the runner and runs it.

        Raises:
            `TypeError`: If the message received is not a JSON string.
        """
        logger.debug(f"Runner {self.device}: start")
        try:
            get_workers_status: Thread = Thread(target=self._get_workers_status)
            get_tasks: Thread = Thread(target=self._get_tasks)
            get_workers_status.daemon = True
            get_workers_status.start()
            get_tasks.daemon = True
            get_tasks.start()

            # Load models
            logger.debug(f"Runner {self.device}: loading models")
            # load_models: List[Thread] = [
            #     Thread(target=self._load_model, args=(model_name,), daemon=True)
            #     for model_name in self.model_list
            # ]
            # for load_model in load_models:
            #     load_model.start()

            # while True:
            #     if not self.models_queue.empty():
            #         model_name, model = self.models_queue.get()
            #         self.models[hash(model_name)] = model
            #         logger.debug(f"Runner {self.device}: loaded model {model_name}")

            for model_name in self.model_list:
                self.models[hash(model_name)] = self._load_model(model_name)
                logger.debug(f"Runner {self.device}: loaded model {model_name}")

            # Create CUDA stream
            self.cuda_stream_for_parameter = torch.cuda.Stream(  # type: ignore
                self.device
            )
            logger.debug(f"Runner {self.device}: create stream")
            while self.do_run:
                time.sleep(100)
        except KeyboardInterrupt as kb_int:
            raise KeyboardInterrupt from kb_int

    def _load_model(self, model_name: str) -> List[Any]:
        """Loads a specific model.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            List[Any]: List of parameters of the model grouped into batches.
        """
        try:
            # Import parameters
            logger.debug(f"Runner {self.device}: loading model {model_name}")
            model_module = importlib.import_module(
                "pipeswitch.task." + model_name
            )
            logger.debug(
                f"Runner {self.device}: importing {model_name} parameters"
            )
            batched_parameter_list = model_module.import_parameters()

            # Preprocess batches
            logger.debug(
                f"Runner {self.device}: preprocessing {model_name} parameters"
            )
            processed_batched_parameter_list: List[Any] = []
            for param, mod_list in batched_parameter_list:
                if param is None:
                    processed_batched_parameter_list.append((None, mod_list))
                else:
                    processed_batched_parameter_list.append(
                        (param.pin_memory(), mod_list)
                    )

            # self.models_queue.put((model_name, processed_batched_parameter_list))
            return processed_batched_parameter_list
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _transfer_parameter(
        self,
        batched_parameter_list: List[Any],
        cuda_stream_for_parameter: torch.cuda.Stream,  # type: ignore
        param_trans_pipe: connection.Connection,
    ) -> None:
        """Transfers model parameters to the workers.

        Args:
            batched_parameter_list (List[Any]):
                List of parameters of the model grouped into batches.
            cuda_stream_for_parameter (torch.cuda.Stream):
                CUDA stream for executing commands.
            param_trans_pipe (connection.Connection):
                Socket endpoint to send the parameters to.
        """
        try:
            param_cuda_list: List[Any] = []
            for param, mod_list in batched_parameter_list:
                with torch.cuda.stream(cuda_stream_for_parameter):
                    if param is not None:
                        param_cuda = param.cuda(non_blocking=True)
                        param_cuda_list.append(param_cuda)
                        e = torch.cuda.Event(enable_timing=True)  # type: ignore
                        e.record()
                        e.synchronize()
                    param_trans_pipe.send(mod_list[0])
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _get_workers_status(self) -> None:
        """Receives worker status updates and updates the worker database"""
        logger.debug(f"Start runner {self.device} worker status update thread")
        try:
            self.comms_server: RedisServer = RunnerCommsServer(
                module_id=self.device,
                host=REDIS_HOST,
                port=REDIS_PORT,
                pub_queue=Queue(),
                sub_queue=Queue(),
            )
            self.comms_server.daemon = True
            self.comms_server.start()
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
        while True:
            try:
                if not self.comms_server.sub_queue.empty():
                    msg: str = self.comms_server.sub_queue.get()
                    status_update: WorkerStatus = jsonpickle.decode(msg)
                    if status_update.device != self.device:
                        logger.debug(
                            "Ignoring worker status update from GPU"
                            f" {status_update.device}"
                        )
                    elif status_update.worker_id != -1:
                        logger.debug("Worker status update received")
                        self.worker_status[
                            status_update.worker_id
                        ] = status_update.status
                        logger.debug(
                            "Worker"
                            f" {status_update.device}-{status_update.worker_id}:"
                            f" status {status_update.status}"
                        )
                        self._update_status()
                    else:
                        logger.debug(f"Ignoring non-worker update msg: {msg}")
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(f"Ignoring invalid update msg: {msg}")
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err

    def _update_status(self) -> None:
        """Updates own runner status based on worker statuses"""
        try:
            logger.debug(f"Updating runner {self.device} status")
            status = State.IDLE
            for _, worker_status in self.worker_status.items():
                if worker_status == State.BUSY:
                    status = State.BUSY
                    break
            if status != self.status:
                self.status = status
                logger.debug(f"Runner {self.device}: status {self.status}")
                runner_status = RunnerStatus(device=self.device, status=status)
                json_status = jsonpickle.encode(runner_status)
                self.comms_server.pub_queue.put(json_status)
                while not self.comms_server.publish():
                    continue
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err

    def _get_tasks(self) -> None:
        try:
            self.task_server: RedisServer = RunnerTaskServer(
                module_id=self.device,
                host=REDIS_HOST,
                port=REDIS_PORT,
                pub_queue=Queue(),
                sub_queue=Queue(),
            )
            self.task_server.daemon = True
            self.task_server.start()
        except KeyboardInterrupt as kb_err:
            raise KeyboardInterrupt from kb_err
        while True:
            try:
                if not self.task_server.sub_queue.empty():
                    msg = self.task_server.sub_queue.get()
                    task: OrderedDict[str, Any] = json.loads(msg)

                    logger.info(
                        f"Runner {self.device}: received task"
                        f" {task['task_name']} from client {task['client_id']}"
                    )

                    # Get current worker
                    _, _, _, term_pipe = self.worker_list[self.cur_w_idx]
                    logger.debug(
                        f"Runner {self.device}: get current worker"
                        f" {self.cur_w_idx}"
                    )

                    # Send terminate signal to current worker
                    term_pipe.send("terminate")

                    # Get next worker to work on request
                    self.cur_w_idx = (self.cur_w_idx + 1) % len(
                        self.worker_list
                    )
                    new_pipe, _, param_trans_pipe_parent, _ = self.worker_list[
                        self.cur_w_idx
                    ]
                    logger.debug(
                        f"Runner {self.device}: notify new worker"
                        f" {self.cur_w_idx}"
                    )

                    # Send request to new worker
                    new_pipe.send(task)
                    logger.debug(f"Runner {self.device}: send data")

                    # Wait for current worker to terminate
                    _ = term_pipe.recv()
                    logger.debug(
                        f"Runner {self.device}: terminate current worker"
                    )

                    # Allocate cache to streams
                    with torch.cuda.stream(self.cuda_stream_for_parameter):
                        torch.cuda.insert_cache_for_param(  # type: ignore
                            self.device
                        )
                    logger.debug(f"Runner {self.device}: insert cache")

                    # Transfer parameters to GPU
                    batched_parameter_list = self.models[
                        hash(task["task_name"])
                    ]
                    self._transfer_parameter(
                        batched_parameter_list,
                        self.cuda_stream_for_parameter,
                        param_trans_pipe_parent,
                    )
                    logger.debug(f"Runner {self.device}: transfer parameters")

                    # Clear status
                    with torch.cuda.stream(self.cuda_stream_for_parameter):
                        torch.cuda.clear_cache(self.device)  # type: ignore
                    logger.debug(f"Runner {self.device}: clear status")

                    # Recv response
                    msg = new_pipe.recv()
                    if msg["status"] == str(State.FAILED):
                        logger.error(
                            f"Runner {self.device}: task"
                            f" {task['task_name']} failed"
                        )
                        self.task_server.pub_queue.put(json.dumps(msg))
                        while not self.task_server.publish():
                            continue
                        self.num_tasks_failed += 1
                    else:
                        logger.debug(
                            f"Runner {self.device}: task results from worker"
                            f" {msg['worker_id']}"
                        )
                        self.task_server.pub_queue.put(json.dumps(msg))
                        while not self.task_server.publish():
                            continue
                        self.num_tasks_complete += 1
            except RuntimeError as runtime_err:
                logger.error(runtime_err)
                logger.error(f"Runner {self.device}: task failed!")
            except TypeError as json_decode_err:
                logger.debug(json_decode_err)
                logger.debug(f"Runner {self.device}: invalid task message")
            except KeyboardInterrupt as kb_err:
                raise KeyboardInterrupt from kb_err
