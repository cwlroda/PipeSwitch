# -*- coding: utf-8 -*-
"""PipeSwitch Runner

This module spawns workers and allocates tasks to available workers.

Todo:
    * None
"""
import time
from threading import Thread
from typing import Any, OrderedDict, List
import multiprocessing as mp
import torch
import traceback

from pipeswitch.common.consts import State
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
        model_list: List[str],
        runner_status_queue: mp.Queue,
        results_queue: mp.Queue,
    ) -> None:
        super().__init__()
        self.do_run: bool = True
        self.mode: str = mode
        self.device: int = device
        self.model_list: List[str] = model_list
        self.runner_status_queue: mp.Queue = runner_status_queue
        self.task_queue: mp.Queue = mp.Queue()
        self.results_queue: mp.Queue = results_queue
        self.status: State = State.STARTUP
        self.worker_status: OrderedDict[int, State] = OrderedDict()
        self.cur_w_idx: int = 0
        self.models = {}
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
                    # task: OrderedDict[str, Any] = json.loads(msg)
                    model_name = task["task_name"]
                    logger.info(
                        f"Runner {self.device}: received task"
                        f" {model_name} from client {task['client_id']}"
                    )
                    self._update_status(State.BUSY)
                    logger.debug(
                        f"Runner {self.device}: CPU debug mode task execution"
                    )
                    self.model = torch.hub.load(
                        "pytorch/vision:v0.10.0",
                        "resnet152",
                        pretrained=True,
                        verbose=False,
                    )
                    # self.data = task["data"]
                    self.model.eval()
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
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
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
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
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
        data = torch.cuda.FloatTensor(self.data)
        input_batch = data.view(-1, 3, 224, 224).cuda(
            device=self.device, non_blocking=True
        )
        with torch.no_grad():
            output = self.model(input_batch)
        return output.sum().item()

    @property
    def exception(self):
        """Returns (exception, traceback) if run raises one."""
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
