import time
import jsonpickle
from typing import Any, List, OrderedDict
from multiprocessing import connection
import traceback
from threading import Thread
from queue import Queue

import torch
import torch.multiprocessing as mp

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, State
from pipeswitch.common.logger import logger
from pipeswitch.common.servers import RedisServer, WorkerCommsServer
from pipeswitch.runner.status import WorkerStatus
from pipeswitch.runner.worker_common import ModelSummary
from pipeswitch.runner.worker_terminate import WorkerTermThd


class WorkerProc(mp.Process):
    def __init__(
        self,
        mode: str,
        device: int,
        worker_id: int,
        model_list: List[str],
        pipe: connection.Connection,
        param_trans_pipe: connection.Connection,
        term_pipe: connection.Connection,
    ):
        super().__init__()
        self.do_run: bool = True
        self.mode: str = mode
        self.device: int = device
        self.worker_id: int = worker_id
        self.status: State = State.STARTUP
        self.model_list: List[str] = model_list
        self.model_map = {}
        self.TERMINATE_SIGNAL = [0]  # 0 Idle, 1 Running, 2 Terminate
        self.pipe: connection.Connection = pipe
        self.param_trans_pipe: connection.Connection = param_trans_pipe
        self.term_pipe: connection.Connection = term_pipe
        self.comms_server: WorkerCommsServer = WorkerCommsServer(
            module_id=self.device,
            worker_id=self.worker_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            pub_queue=mp.Queue(),
            sub_queue=mp.Queue(),
        )
        self.status_queue: "mp.Queue[State]" = mp.Queue()
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        logger.debug(f"Worker {self.device}-{self.worker_id}: start")
        try:
            self.comms_server.create_pub()

            if self.mode == "gpu":
                torch.cuda.set_device(device=self.device)
                # Get shared cache
                torch.cuda.recv_cache(device=self.device)
                logger.debug(
                    f"Worker {self.device}-{self.worker_id}: share GPU memory"
                )

            # Create required variables
            self.complete_queue = mp.Queue()

            # Import models
            for model_name in self.model_list:
                model_summary = ModelSummary(
                    self.device,
                    model_name,
                    self.TERMINATE_SIGNAL,
                    self.param_trans_pipe,
                )
                self.model_map[hash(model_name)] = model_summary
            logger.debug(
                f"Worker {self.device}-{self.worker_id}: import models"
            )

            # ------- start terminate thread -----------
            term_t = WorkerTermThd(
                self.term_pipe, self.complete_queue, self.TERMINATE_SIGNAL
            )
            term_t.daemon = True
            term_t.start()
            logger.debug(
                f"Worker {self.device}-{self.worker_id}: start terminate thread"
            )
            # ------- terminate thread started ---------

            compute = Thread(target=self._compute)
            compute.daemon = True
            compute.start()
            # compute.join()
            while self.do_run:
                # print("worker")
                pass
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

    @property
    def exception(self):
        """Returns (exception, traceback) if run raises one."""
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

    def _compute(self) -> None:
        self._update_status(State.IDLE)
        while True:
            # event loop get a msg then compute
            # after started forward compute
            # last while loop for receiving complete queue trans
            task = self.pipe.recv()
            model_name = task["task_name"]
            self._update_status(State.BUSY)
            model_summary = self.model_map[hash(model_name)]
            self.TERMINATE_SIGNAL[0] = 1
            logger.debug(
                f"Worker {self.device}-{self.worker_id}: get task {model_name}"
            )

            # start doing inference
            # frontend_scheduler will directly put
            # mod_list[0] in to self.complete_queue_trans
            try:
                if self.mode == "gpu":
                    with torch.cuda.stream(
                        model_summary.cuda_stream_for_computation
                    ):
                        logger.debug(
                            f"Worker {self.device}-{self.worker_id}: Dummy"
                            " inference execution"
                        )
                        output = "some random data"
                        time.sleep(5)
                        # output = model_summary.execute(task["data"])
                        # logger.info(f"Get output: {output}")
                        # del output
                else:
                    logger.debug(
                        f"Worker {self.device}-{self.worker_id}: CPU debug mode"
                        " task execution"
                    )
                    output = "some random data"
                    time.sleep(5)
                msg: OrderedDict[str, Any] = {
                    "worker_id": self.worker_id,
                    "client_id": task["client_id"],
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "status": str(State.SUCCESS),
                    "output": output,
                }

                self.pipe.send(msg)
            except RuntimeError as runtime_err:
                logger.error(runtime_err)
                logger.error(
                    f"Worker {self.device}-{self.worker_id}: task failed!"
                )
                msg: OrderedDict[str, Any] = {
                    "worker_id": self.worker_id,
                    "client_id": task["client_id"],
                    "task_name": task["task_name"],
                    "status": str(State.FAILED),
                }
                self.pipe.send(msg)
                # self.complete_queue.put("FAIL")
            except KeyboardInterrupt as kb_int:
                tb = traceback.format_exc()
                self._cconn.send((kb_int, tb))

            # start do cleaning
            self.TERMINATE_SIGNAL[0] = 0
            logger.debug(
                f"Worker {self.device}-{self.worker_id}: task complete"
            )

            model_summary.reset_initialized(model_summary.model)
            self._update_status(State.IDLE)

    def _update_status(self, status: State) -> None:
        try:
            logger.debug(
                f"Updating worker {self.device}-{self.worker_id} status"
            )
            self.status = status
            worker_status = WorkerStatus(
                device=self.device,
                worker_id=self.worker_id,
                status=self.status,
            )
            json_status = jsonpickle.encode(worker_status)
            self.comms_server.pub_queue.put(json_status)
            while not self.comms_server.publish():
                continue
            logger.debug(
                f"Worker {self.device}-{self.worker_id}: status {self.status}"
            )
        except KeyboardInterrupt as kb_int:
            tb = traceback.format_exc()
            self._cconn.send((kb_int, tb))
