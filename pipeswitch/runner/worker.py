import sys
from queue import Queue
import multiprocessing as mp

import torch
import time

from pipeswitch.common.logger import logger
from pipeswitch.common.consts import State
from pipeswitch.runner.status import WorkerStatus
from pipeswitch.runner.worker_common import ModelSummary
from pipeswitch.runner.worker_terminate import WorkerTermThd


class WorkerProc(mp.Process):
    def __init__(
        self,
        device,
        id,
        worker_status_update_queue,
        model_list,
        pipe,
        param_trans_pipe,
        term_pipe,
    ):
        super(WorkerProc, self).__init__()
        self.device = device
        self.id = id
        self.status = State.IDLE
        self.worker_status_update_queue = worker_status_update_queue
        self.model_list = model_list
        self.pipe = pipe
        self.param_trans_pipe = param_trans_pipe
        self.term_pipe = term_pipe

    def run(self):
        logger.debug(f"Worker {self.device}-{self.id}: start")

        # Warm up CUDA and get shared cache
        try:
            torch.cuda.set_device(self.device)
            torch.randn(1024, device=f"cuda:{self.device}")
            time.sleep(1)
            torch.cuda.recv_shared_cache(self.device)  # pylint: disable=no-member
            logger.debug(f"Worker {self.device}-{self.id}: share GPU memory")

            # Create required variables
            model_map = {}
            TERMINATE_SIGNAL = [0]  # 0 Idle, 1 Running, 2 Terminate
            complete_queue = Queue()

            # Import models
            for model_name in self.model_list:
                model_summary = ModelSummary(
                    self.device, model_name, TERMINATE_SIGNAL, self.param_trans_pipe
                )
                model_map[hash(model_name)] = model_summary
            logger.debug(f"Worker {self.device}-{self.id}: import models")

            # ------- start terminate thread -----------
            term_t = WorkerTermThd(self.term_pipe, complete_queue, TERMINATE_SIGNAL)
            term_t.start()
            logger.debug(f"Worker {self.device}-{self.id}: start terminate thread")
            # ------- terminate thread started ---------
            self.update_status(State.IDLE)

            while True:
                # event loop get a msg then compute
                # after started forward compute
                # last while loop for receiving complete queue trans
                task = self.pipe.recv()
                model_name = task["task_name"]
                self.update_status(State.BUSY)
                model_summary = model_map[hash(model_name)]
                TERMINATE_SIGNAL[0] = 1
                logger.debug(f"Worker {self.device}-{self.id}: get task {model_name}")

                # start doing inference
                # frontend_scheduler will directly put
                # mod_list[0] in to self.complete_queue_trans
                try:
                    with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                        output = model_summary.execute(task["data"])
                        logger.info(f"Get output: {output}")
                        del output

                    if "inference" in model_name:
                        self.pipe.send("FNSH")
                except Exception as e:
                    logger.exception(e)
                    self.pipe.send("FAIL")
                    complete_queue.put("FAIL")

                # start do cleaning
                TERMINATE_SIGNAL[0] = 0
                logger.debug(f"Worker {self.device}-{self.id}: task complete")

                model_summary.reset_initialized(model_summary.model)
                self.update_status(State.IDLE)
        except RuntimeError as e:
            logger.error(e)
            sys.exit(1)

    def update_status(self, status):
        if status != self.status:
            self.status = status
            self.worker_status_update_queue.put(
                WorkerStatus(self.device, self.id, self.status)
            )
        logger.debug(f"Worker {self.device}-{self.id}: status {self.status}")
