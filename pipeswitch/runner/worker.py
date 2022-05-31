from queue import Queue
import multiprocessing as mp

import torch
import time

from pipeswitch.common.logger import logger
from pipeswitch.runner.worker_common import ModelSummary
from pipeswitch.runner.worker_terminate import WorkerTermThd
from util.util import timestamp


class WorkerProc(mp.Process):
    def __init__(self, device, model_list, pipe, param_trans_pipe, term_pipe):
        super(WorkerProc, self).__init__()
        self.device = device
        self.model_list = model_list
        self.pipe = pipe
        self.param_trans_pipe = param_trans_pipe
        self.term_pipe = term_pipe

    def run(self):
        timestamp("worker", "start")

        # Warm up CUDA and get shared cache
        try:
            torch.cuda.set_device(self.device)
            torch.randn(1024, device=f"cuda:{self.device}")
            time.sleep(1)
            torch.cuda.recv_shared_cache(self.device)  # pylint: disable=no-member
            timestamp("worker", "share_gpu_memory")

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
            timestamp("worker", "import models")

            # ------- start terminate thread -----------
            term_t = WorkerTermThd(self.term_pipe, complete_queue, TERMINATE_SIGNAL)
            term_t.start()
            timestamp("worker", "start_term_thd")
            # ------- terminate thread started ---------

            while True:
                # event loop get a msg then compute
                # after started forward compute
                # last while loop for receiving complete queue trans
                agent, model_name = self.pipe.recv()
                model_summary = model_map[hash(model_name)]
                TERMINATE_SIGNAL[0] = 1
                timestamp("worker_proc", "get_model")

                data_b = self.pipe.recv()
                timestamp("worker_proc", "get_data")

                # start doing inference
                # frontend_scheduler will directly put
                # mod_list[0] in to self.complete_queue_trans
                try:
                    if "training" in model_name:
                        self.pipe.send("FNSH")
                        agent.send(b"FNSH")
                    with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                        output = model_summary.execute(data_b)
                        print("Get output", output)
                        del output

                    if "inference" in model_name:
                        self.pipe.send("FNSH")
                        agent.send(b"FNSH")
                except Exception as e:
                    print("exception caught")
                    print(e)
                    self.pipe.send("FAIL")
                    agent.send(b"FAIL")
                    complete_queue.put("FAIL")

                # start do cleaning
                TERMINATE_SIGNAL[0] = 0
                timestamp("worker_comp_thd", "complete")

                model_summary.reset_initialized(model_summary.model)
        except RuntimeError as e:
            logger.error(e)
            # raise Exception("Failed to start worker on GPU {self.device}")
