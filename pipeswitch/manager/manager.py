import os
import threading
from collections import OrderedDict
from queue import Queue

import torch
import torch.multiprocessing as mp

from pipeswitch.common.logger import logger
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.manager.servers import RequestsServer, RunnersServer
from pipeswitch.manager.scheduler import Scheduler

from pipeswitch.frontend_schedule import FrontendScheduleThd
from pipeswitch.worker import WorkerProc


class ManagerThd(threading.Thread):
    def __init__(self):
        mp.set_start_method("spawn")
        self.do_run = True

        self.GRA = GPUResourceAllocator()
        self.reqServer = RequestsServer("127.0.0.1", 6379)
        self.runnServer = RunnersServer("127.0.0.1", 6379)
        self.model_list = []
        self.runners = OrderedDict()
        self.scheduler = Scheduler()

    def setup(self):
        logger.info("Manager: Start")
        self.reqServer.start()
        self.runnServer.start()
        self.load_model_list("model_list.txt")
        logger.info(f"Available GPUs: {manager.GRA.auto_acquire_gpus()}")
        self.GRA.warmup_gpus()
        self.create_runners()

    def load_model_list(self, file_name):
        assert os.path.exists(file_name)
        with open(file_name) as f:
            for line in f.readlines():
                self.model_list.append(line.strip())

    def create_runners(self, num_workers=2):
        num_workers = 1
        for id in self.GRA.get_free_gpus():
            try:
                torch.cuda.set_device(id)
                worker_list = []
                for i in range(num_workers):
                    p_parent, p_child = mp.Pipe()
                    param_trans_parent, param_trans_child = mp.Pipe()
                    term_parent, term_child = mp.Pipe()
                    worker = WorkerProc(
                        id, self.model_list, p_child, param_trans_child, term_child
                    )
                    worker.start()
                    torch.cuda.send_shared_cache(id)
                    worker_list.append(
                        (p_parent, worker, param_trans_parent, term_parent)
                    )
                    logger.info(f"Created worker {i+1} in GPU {id}")

                requests_queue = Queue()
                t_sch = FrontendScheduleThd(
                    id, self.model_list, requests_queue, worker_list
                )
                t_sch.start()

                self.runners[id] = t_sch
                logger.info(f"Created scheduler in GPU {id}")
            except Exception as e:
                logger.exception(e)
                logger.warning(f"Manager: Failed to create runner for GPU {id}")
            finally:
                continue
            # break

    def run(self):
        while not self.reqServer.queue.empty():
            request = self.reqServer.queue.get()
            self.scheduler.schedule(self.GRA.get_free_gpus())


if __name__ == "__main__":
    manager = ManagerThd()
    manager.setup()
    logger.info("Manager setup done")

    # if manager.reqServer.pub.ping() and manager.reqServer.sub.ping():
    #     logger.info("reqServer: PONG")
    # else:
    #     logger.error("reqServer: Connection failed!")

    # if manager.instServer.pub.ping() and manager.instServer.sub.ping():
    #     logger.info("instServer: PONG")
    # else:
    #     logger.error("instServer: Connection failed!")

    while manager.do_run:
        try:
            manager.run()
        except Exception as e:
            logger.exception(f"Manager Exception: {e}")
            manager.do_run = False

    manager.GRA.release_gpus()
