import json
import os
import time
import threading
from collections import OrderedDict
from unittest import runner

import torch
import torch.multiprocessing as mp

from pipeswitch.common.logger import logger
from pipeswitch.manager.gpu_resource_allocator import GPUResourceAllocator
from pipeswitch.common.servers import ManagerRequestsServer, ManagerResultsServer
from pipeswitch.manager.scheduler import Scheduler

from pipeswitch.runner.runner import RunnerThd
from pipeswitch.runner.worker import WorkerProc


class ManagerThd(threading.Thread):
    def __init__(self):
        mp.set_start_method("spawn")
        self.do_run = True

        self.GRA = GPUResourceAllocator()
        self.reqServer = ManagerRequestsServer("127.0.0.1", 6379)
        self.resServer = ManagerResultsServer("127.0.0.1", 6379)
        self.model_list = []
        self.runners = OrderedDict()
        self.scheduler = Scheduler()

    def setup(self):
        logger.info("Manager: Start")
        self.reqServer.start()
        self.resServer.start()
        self.scheduler.start()
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
        for id in [0]:  # self.GRA.get_free_gpus():
            torch.cuda.set_device(id)
            worker_status = mp.Queue()
            worker_list = []
            for i in range(num_workers):
                p_parent, p_child = mp.Pipe()
                param_trans_parent, param_trans_child = mp.Pipe()
                term_parent, term_child = mp.Pipe()
                worker = WorkerProc(
                    id,
                    i,
                    worker_status,
                    self.model_list,
                    p_child,
                    param_trans_child,
                    term_child,
                )
                try:
                    worker.start()
                except Exception as e:
                    logger.exception(e)
                    logger.warning(f"Manager: Failed to create runner for GPU {id}")

                torch.cuda.send_shared_cache(id)
                worker_list.append((p_parent, worker, param_trans_parent, term_parent))
                logger.debug(f"Created worker {i+1} in GPU {id}")

            requests_queue = mp.Queue()
            runner = RunnerThd(
                id,
                self.model_list,
                worker_list,
                worker_status,
                requests_queue,
            )
            runner.start()

            self.runners[id] = runner
            logger.info(f"Created runner in GPU {id}")

    def run(self):
        logger.info(f"Manager: Waiting for all runners to be ready")
        while len(self.scheduler.runner_status) != len(self.runners):
            time.sleep(1)
        logger.info(f"Manager: Ready to receive requests")
        while True:
            if not self.reqServer.sub_queue.empty():
                msg = self.reqServer.sub_queue.get()
                try:
                    task = json.loads(msg)
                    logger.info(f"Manager: Received task {task['task_name']}")
                except:
                    logger.debug(f"Ignoring message {msg}")
                    continue

                runner_id = self.scheduler.schedule()
                msg = {
                    "runner_id": runner_id,
                    "client_id": task["client_id"],
                    "task_name": task["task_name"],
                    "data": task["data"],
                }
                logger.info(
                    f"Assigning task {task['task_name']} from client {task['client_id']} to runner {runner_id}"
                )
                json_msg = json.dumps(msg)
                self.reqServer.pub_queue.put(json_msg)
                self.reqServer.publish()


if __name__ == "__main__":
    manager = ManagerThd()
    manager.setup()
    logger.info("Manager setup done")

    while manager.do_run:
        try:
            manager.run()
        except Exception as e:
            logger.exception(f"Manager Exception: {e}")
            manager.do_run = False

    manager.GRA.release_gpus()
