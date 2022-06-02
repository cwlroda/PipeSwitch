import json
import jsonpickle
import threading
from collections import OrderedDict
import torch
import importlib

from pipeswitch.common.logger import logger
from pipeswitch.common.consts import State
from pipeswitch.common.servers import RunnerCommsServer, RunnerTaskServer
from pipeswitch.runner.status import RunnerStatus


class RunnerThd(threading.Thread):
    def __init__(
        self,
        device,
        model_list,
        worker_list,
        worker_status_update_queue,
        requests_queue,
    ):
        super(RunnerThd, self).__init__()
        self.device = device
        self.model_list = model_list
        self.status = State.STARTUP
        self.qin = requests_queue
        self.worker_list = worker_list
        self.worker_status = OrderedDict()
        self.worker_status_update_queue = worker_status_update_queue
        self.cur_w_idx = 0

        self.commsServer = RunnerCommsServer(self.device, "127.0.0.1", 6379)
        self.taskServer = RunnerTaskServer(self.device, "127.0.0.1", 6379)

    def run(self):
        logger.debug(f"Runner {self.device}: start")
        self.commsServer.start()
        self.taskServer.start()

        # Load models
        models = {}
        for model_name in self.model_list:
            models[hash(model_name)] = self._load_model(model_name)
        logger.debug(f"Runner {self.device}: load model")

        # Create CUDA stream
        cuda_stream_for_parameter = torch.cuda.Stream(self.device)
        logger.debug(f"Runner {self.device}: create stream")

        while True:
            self.get_workers_status()
            self.update_status()

            if not self.taskServer.sub_queue.empty():
                msg = self.taskServer.sub_queue.get()
                try:
                    task = json.loads(msg)
                    if int(task["runner_id"]) != self.device:
                        logger.debug(
                            f"Runner {self.device}: ignore task {task['task_name']}"
                        )
                        continue

                    logger.info(
                        f"Runner {self.device}: received task {task['task_name']} from client {task['client_id']}"
                    )

                    # Get current worker
                    _, _, _, term_pipe = self.worker_list[self.cur_w_idx]
                    logger.debug(
                        f"Runner {self.device}: get current worker {self.cur_w_idx}"
                    )

                    # Send terminate signal to current worker
                    term_pipe.send("terminate")

                    # Get next worker to work on request
                    self.cur_w_idx = (self.cur_w_idx + 1) % len(self.worker_list)
                    new_pipe, _, param_trans_pipe_parent, _ = self.worker_list[
                        self.cur_w_idx
                    ]
                    logger.debug(
                        f"Runner {self.device}: notify new worker {self.cur_w_idx}"
                    )

                    # Send request to new worker
                    new_pipe.send(task)
                    logger.debug(f"Runner {self.device}: send data")

                    # Wait for current worker to terminate
                    resp = term_pipe.recv()
                    logger.debug(f"Runner {self.device}: terminate current worker")

                    # Allocate cache to streams
                    with torch.cuda.stream(cuda_stream_for_parameter):
                        torch.cuda.insert_shared_cache_for_parameter(self.device)
                    logger.debug(f"Runner {self.device}: insert cache")

                    # Transfer parameters to GPU
                    batched_parameter_list = models[hash(model_name)]
                    self._transfer_parameter(
                        new_pipe,
                        batched_parameter_list,
                        cuda_stream_for_parameter,
                        param_trans_pipe_parent,
                    )
                    logger.debug(f"Runner {self.device}: transfer parameters")

                    # Clear status
                    with torch.cuda.stream(cuda_stream_for_parameter):
                        torch.cuda.clear_shared_cache(self.device)
                    logger.debug(f"Runner {self.device}: clear status")

                    self.get_workers_status()
                    self.update_status()

                    # Recv response
                    res = new_pipe.recv()
                    logger.debug(f"Runner {self.device}: get response")
                except:
                    logger.debug(f"Runner {self.device}: invalid task message")
                    continue

    def _load_model(self, model_name):
        # Import parameters
        model_module = importlib.import_module("pipeswitch.task." + model_name)
        batched_parameter_list = model_module.import_parameters()

        # Preprocess batches
        processed_batched_parameter_list = []
        for param, mod_list in batched_parameter_list:
            if param is None:
                processed_batched_parameter_list.append((None, mod_list))
            else:
                processed_batched_parameter_list.append((param.pin_memory(), mod_list))

        return processed_batched_parameter_list

    def _transfer_parameter(
        self, pipe, batched_parameter_list, cuda_stream_for_parameter, param_trans_pipe
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

    def get_workers_status(self):
        while not self.worker_status_update_queue.empty():
            worker_status = self.worker_status_update_queue.get()
            assert (
                worker_status.device == self.device
            ), "Worker status update from wrong device"
            self.worker_status[worker_status.id] = worker_status

    def update_status(self):
        status = State.IDLE
        for _, worker_status in self.worker_status.items():
            if worker_status.status == State.BUSY:
                status = State.BUSY
                break
        if status != self.status:
            self.status = status
            logger.debug(f"Runner {self.device}: status {self.status}")
            status = RunnerStatus(self.device, None, status)
            json_status = jsonpickle.encode(status)
            self.commsServer.pub_queue.put(json_status)
            self.commsServer.publish()
