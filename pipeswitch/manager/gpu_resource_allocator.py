import os
from collections import OrderedDict
from pipeswitch.common.logger import logger
import gpustat
import torch


class GPUResourceAllocator:
    def __init__(self):
        self.gpus = self.get_gpus()
        self.cuda_init()

    def cuda_init(self):
        assert torch.cuda.is_available()
        assert (
            len(self.gpus) > 0 and len(self.gpus) <= torch.cuda.device_count()
        ), "No GPUs available"

    def get_gpus(self):
        stats = gpustat.GPUStatCollection.new_query()
        gpus = OrderedDict()
        for gpu in stats:
            if len(gpu["processes"]) == 0:
                gpus[gpu["index"]] = gpu

        return gpus

    def get_free_gpus(self):
        free_gpus = []
        for id, gpu in self.gpus.items():
            if len(gpu["processes"]) == 0:
                free_gpus.append(id)

        return free_gpus

    def _check_gpu(self, id):
        device = torch.device(f"cuda:{id}")
        X_train = torch.FloatTensor([0.0, 1.0, 2.0]).to(device)
        assert X_train.is_cuda, f"GPU {id} is not available"

    def auto_acquire_gpus(self, num_gpus=0):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            return os.environ["CUDA_VISIBLE_DEVICES"]

        free_gpus = self.get_free_gpus()
        if num_gpus == 0:
            num_gpus = len(free_gpus)
        elif num_gpus > len(free_gpus):
            raise Exception(
                "Unable to acquire %d GPUs, there are only %d available."
                % (num_gpus, len(self.free_gpus))
            )

        available_gpus = free_gpus[:num_gpus]
        gpus = ",".join([str(i) for i in available_gpus])
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

        logger.debug("Acquiring GPUs: %s" % os.environ["CUDA_VISIBLE_DEVICES"])
        for id in available_gpus:
            self._check_gpu(id)
        return available_gpus

    def warmup_gpus(self):
        for id in self.get_free_gpus():
            torch.cuda.set_device(id)
            torch.randn(1024, device=f"cuda:{id}")
            torch.cuda.allocate_shared_cache(id)
            logger.debug(f"Allocated shared cache for GPU {id}")

    def release_gpus(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
