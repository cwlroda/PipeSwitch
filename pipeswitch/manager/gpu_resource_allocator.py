# -*- coding: utf-8 -*-
"""PipeSwitch GPU Resource Allocator

This module queries available GPUs and allocates GPU resources
for the PipeSwitch Manager.

Todo:
    * None
"""

import os
from typing import OrderedDict, List
from gpustat import GPUStat, GPUStatCollection  # type: ignore
import torch

from pipeswitch.common.logger import logger


class GPUResourceAllocator:
    """GPU Resource Allocator class.

    It has two main functions:
        - Queries available GPUs and reserves them.
        - Checks if each available GPU can be utilized by PyTorch.

    Attributes:
        gpus (`OrderedDict[int, GPUStat]`):
            Dictionary of all GPUs in the system regardless of availability.
    """

    def __init__(self) -> None:
        self.gpus = self._get_gpus()

    def cuda_init(self) -> None:
        """Checks if available GPUs are visible by PyTorch.

        Raises:
            `AssertionError`: If CUDA is not available.

            `AssertionError`: If the number of GPUs visible by PyTorch
                is not equal to the total number of available GPUs.
        """
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            raise KeyboardInterrupt
        if len(self.gpus) < 1 or torch.cuda.device_count() < 1:
            logger.error("No GPUs available")
            raise KeyboardInterrupt

    def get_free_gpus(self) -> List[int]:
        """Query available GPUs.

        Returns:
            List[int]: List of available GPU ids.
        """
        free_gpus: List[int] = []
        for gpu_id, gpu in self.gpus.items():
            if len(gpu["processes"]) == 0:
                free_gpus.append(gpu_id)

        return free_gpus

    def reserve_gpus(self, num_gpus: int = 0) -> List[int]:
        """Reserves set amount of GPUs.

        Args:
            num_gpus (int, optional): total number of GPUs to reserve. Defaults to 0.

        Returns:
            List[int]: list of IDs of reserved GPUs.
        """
        available_gpus: List[int] = []
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            logger.warning("CUDA_VISIBLE_DEVICES is already set")
            available_gpus = list(
                map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            )
            return available_gpus

        free_gpus: List[int] = self.get_free_gpus()
        if num_gpus == 0:
            num_gpus = len(free_gpus)
        if num_gpus > 0 and num_gpus <= len(free_gpus):
            logger.error(
                f"Unable to acquire {num_gpus} GPUs, there are only"
                f" {len(free_gpus)} available."
            )
            raise KeyboardInterrupt

        available_gpus = free_gpus[:num_gpus]
        gpu_str: str = ",".join([str(i) for i in available_gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        logger.debug(f"Acquiring GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        for gpu_id in available_gpus:
            self._check_gpu(gpu_id)
        return available_gpus

    def warmup_gpus(self, gpus: List[int]) -> None:
        """Warmup GPUs by running a dummy PyTorch function."""
        for gpu_id in gpus:
            torch.cuda.set_device(gpu_id)
            torch.randn(1024, device=f"cuda:{gpu_id}")
            torch.cuda.allocate_cache(gpu_id)  # type: ignore
            logger.debug(f"Allocated shared cache for GPU {gpu_id}")

    def release_gpus(self) -> None:
        """Release all reserved GPUS."""
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            return
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.debug("Releasing all GPUs")

    def _get_gpus(self) -> OrderedDict[int, GPUStat]:
        """Uses gpustat to query all GPUs in the system.

        Returns:
            `OrderedDict[int, GPUStat]`:
                A dictionary with GPU id as key and GPU stats as value.
        """
        stats: GPUStatCollection = GPUStatCollection.new_query()
        gpus: OrderedDict[int, GPUStat] = OrderedDict()
        for gpu in stats:
            if len(gpu["processes"]) == 0:
                gpus[gpu["index"]] = gpu

        return gpus

    def _check_gpu(self, gpu_id: int) -> None:
        """Checks if a GPU can be utilized by PyTorch.

        Args:
            gpu_id (int): GPU id.

        """
        device: torch.device = torch.device(f"cuda:{gpu_id}")
        x_train: torch.Tensor = torch.FloatTensor([0.0, 1.0, 2.0]).to(device)
        if not x_train.is_cuda:
            logger.error(f"GPU {gpu_id} cannot be utilised by PyTorch")
