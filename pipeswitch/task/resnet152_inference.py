import contextlib
import threading
import time

import numpy
import torch

import pipeswitch.task.common as util
import pipeswitch.task.resnet152 as resnet152
from pipeswitch.common.logger import logger

TASK_NAME = "resnet152_inference"


@contextlib.contextmanager
def timer(prefix):
    _start = time.time()
    yield
    _end = time.time()
    logger.debug(f"{prefix} cost {_end - _start}")


def import_data_loader():
    return None


def import_model():
    model = resnet152.import_model()
    model.eval()
    return model


def import_func():
    def inference(device, model, data):
        logger.debug(
            f"{threading.currentThread().getName()} resnet152 inference"
            f" >>>>>>>>>> {time.time()} model status {model.training}"
        )

        with timer("resnet152 inference func"):
            data = torch.FloatTensor(data)
            input_batch = data.view(-1, 3, 224, 224).cuda(
                device=device, non_blocking=True
            )
            with torch.no_grad():
                output = model(input_batch)
            return output.sum().item()

    return inference


def import_task():
    model = import_model()
    func = import_func()
    group_list = resnet152.partition_model(model)
    shape_list = [util.group_to_shape(group) for group in group_list]
    return model, func, shape_list


def import_parameters():
    model = import_model()
    group_list = resnet152.partition_model(model)
    # print(group_list)
    batch_list = [util.group_to_batch(group) for group in group_list]
    return batch_list
