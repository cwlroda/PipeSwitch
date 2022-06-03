import contextlib
import threading
import time

import numpy
import task.common as util
import task.fsdet as fsdet
import torch

TASK_NAME = "fsdet_inference"


@contextlib.contextmanager
def timer(prefix):
    _start = time.time()
    yield
    _end = time.time()
    print(prefix, "cost", _end - _start)


def import_data_loader():
    return None


def import_model():
    model = fsdet.import_model()
    return model


def import_func():
    def inference(model, data_b):
        print(
            threading.currentThread().getName(),
            "fsdet inference >>>>>>>>>>",
            time.time(),
            "model status",
            model.model.training,
        )
        with timer("fsdet inference func"):
            data = numpy.ndarray((1213, 1546, 3), numpy.uint8, data_b)
            print("loaded data")
            output = model(data)
            print("inference done")
            return output

    return inference


def import_task():
    model = import_model()
    func = import_func()
    group_list = fsdet.partition_model(model.model)
    shape_list = [util.group_to_shape(group) for group in group_list]
    return model, func, shape_list


def import_parameters():
    model = import_model()
    group_list = fsdet.partition_model(model.model)
    # print(group_list)
    batch_list = [util.group_to_batch(group) for group in group_list]
    return batch_list
