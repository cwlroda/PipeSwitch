import threading
import time

from pipeswitch.common.consts import Timers
from pipeswitch.common.logger import logger
from pipeswitch.profiling.timer import timer
import pipeswitch.task.common as util
from pipeswitch.task.fsdet import FSDET


TASK_NAME = "fsdet_inference"


class FSDETInference:
    def __init__(self) -> None:
        self.fsdet = FSDET()

    def import_data(self, task_key):
        data = self.fsdet.import_data(task_key)
        return data

    def import_model(self, device=None):
        model = self.fsdet.import_model(device)
        return model

    def import_func(self):
        def inference(model, data):
            logger.debug(
                f"{threading.current_thread().name} fsdet inference >>>>>>>>>>"
                f" {time.time()} model status {model.training}"
            )
            output = self.fsdet(data)
            return output

        return inference

    @timer(Timers.PERF_COUNTER)
    def import_task(self, device):
        model = self.import_model(device)
        func = self.import_func()
        group_list = self.fsdet.partition_model(model)
        shape_list = [util.group_to_shape(group) for group in group_list]
        return model, func, shape_list

    @timer(Timers.PERF_COUNTER)
    def import_parameters(self):
        model = self.import_model()
        group_list = self.fsdet.partition_model(model)
        # print(group_list)
        batch_list = [util.group_to_batch(group) for group in group_list]
        return batch_list


MODEL_CLASS = FSDETInference
