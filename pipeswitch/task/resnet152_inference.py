import threading
import time

import torch

from pipeswitch.common.consts import Timers
from pipeswitch.common.logger import logger
from pipeswitch.profiling.timer import timer
import pipeswitch.task.common as util
from pipeswitch.task.resnet152 import ResNet152


TASK_NAME = "resnet152_inference"


class ResNet152Inference(object):
    def __init__(self) -> None:
        self.resnet152 = ResNet152()

    def import_data_loader(self):
        return None

    def import_model(self):
        model = self.resnet152.import_model()
        model.eval()
        return model

    def import_func(self):
        def inference(device, model, data):
            logger.debug(
                f"{threading.current_thread().name} resnet152 inference"
                f" >>>>>>>>>> {time.time()} model status {model.training}"
            )

            data = torch.FloatTensor(data)
            input_batch = data.view(-1, 3, 224, 224).cuda(
                device=device, non_blocking=True
            )
            with torch.no_grad():
                output = model(input_batch)
            return output.sum().item()

        return inference

    @timer(Timers.PERF_COUNTER)
    def import_task(self):
        model = self.import_model()
        func = self.import_func()
        group_list = self.resnet152.partition_model(model)
        shape_list = [util.group_to_shape(group) for group in group_list]
        return model, func, shape_list

    @timer(Timers.PERF_COUNTER)
    def import_parameters(self):
        model = self.import_model()
        group_list = self.resnet152.partition_model(model)
        # print("partition_model")
        # print(group_list)
        batch_list = [util.group_to_batch(group) for group in group_list]
        # print("group_to_batch")
        return batch_list


MODEL_CLASS = ResNet152Inference
