from tqdm import tqdm
import numpy as np
from threading import Thread

from scalabel_bot.common.consts import Timers
from scalabel_bot.profiling.timer import timer
from scalabel_bot.task.dd3d import DD3D


TASK_NAME = "dd3d_inference"


class DD3DInference:
    def __init__(self) -> None:
        self.dd3d = DD3D()

    def import_model(self, device=None):
        self.dd3d.import_model(device, "inference")

    def import_data(self, task):
        data = self.dd3d.import_data(task)
        return data

    def import_func(self):
        def import_data(task):
            data = self.dd3d.import_data(task)
            return data

        def inference(task, data):
            output = []
            for img in tqdm(
                np.array_split(data, len(data)),
                desc=f"Task {task['projectName']}_{task['taskId']}",
                leave=True,
                position=0,
                unit="items",
            ):
                output.extend(self.dd3d(img))
            return output

        def transfer_model():
            transfer_model = Thread(
                target=self.dd3d.transfer_model, args=("inference",)
            )
            transfer_model.daemon = True
            transfer_model.start()

        return inference
        return import_data, transfer_model, inference

    @timer(Timers.THREAD_TIMER)
    def import_task(self, device):
        self.import_model(device)
        return self.import_func()


MODEL_CLASS = DD3DInference
