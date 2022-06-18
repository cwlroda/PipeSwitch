from tqdm import tqdm

from pipeswitch.common.consts import Timers
from pipeswitch.profiling.timer import timer
from pipeswitch.task.fsdet import FSDET


TASK_NAME = "fsdet_inference"


class FSDETInference:
    def __init__(self) -> None:
        self.fsdet = FSDET()

    def import_data(self, task):
        data = self.fsdet.import_data(task)
        return data

    def import_model(self, device=None):
        self.predictor, model = self.fsdet.import_model(device)
        return model

    def import_func(self):
        def inference(task, data):
            output = [
                self.predictor(img)
                for img in tqdm(
                    data,
                    desc=f"Task {task['task_id']}",
                    leave=True,
                    position=0,
                    unit="items",
                )
            ]
            return output

        return inference

    @timer(Timers.THREAD_TIMER)
    def import_task(self, device):
        model = self.import_model(device)
        func = self.import_func()
        return model, func


MODEL_CLASS = FSDETInference
