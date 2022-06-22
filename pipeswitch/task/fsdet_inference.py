from tqdm import tqdm
from threading import Lock, Thread
import numpy as np

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

    def import_model(self, devices=None):
        self.devices = devices
        self.predictors = self.fsdet.import_model(self.devices)

    def import_func(self):
        def batch_inference(task, data):
            lock = Lock()
            data = list(np.array_split(data, len(self.devices)))
            output_list = []
            jobs = []
            for predictor, data_batch in zip(self.predictors, data):
                prediction = Thread(
                    target=self.inference,
                    args=(
                        task,
                        predictor,
                        data_batch,
                        lock,
                        output_list,
                    ),
                )
                prediction.daemon = True
                prediction.start()
                jobs.append(prediction)
            for prediction in jobs:
                prediction.join()
            return output_list

        return batch_inference

    @timer(Timers.THREAD_TIMER)
    def inference(self, task, predictor, data, lock, output_list):
        output = [
            predictor(img)
            for img in tqdm(
                data,
                desc=f"Task {task['task_id']}",
                leave=True,
                position=0,
                unit="items",
            )
        ]
        lock.acquire()
        output_list.extend(output)
        lock.release()

    @timer(Timers.THREAD_TIMER)
    def import_task(self, devices):
        self.import_model(devices)
        func = self.import_func()
        return func


MODEL_CLASS = FSDETInference
