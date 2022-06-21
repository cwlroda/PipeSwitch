from pipeswitch.common.consts import Timers
from pipeswitch.profiling.timer import timer


class ModelSummary:
    def __init__(self, devices, model_name, model_class):
        """ """
        self._devices = devices
        self._model_name = model_name
        self._model_class = model_class

    @timer(Timers.THREAD_TIMER)
    def execute(self, task, data):
        return self.func(task, data)

    @timer(Timers.THREAD_TIMER)
    def load_model(self):
        (
            self.model,
            self.func,
        ) = self.model_class().import_task(self._devices)

    @timer(Timers.THREAD_TIMER)
    def load_data(self, task):
        return self.model_class().import_data(task)
