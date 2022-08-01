from scalabel_bot.common.consts import Timers
from scalabel_bot.profiling.timer import timer


class ModelSummary:
    def __init__(self, mode, devices, model_name, model_class):
        self._mode = mode
        self._devices = devices
        self._model_name = model_name
        self._model_class = model_class
        self._func = None

    @timer(Timers.PERF_COUNTER)
    def execute(self, task, data):
        output = self._func(task, data)
        # self._transfer()
        return output

    @timer(Timers.PERF_COUNTER)
    def load_model(self):
        # (
        #     self._import_data,
        #     self._transfer,
        #     self._func,
        # )
        self._func = self._model_class().import_task(self._devices)

    @timer(Timers.PERF_COUNTER)
    def load_data(self, task):
        # self._transfer()
        # return self._import_data(task)
        return self._model_class().import_data(task)
