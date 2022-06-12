import threading
import time

import numpy
import task.common as util
import task.fsdet as fsdet

TASK_NAME = "fsdet_inference"


class FSDETInference(object):
    def import_data_loader(self):
        return None

    def import_model(self):
        model = fsdet.import_model()
        return model

    def import_func(self):
        def inference(model, data_b):
            print(
                threading.current_thread().name,
                "fsdet inference >>>>>>>>>>",
                time.time(),
                "model status",
                model.model.training,
            )
            with util.timer("fsdet inference func"):
                data = numpy.ndarray((1213, 1546, 3), numpy.uint8, data_b)
                print("loaded data")
                output = model(data)
                print("inference done")
                return output

        return inference

    def import_task(self):
        model = self.import_model()
        func = self.import_func()
        group_list = fsdet.partition_model(model.model)
        shape_list = [util.group_to_shape(group) for group in group_list]
        return model, func, shape_list

    def import_parameters(self):
        model = self.import_model()
        group_list = fsdet.partition_model(model.model)
        # print(group_list)
        batch_list = [util.group_to_batch(group) for group in group_list]
        return batch_list
