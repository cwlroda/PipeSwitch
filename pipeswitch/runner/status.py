class Status(object):
    def __init__(self, device, id, status):
        self.device = device
        self.id = id
        self.status = status


class RunnerStatus(Status):
    def __init__(self, device, id, status):
        super(RunnerStatus, self).__init__(device, id, status)


class WorkerStatus(Status):
    def __init__(self, device, id, status):
        super(WorkerStatus, self).__init__(device, id, status)
