# -*- coding: utf-8 -*-
"""PipeSwitch Status

This module creates a status class for each runner and worker,
and stores the state of each object.

Todo:
    * None
"""

from pipeswitch.common.consts import State


class RunnerStatus(object):
    """Status class.

    Records the unique id and state of each runner and worker.

    Attributes:
        device (`int`): Unique id of the runner or worker.
        status (`State`): State of the runner or worker.

        worker_id (`int`, optional):
            Unique id of the worker (if multiple workers are instantiated).
            Defaults to -1.
    """

    def __init__(
        self,
        device: int,
        status: State,
        worker_id: int = -1,
    ):
        self._device: int = device
        self._worker_id: int = worker_id
        self._status: State = status

    @property
    def device(self) -> int:
        return self._device

    @property
    def worker_id(self) -> int:
        return self._worker_id

    @property
    def status(self) -> State:
        return self._status
