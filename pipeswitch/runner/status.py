# -*- coding: utf-8 -*-
"""PipeSwitch Status

This module creates a status class for each runner,
and stores the state of each object.

Todo:
    * None
"""

from pipeswitch.common.consts import State


class RunnerStatus(object):
    """Status class.

    Records the unique id and state of each runner.

    Attributes:
        device (`int`): Unique id of the runner or worker.
        status (`State`): State of the runner or worker.
    """

    def __init__(self, device: int, status: State):
        self._device: int = device
        self._status: State = status

    @property
    def device(self) -> int:
        return self._device

    @property
    def status(self) -> State:
        return self._status
