from enum import Enum


class State(Enum):
    def __str__(self):
        return str(self.value)

    """
    Status of the worker or runner.
    """
    STARTUP = "STARTUP"
    # Worker is idle
    IDLE = "IDLE"
    # Worker is running
    BUSY = "BUSY"
    # Worker is terminated
    TERMINATED = "TERMINATED"
