from enum import Enum


class State(Enum):
    """Status of the worker or runner."""

    def __str__(self) -> str:
        return str(self.value)

    STARTUP = "STARTUP"
    # Worker is idle
    IDLE = "IDLE"
    # Worker has been reserved for a task
    RESERVED = "RESERVED"
    # Worker is running
    BUSY = "BUSY"
    # Worker is terminated
    TERMINATED = "TERMINATED"
