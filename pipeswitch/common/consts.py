from enum import Enum
import signal

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
SIGNALS = [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]


class State(Enum):
    """Status of the worker or runner."""

    def __str__(self) -> str:
        return str(self.value)

    # Worker is starting up
    STARTUP = "STARTUP"
    # Worker is idle
    IDLE = "IDLE"
    # Worker has been reserved for a task
    RESERVED = "RESERVED"
    # Worker is running
    BUSY = "BUSY"
    # Worker has completed a task
    SUCCESS = "SUCCESS"
    # Worker has failed to complete a task
    FAILED = "FAILED"
    # Worker ran into some errors
    ERROR = "ERROR"
    # Worker is terminated
    TERMINATED = "TERMINATED"


class ResponseStatus(Enum):
    def __str__(self) -> str:
        return str(self.value)

    OK = "OK"
    WAIT = "WAIT"
    ERROR = "ERROR"


class ConnectionRequest(Enum):
    def __str__(self) -> str:
        return str(self.value)

    CONNECT = "CONNECT"
    DISCONNECT = "DISCONNECT"
