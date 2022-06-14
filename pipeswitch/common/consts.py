from enum import Enum

# Redis host address
REDIS_HOST = "127.0.0.1"
# Redis port number
REDIS_PORT = 6379

# timing log file name
TIMING_LOG_FILE = "profiling/timing_log.txt"
# latency threshold (in ms)
LATENCY_THRESHOLD = 10


class State(Enum):
    """Common status codes."""

    def __str__(self) -> str:
        return str(self.value)

    STARTUP = "STARTUP"
    IDLE = "IDLE"
    RESERVED = "RESERVED"
    BUSY = "BUSY"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"
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


class Timers(Enum):
    def __str__(self) -> str:
        return str(self.value)

    PERF_COUNTER = "PERF_COUNTER"
    PROCESS_TIMER = "PROCESS_TIMER"
    THREAD_TIMER = "THREAD_TIMER"
