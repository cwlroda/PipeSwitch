import time
from enum import Enum
from colorama import Back, Style

from pipeswitch.common.logger import logger

# Redis host address
REDIS_HOST = "127.0.0.1"
# Redis port number
REDIS_PORT = 6379
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


def timer(t: Timers):
    """This function shows the execution time of the function object passed"""

    def timer_wrapper(func):
        if t == Timers.PERF_COUNTER:
            execute_timer = time.perf_counter_ns
        elif t == Timers.PROCESS_TIMER:
            execute_timer = time.process_time_ns
        elif t == Timers.THREAD_TIMER:
            execute_timer = time.thread_time_ns
        else:
            logger.error(f"{func.__qualname__!r}: Timer type not supported")

        def timer_func(*args, **kwargs):
            t1 = execute_timer()
            result = func(*args, **kwargs)
            t2 = execute_timer()
            execution_time = (t2 - t1) / 1000000
            if execution_time <= LATENCY_THRESHOLD:
                color = Back.GREEN
            else:
                color = Back.RED
            logger.verbose(
                f"'{args[0].__class__.__name__}.{func.__name__}':"
                f" {Style.RESET_ALL}{color}"
                f"{execution_time:.6f}{Style.RESET_ALL} ms"
            )
            return result

        return timer_func

    return timer_wrapper
