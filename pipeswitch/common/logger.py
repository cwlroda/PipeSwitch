"""Define logger and its format."""
import os
import logging
from colorama import Fore, Style
import verboselogs

from pipeswitch.common.consts import TIMING_LOG_FILE


class TimingFileHandler(logging.FileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if not record.levelno == logging.VERBOSE:
            return
        super().emit(record)


class TimingFormatter(logging.Formatter):
    """Logging colored formatter"""

    def __init__(self):
        super().__init__()
        self._formats = {
            logging.VERBOSE: "[%(asctime)s,%(msecs)010.6f]\t%(message)s",
        }

    def format(self, record):
        log_fmt = self._formats.get(record.levelno)
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        return formatter.format(record)


class Formatter(logging.Formatter):
    """Logging colored formatter"""

    def __init__(self):
        super().__init__()
        self._formats = {
            logging.SPAM: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                "  %(levelname)s %(filename)s:%(lineno)d"
                f" %(funcName)s]  %(message)s{Style.RESET_ALL}"
            ),
            logging.DEBUG: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Fore.BLUE} %(levelname)s %(filename)s:%(lineno)d"
                f" %(funcName)s] {Fore.MAGENTA} %(message)s{Style.RESET_ALL}"
            ),
            logging.VERBOSE: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Fore.WHITE} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f"{Style.RESET_ALL} %(message)s{Style.RESET_ALL}"
            ),
            logging.INFO: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Style.RESET_ALL}{Fore.CYAN} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.WARNING: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Style.RESET_ALL}{Fore.YELLOW} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.SUCCESS: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Style.RESET_ALL}{Fore.GREEN} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.GREEN + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.ERROR: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f" {Style.RESET_ALL}{Fore.RED} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.CRITICAL: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)s,%(msecs)010.6f"
                f"{Style.RESET_ALL}{Fore.RED} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
        }

    def format(self, record):
        log_fmt = self._formats.get(record.levelno)
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        return formatter.format(record)


verboselogs.install()

logger = logging.getLogger(__name__)
logger.setLevel(logging.SPAM)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(Formatter())

if not os.path.exists(TIMING_LOG_FILE):
    os.makedirs(os.path.dirname(TIMING_LOG_FILE), exist_ok=True)

file_handler = TimingFileHandler(TIMING_LOG_FILE)
file_handler.setLevel(logging.VERBOSE)
file_handler.setFormatter(TimingFormatter())

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
