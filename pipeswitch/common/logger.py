"""Define logger and its format."""
import logging
from colorama import Fore, Style
import verboselogs


class CustomFormatter(logging.Formatter):
    """Logging colored formatter"""

    def __init__(self):
        super().__init__()
        self._formats = {
            logging.SPAM: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                "  %(levelname)s %(filename)s:%(lineno)d"
                f" %(funcName)s]  %(message)s{Style.RESET_ALL}"
            ),
            logging.DEBUG: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Fore.BLUE} %(levelname)s %(filename)s:%(lineno)d"
                f" %(funcName)s] {Fore.MAGENTA} %(message)s{Style.RESET_ALL}"
            ),
            logging.VERBOSE: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Fore.WHITE} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f"{Style.RESET_ALL} %(message)s{Style.RESET_ALL}"
            ),
            logging.INFO: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.CYAN} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.WARNING: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.YELLOW} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.SUCCESS: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.GREEN} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.GREEN + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.ERROR: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.RED} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.CRITICAL: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f"{Style.RESET_ALL}{Fore.RED} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
        }

    def format(self, record):
        log_fmt = self._formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


verboselogs.install()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter())

logger.addHandler(stdout_handler)
