"""Define logger and its format."""
import logging
from colorama import init, Fore, Style


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self.formats = {
            logging.DEBUG: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Fore.BLUE} %(levelname)s %(filename)s:%(lineno)d"
                f" %(funcName)s] {Fore.MAGENTA} %(message)s{Style.RESET_ALL}"
            ),
            logging.INFO: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.GREEN} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
            ),
            logging.WARNING: (
                f"{Fore.WHITE + Style.DIM} [%(asctime)-15s"
                f" {Style.RESET_ALL}{Fore.YELLOW} %(levelname)s"
                " %(filename)s:%(lineno)d %(funcName)s]"
                f" {Fore.MAGENTA + Style.BRIGHT} %(message)s{Style.RESET_ALL}"
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
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


init()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter())

logger.addHandler(stdout_handler)
