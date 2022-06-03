"""Define logger and its format."""
import logging
import coloredlogs  # type: ignore

FORMAT = (
    "[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s]"
    " %(message)s"
)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)
# coloredlogs.install(level=logging.DEBUG, logger=logger)
