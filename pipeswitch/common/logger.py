"""Define logger and its format."""
import coloredlogs
import logging

# FORMAT = (
#     "[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
# )
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG, logger=logger)
