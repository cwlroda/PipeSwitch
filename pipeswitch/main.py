# -*- coding: utf-8 -*-
"""PipeSwitch Run Script

This module is the main entry point for the PipeSwitch Manager.

Run this file from the main directory of the project::

    $ python3 main.py <num_gpus>

For profiling, run:

    $ py-spy top --pid <PID> --subprocesses

Todo:
    * None
"""

import os
import sys
import torch.multiprocessing as mp

from pipeswitch.common.logger import logger
from pipeswitch.manager.manager import Manager
import traceback


def launch():
    try:
        logger.info(f"PID: {os.getpid()}")
        os.system("redis-server redis.conf")
        mp.set_start_method("forkserver")
        num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        manager: Manager = Manager(num_gpus=num_gpus)
        manager.run()
    except TypeError as type_err:
        logger.warning(type_err)
        logger.warning(traceback.format_exc())
        logger.warning("Handling stray event loops")

    logger.info("Main: Exiting")


if __name__ == "__main__":
    launch()
