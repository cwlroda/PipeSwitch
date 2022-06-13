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
from pipeswitch.manager.manager import PipeSwitchManager


def launch():
    try:
        logger.info(f"PID: {os.getpid()}")
        # os.system("redis-server redis.conf")
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        mode = sys.argv[1]
        num_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        manager: PipeSwitchManager = PipeSwitchManager(
            mode=mode, num_gpus=num_gpus
        )
        manager.daemon = True
        manager.start()
        manager.join()
    except KeyboardInterrupt as _:
        manager.shutdown()


if __name__ == "__main__":
    launch()
