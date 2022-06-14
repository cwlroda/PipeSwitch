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
from redis import exceptions, Redis

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, TIMING_LOG_FILE
from pipeswitch.common.logger import logger
from pipeswitch.manager.manager import PipeSwitchManager


def clear_timing_log():
    with open(TIMING_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")


def launch():
    try:
        logger.info(f"PID: {os.getpid()}")
        clear_timing_log()
        redis = Redis(host=REDIS_HOST, port=REDIS_PORT)
        if not redis.ping():
            logger.warning("Cannot connect to Redis server.")
            logger.warning("Please restart Redis.")
            return
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
    except exceptions.ConnectionError as conn_err:
        logger.error(conn_err)
        logger.warning("Redis server is not running")
        logger.warning("Please start it before running the script.")
    except KeyboardInterrupt as _:
        manager.shutdown()
        # os.system("redis-cli shutdown")


if __name__ == "__main__":
    launch()
