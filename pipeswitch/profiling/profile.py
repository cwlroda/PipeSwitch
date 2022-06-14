import os
import time
from argparse import ArgumentParser
from colorama import Back, Style

from pipeswitch.common.consts import LATENCY_THRESHOLD, TIMING_LOG_FILE


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="FsDet demo for builtin models")

    parser.add_argument(
        "--sort",
        type=str,
        default="chrono",
        help=(
            "How profiling results are sorted: 'asc', 'desc', or 'chrono'."
            " Default is 'chrono'"
        ),
    )
    return parser


def launch():
    try:
        args: ArgumentParser = get_parser().parse_args()
        last_modified_time = -1

        while True:
            stats = os.stat(TIMING_LOG_FILE)
            if last_modified_time != stats.st_mtime:
                last_modified_time = stats.st_mtime

                with open(
                    file=TIMING_LOG_FILE, mode="r", encoding="utf-8"
                ) as f:
                    timing_list = [line.split("'") for line in f.readlines()]

                    if args.sort == "asc":
                        timing_list.sort(
                            key=lambda x: float(x[2].strip().split(" ")[0]),
                        )
                    elif args.sort == "desc":
                        timing_list.sort(
                            key=lambda x: float(x[2].strip().split(" ")[0]),
                            reverse=True,
                        )

                    os.system("clear")
                    print(
                        f"{'Timestamp':<41}{'Function':<45}{'Time (ms)':>17}\n"
                    )

                    for timing in timing_list:
                        t = float(timing[2].strip().split(" ")[0])

                        if t <= LATENCY_THRESHOLD:
                            color = Back.GREEN
                        elif LATENCY_THRESHOLD < t <= LATENCY_THRESHOLD * 5:
                            color = Back.YELLOW
                        else:
                            color = Back.RED

                        timing[2] = f"{(color + str(t) + Style.RESET_ALL):>25}"
                        print(f"{timing[0]} {timing[1]:<45} {timing[2]}")

            time.sleep(1)

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    launch()
