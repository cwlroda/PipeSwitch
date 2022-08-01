# -*- coding: utf-8 -*-
import json
from time import perf_counter_ns, sleep
from argparse import ArgumentParser
from uuid import uuid4
from queue import Queue
from typing import Dict
from threading import Thread

from scalabel_bot.common.consts import (
    REDIS_HOST,
    REDIS_PORT,
    State,
    Timers,
)

from scalabel_bot.common.logger import logger
from scalabel_bot.common.message import Message
from scalabel_bot.profiling.timer import timer
from scalabel_bot.server.stream import (
    ClientRequestsStream,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="FsDet demo for builtin models")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet152",
        help="Name of model for processing requests. Default is 'resnet152'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size of data. Default is 8.",
    )
    parser.add_argument(
        "--it",
        type=int,
        default=1,
        help="Number of iterations of requests. Default is 1.",
    )
    return parser


class Client:
    @timer(Timers.PERF_COUNTER)
    def __init__(self, model_name, batch_size, num_it) -> None:
        super().__init__()
        self._name: str = self.__class__.__name__
        self._client_id: str = "scalabel"
        self._results_queue: "Queue[Message]" = Queue()
        self._req_server: ClientRequestsStream = ClientRequestsStream(
            idx=self._client_id,
            host=REDIS_HOST,
            port=REDIS_PORT,
            sub_queue=self._results_queue,
        )
        self._model_name: str = model_name
        self._batch_size: int = batch_size
        self._num_it: int = num_it
        self._task_queue: "Queue[Message]" = Queue()
        self._pending_tasks: Dict[str, Message] = {}

    @timer(Timers.PERF_COUNTER)
    def run(self) -> None:
        try:
            self._req_server.run()
            self._prepare_requests()
            while not self._req_server.ready:
                sleep(0.1)
            send_requests = Thread(target=self._send_requests)
            send_requests.daemon = True
            send_requests.start()
            self._receive_results()
        except KeyboardInterrupt:
            self._shutdown()

    @timer(Timers.PERF_COUNTER)
    def _prepare_requests(self) -> None:
        for i in range(self._num_it):
            if self._model_name == "test":
                if i % 3 == 0:
                    model_name = "opt"
                elif i % 2 == 0:
                    model_name = "fsdet"
                else:
                    model_name = "dd3d"
            else:
                model_name = self._model_name
            task_id: str = str(uuid4())
            if model_name == "fsdet":
                task_type = "box2d"
            elif model_name == "dd3d":
                task_type = "box3d"
            elif model_name == "opt":
                task_type = "textgen"
            task_key: str = "projects/bot-batch/saved/000000"
            # task_key: str = "projects/image/saved/000025"

            if model_name == "opt":
                items = [
                    {
                        "prompt": "Paris is the capital city of France.",
                        "length": 100,
                    },
                    {
                        "prompt": (
                            "Computer science is the study of computation and"
                        ),
                        "length": 100,
                    },
                    {
                        "prompt": (
                            "The University of California, Berkeley is a public"
                        ),
                        "length": 100,
                    },
                    {
                        "prompt": (
                            "Ion Stoica is a Romanian-American computer"
                            " scientist specializing in"
                        ),
                        "length": 100,
                    },
                    {
                        "prompt": "Today is a good day and I want to",
                        "length": 100,
                    },
                    {
                        "prompt": "What is the valuation of Databricks?",
                        "length": 100,
                    },
                    {
                        "prompt": "Which country has the most population?",
                        "length": 100,
                    },
                    {
                        "prompt": (
                            "What do you think about the future of"
                            " Cryptocurrency?"
                        ),
                        "length": 100,
                    },
                    {
                        "prompt": (
                            "What do you think about the meaning of life?"
                        ),
                        "length": 100,
                    },
                    {
                        "prompt": "Donald Trump is the president of",
                        "length": 100,
                    },
                ]
            else:
                items = [
                    {
                        "attributes": {},
                        "intrinsics": {
                            "center": [771.31406, 360.79945],
                            "focal": [1590.83437, 1592.79032],
                        },
                        "labels": [],
                        "name": "bot-batch",
                        "sensor": -1,
                        "timestamp": -1,
                        "url": "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/synscapes/img/rgb/1.png",
                        "videoName": "",
                    }
                ]
            task: Message = {
                "type": "inference",
                "clientId": self._client_id,
                "projectName": "bot-batch",
                "taskId": task_id,
                "taskType": task_type,
                "taskKey": task_key,
                "dataSize": 10 if model_name == "opt" else 50,
                "items": items,
                "modelName": model_name,
                "channel": self._req_server.sub_stream,
            }
            self._task_queue.put(task)

    def _send_requests(self) -> None:
        self._start_time = perf_counter_ns()
        while True:
            task: Message = self._task_queue.get()
            self._send_request(task)

    @timer(Timers.THREAD_TIMER)
    def _send_request(self, task) -> None:
        logger.info(
            f"{self._name} {self._client_id}: sending task"
            f" {task['modelName']} {task['taskType']} with id"
            f" {task['taskId']}"
        )
        self._pending_tasks[task["taskId"]] = task
        stream: str = "REQUESTS"
        msg: Dict[str, str] = {"message": json.dumps(task)}
        self._req_server.publish(stream, msg)

    @timer(Timers.PERF_COUNTER)
    def _receive_results(self) -> None:
        task_count = 0
        while task_count != self._num_it:
            result: Message = self._results_queue.get()
            if result["taskId"] in self._pending_tasks.keys():
                if result["status"] == State.SUCCESS.value:
                    logger.success(
                        f"{self._name} {self._client_id}: received task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']} result"
                    )
                    logger.spam(result["output"])
                    task_count += 1
                    logger.info(
                        f"{self._name} {self._client_id}: completed task(s)"
                        f" {task_count}/{self._num_it}"
                    )
                    self._pending_tasks.pop(result["taskId"])
                    # TODO: store results in the client
                else:
                    # TODO: handle failed task
                    # TODO: push failed task back to the task queue and resend
                    logger.error(
                        f"{self._name} {self._client_id}: task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']} failed"
                    )
                    logger.debug(
                        f"{self._name} {self._client_id}: retrying task"
                        f" {result['modelName']} {result['taskType']} with"
                        f" id {result['taskId']}"
                    )
                    self._task_queue.put(self._pending_tasks[result["taskId"]])
        total_time = perf_counter_ns() - self._start_time
        logger.info(f"Total time taken: {total_time / 1000000} ms")
        logger.info(
            f"Average task time: {total_time / self._num_it / 1000000} ms"
        )

    def _shutdown(self) -> None:
        logger.warning(f"{self._name} {self._client_id}: shutting down")


def launch():
    args: ArgumentParser = get_parser().parse_args()
    client = Client(args.model, args.batch_size, args.it)
    client.run()


if __name__ == "__main__":
    launch()
