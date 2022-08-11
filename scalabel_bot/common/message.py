from typing import Any, List, Dict, TypedDict


class ConnectionMessage(TypedDict, total=False):
    clientId: str
    channel: str
    request: str


class TaskMessage(TypedDict, total=False):
    type: str
    clientId: str
    projectName: str
    taskId: str
    taskType: str
    taskKey: str
    dataSize: int
    items: List[Dict[str, Any]]
    ect: int
    wait: int
    channel: str
    output: object | None
    status: object
