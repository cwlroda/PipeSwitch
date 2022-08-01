from typing import Any, List, Dict, TypedDict


class Message(TypedDict, total=False):
    type: str
    clientId: str
    projectName: str
    taskId: str
    taskType: str
    taskKey: str
    dataSize: int
    items: List[Dict[str, Any]]
    ect: int
    channel: str
    output: object
    status: object
