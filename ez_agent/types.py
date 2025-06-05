from collections.abc import Mapping
from typing import Literal, TypeAlias, TypedDict, NotRequired


JSONType: TypeAlias = (
    Mapping[str, "JSONType"] | list["JSONType"] | str | int | float | bool | None
)

MessageContent: TypeAlias = dict[str, list[dict[str, str]]] | str


class ToolCall(TypedDict):
    type: Literal["function"]
    function: dict[str, str]
    id: str


class Message(TypedDict):

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent | None
    name: NotRequired[str]
    time: NotRequired[int]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]
