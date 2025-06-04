from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias, TypedDict, NotRequired


_JSONType: TypeAlias = (
    Mapping[str, "JSONType"] | Iterable["JSONType"] | str | int | float | bool | None
)
JSONType = dict[str, _JSONType] | list[_JSONType]

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
