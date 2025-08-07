from collections.abc import Mapping, Iterable
from typing_extensions import TypeAlias, NotRequired
from volcenginesdkarkruntime.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionContentPartParam,
)


JSONType: TypeAlias = (
    Mapping[str, "JSONType"] | list["JSONType"] | str | int | float | bool | None
)


class SystemMessageParam(ChatCompletionSystemMessageParam):
    time: NotRequired[int]


class UserMessageParam(ChatCompletionUserMessageParam):
    time: NotRequired[int]


class AssistantMessageParam(ChatCompletionAssistantMessageParam):
    time: NotRequired[int]


class ToolMessageParam(ChatCompletionToolMessageParam):
    time: NotRequired[int]


class FunctionMessageParam(ChatCompletionFunctionMessageParam):
    time: NotRequired[int]


MessageParam: TypeAlias = (
    SystemMessageParam
    | UserMessageParam
    | AssistantMessageParam
    | ToolMessageParam
    | FunctionMessageParam
)

ToolCallParam: TypeAlias = ChatCompletionMessageToolCallParam
ContentPartParam: TypeAlias = ChatCompletionContentPartParam
MessageContent: TypeAlias = Iterable[ContentPartParam] | str
