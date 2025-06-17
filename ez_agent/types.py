from collections.abc import Mapping, Iterable
from typing import TypeAlias, NotRequired
from openai.types.chat import (
    ChatCompletionDeveloperMessageParam,
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


class DeveloperMessageParam(ChatCompletionDeveloperMessageParam):
    time: NotRequired[int]


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
    DeveloperMessageParam
    | SystemMessageParam
    | UserMessageParam
    | AssistantMessageParam
    | ToolMessageParam
    | FunctionMessageParam
)

ToolCallParam: TypeAlias = ChatCompletionMessageToolCallParam
ContentPartParam: TypeAlias = ChatCompletionContentPartParam
MessageContent: TypeAlias = Iterable[ContentPartParam] | str
