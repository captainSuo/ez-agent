from collections.abc import Mapping, Iterable
from typing import TypeAlias, TypedDict, NotRequired
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


class TimedMessage(TypedDict):

    time: NotRequired[int]


class DeveloperMessageParam(TimedMessage, ChatCompletionDeveloperMessageParam):
    pass


class SystemMessageParam(TimedMessage, ChatCompletionSystemMessageParam):
    pass


class UserMessageParam(TimedMessage, ChatCompletionUserMessageParam):
    pass


class AssistantMessageParam(TimedMessage, ChatCompletionAssistantMessageParam):
    pass


class ToolMessageParam(TimedMessage, ChatCompletionToolMessageParam):
    pass


class FunctionMessageParam(TimedMessage, ChatCompletionFunctionMessageParam):
    pass


MessageParam: TypeAlias = (
    DeveloperMessageParam
    | SystemMessageParam
    | UserMessageParam
    | AssistantMessageParam
    | ToolMessageParam
    | FunctionMessageParam
)

ToolCallParam = ChatCompletionMessageToolCallParam
ContentPartParam = ChatCompletionContentPartParam
MessageContent: TypeAlias = Iterable[ContentPartParam] | str
