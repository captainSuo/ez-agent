from abc import ABC, abstractmethod
from typing import TypeAlias
from collections.abc import Awaitable
from volcenginesdkarkruntime.types.chat import ChatCompletionToolParam

FunctionParameters: TypeAlias = dict[str, object]


class Tool(ABC):
    foldable = False
    name: str = "undefined"
    parameters: FunctionParameters = {}
    description: str = ""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> str | Awaitable[str]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> ChatCompletionToolParam:
        pass
