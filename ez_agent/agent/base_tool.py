from abc import ABC, abstractmethod
from collections.abc import Awaitable
from openai.types import FunctionParameters
from openai.types.chat import ChatCompletionToolParam


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
