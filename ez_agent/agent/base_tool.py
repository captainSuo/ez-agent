from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any
from ..types import JSONType


class Tool(ABC):
    foldable = False
    name: str = "undefined"
    parameters: dict[str, Any] = {}
    description: str = ""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> str | Awaitable[str]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> JSONType:
        pass
