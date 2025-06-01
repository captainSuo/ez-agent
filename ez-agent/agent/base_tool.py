from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    foldable = False
    name: str = "undefined"
    parameters: dict[str, Any] = {}
    description: str = ""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass
