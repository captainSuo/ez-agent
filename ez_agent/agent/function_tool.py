from .base_tool import Tool
from abc import ABC
from typing import Any
from collections.abc import Callable, Awaitable
from openai.types import FunctionDefinition, FunctionParameters
from openai.types.chat import ChatCompletionToolParam
import inspect


class BaseFunctionTool(Tool, ABC):

    def __init__(self, func: Callable[..., str | Awaitable[str]]) -> None:
        if isinstance(func, classmethod):
            raise TypeError(
                "FunctionTool cannot be used as a class method, please use @staticmethod to decorate the function."
            )
        elif not callable(func):
            raise TypeError("FunctionTool must be a callable object.")

        sig = inspect.signature(func)
        props: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = "string"

            # 获取参数类型注解
            if param.annotation != inspect.Parameter.empty:

                # 处理基本类型
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int or param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

                # 处理嵌套类型
                elif hasattr(param.annotation, "__origin__"):
                    origin = param.annotation.__origin__

                    if origin == list:
                        param_type = "array"
                        if hasattr(param.annotation, "__args__"):
                            item_type = param.annotation.__args__[0]
                            item_schema = {}

                            # 处理列表中元素的类型
                            if item_type == str:
                                item_schema = {"type": "string"}
                            elif item_type == int or item_type == float:
                                item_schema = {"type": "number"}
                            elif item_type == bool:
                                item_schema = {"type": "boolean"}
                            elif (
                                item_type == dict
                                or getattr(item_type, "__origin__", None) == dict
                            ):
                                item_schema = {"type": "object"}

                            # 如果项类型已确定，添加 items 字段
                            if item_schema:
                                props[param_name] = {
                                    "type": "array",
                                    "description": f"{param_name} parameter",
                                    "items": item_schema,
                                }
                                continue

                else:  # 其他类型默认为 object
                    param_type = "object"

            props[param_name] = {
                "type": param_type,
                "description": f"{param_name} parameter",
            }

            # 如果有必需的参数，添加到 required 字段
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters: FunctionParameters = {
            "type": "object",
            "properties": props,
        }

        if required:
            parameters["required"] = required

        self.parameters = parameters
        self.description: str = func.__doc__.strip() if func.__doc__ else ""
        self.name = func.__name__
        self.__annotations__ = func.__annotations__
        self.__signature__ = inspect.signature(func)
        self.__module__ = func.__module__
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__


class FunctionTool(BaseFunctionTool):
    def __init__(self, func: Callable[..., str]) -> None:
        super().__init__(func)
        self._func: Callable[..., str] = func

    def __call__(self, *args, **kwds) -> str:
        return self._func(*args, **kwds)

    def __get__(self, obj, objtype):
        @FunctionTool
        def tool(*args, **kwds):
            return self._func(obj, *args, **kwds)

        tool.name = self.name
        tool.parameters = self.parameters
        tool.parameters["properties"] = dict(
            list(self.parameters["properties"].items())[1:]  # type: ignore
        )
        tool.parameters["required"] = self.parameters["required"][1:]  # type: ignore
        tool.description = self.description
        return tool

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name}, description={self.description}, parameters={self.parameters})"

    def to_dict(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FoldableFunctionTool(FunctionTool):
    foldable = True

    def __init__(self, func: Callable):
        super().__init__(func)
        self.description += (
            "This tool is foldable, which means the result of this tool "
            "will be hidden when a new round of conversation starts."
        )


class AsyncFunctionTool(BaseFunctionTool):
    def __init__(self, func: Callable[..., Awaitable[str]]):
        super().__init__(func)
        self._func: Callable[..., Awaitable[str]] = func

    def __call__(self, *args, **kwargs) -> Awaitable[str]:
        return self._func(*args, **kwargs)


class FoldableAsyncFunctionTool(FoldableFunctionTool, AsyncFunctionTool):
    def __init__(self, func: Callable[..., Awaitable[str]]):
        super().__init__(func)
