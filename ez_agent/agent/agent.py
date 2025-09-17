# 已停止维护
from deprecated import deprecated

import time
from typing import Self, cast, Literal
from collections.abc import Callable, Generator
from copy import deepcopy
from contextlib import contextmanager
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime._streaming import Stream
from volcenginesdkarkruntime.types.chat.completion_create_params import Thinking
from volcenginesdkarkruntime.types.chat import ChatCompletion, ChatCompletionChunk

from ez_agent.types import (
    AssistantMessageParam,
    MessageContent,
    MessageParam,
    ToolCallParam,
    UserMessageParam,
)
from .base_tool import Tool


@deprecated(reason="Agent is deprecated, please use AsyncAgent instead", version="0.2.0a1", category=DeprecationWarning)
class Agent:

    def __init__(
        self: Self,
        model: str,
        api_key: str,
        base_url: str,
        instructions: str = "",
        tools: list[Tool] | None = None,
        frequency_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        thinking: bool | None = None,
        message_expire_time: int | None = None,
    ) -> None:
        self._tools: dict[str, Tool] | None = {tool.name: tool for tool in tools} if tools else None
        self._client: Ark = Ark(api_key=api_key, base_url=base_url)
        self._api_key: str = api_key
        self._base_url: str = base_url

        self.model: str = model
        self.instructions: str = instructions
        self.messages: list[MessageParam] = [{"role": "system", "content": instructions}]
        self.response_handlers: list[Callable[[AssistantMessageParam], None]] = []
        self.stream_chunk_handlers: list[Callable[[str], None]] = []
        self.tool_call_handlers: list[Callable[[ToolCallParam], None]] = []
        self.reasoning_handlers: list[Callable[[str], None]] = []
        self.stream_reasoning_handlers: list[Callable[[str], None]] = []

        self.frequency_penalty: float | None = frequency_penalty
        self.temperature: float | None = temperature
        self.top_p: float | None = top_p
        self.max_tokens: int | None = max_tokens
        self.max_completion_tokens: int | None = max_completion_tokens
        self.thinking: bool | None = thinking

        self.message_expire_time: int | None = message_expire_time

    @property
    def client(self) -> Ark:
        return self._client

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values()) if self._tools else []

    @tools.setter
    def tools(self, value: list[Tool] | None):
        self._tools = {tool.name: tool for tool in value} if value else None

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name) if self._tools else None

    def send_messages(self) -> AssistantMessageParam:
        thinking_mapping: dict[bool | None, Literal["enabled", "disabled", "auto"]] = {
            True: "enabled",
            False: "disabled",
            None: "auto",
        }
        thinking_param: Thinking = {"type": thinking_mapping.get(self.thinking) or "auto"}
        response: ChatCompletion | Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=([tool.to_dict() for tool in self._tools.values()] if self._tools else None),
            tool_choice="auto" if self._tools else "none",
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
            thinking=thinking_param,
        )
        assert isinstance(response, ChatCompletion)
        result: AssistantMessageParam = cast(AssistantMessageParam, response.choices[0].message.to_dict())
        result["time"] = response.created
        reasoning_content = response.choices[0].message.reasoning_content
        if reasoning_content:
            for reasoning_handler in self.reasoning_handlers:
                reasoning_handler(reasoning_content)
        for response_handler in self.response_handlers:
            response_handler(result)
        return result

    def get_response(self) -> MessageContent | None:
        response: AssistantMessageParam = self.send_messages()
        tool_calls: list[ToolCallParam] | None = (
            cast(list[ToolCallParam], response.get("tool_calls")) if response.get("tool_calls") else None
        )
        self.messages.append(response)
        if tool_calls:
            self.call_tool(tool_calls)
            return self.get_response()
        return response.get("content")  # type: ignore

    def send_messages_stream(self) -> Generator[ChatCompletionChunk, None, None]:
        thinking_mapping: dict[bool | None, Literal["enabled", "disabled", "auto"]] = {
            True: "enabled",
            False: "disabled",
            None: "auto",
        }
        thinking_param: Thinking = {"type": thinking_mapping.get(self.thinking) or "auto"}
        response: ChatCompletion | Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=([tool.to_dict() for tool in self._tools.values()] if self._tools else None),
            tool_choice="auto" if self._tools else "none",
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            thinking=thinking_param,
        )
        assert isinstance(response, Stream)
        for chunk in response:
            if chunk.choices[0].finish_reason == "stop":
                break
            yield chunk

    def get_response_stream(self) -> MessageContent | None:
        response = self.send_messages_stream()
        collected_chunks: list[ChatCompletionChunk] = []
        collected_messages: list[str] = []
        collected_reasoning_messages: list[str] = []
        tool_calls_by_id: dict[int, ToolCallParam] = {}

        for chunk in response:
            collected_chunks.append(chunk)
            if chunk.choices[0].delta.content:
                collected_messages.append(chunk.choices[0].delta.content)
                for stream_chunk_handler in self.stream_chunk_handlers:
                    stream_chunk_handler(chunk.choices[0].delta.content)

            if chunk.choices[0].delta.reasoning_content:
                collected_reasoning_messages.append(chunk.choices[0].delta.reasoning_content)
                for resoning_stream_handler in self.stream_reasoning_handlers:
                    resoning_stream_handler(chunk.choices[0].delta.reasoning_content)

            # 处理工具调用
            if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    call_id = tool_call.index

                    if call_id not in tool_calls_by_id:
                        tool_calls_by_id[call_id] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # 更新工具调用信息
                    current_tool = tool_calls_by_id[call_id]
                    if hasattr(tool_call, "function"):
                        if not tool_call.function:
                            continue
                        function_data = current_tool["function"]
                        if hasattr(tool_call.function, "name") and tool_call.function.name:
                            function_data["name"] = tool_call.function.name

                        if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                            function_data["arguments"] += tool_call.function.arguments

                    if hasattr(tool_call, "id") and tool_call.id:
                        current_tool["id"] = tool_call.id

        # 转换工具调用字典为列表
        tool_calls: list[ToolCallParam] = []
        for tool_call in tool_calls_by_id.values():
            tool_calls.append(tool_call)

        full_content = "".join(collected_messages)
        message: AssistantMessageParam = {
            "role": "assistant",
            "content": full_content,
            "time": collected_chunks[-1].created,
        }

        if collected_reasoning_messages:
            reasoning_content: str = "".join(collected_reasoning_messages)
            for reasoning_handler in self.reasoning_handlers:
                reasoning_handler(reasoning_content)

        for response_handler in self.response_handlers:
            response_handler(message)

        if tool_calls:
            message["tool_calls"] = tool_calls
            self.messages.append(message)
            self.call_tool(tool_calls)

            return self.get_response_stream()
        else:
            self.messages.append(message)
            return message.get("content")  # type: ignore

    def call_tool(self, tool_calls: list[ToolCallParam]) -> None:
        # 因为模型会输出 ture/false 而不是 True/False，所以需要转换
        true: bool = True  # type: ignore
        false: bool = False  # type: ignore
        if not self._tools:
            return
        for tool_call in tool_calls:
            called_tool = self._tools[tool_call["function"]["name"]]
            result = str(called_tool(**eval(tool_call["function"]["arguments"])))

            self.messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"],
                    "time": int(time.time()),
                }
            )
            for tool_call_handler in self.tool_call_handlers:
                tool_call_handler(tool_call)

    def _fold_previous_tool_results(self) -> None:
        if not self._tools:
            return
        for index, _message in enumerate(self.messages):
            if not _message.get("tool_calls"):
                continue
            for tool_call in _message["tool_calls"]:  # type: ignore
                tool_name: str = tool_call["function"]["name"]
                if not self._tools.get(tool_name):
                    continue
                if not self._tools[tool_name].foldable:
                    continue
                for i in range(index + 1, len(self.messages)):
                    if self.messages[i].get("role") != "tool":
                        continue
                    if self.messages[i].get("tool_call_id") == tool_call["id"]:
                        self.messages[i] = {
                            "role": "tool",
                            "content": "The result has been folded",
                            "tool_call_id": tool_call["id"],
                        }
                        break

    def run(
        self,
        content: MessageContent,
        user_name: str | None = None,
        stream: bool = False,
    ) -> str | None:
        if self.message_expire_time:
            self.clear_msg_by_time(self.message_expire_time)
        self._fold_previous_tool_results()
        import time

        user_message: UserMessageParam = {
            "role": "user",
            "content": content,
            "time": int(time.time()),
        }
        if user_name:
            user_message["name"] = user_name
        self.messages.append(user_message)

        if stream:
            return str(self.get_response_stream())
        else:
            return str(self.get_response())

    def save_messages(self, file_path: str) -> None:
        import json

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load_messages(self, file_path: str) -> None:
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

    def copy(self) -> Self:
        """深拷贝，用于多线程安全"""
        _agent = Agent.__new__(self.__class__)
        _agent.__dict__ = deepcopy(self.__dict__)

        _agent.messages = deepcopy(self.messages)
        _agent._tools = self._tools.copy() if self._tools else None
        _agent.tool_call_handlers = self.tool_call_handlers.copy()
        _agent.response_handlers = self.response_handlers.copy()
        _agent.reasoning_handlers = self.reasoning_handlers.copy()
        _agent.stream_chunk_handlers = self.stream_chunk_handlers.copy()
        _agent.stream_reasoning_handlers = self.stream_reasoning_handlers.copy()
        return _agent

    @contextmanager
    def safe_modify(self, merge_messages: bool = True) -> Generator[Self, None, None]:
        """
        线程安全地更改messages，会在一轮对话结束后再追加更新的消息，并且不会改变其他属性。
        注意：过期的消息仍然会被清理
        """
        if self.message_expire_time:
            self.clear_msg_by_time(self.message_expire_time)
        _agent: Self = self.copy()
        yield _agent
        if merge_messages:
            added_messages: list[MessageParam] = _agent.messages
            for message in self.messages:
                if not message in added_messages:
                    break
                added_messages.remove(message)
            self.messages.extend(added_messages)

    def clear_msg(self) -> None:
        """清空消息，仅保留系统消息"""
        self.messages = [self.messages[0]]

    def clear_msg_by_time(self, expire_time: int) -> None:
        """
        清空消息，仅保留系统消息和最近若干秒内的消息

        :param expire_time: 过期时间，单位为秒
        """
        import time

        for message in self.messages[1:]:
            if int(time.time()) - message.get("time", 0) > expire_time:
                self.messages.remove(message)

    def add_response_handler(self, handler: Callable[[AssistantMessageParam], None]) -> None:
        """添加一个响应处理函数，当收到模型响应时，会调用该函数。函数的第一个（且是唯一一个）参数应当是模型输出的消息，以字典形式返回"""
        self.response_handlers.append(handler)

    def remove_response_handler(self, handler: Callable[[AssistantMessageParam], None]) -> None:
        self.response_handlers.remove(handler)

    def add_stream_chunk_handler(self, handler: Callable[[str], None]) -> None:
        """添加一个流式响应处理函数，当收到模型响应时，会调用该函数。只有在stream=True时，才会生效。函数的第一个（且是唯一一个）参数应当是模型输出的单个词语，以字符串形式返回"""
        self.stream_chunk_handlers.append(handler)

    def remove_stream_chunk_handler(self, handler: Callable[[str], None]):
        self.stream_chunk_handlers.remove(handler)

    def add_tool_call_handler(self, handler: Callable[[ToolCallParam], None]) -> None:
        """添加一个工具调用处理函数，当收到模型调用请求时，会调用该函数。函数的第一个（且是唯一一个）参数应当是模型的工具调用，以字典形式返回"""
        self.tool_call_handlers.append(handler)

    def remove_tool_call_handler(self, handler: Callable[[ToolCallParam], None]) -> None:
        self.tool_call_handlers.remove(handler)

    def add_reasoning_handler(self, handler: Callable[[str], None]) -> None:
        """添加一个推理处理函数，当收到模型推理请求时，会调用该函数。函数的第一个（且是唯一一个）参数应当是模型的推理请求，以字符串形式返回"""
        self.reasoning_handlers.append(handler)

    def remove_reasoning_handler(self, handler: Callable[[str], None]) -> None:
        self.reasoning_handlers.remove(handler)

    def add_stream_reasoning_handler(self, handler: Callable[[str], None]) -> None:
        """添加一个流式推理处理函数，当收到模型推理请求时，会调用该函数。只有在stream=True时，才会生效。函数的第一个（且是唯一一个）参数应当是模型的推理请求，以字符串形式返回"""
        self.stream_reasoning_handlers.append(handler)

    def remove_stream_reasoning_handler(self, handler: Callable[[str], None]) -> None:
        self.stream_reasoning_handlers.remove(handler)
