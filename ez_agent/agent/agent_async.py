import logging, time
from typing import Self, cast
from collections.abc import AsyncGenerator, Awaitable, Callable
from copy import deepcopy
from contextlib import asynccontextmanager
from openai import AsyncOpenAI, NOT_GIVEN, AsyncStream, NotGiven
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call_param import Function
from .base_tool import Tool
from .mcp_tool import MCPClient
from ..types import (
    JSONType,
    AssistantMessageParam,
    MessageContent,
    MessageParam,
    ToolCallParam,
    ToolMessageParam,
    UserMessageParam,
)

logger = logging.getLogger(__name__)


class AsyncAgent:

    def __init__(
        self: Self,
        model: str,
        api_key: str,
        base_url: str,
        instructions: str = "",
        tools: list[Tool] | None = None,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        message_expire_time: int | None = None,
    ) -> None:
        self._tools: dict[str, Tool] | None = (
            {tool.name: tool for tool in tools} if tools else None
        )
        self._client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._api_key: str = api_key
        self._base_url: str = base_url

        self.model: str = model
        self.instructions: str = instructions
        self.messages: list[MessageParam] = [
            {"role": "system", "content": instructions}
        ]
        self.response_handlers: list[
            Callable[[AssistantMessageParam], Awaitable[None] | None]
        ] = []
        self.stream_chunk_handlers: list[Callable[[str], Awaitable[None] | None]] = []
        self.tool_call_handlers: list[
            Callable[[ToolCallParam], Awaitable[None] | None]
        ] = []
        self._mcp_clients: list[MCPClient] = []

        self.frequency_penalty: float | None | NotGiven = frequency_penalty
        self.temperature: float | None | NotGiven = temperature
        self.top_p: float | None | NotGiven = top_p
        self.max_tokens: int | None | NotGiven = max_tokens
        self.max_completion_tokens: int | None | NotGiven = max_completion_tokens

        self.message_expire_time: int | None = message_expire_time

    @property
    def client(self) -> AsyncOpenAI:
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

    async def send_messages(self) -> AssistantMessageParam:
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=(
                [tool.to_dict() for tool in self._tools.values()]
                if self._tools
                else NOT_GIVEN
            ),
            tool_choice="auto" if self._tools else "none",
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=False,
        )
        result: AssistantMessageParam = cast(
            AssistantMessageParam, response.choices[0].message.to_dict()
        )
        result["time"] = response.created
        for response_handler in self.response_handlers:
            awaitable: Awaitable[None] | None = response_handler(result)
            if awaitable:
                await awaitable
        return result

    async def get_response(self) -> MessageContent | None:
        response: AssistantMessageParam = await self.send_messages()
        tool_calls: list[ToolCallParam] | None = (
            cast(list[ToolCallParam], response.get("tool_calls"))
            if response.get("tool_calls")
            else None
        )
        self.messages.append(response)
        if tool_calls:
            await self.call_tool(tool_calls)
            return await self.get_response()
        return response.get("content")  # type: ignore

    async def send_messages_stream(self) -> AsyncGenerator[ChatCompletionChunk, None]:
        response: AsyncStream[ChatCompletionChunk] = (
            await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=(
                    [tool.to_dict() for tool in self._tools.values()]
                    if self._tools
                    else NOT_GIVEN
                ),
                tool_choice="auto" if self._tools else "none",
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_tokens,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
        )
        async for chunk in response:
            if chunk.choices[0].finish_reason == "stop":
                break
            yield chunk

    async def get_response_stream(self) -> MessageContent | None:
        response: AsyncGenerator[ChatCompletionChunk, None] = (
            self.send_messages_stream()
        )
        collected_chunks: list[ChatCompletionChunk] = []
        collected_messages: list[str] = []
        tool_calls_by_id: dict[int, ToolCallParam] = {}

        async for chunk in response:
            collected_chunks.append(chunk)
            if chunk.choices[0].delta.content:
                collected_messages.append(chunk.choices[0].delta.content)
                for stream_chunk_handler in self.stream_chunk_handlers:
                    awaitable = stream_chunk_handler(chunk.choices[0].delta.content)
                    if awaitable:
                        await awaitable

            # 处理工具调用
            if (
                hasattr(chunk.choices[0].delta, "tool_calls")
                and chunk.choices[0].delta.tool_calls
            ):
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
                        function_data: Function = current_tool["function"]
                        if (
                            hasattr(tool_call.function, "name")
                            and tool_call.function.name
                        ):
                            function_data["name"] = tool_call.function.name

                        if (
                            hasattr(tool_call.function, "arguments")
                            and tool_call.function.arguments
                        ):
                            function_data["arguments"] += tool_call.function.arguments

                    if hasattr(tool_call, "id") and tool_call.id:
                        current_tool["id"] = tool_call.id

        # 转换工具调用字典为列表
        tool_calls: list[ToolCallParam] = []
        for tool_call in tool_calls_by_id.values():
            tool_calls.append(tool_call)

        full_content: str = "".join(collected_messages)
        message: AssistantMessageParam = {
            "role": "assistant",
            "content": full_content,
            "time": collected_chunks[-1].created,
        }
        for response_handler in self.response_handlers:
            awaitable: Awaitable[None] | None = response_handler(message)
            if awaitable:
                await awaitable

        if tool_calls:
            message["tool_calls"] = tool_calls
            self.messages.append(message)
            await self.call_tool(tool_calls)

            return await self.get_response_stream()
        else:
            self.messages.append(message)
            return message.get("content")  # type: ignore

    async def call_tool(self, tool_calls: list[ToolCallParam]) -> None:
        # 记录时间
        time: int | None = self.messages[-1].get("time")
        # 因为模型会输出 ture/false 而不是 True/False，所以需要转换
        true: bool = True  # type: ignore
        false: bool = False  # type: ignore
        if not self._tools:
            return
        for tool_call in tool_calls:
            called_tool = self._tools[tool_call["function"]["name"]]
            result = called_tool(**eval(tool_call["function"]["arguments"]))
            if isinstance(result, Awaitable):
                result = await result

            message: ToolMessageParam = {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call["id"],
            }
            if time:
                message["time"] = time
            self.messages.append(message)
            for tool_call_handler in self.tool_call_handlers:
                awaitable: Awaitable[None] | None = tool_call_handler(tool_call)
                if awaitable:
                    await awaitable

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

    async def run(
        self: Self,
        content: MessageContent,
        user_name: str | NotGiven = NOT_GIVEN,
        stream: bool = False,
    ) -> str | None:
        if self.message_expire_time:
            self.clear_msg_by_time(self.message_expire_time)
        self._fold_previous_tool_results()

        user_message: UserMessageParam = {
            "role": "user",
            "content": content,
            "time": int(time.time()),
        }
        if user_name:
            user_message["name"] = user_name
        self.messages.append(user_message)

        if stream:
            return str(await self.get_response_stream())
        else:
            return str(await self.get_response())

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
        _agent = AsyncAgent.__new__(self.__class__)
        _agent._tools = self._tools
        _agent._client = self._client
        _agent._api_key = self._api_key
        _agent._base_url = self._base_url

        _agent.model = self.model
        _agent.instructions = self.instructions

        _agent.frequency_penalty = self.frequency_penalty
        _agent.temperature = self.temperature
        _agent.top_p = self.top_p
        _agent.max_tokens = self.max_tokens
        _agent.max_completion_tokens = self.max_completion_tokens
        _agent.message_expire_time = self.message_expire_time

        _agent.messages = deepcopy(self.messages)
        _agent.response_handlers = self.response_handlers.copy()
        _agent.stream_chunk_handlers = self.stream_chunk_handlers.copy()
        _agent.tool_call_handlers = self.tool_call_handlers.copy()
        return _agent

    @asynccontextmanager
    async def safe_modify(self, merge_messages: bool = True) -> AsyncGenerator[Self]:
        """
        线程安全地更改messages，会在一轮对话结束后再追加更新的消息，并且不会改变其他属性。
        注意：过期的消息仍然会被清理
        """
        if self.message_expire_time:
            self.clear_msg_by_time(self.message_expire_time)
        _agent: "AsyncAgent" = self.copy()
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

    async def connect_to_mcp_server(self, params: dict[str, JSONType]) -> None:
        """连接到MCP服务器"""
        mcp_client = MCPClient()
        self._mcp_clients.append(mcp_client)
        await mcp_client.connect_to_server(params)
        for tool in mcp_client.available_tools:
            await tool.init()
            if self._tools:
                self._tools[tool.name] = tool
            else:
                self._tools = {tool.name: tool}

    async def load_mcp_config(self, config_file: str) -> None:
        """加载MCP配置文件"""
        import json, os

        if not os.path.exists(config_file):
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(file=config_file, mode="w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return

        with open(file=config_file, mode="r", encoding="utf-8") as f:
            config: JSONType = json.load(f)
        assert isinstance(config, dict), "config file must be a json object"
        if not config.get("mcpServers"):
            return
        for params in config.get("mcpServers").values():  # type: ignore
            assert isinstance(params, dict), "mcpServers params must be a dict"
            await self.connect_to_mcp_server(params)
        logger.info(f"Loaded MCP config from {config_file}")

    async def cleanup(self) -> None:
        """清理连接，释放资源"""
        if not self._mcp_clients:
            return
        for mcp_client in self._mcp_clients:
            await mcp_client.cleanup()
        self._mcp_clients.clear()
        logger.info(f"MCP clients cleaned up")

    def add_response_handler(
        self, handler: Callable[[AssistantMessageParam], Awaitable[None] | None]
    ) -> None:
        """添加一个响应处理函数，当收到模型响应时，会调用该函数。函数的第一个（且是唯一一个）参数应当是模型输出的消息，以字典形式返回"""
        self.response_handlers.append(handler)

    def remove_response_handler(
        self, handler: Callable[[AssistantMessageParam], Awaitable[None] | None]
    ) -> None:
        self.response_handlers.remove(handler)

    def add_stream_chunk_handler(
        self, handler: Callable[[str], Awaitable[None] | None]
    ) -> None:
        """添加一个流式响应处理函数，当收到模型响应时，会调用该函数。只有在stream=True时，才会生效。函数的第一个（且是唯一一个）参数应当是模型输出的单个词语，以字符串形式返回"""
        self.stream_chunk_handlers.append(handler)

    def remove_stream_chunk_handler(
        self, handler: Callable[[str], Awaitable[None] | None]
    ) -> None:
        self.stream_chunk_handlers.remove(handler)

    def add_tool_call_handler(
        self, handler: Callable[[ToolCallParam], Awaitable[None] | None]
    ) -> None:
        """添加一个工具调用处理函数，当收到模型调用请求时，会调用该函数。函数的第一个（且是唯一一个）参数应当是模型的工具调用，以字典形式返回"""
        self.tool_call_handlers.append(handler)

    def remove_tool_call_handler(
        self, handler: Callable[[ToolCallParam], Awaitable[None] | None]
    ) -> None:
        self.tool_call_handlers.remove(handler)
