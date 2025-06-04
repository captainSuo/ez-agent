from collections.abc import Awaitable
import logging
from .base_tool import Tool
from typing import Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, McpError
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self._server_params: dict[str, Any] | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, params: dict[str, Any] | None) -> None:
        if not params:
            return
        if "url" in params:
            await self.connect_to_sse_server(**params)
        elif "command" in params:
            await self.connect_to_stdio_server(**params)
        else:
            raise ValueError(f"Invalid parameters: {params}")

    async def connect_to_stdio_server(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        encoding: str = "utf-8",
    ):
        """Connect to an MCP server

        Args:
            _server_params: parameters for the server
        """

        self._server_params = {
            "command": command,
            "args": args,
            "env": env,
            "cwd": cwd,
            "encoding": encoding,
        }
        _server_params = StdioServerParameters(
            command=command, args=args, env=env, cwd=cwd, encoding=encoding
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(_server_params)
        )
        self.read, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )

        if not self.session:
            raise ValueError("Failed to create session")

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        self.tool_list = [tool.name for tool in tools]
        logger.info(
            f"Successfully connected to MCP server with tools: {self.tool_list}"
        )

    async def connect_to_sse_server(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ):
        """Connect to an MCP server

        Args:
            url: URL of the server
            headers: headers to send with the request
        """
        self._server_params = {
            "url": url,
            "headers": headers,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout,
        }
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(
                url=url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
            )
        )
        self.read, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )

        if not self.session:
            raise ValueError("Failed to create session")

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        self.tool_list = [tool.name for tool in tools]
        logger.info(
            f"Successfully connected to MCP server with tools: {self.tool_list}"
        )

    @property
    def available_tools(self) -> list["MCPTool"]:
        """Get a list of available tools"""
        return [MCPTool(tool, self) for tool in self.tool_list]

    async def reconnect(self):
        """Reconnect to the server"""
        logger.info(f"Reconnecting to MCP server: {self._server_params}")
        if not self.session:
            raise ValueError("MCPClient not connected yet")
        await self.cleanup()
        await self.connect_to_server(self._server_params)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        logger.info("MCPClient cleaned up")


class MCPTool(Tool):
    def __init__(self, name: str, client: MCPClient):
        self.name: str = name
        self.client: MCPClient = client
        self._initialized: bool = False

    async def init(self) -> None:
        """初始化工具信息，否则语言模型无法正确调用"""
        if not self.client.session:
            raise ValueError("MCPClient not connected")
        tool_list = (await self.client.session.list_tools()).tools
        for tool in tool_list:
            if tool.name == self.name:
                self.description: str = tool.description if tool.description else ""
                self.parameters: dict[str, Any] = tool.inputSchema
                self._initialized = True
                break
        else:
            raise ValueError(f"Tool {self.name} not found")

    def to_dict(self) -> dict[str, Any]:
        if not self._initialized:
            raise ValueError(f"Tool '{self.name}' not initialized")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return (
            f"MCPTool("
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"parameters={self.parameters!r})"
        )

    def __call__(self, *args, **kwargs) -> Awaitable[str]:
        return self._call(*args, **kwargs)

    async def _call(self, *args, **arguments) -> str:
        if args:
            raise ValueError(
                "MCPTool does not support positional arguments, try keyword arguments"
            )
        if not self.client.session:
            raise ValueError("MCPClient not connected")
        try:
            result = await self.client.session.call_tool(
                name=self.name, arguments=arguments
            )
        except McpError:
            try:
                logging.warning("MCP request failed, trying to reconnect to MCP server")
                await self.client.reconnect()
                result = await self.client.session.call_tool(
                    name=self.name, arguments=arguments
                )
            except McpError as e:
                logger.exception("MCP request failed after reconnecting")
                return f"MCP request failed after reconnecting: {e}"
        except Exception as e:
            logger.exception(f"Error when calling tool: {e}")
            return f"Error when calling tool: {e}"
        return str(result.content)


class FoldableMCPTool(MCPTool):
    foldable = True

    def __init__(self, name: str, client: MCPClient):
        super().__init__(name, client)
        self.description += (
            "This tool is foldable, which means the result of this tool "
            "will be hidden when a new round of conversation starts."
        )
