import logging
from openai import NOT_GIVEN, NotGiven
from typing import Self, override
from ..agent.base_tool import Tool
from ..agent.agent_async import AsyncAgent
from ..types import MessageContent

logger = logging.getLogger(__name__)


class DebugAgent(AsyncAgent):

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
        super().__init__(
            model,
            api_key,
            base_url,
            instructions,
            tools,
            frequency_penalty,
            temperature,
            top_p,
            max_tokens,
            max_completion_tokens,
            message_expire_time,
        )
        self.add_response_handler(
            lambda response: logger.debug(f"Agent response: {response!r}")
        )
        self.add_tool_call_handler(
            lambda tool_call: logger.debug(f"Tool called: {tool_call!r}")
        )

    @override
    async def run(
        self: Self,
        content: MessageContent,
        user_name: str | NotGiven = NOT_GIVEN,
        stream: bool = False,
    ) -> str | None:
        try:
            return await super().run(content, user_name, stream)
        except Exception as e:
            logger.exception(f"Error running agent: {e}")
