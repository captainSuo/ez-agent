import asyncio
from typing import Self
from rich import print
from PIL import Image

from ez_agent.agent.base_tool import Tool
from ..agent.agent_async import AsyncAgent
from ..agent.function_tool import AsyncFunctionTool
from .screenshot import take_screenshot, encode_image
from .action_parser import (
    parsing_response_to_pyautogui_code,
    parse_action_to_structure_output,
)

# logging.basicConfig(level=logging.INFO)

printing_reasoning = False


class GUIAgent(AsyncAgent):
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
            thinking,
            message_expire_time,
        )
        self.add_tool(self.action)

    @AsyncFunctionTool
    async def action(self, action_str: str) -> str:
        """
                Args:
                    action_str (str): action string, using pyautogui

        # Action space:
            click(point='<point>x1 y1</point>')
            left_double(point='<point>x1 y1</point>')
            right_single(point='<point>x1 y1</point>')
            drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
            hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
            type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
            scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
            wait() #Sleep for 5s and take a screenshot to check for any changes.
            finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
        """
        image = Image.open("screenshot.png")
        original_image_width, original_image_height = image.size
        model_type = "doubao"
        parsed_dict = parse_action_to_structure_output(
            action_str, 1000, original_image_height, original_image_width, model_type
        )
        parsed_pyautogui_code: str = parsing_response_to_pyautogui_code(
            parsed_dict, original_image_height, original_image_width
        )
        exec(parsed_pyautogui_code)
        await asyncio.sleep(0.5)
        take_screenshot()
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image()}"}},
        ]
        return content  # type: ignore

    async def start(self):
        try:
            while True:
                input_str = input(">>> ")
                take_screenshot()
                content = [
                    {"type": "text", "text": input_str},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image()}"}},
                ]
                await self.run(content, stream=True)
                self.save_messages("messages.json")
                print()
        finally:
            await self.cleanup()
