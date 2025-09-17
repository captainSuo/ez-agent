from PIL import ImageGrab
import os
import base64
import logging
from PIL.Image import Image

logger = logging.getLogger(__name__)


def take_screenshot(save_path: str = "screenshot.png") -> Image | None:
    """
    截取整个屏幕并保存到指定路径

    Args:
        save_path (str, optional): 保存路径。如果为None，则使用默认文件名。

    Returns:
        PIL.Image.Image: 截图图像对象

    Raises:
        Exception: 截图失败时抛出异常
    """
    try:
        # 截取整个屏幕:cite[1]:cite[10]
        screenshot: Image = ImageGrab.grab()

        # 如果提供了保存路径，则保存截图
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            screenshot.save(save_path)
            logging.debug(f"截图已保存至: {save_path}")
        else:
            # 使用默认文件名保存
            screenshot.save("screenshot.png")
            logging.debug("截图已保存至: screenshot.png")

        return screenshot

    except Exception as e:
        logging.exception(f"截图失败: {str(e)}")


def encode_image(image_path: str = "screenshot.png") -> str:
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")
    return image
