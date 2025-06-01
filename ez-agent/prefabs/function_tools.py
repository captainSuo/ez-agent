from ..agent.function_tool import FunctionTool, FoldableFunctionTool


# 预设工具
@FoldableFunctionTool
def python_script_tool(script: str) -> str:
    """A tool that allows you to run python script and returns the full output. Tips: You can use `print` to check the result"""
    import subprocess

    result = subprocess.run(
        ["python", "-E", "-c", script], capture_output=True, text=True
    )
    return result.stdout + result.stderr


@FunctionTool
def get_time_tool() -> str:
    """A tool that can get the current time"""
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@FunctionTool
def open_website_tool(url: str) -> str:
    """A tool that allows user to browse the website"""
    import webbrowser

    webbrowser.open(url)
    return f"Successfully opened {url}"


@FoldableFunctionTool
def browse_web_tool(url: str) -> str:
    """A tool that allows you to browse the website"""
    import httpx, bs4  # type: ignore

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = httpx.get(url, headers=headers)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    return soup.get_text()
