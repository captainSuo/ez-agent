from ..agent.function_tool import (
    FunctionTool,
    FoldableFunctionTool,
    AsyncFunctionTool,
)


# 预设工具
@AsyncFunctionTool
async def python_script_tool(script: str) -> str:
    """
    A tool that allows you to run python script and returns the full output. Tips: You can use `print` to check the result
    Using os, sys and subprocess is not allowed. File operations are not allowed.
    All cases of dangerous code is not allowed.
    Running time is limited to 10 seconds. Memeory usage is limited to 32MB.
    """
    import asyncio

    MEMORY_LIMIT_MB = 32
    TIMEOUT = 10

    cmd = ["python", "-E", "-c", script]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=TIMEOUT)

    import locale

    ENCODING = locale.getpreferredencoding()

    return stdout.decode(ENCODING) + stderr.decode(ENCODING)


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


import re


def detect_unsafe_file_operations(code):
    """
    检测不安全的文件操作（主要是写入和删除）
    """
    import re

    unsafe_operations = {
        "file_write_open": [
            # 以写模式打开文件
            r"open\s*\(\s*[^,)]+[^)]*mode\s*=\s*[\'\"][wax\+]",
            r"open\s*\(\s*[^,)]+[^)]*[\'\"][wax\+][\w\+]*[\'\"]",
        ],
        "file_deletion": [
            # 文件删除操作
            r"os\.(remove|unlink)\s*\(",
            r"Path\s*\([^)]*\)\.unlink\s*\(",
            r"shutil\.rmtree\s*\(",
        ],
        "file_modification": [
            # 文件重命名/移动
            r"os\.rename\s*\(",
            r"shutil\.move\s*\(",
        ],
        "directory_operations": [
            # 目录操作
            r"os\.(mkdir|makedirs|rmdir)\s*\(",
        ],
        "serialization_write": [
            # 序列化写入文件
            r"(json|pickle|yaml)\.dump\s*\([^,)]+,\s*[^,)]+\)",
        ],
    }

    detected = {}

    for category, patterns in unsafe_operations.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, code, re.IGNORECASE)
            if found:
                matches.extend(found)
        if matches:
            detected[category] = list(set(matches))

    return detected


def has_unsafe_file_operations(code):
    """
    检查代码是否包含不安全的文件操作
    """
    detected = detect_unsafe_file_operations(code)
    return len(detected) > 0


@AsyncFunctionTool
async def python_script_tool_safer(script: str) -> str:
    """
    A tool that allows you to run python script and returns the full output. Tips: You can use `print` to check the result
    Using os, sys and subprocess is not allowed. File operations are not allowed.
    All cases of dangerous code is not allowed.
    Running time is limited to 10 seconds. Memeory usage is limited to 32MB.
    """
    import asyncio, re, psutil

    pattern = r"\b(os|sys|subprocess)\b"
    if re.search(pattern, script):
        return "Invalid script: using os, sys and subprocess is not allowed."

    if has_unsafe_file_operations(script):
        return "Invalid script: file operations are not allowed."

    MEMORY_LIMIT_MB = 32
    TIMEOUT = 10

    cmd = ["python", "-E", "-c", script]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    memory_exceeded = False

    async def monitor_memory_async():
        nonlocal memory_exceeded
        """异步监控内存使用"""
        while process.returncode is None:
            try:
                # 获取内存信息
                ps_process = psutil.Process(process.pid)
                memory_info = ps_process.memory_info()

                if memory_info.rss > MEMORY_LIMIT_MB * 1024 * 1024:
                    process.terminate()
                    memory_exceeded = True
                    break

                await asyncio.sleep(0.1)  # 每100ms检查一次
            except (psutil.NoSuchProcess, ProcessLookupError):
                break

    # 创建监控任务
    monitor_task = asyncio.create_task(monitor_memory_async())

    try:
        # 等待进程完成或超时
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=TIMEOUT
        )

        # 取消监控任务
        monitor_task.cancel()

        if memory_exceeded:
            return f"Memory exceeded the limit"

        import locale

        ENCODING = locale.getpreferredencoding()

        return stdout.decode(ENCODING) + stderr.decode(ENCODING)

    except asyncio.TimeoutError:
        process.terminate()
        monitor_task.cancel()
        await process.wait()
        return "Time exceeded the limit"
