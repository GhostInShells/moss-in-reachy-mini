import asyncio
import os
import sys
from collections import defaultdict
from typing import Optional, List, Dict, Literal

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from ghoshell_moss import Channel, ChannelRuntime, PyChannel
from pydantic import BaseModel, PrivateAttr

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


class Task(BaseModel):
    session_id: str
    status: Literal["running", "finished"]
    system_prompt: str
    prompt: str
    result: str

    _asyncio_task: asyncio.Task = PrivateAttr()

async def read_stream(stream, stream_name):
    """
    正确的流读取函数：循环读取直到EOF，避免管道阻塞
    """
    if stream is None:
        return
    # 按块读取流，直到EOF（read()返回空字节串）
    while True:
        try:
            # 每次读取4KB数据，避免一次性读取过大
            chunk = await stream.read(4096)
            if not chunk:  # 读到EOF，流已关闭
                break
            # 解码输出（根据系统编码调整，默认utf-8，兼容中文/特殊字符）
            text = chunk.decode('utf-8', errors='replace')
            # 打印到终端（也可根据需求保存到变量）
            print(f"[{stream_name}] {text}", end='', flush=True)
        except Exception as e:
            print(f"读取{stream_name}出错: {e}", file=sys.stderr)
            break


class ClaudeChannel(Channel):

    def __init__(
            self,
            name: str,
            description: str,
            system_prompt: str="",
            dangerously_skip_permissions: bool=False,
    ):
        self._uid = uuid()
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._dangerously_skip_permissions = dangerously_skip_permissions
        self._runtime: Optional[ChannelRuntime] = None

        self._tasks = defaultdict(lambda: Task(
            session_id="",
            status="running",
            system_prompt=self._system_prompt,
            prompt="",
            result="",
        ))

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._description

    async def run(self, prompt: str, session_id: str=""):
        """
        claude -p {prompt} --json [--session-id {session_id}] [--allow-dangerously-skip-permissions]

        :param prompt:
        :param session_id:
        """
        first_time = False
        if not session_id:
            first_time = True
            session_id = uuid()
        task = self._tasks[session_id]
        if task.status == "running":
            raise ValueError(f"session_id={session_id} is running now")
        task.prompt = prompt
        task.result = ""
        task._asyncio_task = asyncio.create_task(self._run(prompt, session_id, first_time))

    async def _run(self, prompt: str, session_id: str, first_time: bool):
        # With options
        options = ClaudeAgentOptions(
            # system_prompt=self._system_prompt,
            permission_mode="bypassPermissions",
            # env={
            #     "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/compatible",
            #     "ANTHROPIC_AUTH_TOKEN": "1c0898fa-e5d9-4a64-8225-906932461234",
            #     "API_TIMEOUT_MS": "600000",
            #     "ANTHROPIC_MODEL": "deepseek-v3-2-251201",
            #     "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-v3-2-251201",
            #     "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # },
            # output_format="json",
            resume="" if first_time else session_id,
            # cli_path="/opt/homebrew/bin/claude"
        )
        async for message in query(prompt=prompt, options=options):
            print(message)

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_running():
            raise RuntimeError(f"{self._name} already running")

        channel = PyChannel(name=self._name, description=self._description, blocking=True)

        channel.build.command()(self.run)

        self._runtime = channel.bootstrap(container=container)
        return self._runtime

