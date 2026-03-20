import asyncio
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Optional, List, Literal, Dict, Coroutine, Any, Callable

from anthropic.types.beta import BetaThinkingConfigEnabledParam
from ghoshell_common.contracts import Storage, FileStorage, Workspace, LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import Channel, ChannelRuntime, PyChannel, Message, Text
from ghoshell_moss.core.concepts.command import CommandTaskResult
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_ai import Tool, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai_backends import LocalBackend
from pydantic_deep import create_default_deps, create_deep_agent

from framework.abcd.agent_event import ProgramInputAgentEvent
from framework.abcd.agent_hub import EventBus
from framework.apps.volc_websearch import VolcWebsearchChannel


class TaskOutput(BaseModel):
    filenames: List[str] = Field(default_factory=list, description="The filename of the output file")
    summary: str = Field(default="", description="The summary of the task")
    tips: List[str] = Field(default_factory=list, description="The tips of the task")



class TaskOutputWithFileContents(TaskOutput):
    file_contents: Dict[str, str] = Field(description="The file contents of the files output")


class Task(BaseModel):
    id: str = Field(default_factory=uuid, description="The unique id of the task")
    name: str = Field(description="The name of the task")
    prompt: str = Field(description="The prompt of the task")
    status: Literal["doing", "done", "cancelled", "error"] = Field(description="The status of the task")

    _running: asyncio.Task = PrivateAttr()
    output: TaskOutput = Field(default_factory=TaskOutput, description="The output of the task")
    error: str = Field(default="", description="The error output of the task")


class AgentTaskChannel(Channel):

    def __init__(
            self,
            name: str,
            description: str,
            instructions: str,
            tools: List[Tool]=None,
            storage: FileStorage=None,
            task_done_callback: Callable[[Task, TaskOutputWithFileContents], Coroutine[Any, Any, None]]=None,
            logger: LoggerItf=None,
    ) -> None:
        self._id = uuid()
        self._name = name
        self._description = description
        self._tools = tools or []
        self._instructions = instructions
        self._storage = storage
        self.logger = logger or logging.getLogger("AgentTaskChannel")

        root_dir = storage.abspath()
        self.deps = create_default_deps(backend=LocalBackend(root_dir=root_dir))
        self.agent = create_deep_agent(
            instructions=instructions + self.inner_instructions(),
            model="anthropic:doubao-seed-1-6-251015",
            include_todo=True,  # Task planning
            include_filesystem=True,  # File read/write/edit/execute
            include_subagents=True,  # Delegate to subagents
            include_skills=True,  # Domain-specific skills from SKILL.md files
            tools=self._tools,
            output_type=TaskOutput,
            model_settings=AnthropicModelSettings(
                anthropic_thinking=BetaThinkingConfigEnabledParam(
                    budget_tokens=1000000,
                    type="enabled"
                )
            ),
        )

        self.task_done_callback = task_done_callback

        self._runtime: Optional[ChannelRuntime] = None

    def _get_task(self, task_id: str) -> Optional[Task]:
        tasks = self._list_tasks()
        for task in tasks:
            if task.id == task_id:
                return task
        return None

    def _list_tasks(self) -> List[Task]:
        if not self._storage.exists(".tasks.json"):
            return []
        meta_bytes = self._storage.get(".tasks.json")
        ll = json.loads(meta_bytes)
        res = []
        for item in ll:
            res.append(Task.model_validate(item))
        return res

    def _append_tasks(self, task: Task):
        tasks = self._list_tasks()
        tasks.append(task)

        prepare = []
        for t in tasks:
            prepare.append(t.model_dump())

        self._storage.put(".tasks.json", json.dumps(prepare, ensure_ascii=False, indent=4).encode("utf-8"))

    def _update_tasks(self, *tasks: Task):
        current_tasks = self._list_tasks()
        for i, t in enumerate(current_tasks):
            for task in tasks:
                if t.id == task.id:
                    current_tasks[i] = task

        prepare = []
        for t in current_tasks:
            prepare.append(t.model_dump())

        self._storage.put(".tasks.json", json.dumps(prepare, ensure_ascii=False, indent=4).encode("utf-8"))

    @staticmethod
    def inner_instructions() -> str:
        prompts = [
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        return "\n".join(prompts)

    def id(self) -> str:
        return self._id

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    async def _run(self, task: Task):
        try:
            result = await self.agent.run(task.prompt, deps=self.deps)
            task.output = result.output
            task.status = "done"
        except asyncio.CancelledError as e:
            task.status = "cancelled"
            self.logger.warning(f"Task cancelled: {e}")
        except Exception as e:
            task.status = "error"
            task.error = str(traceback.format_exc())
            self.logger.error(f"Error running task {task.id}: {e}")
        finally:
            self._update_tasks(task)
            if self.task_done_callback:
                output = await self.get_output(task_id=task.id)
                await self.task_done_callback(task, output)

    async def start(self, name: str, prompt: str):
        """
        Start a task with the given prompt in background, Once task finished, will call back agent.

        :param name: The name of the task
        :param prompt: The prompt of the task to start
        """
        task = Task(
            name=name,
            prompt=prompt,
            status="doing",
        )
        self._append_tasks(task)
        task._running = asyncio.create_task(self._run(task))

    async def rerun(self, task_id: str) -> CommandTaskResult:
        """
        Rerun the task with the given id.

        :param task_id: The unique id of the task
        """
        task = self._get_task(task_id=task_id)
        if not task:
            return CommandTaskResult(
                result=f"任务{task_id}不存在",
                observe=True,
            )
        task._running = asyncio.create_task(self._run(task))
        task.status = "doing"
        self._update_tasks(task)
        return CommandTaskResult(
            result=f"任务{task_id}已重新运行",
            observe=False,
        )

    async def get_output(self, task_id: str) -> Optional[TaskOutputWithFileContents]:
        """
        Get the output of the task with the given id.

        :param task_id: The unique id of the task
        :return: The output of the task with the given id, if it exists and is done.
        """
        task = self._get_task(task_id=task_id)

        if task:
            file_contents = {}
            for filename in task.output.filenames:
                content = self._storage.get(filename)
                file_contents[filename] = content.decode("utf-8")

            return TaskOutputWithFileContents(
                filenames=task.output.filenames,
                summary=task.output.summary,
                tips=task.output.tips,
                file_contents=file_contents,
            )
        return None

    async def context_messages(self):
        msg = Message.new(role="user", name=f"__agent_task_{self._name}__")

        tasks = self._list_tasks()

        doing_tasks = [task for task in tasks if task.status == "doing"]
        done_tasks = [task for task in tasks if task.status == "done"]

        contents = []
        for task in tasks:
            contents.append(Text(text=f"任务ID：{task.id}，任务名：{task.name}，任务描述：{task.prompt}，状态：{task.status}"))

        msg.with_content(
            Text(text=f"当前正在任务总数: {len(tasks)}，正在处理任务数量: {len(doing_tasks)}，已完成任务数量: {len(done_tasks)}"),
            *contents,
        )
        return [msg]

    async def start_up(self):
        tasks = self._list_tasks()

        updated = []
        for t in tasks:
            if t.status == "doing":
                t.status = "cancelled"
                updated.append(t)

        self._update_tasks(*updated)

    async def close(self):
        tasks = self._list_tasks()

        updated = []
        for t in tasks:
            if t.status == "doing":
                t.status = "cancelled"
                updated.append(t)

        self._update_tasks(*updated)

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime is not None and self._runtime.is_running():
            return self._runtime

        chan = PyChannel(name=self.name(), description=self.description())
        chan.build.command()(self.start)
        chan.build.command()(self.rerun)
        # chan.build.command()(self.get_output)
        chan.build.context_messages(self.context_messages)

        chan.build.start_up(self.start_up)
        chan.build.close(self.close)

        self._runtime = chan.bootstrap(container=container)
        return self._runtime


class AgentTaskChannelProvider(Provider[AgentTaskChannel]):

    def __init__(self, name: str, description: str, instructions: str, agent_id: str):
        self._name = name
        self._description = description
        self._instructions = instructions
        self._agent_id = agent_id

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ws = con.force_fetch(Workspace)
        storage: FileStorage|Storage = ws.runtime().sub_storage("agent_tasks")

        websearch_chan = con.force_fetch(VolcWebsearchChannel)

        async def websearch(ctx: RunContext, query: str) -> CommandTaskResult:
            """Get the latitude and longitude of a location.

            Args:
                ctx: The context.
                query: A description of a location.
            """
            # NOTE: the response here will be random, and is not related to the location description.
            return await websearch_chan.websearch(query)

        tool = Tool(
            function=websearch,
            name=websearch_chan.name(),
            description=websearch_chan.description(),
        )

        eventbus = con.force_fetch(EventBus)
        async def task_done_callback(task: Task, task_output: TaskOutputWithFileContents):
            await eventbus.put(ProgramInputAgentEvent(
                message=Message.new(role="user", name=f"__agent_task_{task.name}__").with_content(
                    Text(text=f"任务ID：{task.id}，任务名：{task.name}，任务描述：{task.prompt}，状态：{task.status}"),
                    Text(text=f"任务总结：{task_output.summary} 任务提示：{task_output.tips}"),
                    *[
                        Text(text=f"任务生成文件名：{filename}，任务生成文件内容：{content}")
                        for filename, content in task_output.file_contents.items()
                    ]
                ),
                agent_id=self._agent_id,
                priority=1,  # 优先
            ))

        logger = con.get(LoggerItf)
        return AgentTaskChannel(
            name=self._name,
            description=self._description,
            instructions=self._instructions,
            tools=[tool],
            storage=storage,
            task_done_callback=task_done_callback,
            logger=logger,
        )

