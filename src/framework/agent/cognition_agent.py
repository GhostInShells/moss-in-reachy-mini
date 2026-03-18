import asyncio
import datetime
from typing import Union, Self, List

from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import MOSSShell, Message, Text

from framework.abcd.agent import AgentConfig, Response
from framework.abcd.agent_event import AgentEvent, ProgramInputAgentEvent
from framework.abcd.agent_hook import AgentHook
from framework.abcd.agent_hub import EventBus
from framework.abcd.session import Session
from framework.agent.agent_hook import BaseAgentHook
from framework.agent.main_agent import BaseMainAgent
from framework.apps.session.storage_session import StorageSession
from framework.apps.todolist import TodoList


class CognitionSession(StorageSession):
    pass


class CognitionAgent(BaseMainAgent):
    """
    Cognition Agent.
    """
    def __init__(self, container: IoCContainer, config: AgentConfig, shell: MOSSShell, main_session: Session, cognition_session: CognitionSession):
        super().__init__(container, config, shell, main_session)

        self.cognition_session = cognition_session

    def _parse_event(self, event: AgentEvent) -> Union[AgentEvent, None]:
        return event

    @classmethod
    def new(cls, container: IoCContainer, config: AgentConfig) -> Self:
        main_session = container.force_fetch(Session)
        shell = container.force_fetch(MOSSShell)
        cognition_session = container.force_fetch(CognitionSession)

        ins = cls(container=container, config=config, shell=shell, main_session=main_session, cognition_session=cognition_session)

        todolist = container.force_fetch(TodoList)
        eventbus = container.force_fetch(EventBus)
        agent_hook = CognitionAgentHook(todolist, eventbus, agent_id=ins.info().id)

        ins.set_state_hook(agent_hook)
        return ins

    async def _finish_response(self, response: Response) -> None:
        """
        仅和主脑共享session，但是自己不主动更新session
        """
        inputs = response.inputted()
        outputs = response.buffered()
        # 判断 outputs 不为空, 就再次保存.
        if inputs or outputs:
            await self.cognition_session.save_turn(inputs, outputs)

    async def make_prompts(self) -> List[Message]:
        """
        语法糖, 用来快速定义 prompt 对象.
        """
        super_prompts = await super().make_prompts()
        cognition_prompts = await self.cognition_session.get_session_history()
        return super_prompts + cognition_prompts


class CognitionAgentHook(BaseAgentHook):
    def __init__(self, todolist: TodoList, eventbus: EventBus, agent_id: str):
        super().__init__()
        self.todolist = todolist
        self.eventbus = eventbus
        self.agent_id = agent_id

    def get_hook(self):
        return self

    async def on_self_enter(self):
        pass

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self.todolist and self.todolist.todo_todos:
            message = Message.new(role="user")
            message.with_content(
                Text(
                    text="按 todolist 顺序执行下一个未完成的叶子任务，并且需要用很短的话让用户知道当前在干什么，任务结果需要用mark_as_done来传递"),
            )
            await self.eventbus.put(ProgramInputAgentEvent(
                message=message,
                priority=0,  # 正常事件队列
                agent_id=self.agent_id,
            ))
            return
