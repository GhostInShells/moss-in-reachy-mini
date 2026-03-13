import logging

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Message, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_event import UserInputAgentEvent, CTMLAgentEvent
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.apps.live.douyin_live import DouyinLive
from framework.apps.todolist import TodoList
from moss_in_reachy_mini.state import MiniStateHook


class LiveState(MiniStateHook):

    """
    直播状态
    """

    NAME = "live"
    out_switchable = False

    def __init__(
            self,
            mini: ReachyMini,
            douyin_live: DouyinLive,
            eventbus: EventBus,
            todolist: TodoList=None,
            logger: LoggerItf=None,
    ):
        super().__init__()
        self.mini = mini
        self.douyin_live = douyin_live
        self.logger = logger or logging.getLogger("WakenState")
        self.eventbus = eventbus
        self.todolist = todolist

    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.douyin_live.resume()
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:head_reset />"
        ))

    async def on_self_exit(self):
        await self.douyin_live.pause()

    async def _run_idle_move(self):
        if self.todolist and self.todolist.todo_todos:
            message = Message.new(role="user")
            message.with_content(
                Text(text="按 todolist 顺序执行下一个未完成的叶子任务，并且需要用很短的话让用户知道当前在干什么"),
            )
            await self.eventbus.put(UserInputAgentEvent(
                message=message,
                priority=0,  # 正常事件队列
            ))
            return


class LiveStateProvider(Provider[LiveState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> LiveState:
        mini = con.force_fetch(ReachyMini)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)
        todolist = con.get(TodoList)
        douyin_live = con.get(DouyinLive)

        return LiveState(
            mini=mini,
            douyin_live=douyin_live,
            eventbus=eventbus,
            todolist=todolist,
            logger=logger,
        )