import logging
import random

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import PyChannel, Message, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_hub import EventBus
from framework.abcd.agent_event import UserInputAgentEvent
from framework.apps.live.douyin_live import DouyinLive
from framework.apps.todolist import TodoList
from moss_in_reachy_mini.components.antennas import Antennas
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.components.vision import Vision
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
            body: Body,
            head: Head,
            antennas: Antennas,
            vision: Vision,
            eventbus: EventBus,
            todolist: TodoList=None,
            logger: LoggerItf=None,
    ):
        super().__init__()
        self.mini = mini
        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision
        self.logger = logger or logging.getLogger("WakenState")
        self.eventbus = eventbus
        self.todolist = todolist

    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.head.reset()

    async def on_self_exit(self):
        pass

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

    def as_channel(self):
        chan = PyChannel(name="live", description="当前状态是直播状态，不可以切换为其他状态")
        chan.build.command(doc=self.body.dance_docstring)(self.body.dance)
        chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        chan.build.command(name="head_move")(self.head.move)
        chan.build.command(name="head_reset")(self.head.reset)
        chan.build.command(name="antennas_move")(self.antennas.move)
        chan.build.command(name="antennas_reset")(self.antennas.reset)
        chan.build.idle(self.head.on_idle)
        return chan


class LiveStateProvider(Provider[LiveState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> LiveState:
        mini = con.force_fetch(ReachyMini)
        body = con.force_fetch(Body)
        head = con.force_fetch(Head)
        vision = con.force_fetch(Vision)
        antennas = con.force_fetch(Antennas)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)
        todolist = con.get(TodoList)

        return LiveState(
            mini=mini,
            body=body,
            head=head,
            antennas=antennas,
            vision=vision,
            eventbus=eventbus,
            todolist=todolist,
            logger=logger,
        )