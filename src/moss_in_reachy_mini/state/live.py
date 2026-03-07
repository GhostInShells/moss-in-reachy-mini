import logging
import random

from ghoshell_common.contracts import LoggerItf, Workspace, WorkspaceConfigs, FileStorage, Storage
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import PyChannel, Message, Text
from reachy_mini import ReachyMini

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import ReactAgentEvent, CTMLAgentEvent
from framework.channels.todolist_channel import TodoList
from framework.live.douyin_live import DouyinLive, DouyinLiveConfig
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
            config: DouyinLiveConfig,
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
        self._eventbus = eventbus
        self._config = config
        self._todolist = todolist

        self.douyin_live = DouyinLive(config)

    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.head.reset()
        if self._config.live_id == "":
            await self._eventbus.put(CTMLAgentEvent(
                ctml='<reachy_mini:switch_state state_name="waken" force="true" />'
            ).to_agent_event())
            return

        self.douyin_live.start()

    async def on_self_exit(self):
        self.douyin_live.stop()

    async def _run_idle_move(self):
        if self._todolist:
            todos = self._todolist.todo_todos
            if todos:
                await self._eventbus.put(ReactAgentEvent(
                    messages=[Message.new(role="system").with_content(
                        Text(text=f"你需要继续完成todolist，同时你要关注直播间互动，你需要用很短的话让用户知道当前在干什么")
                    )]
                ).to_agent_event())

        # 检查是否有新的事件
        events = self.douyin_live.get_agent_events()
        for event in events:
            await self._eventbus.put(event.to_agent_event())

        # 检查是否需要触发空闲React
        if self._idle_move_duration > self._config.idle_react_threshold:
            await self._eventbus.put(ReactAgentEvent(
                messages=[Message.new(role="system").with_content(
                    Text(text=random.choice(self._config.idle_prompts))
                )]
            ).to_agent_event())

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text=f"你正在抖音里进行直播，当前观看人数：{self.douyin_live.current_users}，累计观看人数：{self.douyin_live.total_users}")
        )
        return [msg]

    def as_channel(self):
        chan = PyChannel(name="douyin_live", description="当前状态是直播状态，不可以切换为其他状态")
        chan.build.context_messages(self.context_messages)
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
        _storage: FileStorage|Storage = con.force_fetch(Workspace).configs().sub_storage("douyin_live")
        config = WorkspaceConfigs(_storage).get_or_create(DouyinLiveConfig())
        logger = con.get(logging.Logger)
        todolist = con.get(TodoList)

        return LiveState(
            mini=mini,
            body=body,
            head=head,
            antennas=antennas,
            vision=vision,
            eventbus=eventbus,
            config=config,
            todolist=todolist,
            logger=logger,
        )