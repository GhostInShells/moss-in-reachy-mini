import logging

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Message, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_event import ProgramInputAgentEvent, CTMLAgentEvent
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.apps.live.douyin_live import DouyinLive
from framework.apps.todolist import TodoList
from moss_in_reachy_mini.state import BaseAgentHook


class LiveState(BaseAgentHook):

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
        self.logger = logger or logging.getLogger("LiveState")
        self.eventbus = eventbus
        self.todolist = todolist

    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.douyin_live.resume()
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:head_reset idle_mode=\"breathing\" />"
        ))

    async def on_self_exit(self):
        await self.douyin_live.pause()

    async def _run_idle_move(self):
        message = Message.new(role="user", name="__douyin_live__")
        # 5秒内被打断就略过
        if self._idle_move_duration < self.douyin_live.config.idle_task_threshold:
            return
        # ============ 添加直播间的事件更新信息 ============
        recent_unprocessed_events = await self.douyin_live.get_unprocessed_events()
        if recent_unprocessed_events:
            message.with_content(
                Text(text="====== 抖音直播间事件 start ======"),
                Text(text=self.douyin_live.config.idle_task_prompt),
                Text(text=f"\n发现{len(recent_unprocessed_events)}个未处理事件，请回应："),
                Text(text=f"当前在线人数：{self.douyin_live.current_users}"),
                Text(text=f"事件列表："),
                *[Text(text=f"- {event.to_natural()}") for event in recent_unprocessed_events],
                Text(text=f"\n请对以下事件进行回应。"),
                Text(text="====== 抖音直播间事件 end ======"),
            )
        if not message.is_empty():
            await self.eventbus.put(ProgramInputAgentEvent(
                message=message,
                agent_id="", # 默认主脑
                priority=0,  # 普通队列，可被高优事件打断
                overdue=20,
            ))

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