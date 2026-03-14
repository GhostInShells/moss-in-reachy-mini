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


class EnrollingState(MiniStateHook):

    """
    人脸录入状态
    """

    NAME = "enrolling"
    out_switchable = False

    def __init__(
            self,
            mini: ReachyMini,
            eventbus: EventBus,
            logger: LoggerItf=None,
    ):
        super().__init__()
        self.mini = mini
        self.logger = logger or logging.getLogger("EnrollingState")
        self.eventbus = eventbus

    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:head_reset />"
        ))

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        pass


class EnrollingStateProvider(Provider[EnrollingState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> EnrollingState:
        mini = con.force_fetch(ReachyMini)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)

        return EnrollingState(
            mini=mini,
            eventbus=eventbus,
            logger=logger,
        )