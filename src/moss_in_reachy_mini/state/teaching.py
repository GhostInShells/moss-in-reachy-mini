import logging

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from reachy_mini import ReachyMini

from framework.abcd.agent_event import CTMLAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.state.abcd import MiniStateHook


class TeachingState(MiniStateHook):

    NAME = "teaching"

    def __init__(
        self,
        mini: ReachyMini,
        eventbus: EventBus,
        logger: LoggerItf=None,
    ):
        super().__init__()
        self.mini = mini
        self.logger = logger or logging.getLogger("TeachingState")

        self.eventbus = eventbus

        self._time_to_boring = 60 * 5 # 5分钟


    async def on_self_enter(self):
        self.mini.enable_motors()
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:head reset />"
        ))

    async def on_self_exit(self):
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:head reset />"
        ))

    async def _run_idle_move(self):
        if self.eventbus:
            pass

    async def start_idle_move(self):
        await super().start_idle_move()

    async def cancel_idle_move(self):
        await super().cancel_idle_move()



class TeachingStateProvider(Provider[TeachingState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> TeachingState:
        mini = con.force_fetch(ReachyMini)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)

        return TeachingState(
            mini=mini,
            eventbus=eventbus,
            logger=logger,
        )
