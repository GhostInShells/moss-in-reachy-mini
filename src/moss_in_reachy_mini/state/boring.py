import logging
import math
import random

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from reachy_mini import ReachyMini

from framework.abcd.agent_event import CTMLAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.state.abcd import MiniStateHook


class BoringState(MiniStateHook):

    NAME = "boring"

    def __init__(self, mini: ReachyMini, eventbus: EventBus, logger: LoggerItf=None):
        super().__init__()
        self.mini = mini

        self.eventbus = eventbus
        self.logger = logger or logging.getLogger("BoringState")

        self._time_to_sleep = 30 # 30秒
        self._emotion_prob = 0.03 # 目标：每秒有3%的概率触发函数

    async def on_self_enter(self):
        self.mini.enable_motors()
        # Boring只能靠自己来触发idle move
        await self.start_idle_move()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_sleep:
            await self.eventbus.put(CTMLAgentEvent(
                ctml='<reachy_mini:switch_state state_name="asleep" />'
            ))

        loop_times_per_second = 1 / self._idle_move_elapsed  # 每秒循环的次数
        per_loop_prob = 1 - math.pow(1 - self._emotion_prob, 1 / loop_times_per_second)  # 每次循环的概率
        if per_loop_prob >= 0 and  random.random() < per_loop_prob:
            emotion = random.choice(["sleep1", "boredom1", "boredom2"])
            await self.eventbus.put(CTMLAgentEvent(
                ctml=f'<reachy_mini:emotion name="{emotion}" />'
            ))
            self._emotion_prob -= 0.003  # 每次触发后降低概率

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        self._emotion_prob = 0.03  # 重置触发概率


class BoringStateProvider(Provider[BoringState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> BoringState:
        mini = con.force_fetch(ReachyMini)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)

        return BoringState(
            mini=mini,
            eventbus=eventbus,
            logger=logger,
        )
