import logging
import math
import random

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Text, Message, PyChannel
from reachy_mini import ReachyMini

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import CTMLAgentEvent
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.vision import Vision
from moss_in_reachy_mini.state.abcd import MiniStateHook


class BoringState(MiniStateHook):

    NAME = "boring"

    def __init__(self, mini: ReachyMini, body: Body, vision: Vision, eventbus: EventBus, logger: LoggerItf=None):
        super().__init__()
        self.mini = mini
        self.body = body
        self.vision = vision

        self.eventbus = eventbus

        self.logger = logger or logging.getLogger("BoringState")

        self._time_to_sleep = 30 # 30秒
        self._emotion_prob = 0.03 # 目标：每秒有3%的概率触发函数

    async def on_self_enter(self):
        # Boring只能靠自己来触发idle move
        await self.start_idle_move()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_sleep:
            await self.eventbus.put(CTMLAgentEvent(
                ctml='<reachy_mini:switch_state state_name="asleep" />'
            ).to_agent_event())

        loop_times_per_second = 1 / self._idle_move_elapsed  # 每秒循环的次数
        per_loop_prob = 1 - math.pow(1 - self._emotion_prob, 1 / loop_times_per_second)  # 每次循环的概率
        if per_loop_prob >= 0 and  random.random() < per_loop_prob:
            emotion = random.choice(["sleep1", "boredom1", "boredom2"])
            await self.body.emotion(emotion)
            self._emotion_prob -= 0.003  # 每次触发后降低概率

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        self._emotion_prob = 0.03  # 重置触发概率

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text="你现在处于Boring状态"),
        )
        vision_message = await self.vision.context_messages()
        return [msg] + vision_message

    def as_channel(self):
        boring_chan = PyChannel(name=BoringState.NAME, description=f"current state is boring", block=True)
        boring_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        boring_chan.build.command()(self.vision.look)
        boring_chan.build.with_context_messages(self.context_messages)
        return boring_chan

class BoringStateProvider(Provider[BoringState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> BoringState:
        mini = con.force_fetch(ReachyMini)
        body = con.force_fetch(Body)
        vision = con.force_fetch(Vision)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)

        return BoringState(
            mini=mini,
            body=body,
            vision=vision,
            eventbus=eventbus,
            logger=logger,
        )
